import asyncio

import httpx
import pytest

from app.llm_calls import RateLimiter, call_with_retry, rate_limit_info


class FakeClock:
    """Horloge simulée : sleep avance le temps sans attendre."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def _http_error(status: int, retry_after: str | None = None, per_minute: int | None = None) -> httpx.HTTPStatusError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    if per_minute is not None:
        headers |= {"x-ratelimit-limit-req-minute": str(per_minute), "x-ratelimit-remaining-req-minute": "0"}
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError("erreur", request=request, response=response)


class TestRateLimiter:
    def test_starts_are_spaced_by_one_over_rps(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(0.5, 4, clock=clock, sleep=clock.sleep)
        starts: list[float] = []

        async def call() -> None:
            async with limiter.slot():
                starts.append(clock())

        async def main() -> None:
            await asyncio.gather(*(call() for _ in range(4)))

        asyncio.run(main())

        assert starts == [0.0, 2.0, 4.0, 6.0]

    def test_concurrency_is_capped(self) -> None:
        limiter = RateLimiter(0, 2)
        active = peak = 0

        async def call(release: asyncio.Event) -> None:
            nonlocal active, peak
            async with limiter.slot():
                active += 1
                peak = max(peak, active)
                await release.wait()
                active -= 1

        async def main() -> None:
            release = asyncio.Event()
            tasks = [asyncio.create_task(call(release)) for _ in range(5)]
            for _ in range(5):
                await asyncio.sleep(0)
            assert active == 2
            release.set()
            await asyncio.gather(*tasks)

        asyncio.run(main())

        assert peak == 2

    def test_several_calls_can_run_while_starts_stay_spaced(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(1.0, 3, clock=clock, sleep=clock.sleep)
        active = peak = 0

        async def call() -> None:
            nonlocal active, peak
            async with limiter.slot():
                active += 1
                peak = max(peak, active)
                await clock.sleep(5)  # appel long : les suivants démarrent pendant ce temps
                active -= 1

        async def main() -> None:
            await asyncio.gather(*(call() for _ in range(3)))

        asyncio.run(main())

        assert peak == 3

    def test_zero_rps_means_no_spacing(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(0, 4, clock=clock, sleep=clock.sleep)

        async def main() -> None:
            for _ in range(3):
                async with limiter.slot():
                    pass

        asyncio.run(main())

        assert clock.sleeps == []


class TestRetry:
    def _run(self, outcomes: list[object], clock: FakeClock, **kwargs: object) -> tuple[object, int]:
        attempts = 0

        async def call() -> object:
            nonlocal attempts
            attempts += 1
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        limiter = RateLimiter(0, 4, clock=clock, sleep=clock.sleep)
        options = {"rate_limit_retries": 3, "backoff_s": 2.0, "sleep": clock.sleep} | kwargs
        result = asyncio.run(call_with_retry(call, limiter, **options))  # type: ignore[arg-type]
        return result, attempts

    def test_429_respects_retry_after_then_backs_off(self) -> None:
        clock = FakeClock()

        result, attempts = self._run([_http_error(429, "7"), _http_error(429), "ok"], clock)

        assert (result, attempts) == ("ok", 3)
        assert clock.sleeps == [7.0, 4.0]

    def test_429_backoff_grows(self) -> None:
        clock = FakeClock()

        self._run([_http_error(429), _http_error(429), _http_error(429), "ok"], clock)

        assert clock.sleeps == [2.0, 4.0, 8.0]

    def test_429_gives_up_after_max_retries(self) -> None:
        clock = FakeClock()
        outcomes: list[object] = [_http_error(429) for _ in range(10)]

        with pytest.raises(httpx.HTTPStatusError):
            self._run(outcomes, clock, rate_limit_retries=2)

        assert len(outcomes) == 7  # 1 tentative + 2 nouvelles

    def test_429_found_in_exception_chain(self) -> None:
        clock = FakeClock()
        try:
            raise _http_error(429, "1.5")
        except httpx.HTTPStatusError as cause:
            wrapped = RuntimeError("enveloppée")
            wrapped.__cause__ = cause

        result, attempts = self._run([wrapped, "ok"], clock)

        assert (result, attempts) == ("ok", 2)
        assert clock.sleeps == [1.5]

    def test_other_errors_get_a_single_retry(self) -> None:
        clock = FakeClock()

        with pytest.raises(ConnectionError):
            self._run([ConnectionError("1"), ConnectionError("2"), "ok"], clock)

        result, attempts = self._run([_http_error(500), "ok"], FakeClock())
        assert (result, attempts) == ("ok", 2)

    def test_429_without_retry_after_waits_the_per_minute_window_and_slows_down(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(0.5, 4, clock=clock, sleep=clock.sleep)
        starts: list[float] = []
        outcomes: list[object] = [_http_error(429, per_minute=2), "ok", "ok", "ok"]

        async def call() -> object:
            starts.append(clock())
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        async def main() -> None:
            await call_with_retry(call, limiter, rate_limit_retries=3, backoff_s=2.0, sleep=clock.sleep)
            # Appels suivants de la même analyse : au rythme imposé (2 par minute).
            for _ in range(2):
                await call_with_retry(call, limiter, rate_limit_retries=3, backoff_s=2.0, sleep=clock.sleep)

        asyncio.run(main())

        assert clock.sleeps[0] == 30.0  # 60 / 2, pas 2 s
        assert limiter.interval == 30.0
        assert [b - a for a, b in zip(starts, starts[1:], strict=False)] == [30.0, 30.0, 30.0]
        limiter.restore()
        assert limiter.interval == 2.0  # retour à LLM_RPS = 0,5

    def test_429_on_exhausted_token_budget_waits_the_whole_minute_and_holds_the_limiter(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(0.5, 4, clock=clock, sleep=clock.sleep)
        starts: list[float] = []
        # 429 de la limite de tokens : pas d'en-tête de requêtes, quota restant à 0.
        token_429 = _http_error(429)
        token_429.response.headers.update(
            {"x-ratelimit-limit-tokens-minute": "20000", "x-ratelimit-remaining-tokens-minute": "0"}
        )
        outcomes: list[object] = [token_429, "ok", "ok"]

        async def call() -> object:
            starts.append(clock())
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        async def main() -> None:
            await call_with_retry(call, limiter, rate_limit_retries=3, backoff_s=2.0, sleep=clock.sleep)
            await call_with_retry(call, limiter, rate_limit_retries=3, backoff_s=2.0, sleep=clock.sleep)

        asyncio.run(main())

        assert clock.sleeps[0] == 60.0  # fin de la fenêtre, pas 2 s
        assert starts[1] >= 60.0  # aucun départ avant la fin de la minute
        assert limiter.interval == 2.0  # débit inchangé ensuite
        assert starts[2] - starts[1] == 2.0

    def test_429_with_remaining_tokens_keeps_the_backoff(self) -> None:
        clock = FakeClock()
        error = _http_error(429)
        error.response.headers.update({"x-ratelimit-remaining-tokens-minute": "287"})

        result, _ = self._run([error, "ok"], clock)

        assert result == "ok" and clock.sleeps == [2.0]

    def test_retry_after_takes_precedence_over_per_minute_window(self) -> None:
        clock = FakeClock()

        result, _ = self._run([_http_error(429, "5", per_minute=2), "ok"], clock)

        assert result == "ok" and clock.sleeps == [5.0]

    def test_retries_go_through_the_limiter(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(0.5, 4, clock=clock, sleep=clock.sleep)
        starts: list[float] = []
        outcomes: list[object] = [ValueError("sortie invalide"), "ok"]

        async def call() -> object:
            starts.append(clock())
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        asyncio.run(call_with_retry(call, limiter, rate_limit_retries=3, backoff_s=2.0, sleep=clock.sleep))

        assert starts == [0.0, 2.0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("3", 3.0), ("0.5", 0.5), (None, None), ("pas une date", None)],
)
def test_retry_after_parsing(value: str | None, expected: float | None) -> None:
    assert rate_limit_info(_http_error(429, value)) == (429, expected)


def test_non_http_error_has_no_status() -> None:
    assert rate_limit_info(ValueError("x")) == (None, None)
