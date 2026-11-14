"""Limites des fournisseurs (Groq surtout) : tokens par minute, quota journalier, accès refusé.

Groq n'envoie aucun en-tête pour le quota journalier de tokens : seul le message du 429 le
nomme (« … on tokens per day (TPD) … »). x-ratelimit-limit-requests y compte les requêtes
du jour, et x-ratelimit-*-tokens les tokens de la minute."""

import asyncio
from typing import Any

import groq
import httpx
import pytest

from app.llm_calls import (
    ProviderAccessDeniedError,
    ProviderQuotaExceededError,
    RateLimiter,
    call_with_retry,
    parse_go_duration,
)
from app.main import ProviderBlock
from app.requirements import extract_requirements
from tests.conftest import FakeStructuredChatModel, make_settings
from tests.test_analyze_stream import make_app, parse_sse, run_post, stream_settings
from tests.test_llm_calls import FakeClock
from tests.test_matcher import EXTRACTED

GROQ_HEADERS = {
    "x-ratelimit-limit-requests": "1000",  # par jour
    "x-ratelimit-remaining-requests": "990",
    "x-ratelimit-limit-tokens": "8000",  # par minute
    "x-ratelimit-remaining-tokens": "1200",
    "x-ratelimit-reset-requests": "2m52.8s",
    "x-ratelimit-reset-tokens": "7.5s",
}
TPM_MESSAGE = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_test` service tier `on_demand` "
    "on tokens per minute (TPM): Limit 8000, Used 6900, Requested 2500. Please try again in 10.4s."
)
TPD_MESSAGE = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_test` service tier `on_demand` "
    "on tokens per day (TPD): Limit 200000, Used 199100, Requested 2500. Please try again in 7m30s."
)
RPD_MESSAGE = "Rate limit reached for model `openai/gpt-oss-120b` on requests per day (RPD): Limit 1000, Used 1000."


def groq_error(status: int, message: str, headers: dict[str, str] | None = None) -> groq.APIStatusError:
    """Erreur telle que la lève le SDK Groq (corps JSON, en-têtes de la réponse)."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    error_class = {429: groq.RateLimitError, 403: groq.PermissionDeniedError, 401: groq.AuthenticationError}
    return error_class[status](f"Error code: {status} - {body}", response=response, body=body)


def run_calls(errors: list[BaseException], **kwargs: Any) -> tuple[Any, list[int], FakeClock, RateLimiter]:
    """call_with_retry sur un appel qui lève `errors` dans l'ordre, puis réussit."""
    clock = FakeClock()
    limiter = RateLimiter(0, 4, clock=clock, sleep=clock.sleep)
    attempts: list[int] = []

    async def call() -> str:
        attempts.append(1)
        if len(attempts) <= len(errors):
            raise errors[len(attempts) - 1]
        return "ok"

    async def run() -> Any:
        return await call_with_retry(call, limiter, rate_limit_retries=5, backoff_s=2.0, **kwargs)

    return asyncio.run(run()), attempts, clock, limiter


class TestMinuteLimits:
    def test_tokens_per_minute_waits_retry_after_and_pauses_the_limiter(self) -> None:
        error = groq_error(429, TPM_MESSAGE, GROQ_HEADERS | {"retry-after": "11"})

        result, attempts, clock, limiter = run_calls([error])

        assert (result, len(attempts)) == ("ok", 2)
        assert clock.sleeps == [11.0]
        # Pendant l'attente, le limiteur partagé ne lance aucun autre appel.
        assert limiter._next_start >= 11.0

    def test_tokens_per_minute_without_retry_after_uses_reset_tokens(self) -> None:
        headers = GROQ_HEADERS | {"x-ratelimit-remaining-tokens": "0"}

        _, attempts, clock, _ = run_calls([groq_error(429, "rate limited", headers)])

        assert len(attempts) == 2
        assert clock.sleeps == [7.5]

    def test_requests_per_day_header_is_not_mistaken_for_a_per_minute_limit(self) -> None:
        # x-ratelimit-limit-requests de Groq compte les requêtes du jour : il ne doit pas
        # ralentir le limiteur à 60 / 1000 s comme x-ratelimit-limit-req-minute (Mistral).
        _, attempts, clock, limiter = run_calls([groq_error(429, "rate limited", GROQ_HEADERS)])

        assert len(attempts) == 2
        assert clock.sleeps == [2.0]  # attente croissante, faute d'indication
        assert limiter.interval == limiter.base_interval


class TestDailyQuota:
    @pytest.mark.parametrize("message", [TPD_MESSAGE, RPD_MESSAGE])
    def test_daily_quota_stops_at_once_with_a_clear_message(self, message: str) -> None:
        error = groq_error(429, message, GROQ_HEADERS | {"retry-after": "450"})

        with pytest.raises(ProviderQuotaExceededError) as raised:
            run_calls([error, error, error])

        assert str(raised.value).startswith("Quota journalier du fournisseur LLM atteint : réessayez demain")
        assert "environ 8 min" in str(raised.value)
        assert raised.value.retry_after == 450

    def test_daily_quota_is_never_retried(self) -> None:
        error = groq_error(429, TPD_MESSAGE, GROQ_HEADERS)
        clock = FakeClock()
        limiter = RateLimiter(0, 4, clock=clock, sleep=clock.sleep)
        attempts: list[int] = []

        async def call() -> None:
            attempts.append(1)
            raise error

        async def run() -> None:
            await call_with_retry(call, limiter, rate_limit_retries=5, backoff_s=2.0)

        with pytest.raises(ProviderQuotaExceededError):
            asyncio.run(run())
        assert attempts == [1]
        assert clock.sleeps == []

    def test_tokens_per_minute_is_not_a_daily_quota(self) -> None:
        _, attempts, _, _ = run_calls([groq_error(429, TPM_MESSAGE, GROQ_HEADERS | {"retry-after": "3"})])

        assert len(attempts) == 2


class TestAccessDenied:
    def test_403_names_the_model_and_is_not_retried(self) -> None:
        error = groq_error(403, "This model is not available in your subscription tier")

        with pytest.raises(ProviderAccessDeniedError) as raised:
            run_calls([error, error], model="openai/gpt-oss-120b")

        message = str(raised.value)
        assert "refuse l'accès au modèle openai/gpt-oss-120b" in message
        assert "abonnement ou modèle non autorisé" in message
        assert "LLM_MODEL et MATCH_LLM_MODEL" in message
        assert raised.value.status == 403

    def test_401_points_to_the_api_key(self) -> None:
        with pytest.raises(ProviderAccessDeniedError, match="refuse la clé API"):
            run_calls([groq_error(401, "Invalid API Key")])

    def test_extraction_lets_the_clear_message_through(self) -> None:
        settings = make_settings()
        llm = FakeStructuredChatModel(responses=[groq_error(403, "tier_not_allowed")])

        with pytest.raises(ProviderAccessDeniedError):
            asyncio.run(extract_requirements("Stage : Python.", llm, settings, RateLimiter(0, 4)))
        assert len(llm.calls) == 1  # aucune nouvelle tentative


class TestAnalyzeStream:
    def test_daily_quota_during_evaluation_stops_the_analysis(self, sectioned_cv_pdf: bytes) -> None:
        quota = groq_error(429, TPD_MESSAGE, GROQ_HEADERS | {"retry-after": "450"})
        match = FakeStructuredChatModel(responder=lambda _: quota)
        app = make_app(stream_settings(llm_concurrency=1), FakeStructuredChatModel(responses=[EXTRACTED]), match)

        events = parse_sse(run_post(app, sectioned_cv_pdf).text)

        assert events[-1].event == "error"
        assert events[-1].data["code"] == "quota_exceeded"
        assert events[-1].data["message"].startswith("Quota journalier du fournisseur LLM atteint : réessayez demain")
        # Pas d'exigence marquée en erreur une à une : l'analyse s'arrête dès le premier refus.
        assert "requirement_evaluated" not in [e.event for e in events]
        assert len(match.calls) == 1

    def test_next_analysis_is_refused_before_the_stream(self, sectioned_cv_pdf: bytes) -> None:
        quota = groq_error(429, TPD_MESSAGE, GROQ_HEADERS | {"retry-after": "450"})
        extraction = FakeStructuredChatModel(responses=[quota])
        app = make_app(stream_settings(), extraction)
        run_post(app, sectioned_cv_pdf)

        response = run_post(app, sectioned_cv_pdf)

        assert response.status_code == 429
        assert response.headers["content-type"].startswith("application/json")
        assert "Quota journalier du fournisseur LLM atteint" in response.json()["detail"]
        assert len(extraction.calls) == 1  # le fournisseur n'est pas rappelé
        assert app.state.analysis_slots.active == 0

    def test_access_denied_becomes_an_error_event_then_502_before_the_stream(self, sectioned_cv_pdf: bytes) -> None:
        extraction = FakeStructuredChatModel(responses=[groq_error(403, "tier_not_allowed")])
        app = make_app(stream_settings(), extraction)

        first = parse_sse(run_post(app, sectioned_cv_pdf).text)
        second = run_post(app, sectioned_cv_pdf)

        assert first[-1].event == "error"
        assert first[-1].data["code"] == "provider_access_denied"
        assert "abonnement ou modèle non autorisé" in first[-1].data["message"]
        assert second.status_code == 502
        assert "abonnement ou modèle non autorisé" in second.json()["detail"]
        assert len(extraction.calls) == 1


class TestProviderBlock:
    def test_quota_block_ends_when_the_provider_says_so(self) -> None:
        clock = FakeClock()
        block = ProviderBlock(clock=clock)
        block.record(ProviderQuotaExceededError(retry_after=450))

        assert block.current() is not None
        clock.now = 449
        assert block.current() is not None
        clock.now = 450
        assert block.current() is None

    def test_quota_without_retry_after_blocks_one_hour(self) -> None:
        clock = FakeClock()
        block = ProviderBlock(clock=clock)
        block.record(ProviderQuotaExceededError())

        clock.now = 3599
        assert block.current() is not None
        clock.now = 3600
        assert block.current() is None

    def test_access_denied_lasts_until_restart(self) -> None:
        clock = FakeClock()
        block = ProviderBlock(clock=clock)
        block.record(ProviderAccessDeniedError(403, "m"))

        clock.now = 10**9
        assert isinstance(block.current(), ProviderAccessDeniedError)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7.66s", 7.66), ("1m26.4s", 86.4), ("2h3m4s", 7384.0), ("120ms", 0.12), ("", None), ("bientôt", None)],
)
def test_groq_durations(value: str, expected: float | None) -> None:
    assert parse_go_duration(value) == expected
