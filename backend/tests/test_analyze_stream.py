"""POST /analyze : flux SSE branché sur analyze() (faux LLM, faux embeddings, aucun réseau)."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from app.config import Settings
from app.llm import match_llm_model
from app.llm_calls import RateLimiter
from app.main import TOO_MANY_ANALYSES, create_app
from app.matcher import analyze
from app.schemas import AnalysisStatus, MatchEvent
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_llm_calls import FakeClock
from tests.test_matcher import EXTRACTED, _responder

OFFER = "Offre : Python 3 ans, Kubernetes, React, Rust, Go, rigueur."
TOTAL = len(EXTRACTED.requirements)


# --- Outils --------------------------------------------------------------------------


class AsyncFakeModel(FakeStructuredChatModel):
    """Comme FakeStructuredChatModel, mais la réponse est une coroutine : un appel en
    attente est annulable (comme un vrai appel réseau)."""

    async_responder: Callable[[list[BaseMessage]], Awaitable[Any]] | None = None
    started: list[int] = Field(default_factory=list)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        async def respond(messages: Sequence[BaseMessage]) -> Any:
            self.calls.append(list(messages))
            assert self.async_responder is not None
            response = await self.async_responder(list(messages))
            if isinstance(response, BaseException):
                raise response
            return response

        def unused(_: Any) -> Any:
            raise AssertionError("appel synchrone inattendu")

        return RunnableLambda(unused, afunc=respond)


def token_429() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    response = httpx.Response(
        429,
        headers={"x-ratelimit-limit-tokens-minute": "20000", "x-ratelimit-remaining-tokens-minute": "0"},
        request=request,
    )
    return httpx.HTTPStatusError("429", request=request, response=response)


@dataclass
class SSE:
    event: str
    id: int
    data: dict[str, Any]


def parse_sse(raw: str) -> list[SSE]:
    """Événements du flux ; les commentaires (pings « : … ») sont ignorés."""
    events: list[SSE] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            fields[name] = value.removeprefix(" ")
        if "event" in fields:
            events.append(SSE(fields["event"], int(fields["id"]), json.loads(fields["data"])))
    return events


def make_app(
    settings: Settings,
    extraction: FakeStructuredChatModel,
    match: FakeStructuredChatModel | None = None,
    embeddings_factory: Callable[[Settings], Embeddings] = lambda _: KeywordEmbeddings(),
    **kwargs: Any,
) -> Any:
    match_model = match or FakeStructuredChatModel(responder=_responder)

    def llm_factory(s: Settings) -> FakeStructuredChatModel:
        return match_model if s.llm_model == match_llm_model(settings) else extraction

    return create_app(settings, embeddings_factory=embeddings_factory, llm_factory=llm_factory, **kwargs)


def files(pdf: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("cv.pdf", pdf, "application/pdf")}


async def post_analyze(app: Any, pdf: bytes, offer: str = OFFER) -> httpx.Response:
    """Client httpx asynchrone en streaming ; le corps est lu en entier."""
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/analyze", files=files(pdf), data={"job_offer": offer}) as response,
    ):
        await response.aread()
        return response


def run_post(app: Any, pdf: bytes, offer: str = OFFER) -> httpx.Response:
    return asyncio.run(post_analyze(app, pdf, offer))


def stream_settings(**overrides: Any) -> Settings:
    return make_settings(sse_ping_s=0, **overrides)


# --- Flux nominal --------------------------------------------------------------------


def test_events_order_format_and_increasing_ids(sectioned_cv_pdf: bytes) -> None:
    app = make_app(stream_settings(), FakeStructuredChatModel(responses=[EXTRACTED]))

    response = run_post(app, sectioned_cv_pdf)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    kinds = [e.event for e in events]
    assert kinds[0] == "started"  # dès l'ouverture du flux
    assert events[0].data["phases"] == ["indexing", "extraction", "evaluation"]
    assert events[0].data["timeout_s"] == 600
    assert set(kinds[1:3]) == {"requirements_extracted", "cv_indexed"}
    assert kinds[3:-1] == ["requirement_evaluated"] * TOTAL
    assert kinds[-1] == "report"  # dernier événement : le flux se ferme ensuite
    assert [e.id for e in events] == list(range(1, len(events) + 1))
    assert all(e.data["type"] == e.event for e in events)  # data = JSON du modèle Pydantic
    assert [e.data["done"] for e in events[3:-1]] == list(range(1, TOTAL + 1))
    assert {e.data["total"] for e in events[3:-1]} == {TOTAL}
    report = events[-1].data["report"]
    assert len(report["requirements"]) == TOTAL and report["score"] is not None
    assert app.state.analysis_slots.active == 0


def test_heartbeat_ping_comments_during_long_waits(sectioned_cv_pdf: bytes) -> None:
    async def slow(messages: list[BaseMessage]) -> Any:
        await asyncio.sleep(0.1)  # longue attente du fournisseur, à l'échelle du test
        return _responder(messages)

    app = make_app(
        make_settings(sse_ping_s=0.02),
        FakeStructuredChatModel(responses=[EXTRACTED]),
        AsyncFakeModel(async_responder=slow),
    )

    raw = run_post(app, sectioned_cv_pdf).text

    assert make_settings().sse_ping_s == 15  # défaut
    assert ": ping" in raw  # commentaire SSE : ignoré par les clients, garde la connexion
    assert parse_sse(raw)[-1].event == "report"


# --- Validations avant le flux --------------------------------------------------------


def _assert_plain_http_error(response: httpx.Response, code: int, app: Any) -> None:
    assert response.status_code == code
    assert response.headers["content-type"].startswith("application/json")  # aucun flux ouvert
    assert response.json()["detail"]
    assert app.state.analysis_slots.active == 0


def test_invalid_pdf_returns_400_without_stream(invalid_bytes: bytes) -> None:
    app = make_app(stream_settings(), FakeStructuredChatModel(responses=[EXTRACTED]))

    _assert_plain_http_error(run_post(app, invalid_bytes), 400, app)


def test_too_large_pdf_returns_413_without_stream(sectioned_cv_pdf: bytes) -> None:
    app = make_app(stream_settings(max_pdf_size_bytes=100), FakeStructuredChatModel(responses=[EXTRACTED]))

    _assert_plain_http_error(run_post(app, sectioned_cv_pdf), 413, app)


def test_too_long_offer_returns_413_without_stream(sectioned_cv_pdf: bytes) -> None:
    app = make_app(stream_settings(max_job_offer_chars=10), FakeStructuredChatModel(responses=[EXTRACTED]))

    _assert_plain_http_error(run_post(app, sectioned_cv_pdf), 413, app)


def test_empty_offer_returns_422_without_stream(sectioned_cv_pdf: bytes) -> None:
    app = make_app(stream_settings(), FakeStructuredChatModel(responses=[EXTRACTED]))

    _assert_plain_http_error(run_post(app, sectioned_cv_pdf, offer=""), 422, app)


def test_missing_key_returns_500_without_stream(sectioned_cv_pdf: bytes) -> None:
    app = create_app(stream_settings(mistral_api_key=None), embeddings_factory=lambda _: KeywordEmbeddings())

    response = run_post(app, sectioned_cv_pdf)

    _assert_plain_http_error(response, 500, app)
    assert "MISTRAL_API_KEY" in response.json()["detail"]


# --- Événements status, error, timeout -----------------------------------------------


def test_status_event_when_a_429_makes_the_analysis_wait(sectioned_cv_pdf: bytes) -> None:
    clock = FakeClock()
    limiter = RateLimiter(0, 4, clock=clock, sleep=clock.sleep)
    extraction = FakeStructuredChatModel(responses=[token_429(), EXTRACTED])
    app = make_app(stream_settings(), extraction, limiter=limiter)

    events = parse_sse(run_post(app, sectioned_cv_pdf).text)

    status = [e for e in events if e.event == "status"]
    assert len(status) == 1
    assert status[0].data == {
        "type": "status",
        "reason": "rate_limit",
        "wait_s": 60.0,
        "message": "En attente du quota du fournisseur, reprise dans 60 s",
    }
    assert 60.0 in clock.sleeps  # horloge simulée : aucune vraie attente
    assert events[-1].event == "report"


def test_fatal_error_becomes_error_event_then_stream_closes(sectioned_cv_pdf: bytes) -> None:
    failing = FakeStructuredChatModel(responses=[ConnectionError("x"), ConnectionError("x")])
    app = make_app(stream_settings(), failing)

    response = run_post(app, sectioned_cv_pdf)

    assert response.status_code == 200  # le flux était déjà ouvert
    events = parse_sse(response.text)
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "llm_provider_error" and events[-1].data["message"]
    assert "report" not in [e.event for e in events]
    assert [e.id for e in events] == list(range(1, len(events) + 1))
    assert app.state.analysis_slots.active == 0


def test_unexpected_error_does_not_leak_details(sectioned_cv_pdf: bytes) -> None:
    class Broken(KeywordEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("secret interne")

    app = make_app(
        stream_settings(), FakeStructuredChatModel(responses=[EXTRACTED]), embeddings_factory=lambda _: Broken()
    )

    events = parse_sse(run_post(app, sectioned_cv_pdf).text)

    assert events[-1].data["code"] == "embeddings_error"
    assert "secret" not in events[-1].data["message"]


def test_timeout_cancels_tasks_sends_error_and_frees_slot(sectioned_cv_pdf: bytes) -> None:
    cancelled: list[str] = []
    evaluation_started = asyncio.Event()

    async def hang(messages: list[BaseMessage]) -> Any:
        evaluation_started.set()
        try:
            await asyncio.Event().wait()  # le fournisseur ne répond jamais
        except asyncio.CancelledError:
            cancelled.append("évaluation")
            raise

    async def simulated_deadline(seconds: float) -> None:
        # Horloge simulée : les 600 s « passent » dès que les évaluations sont lancées.
        assert seconds == 600
        await evaluation_started.wait()

    app = make_app(
        stream_settings(),
        FakeStructuredChatModel(responses=[EXTRACTED]),
        AsyncFakeModel(async_responder=hang),
        sleep=simulated_deadline,
    )

    events = parse_sse(run_post(app, sectioned_cv_pdf).text)

    assert events[-1].event == "error" and events[-1].data["code"] == "timeout"
    assert "600" in events[-1].data["message"]
    assert "requirement_evaluated" not in [e.event for e in events]
    assert cancelled  # les appels en cours sont annulés
    assert app.state.analysis_slots.active == 0


# --- Deux analyses simultanées : la pause du limiteur partagé prévient les deux -----------


def test_shared_limiter_pause_notifies_every_analysis_with_waiting_calls(sectioned_cv_pdf: bytes) -> None:
    clock = FakeClock()
    # Un appel à la fois : pendant le premier appel d'extraction, celui de l'autre analyse
    # attend dans le limiteur.
    limiter = RateLimiter(0, 1, clock=clock, sleep=clock.sleep)
    settings = make_settings()
    first_call = True

    async def extraction_responder(messages: list[BaseMessage]) -> Any:
        nonlocal first_call
        if first_call:
            first_call = False
            for _ in range(1000):  # attendre que l'autre analyse ait un appel en attente
                if limiter.waiting_monitors():
                    break
                await asyncio.sleep(0.001)
            assert limiter.waiting_monitors(), "l'autre analyse n'attend pas"
            return token_429()
        return EXTRACTED

    extraction = AsyncFakeModel(async_responder=extraction_responder)

    async def run_one() -> list[MatchEvent]:
        stream = analyze(
            sectioned_cv_pdf,
            OFFER,
            settings=settings,
            embeddings=KeywordEmbeddings(),
            extraction_llm=extraction,
            match_llm=FakeStructuredChatModel(responder=_responder),
            limiter=limiter,
        )
        return [event async for event in stream]

    async def main() -> list[list[MatchEvent]]:
        return list(await asyncio.gather(run_one(), run_one()))

    first, second = asyncio.run(main())

    for events in (first, second):
        statuses = [e for e in events if isinstance(e, AnalysisStatus)]
        assert len(statuses) == 1 and statuses[0].wait_s == 60.0
        assert events[-1].type == "report"


# --- Déconnexion -----------------------------------------------------------------------


async def drive_until_disconnect(app: Any, pdf: bytes, evaluated_before_disconnect: int) -> bytes:
    """Pilote l'application ASGI directement (httpx.ASGITransport attend la fin de la
    réponse) : le client se déconnecte après N événements requirement_evaluated."""
    request = httpx.Request("POST", "http://test/analyze", files=files(pdf), data={"job_offer": OFFER})
    body = request.read()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/analyze",
        "raw_path": b"/analyze",
        "query_string": b"",
        "root_path": "",
        "client": ("test", 1234),
        "server": ("test", 80),
        "headers": [(k.lower().encode(), v.encode()) for k, v in request.headers.items()],
    }
    received = bytearray()
    disconnected = asyncio.Event()
    body_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            received.extend(message.get("body", b""))
            if received.count(b"event: requirement_evaluated") >= evaluated_before_disconnect:
                disconnected.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return bytes(received)


def test_disconnect_cancels_evaluations_and_no_new_llm_call(
    sectioned_cv_pdf: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    calls_after_disconnect: list[int] = []

    async def slow(messages: list[BaseMessage]) -> Any:
        await asyncio.sleep(0.02)
        return _responder(messages)

    match = AsyncFakeModel(async_responder=slow)
    # Un appel à la fois : les évaluations suivantes attendent dans le limiteur.
    app = make_app(
        stream_settings(),
        FakeStructuredChatModel(responses=[EXTRACTED]),
        match,
        limiter=RateLimiter(0, 1),
    )

    async def main() -> bytes:
        raw = await drive_until_disconnect(app, sectioned_cv_pdf, evaluated_before_disconnect=2)
        calls_after_disconnect.append(len(match.calls))
        await asyncio.sleep(0.2)  # plus rien ne doit partir
        calls_after_disconnect.append(len(match.calls))
        return raw

    with caplog.at_level(logging.INFO, logger="app.matcher"):
        raw = asyncio.run(main())

    events = parse_sse(raw.decode())
    assert "report" not in [e.event for e in events]
    assert calls_after_disconnect[0] == calls_after_disconnect[1]  # aucun nouvel appel
    assert calls_after_disconnect[0] < TOTAL  # des appels ont été évités
    interrupted = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Analyse interrompue")]
    assert interrupted and "jamais lancé(s) (évités)" in interrupted[0]
    avoided = int(re.search(r"(\d+) jamais lancé", interrupted[0]).group(1))  # type: ignore[union-attr]
    assert avoided >= 1 and avoided + calls_after_disconnect[0] <= TOTAL + 1
    assert app.state.analysis_slots.active == 0


# --- Limite d'analyses simultanées -----------------------------------------------------


def test_concurrency_limit_returns_429_and_slot_is_freed(sectioned_cv_pdf: bytes) -> None:
    release = asyncio.Event()

    async def gated(messages: list[BaseMessage]) -> Any:
        await release.wait()
        return _responder(messages)

    app = make_app(
        stream_settings(analysis_concurrency=1),
        FakeStructuredChatModel(responses=[EXTRACTED, EXTRACTED]),
        AsyncFakeModel(async_responder=gated),
    )

    async def main() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        first = asyncio.create_task(post_analyze(app, sectioned_cv_pdf))
        for _ in range(1000):
            if app.state.analysis_slots.active == 1:
                break
            await asyncio.sleep(0.001)
        rejected = await post_analyze(app, sectioned_cv_pdf)
        release.set()
        done = await first
        again = await post_analyze(app, sectioned_cv_pdf)  # créneau libéré
        return rejected, done, again

    rejected, done, again = asyncio.run(main())

    assert rejected.status_code == 429 and rejected.json()["detail"] == TOO_MANY_ANALYSES
    assert parse_sse(done.text)[-1].event == "report"
    assert again.status_code == 200 and parse_sse(again.text)[-1].event == "report"
    assert app.state.analysis_slots.active == 0


def test_slot_is_freed_after_validation_error_and_stream_error(sectioned_cv_pdf: bytes, invalid_bytes: bytes) -> None:
    extraction = FakeStructuredChatModel(responses=[ConnectionError("x"), ConnectionError("x"), EXTRACTED])
    app = make_app(stream_settings(analysis_concurrency=1), extraction)

    assert run_post(app, invalid_bytes).status_code == 400
    assert parse_sse(run_post(app, sectioned_cv_pdf).text)[-1].event == "error"
    assert parse_sse(run_post(app, sectioned_cv_pdf).text)[-1].event == "report"


# --- Préparation : embeddings construits une seule fois ------------------------------------


def test_embeddings_are_built_once_for_several_analyses(sectioned_cv_pdf: bytes) -> None:
    built: list[int] = []

    def factory(_: Settings) -> Embeddings:
        built.append(1)
        return KeywordEmbeddings()

    app = make_app(stream_settings(), FakeStructuredChatModel(responses=[EXTRACTED]), embeddings_factory=factory)

    for _ in range(3):
        assert parse_sse(run_post(app, sectioned_cv_pdf).text)[-1].event == "report"

    assert len(built) == 1


def test_local_model_is_loaded_once_per_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import app.embeddings as embeddings_module

    loaded: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str) -> None:
            loaded.append(model_name)

        def embed(self, texts: list[str]) -> Any:
            import numpy as np

            return [np.zeros(4) for _ in texts]

    monkeypatch.setattr(embeddings_module, "TextEmbedding", FakeTextEmbedding)
    embeddings_module._cached_model.cache_clear()
    try:
        for _ in range(3):  # une instance par analyse, comme avant la mise en cache par app
            embeddings_module.LocalEmbeddings("modele-test", tmp_path).embed_documents(["a", "b"])
    finally:
        embeddings_module._cached_model.cache_clear()

    assert loaded == ["modele-test"]
