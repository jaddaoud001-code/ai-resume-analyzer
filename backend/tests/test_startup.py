"""Démarrage : chargement du modèle d'embeddings (lifespan), /health, /config ; débit du
limiteur partagé entre analyses qui se chevauchent."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage

from app.config import Settings
from app.llm_calls import RateLimiter
from app.main import create_app
from app.matcher import analyze
from app.schemas import MatchEvent
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_analyze_stream import OFFER, AsyncFakeModel, make_app, parse_sse
from tests.test_llm_calls import FakeClock, _http_error
from tests.test_matcher import EXTRACTED, _responder


class LoadableEmbeddings(KeywordEmbeddings):
    """Faux modèle local : `load` compte les appels, peut échouer ou bloquer."""

    def __init__(self, fail_times: int = 0, gate: threading.Event | None = None) -> None:
        self.loads = 0
        self.fail_times = fail_times
        self.gate = gate

    def load(self) -> None:
        self.loads += 1
        if self.gate is not None:
            self.gate.wait(5)
        if self.loads <= self.fail_times:
            raise OSError("téléchargement interrompu")


def _wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition jamais atteinte"
        time.sleep(0.01)


def _ready(client: TestClient) -> bool:
    return bool(client.get("/health").json()["embeddings_ready"])


def _app_with(embeddings: LoadableEmbeddings, built: list[int] | None = None, **settings: Any) -> Any:
    def factory(_: Settings) -> LoadableEmbeddings:
        if built is not None:
            built.append(1)
        return embeddings

    return make_app(
        make_settings(sse_ping_s=0, **settings),
        FakeStructuredChatModel(responses=[EXTRACTED, EXTRACTED]),
        embeddings_factory=factory,
    )


class TestEmbeddingsWarmup:
    def test_model_is_loaded_once_at_startup_and_duration_logged(
        self, sectioned_cv_pdf: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        embeddings, built = LoadableEmbeddings(), []
        app = _app_with(embeddings, built)

        with caplog.at_level(logging.INFO, logger="app.main"), TestClient(app) as client:
            _wait_until(lambda: _ready(client))
            for _ in range(2):
                response = client.post(
                    "/analyze",
                    files={"file": ("cv.pdf", sectioned_cv_pdf, "application/pdf")},
                    data={"job_offer": OFFER},
                )
                assert parse_sse(response.text)[-1].event == "report"

        assert built == [1]  # embeddings construits une seule fois pour l'application
        assert any(r.getMessage().startswith("Modèle d'embeddings chargé en ") for r in caplog.records)

    def test_health_reports_loading_then_ready(self) -> None:
        gate = threading.Event()
        app = _app_with(LoadableEmbeddings(gate=gate))

        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok", "embeddings_ready": False}
            gate.set()
            _wait_until(lambda: _ready(client))

    def test_startup_failure_does_not_prevent_startup_and_first_analysis_retries(
        self, sectioned_cv_pdf: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        embeddings = LoadableEmbeddings(fail_times=1)
        app = _app_with(embeddings)

        with caplog.at_level(logging.ERROR, logger="app.main"), TestClient(app) as client:
            _wait_until(lambda: embeddings.loads == 1)
            _wait_until(lambda: any("impossible" in r.getMessage() for r in caplog.records))
            assert client.get("/health").json() == {"status": "ok", "embeddings_ready": False}

            response = client.post(
                "/analyze",
                files={"file": ("cv.pdf", sectioned_cv_pdf, "application/pdf")},
                data={"job_offer": OFFER},
            )

            assert parse_sse(response.text)[-1].event == "report"  # le 2e chargement a réussi
            assert _ready(client)

    def test_persistent_failure_returns_503_before_the_stream(self, sectioned_cv_pdf: bytes) -> None:
        app = _app_with(LoadableEmbeddings(fail_times=99))

        with TestClient(app) as client:
            response = client.post(
                "/analyze",
                files={"file": ("cv.pdf", sectioned_cv_pdf, "application/pdf")},
                data={"job_offer": OFFER},
            )

        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert "Modèle d'embeddings indisponible (OSError)" in response.json()["detail"]
        assert app.state.analysis_slots.active == 0

    def test_external_provider_has_nothing_to_load(self) -> None:
        built: list[int] = []

        def factory(_: Settings) -> KeywordEmbeddings:
            built.append(1)
            return KeywordEmbeddings()

        app = create_app(
            make_settings(embeddings_provider="mistral", mistral_api_key="sk-test"), embeddings_factory=factory
        )
        with TestClient(app) as client:
            assert client.get("/health").json()["embeddings_ready"] is True

        assert built == []  # aucun préchargement, donc aucune construction au démarrage


def test_public_config_exposes_limits() -> None:
    app = create_app(make_settings(max_pdf_size_bytes=1234, max_pdf_pages=3, max_job_offer_chars=500))

    with TestClient(app) as client:
        response = client.get("/config")

    assert response.status_code == 200
    assert response.json() == {"max_pdf_size_bytes": 1234, "max_pdf_pages": 3, "max_job_offer_chars": 500}


# --- Débit du limiteur avec deux analyses qui se chevauchent ------------------------------


def test_normal_rate_is_restored_only_when_no_analysis_is_active(sectioned_cv_pdf: bytes) -> None:
    clock = FakeClock()
    limiter = RateLimiter(0.5, 4, clock=clock, sleep=clock.sleep)
    settings = make_settings()
    gate = asyncio.Event()

    async def gated_extraction(messages: list[BaseMessage]) -> Any:
        await gate.wait()
        return EXTRACTED

    def run(extraction: FakeStructuredChatModel) -> Any:
        async def collect() -> list[MatchEvent]:
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

        return collect()

    async def main() -> tuple[float, int, float, int]:
        # B démarre et reste active (son extraction attend) ; A reçoit un 429 « 2 req/min ».
        second = asyncio.create_task(run(AsyncFakeModel(async_responder=gated_extraction)))
        await asyncio.sleep(0)
        first = await run(FakeStructuredChatModel(responses=[_http_error(429, per_minute=2), EXTRACTED]))
        assert first[-1].type == "report"
        during = (limiter.interval, limiter.active_analyses)
        gate.set()
        assert (await second)[-1].type == "report"
        return (*during, limiter.interval, limiter.active_analyses)

    interval_after_first, active_after_first, interval_after_both, active_after_both = asyncio.run(main())

    assert (interval_after_first, active_after_first) == (30.0, 1)  # B active : toujours ralenti
    assert (interval_after_both, active_after_both) == (2.0, 0)  # plus aucune : LLM_RPS = 0,5
