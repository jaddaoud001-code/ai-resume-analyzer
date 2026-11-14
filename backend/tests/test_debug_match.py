from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.config import Settings
from app.llm import match_llm_model
from app.main import create_app
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_matcher import EXTRACTED, _responder

OFFER = "Offre : Python 3 ans, Kubernetes, React, Rust, Go, rigueur."


def _client(settings: Settings, extraction: FakeStructuredChatModel) -> TestClient:
    match = FakeStructuredChatModel(responder=_responder)

    def llm_factory(s: Settings) -> FakeStructuredChatModel:
        return match if s.llm_model == match_llm_model(settings) else extraction

    return TestClient(create_app(settings, embeddings_factory=lambda _: KeywordEmbeddings(), llm_factory=llm_factory))


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with _client(settings, FakeStructuredChatModel(responses=[EXTRACTED])) as test_client:
        yield test_client


def _post(client: TestClient, pdf: bytes, offer: str = OFFER) -> Response:
    return client.post("/debug/match", files={"file": ("cv.pdf", pdf, "application/pdf")}, data={"job_offer": offer})


def test_returns_report_events_and_total_time(client: TestClient, sectioned_cv_pdf: bytes) -> None:
    response = _post(client, sectioned_cv_pdf)

    assert response.status_code == 200
    body = response.json()
    assert [e["type"] for e in body["events"]][-1] == "report"
    assert len(body["events"]) == 10
    assert body["events"][0]["type"] == "started"
    assert body["total_seconds"] >= body["events"][-1]["t_s"] >= 0
    report = body["report"]
    assert report["score"] == 42
    assert [r["label"] for r in report["missing_mandatory"]] == ["Go"]
    assert report["rejected_evidence_count"] == 1
    assert set(report["timings"]) == {"extraction_s", "indexing_s", "retrieval_s", "evaluation_s"}
    python = next(r for r in report["requirements"] if r["requirement"]["label"] == "Python")
    assert python["level"] == "solide" and python["evidence"][0]["quote"].startswith("Conception d'API")


def test_invalid_pdf_returns_400(client: TestClient, invalid_bytes: bytes) -> None:
    assert _post(client, invalid_bytes).status_code == 400


def test_offer_too_long_returns_413(sectioned_cv_pdf: bytes) -> None:
    settings = make_settings(max_job_offer_chars=10)
    with _client(settings, FakeStructuredChatModel(responses=[EXTRACTED])) as client:
        assert _post(client, sectioned_cv_pdf).status_code == 413


def test_empty_offer_returns_422(client: TestClient, sectioned_cv_pdf: bytes) -> None:
    assert _post(client, sectioned_cv_pdf, offer="").status_code == 422


def test_extraction_failure_returns_502(settings: Settings, sectioned_cv_pdf: bytes) -> None:
    failing = FakeStructuredChatModel(responses=[ConnectionError("x"), ConnectionError("x")])
    with _client(settings, failing) as client:
        response = _post(client, sectioned_cv_pdf)

    assert response.status_code == 502


def test_missing_key_returns_500(sectioned_cv_pdf: bytes) -> None:
    settings = make_settings(groq_api_key=None)
    with TestClient(create_app(settings, embeddings_factory=lambda _: KeywordEmbeddings())) as client:
        response = _post(client, sectioned_cv_pdf)

    assert response.status_code == 500
    assert "GROQ_API_KEY" in response.json()["detail"]
