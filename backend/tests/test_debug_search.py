from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.main import create_app
from tests.conftest import KeywordEmbeddings, make_settings


def _search(client: TestClient, pdf: bytes, query: str, k: int | None = None) -> Response:
    data = {"query": query} | ({"k": str(k)} if k is not None else {})
    return client.post("/debug/search", files={"file": ("cv.pdf", pdf, "application/pdf")}, data=data)


@pytest.fixture
def search_client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, embeddings_factory=lambda _: KeywordEmbeddings())) as client:
        yield client


def test_returns_provider_chunks_and_ranked_results(search_client: TestClient, sectioned_cv_pdf: bytes) -> None:
    response = _search(search_client, sectioned_cv_pdf, "React TypeScript", k=2)

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "local"
    assert body["model"] == "intfloat/multilingual-e5-large"
    assert body["chunks"][0]["section"] == "EN-TÊTE"
    assert {c["section"] for c in body["chunks"]} >= {"EXPÉRIENCE", "FORMATION", "LANGUES"}
    assert len(body["results"]) == 2
    assert "Initech" in body["results"][0]["chunk"]["text"]
    assert body["results"][0]["score"] >= body["results"][1]["score"]


def test_default_k_comes_from_settings(search_client: TestClient, sectioned_cv_pdf: bytes) -> None:
    body = _search(search_client, sectioned_cv_pdf, "Python").json()

    assert len(body["results"]) == 4


def test_response_contains_no_personal_data(search_client: TestClient, sectioned_cv_pdf: bytes) -> None:
    raw = _search(search_client, sectioned_cv_pdf, "contact").text

    for leaked in ("Camille", "CAMILLE", "camille", "+33", "github.com"):
        assert leaked not in raw


def test_missing_key_returns_500_with_clear_message(sectioned_cv_pdf: bytes) -> None:
    configured = make_settings(embeddings_provider="mistral", mistral_api_key=None)

    with TestClient(create_app(configured)) as client:
        response = _search(client, sectioned_cv_pdf, "Python")

    assert response.status_code == 500
    assert "MISTRAL_API_KEY" in response.json()["detail"]


class _FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise ConnectionError("fournisseur injoignable")

    def embed_query(self, text: str) -> list[float]:
        raise ConnectionError("fournisseur injoignable")


def test_external_provider_failure_returns_502(sectioned_cv_pdf: bytes) -> None:
    configured = make_settings(embeddings_provider="openai", openai_api_key="sk-test")

    with TestClient(create_app(configured, embeddings_factory=lambda _: _FailingEmbeddings())) as client:
        response = _search(client, sectioned_cv_pdf, "Python")

    assert response.status_code == 502
    assert response.json() == {"detail": "Erreur du fournisseur d'embeddings openai : ConnectionError."}


def test_invalid_pdf_returns_400(search_client: TestClient, invalid_bytes: bytes) -> None:
    assert _search(search_client, invalid_bytes, "Python").status_code == 400


@pytest.mark.parametrize(("query", "k"), [("", None), ("Python", 0), ("Python", 21)])
def test_invalid_parameters_return_422(
    search_client: TestClient, sectioned_cv_pdf: bytes, query: str, k: int | None
) -> None:
    assert _search(search_client, sectioned_cv_pdf, query, k).status_code == 422
