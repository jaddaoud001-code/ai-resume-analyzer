from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from httpx import Response

from app.config import Settings
from app.requirements import ExtractedRequirement, ExtractedRequirements
from tests.conftest import FakeStructuredChatModel, create_test_app, make_settings

EXTRACTED = ExtractedRequirements(
    job_title="Stagiaire ingénieur IA",
    requirements=[
        ExtractedRequirement(
            label="Java",
            category="compétence technique",
            importance="obligatoire",
            min_years=2,
            search_queries=["Java", "Spring Boot", "développement Java"],
        )
    ],
)


def _post(settings: Settings, responses: list[Any], offer: str = "Stage IA. Java requis.") -> Response:
    model = FakeStructuredChatModel(responses=responses)
    factory: Callable[[Settings], FakeStructuredChatModel] = lambda _: model  # noqa: E731
    with TestClient(create_test_app(settings, llm_factory=factory)) as client:
        return client.post("/debug/requirements", json={"job_offer": offer})


def test_returns_job_requirements(settings: Settings) -> None:
    response = _post(settings, [EXTRACTED])

    assert response.status_code == 200
    assert response.json() == {
        "job_title": "Stagiaire ingénieur IA",
        "requirements": [
            {
                "id": "R1",
                "label": "Java",
                "category": "compétence technique",
                "importance": "obligatoire",
                "min_years": 2,
                "search_queries": ["Java", "Spring Boot", "développement Java"],
            }
        ],
    }


def test_offer_too_long_returns_413_even_without_key(settings: Settings) -> None:
    no_key = make_settings(mistral_api_key=None, max_job_offer_chars=50)

    with TestClient(create_test_app(no_key)) as client:
        response = client.post("/debug/requirements", json={"job_offer": "x" * 51})

    assert response.status_code == 413
    assert "50 caractères" in response.json()["detail"]


def test_empty_offer_returns_422(settings: Settings) -> None:
    assert _post(settings, [], offer="").status_code == 422


def test_no_requirement_returns_422(settings: Settings) -> None:
    response = _post(settings, [ExtractedRequirements(job_title="Stage", requirements=[])])

    assert response.status_code == 422
    assert response.json() == {"detail": "Aucune exigence identifiée dans l'offre."}


def test_missing_key_returns_500() -> None:
    with TestClient(create_test_app(make_settings(mistral_api_key=None))) as client:
        response = client.post("/debug/requirements", json={"job_offer": "Stage IA."})

    assert response.status_code == 500
    assert "MISTRAL_API_KEY" in response.json()["detail"]


def test_provider_failure_returns_502_after_one_retry(settings: Settings) -> None:
    response = _post(settings, [ConnectionError("réseau"), ConnectionError("réseau")])

    assert response.status_code == 502
    assert response.json() == {"detail": "Le fournisseur LLM n'a pas répondu correctement (ConnectionError)."}
