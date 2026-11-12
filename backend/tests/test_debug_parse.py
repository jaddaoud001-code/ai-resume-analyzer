import hashlib
import logging

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.config import Settings
from app.main import create_app


def _upload(client: TestClient, data: bytes, filename: str = "cv.pdf") -> Response:
    return client.post("/debug/parse", files={"file": (filename, data, "application/pdf")})


def test_returns_parsed_cv(client: TestClient, simple_cv_pdf: bytes) -> None:
    response = _upload(client, simple_cv_pdf)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"text", "page_count", "char_count"}
    assert body["page_count"] == 1
    assert body["char_count"] == len(body["text"])
    assert "full-stack" in body["text"]


def test_logs_fingerprints_at_debug_level(
    client: TestClient, simple_cv_pdf: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.main")

    text = _upload(client, simple_cv_pdf).json()["text"]

    [message] = [r.getMessage() for r in caplog.records if r.name == "app.main"]
    assert f"text_sha256={hashlib.sha256(text.encode()).hexdigest()[:12]}" in message
    assert f"pdf_sha256={hashlib.sha256(simple_cv_pdf).hexdigest()[:12]}" in message
    assert "host=testserver" in message
    # Aucun contenu du CV dans les logs.
    assert "Jean Dupont" not in caplog.text


def test_invalid_file_returns_400(client: TestClient, invalid_bytes: bytes) -> None:
    response = _upload(client, invalid_bytes, filename="notes.txt")

    assert response.status_code == 400
    assert response.json() == {"detail": "Le fichier n'est pas un PDF valide ou est corrompu."}


def test_encrypted_pdf_returns_400(client: TestClient, encrypted_pdf: bytes) -> None:
    response = _upload(client, encrypted_pdf)

    assert response.status_code == 400
    assert "mot de passe" in response.json()["detail"]


def test_image_only_pdf_returns_400(client: TestClient, image_only_pdf: bytes) -> None:
    response = _upload(client, image_only_pdf)

    assert response.status_code == 400
    assert "Aucun texte exploitable" in response.json()["detail"]


def test_too_many_pages_returns_413(client: TestClient, too_many_pages_pdf: bytes) -> None:
    response = _upload(client, too_many_pages_pdf)

    assert response.status_code == 413
    assert "maximum : 10" in response.json()["detail"]


def test_file_too_large_returns_413(settings: Settings, simple_cv_pdf: bytes) -> None:
    tiny = settings.model_copy(update={"max_pdf_size_bytes": 1024 * 1024 // 2})
    oversized = simple_cv_pdf + b"\0" * tiny.max_pdf_size_bytes

    with TestClient(create_app(tiny)) as client:
        response = _upload(client, oversized)

    assert response.status_code == 413
    assert response.json() == {"detail": "Le fichier dépasse la taille maximale de 0.5 Mo."}


def test_missing_file_returns_422(client: TestClient) -> None:
    assert client.post("/debug/parse").status_code == 422


def test_route_absent_when_debug_disabled(settings: Settings, simple_cv_pdf: bytes) -> None:
    disabled = settings.model_copy(update={"enable_debug_routes": False})

    with TestClient(create_app(disabled)) as client:
        assert _upload(client, simple_cv_pdf).status_code == 404
