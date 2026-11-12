"""Garde-fou : variables du .env mal nommées (ignorées par Settings) ou définies deux fois."""

import logging
from pathlib import Path

import pytest

import app.main as main_module
from app.config import Settings, closest_setting, env_file_names, env_file_warnings
from tests.conftest import make_settings

KNOWN = [name.upper() for name in Settings.model_fields]


def _env(tmp_path: Path, content: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    return path


def test_unknown_variable_with_our_prefix_is_reported_with_closest_name(tmp_path: Path) -> None:
    path = _env(tmp_path, "LLM_RPS=0.1\nMATCH_MAX_RETRIES=5\n")

    assert env_file_warnings(path) == [
        "Variable MATCH_MAX_RETRIES du .env inconnue, ignorée ; vouliez-vous dire LLM_RATE_LIMIT_RETRIES ?"
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("MATCH_MAX_RETRIES", "LLM_RATE_LIMIT_RETRIES"),
        ("LLM_RSP", "LLM_RPS"),
        ("MATCH_SEED", "LLM_SEED"),
        ("EMBEDDING_PROVIDER", "EMBEDDINGS_PROVIDER"),
        ("MAX_PDF_SIZE", "MAX_PDF_SIZE_BYTES"),
        ("REQUIREMENT_CACHE_SIZE", "REQUIREMENTS_CACHE_SIZE"),
        ("MISTRAL_KEY", "MISTRAL_API_KEY"),
        ("LLM_FOO_BAR", None),  # rien d'assez proche : pas de suggestion hasardeuse
    ],
)
def test_closest_setting(name: str, expected: str | None) -> None:
    assert closest_setting(name, KNOWN) == expected


def test_known_variables_and_foreign_prefixes_are_not_reported(tmp_path: Path) -> None:
    path = _env(
        tmp_path,
        "# MATCH_OLD=commentaire\n"
        "llm_rps=0.1\n"  # casse ignorée, comme Settings
        "export LLM_RATE_LIMIT_RETRIES=5\n"
        "MISTRAL_API_KEY=sk-test\n"
        "PATH_EXTRA=/opt\n"  # préfixe étranger à l'application
        "NODE_ENV=dev\n",
    )

    assert env_file_warnings(path) == []


def test_singular_prefix_typo_is_still_checked(tmp_path: Path) -> None:
    warnings = env_file_warnings(_env(tmp_path, "EMBEDDING_PROVIDER=local\n"))

    assert warnings == [
        "Variable EMBEDDING_PROVIDER du .env inconnue, ignorée ; vouliez-vous dire EMBEDDINGS_PROVIDER ?"
    ]


def test_duplicate_variable_is_reported(tmp_path: Path) -> None:
    warnings = env_file_warnings(_env(tmp_path, "LLM_RPS=0.5\nLLM_MODEL=x\nLLM_RPS=0.1\n"))

    assert warnings == ["Variable LLM_RPS définie 2 fois dans le .env : seule la dernière compte"]


def test_values_never_appear_in_warnings(tmp_path: Path) -> None:
    warnings = env_file_warnings(_env(tmp_path, "MISTRAL_KEY=sk-secret-value\nMISTRAL_KEY=sk-other\n"))

    assert warnings and not any("sk-" in w for w in warnings)


def test_missing_file_gives_no_warning(tmp_path: Path) -> None:
    assert env_file_names(tmp_path / "absent.env") == []
    assert env_file_warnings(tmp_path / "absent.env") == []


def test_create_app_logs_warnings_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main_module, "ENV_FILE", _env(tmp_path, "MATCH_MAX_RETRIES=5\n"))
    monkeypatch.setattr(main_module, "get_settings", lambda: make_settings())

    with caplog.at_level(logging.WARNING, logger="app.config"):
        main_module.create_app()

    assert any("MATCH_MAX_RETRIES" in r.message and "LLM_RATE_LIMIT_RETRIES" in r.message for r in caplog.records)


def test_create_app_with_explicit_settings_does_not_read_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main_module, "ENV_FILE", _env(tmp_path, "MATCH_MAX_RETRIES=5\n"))

    with caplog.at_level(logging.WARNING, logger="app.config"):
        main_module.create_app(make_settings())

    assert not caplog.records
