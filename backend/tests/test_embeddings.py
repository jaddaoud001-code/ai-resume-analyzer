from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pytest
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from app import embeddings as embeddings_module
from app.embeddings import LocalEmbeddings, MissingAPIKeyError, embeddings_model, get_embeddings
from tests.conftest import make_settings


def test_local_is_the_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Valeurs par défaut du code, pas celles de l'environnement (la CI fixe EMBEDDINGS_CACHE_DIR).
    for variable in ("EMBEDDINGS_PROVIDER", "EMBEDDINGS_MODEL", "EMBEDDINGS_CACHE_DIR"):
        monkeypatch.delenv(variable, raising=False)
    embeddings = get_embeddings(make_settings())

    assert isinstance(embeddings, LocalEmbeddings)
    assert embeddings.model_name == "intfloat/multilingual-e5-large"
    assert embeddings.cache_dir == Path("~/.cache/ai-resume-analyzer/fastembed").expanduser()


@pytest.mark.parametrize(
    ("provider", "key_field", "expected_type", "default_model"),
    [
        ("openai", "openai_api_key", OpenAIEmbeddings, "text-embedding-3-small"),
        ("mistral", "mistral_api_key", MistralAIEmbeddings, "mistral-embed"),
    ],
)
def test_external_providers(provider: str, key_field: str, expected_type: type, default_model: str) -> None:
    configured = make_settings(embeddings_provider=provider, **{key_field: "sk-test"})

    embeddings = get_embeddings(configured)

    assert isinstance(embeddings, expected_type)
    assert embeddings.model == default_model  # type: ignore[attr-defined]


def test_model_name_can_be_overridden() -> None:
    configured = make_settings(embeddings_model="custom/model")

    assert embeddings_model(configured) == "custom/model"
    assert get_embeddings(configured).model_name == "custom/model"  # type: ignore[attr-defined]


@pytest.mark.parametrize(("provider", "variable"), [("openai", "OPENAI_API_KEY"), ("mistral", "MISTRAL_API_KEY")])
@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_raises_clear_error(provider: str, variable: str, key: str | None) -> None:
    configured = make_settings(embeddings_provider=provider, openai_api_key=key, mistral_api_key=key)

    with pytest.raises(MissingAPIKeyError, match=variable):
        get_embeddings(configured)


def test_key_of_other_provider_is_not_required() -> None:
    configured = make_settings(embeddings_provider="mistral", mistral_api_key="sk-test", openai_api_key=None)

    assert isinstance(get_embeddings(configured), MistralAIEmbeddings)


class _FakeTextEmbedding:
    def __init__(self) -> None:
        self.received: list[str] = []

    def embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        for text in texts:
            self.received.append(text)
            yield np.array([1.0, 0.0])


def test_e5_prefixes_are_added(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakeTextEmbedding()
    monkeypatch.setattr(embeddings_module, "_load_model", lambda name, cache_dir: fake)
    embeddings = LocalEmbeddings("intfloat/multilingual-e5-large", tmp_path)

    assert embeddings.embed_documents(["Développeur React"]) == [[1.0, 0.0]]
    assert embeddings.embed_query("expérience React") == [1.0, 0.0]
    assert fake.received == ["passage: Développeur React", "query: expérience React"]


def test_no_prefix_for_other_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakeTextEmbedding()
    monkeypatch.setattr(embeddings_module, "_load_model", lambda name, cache_dir: fake)
    embeddings = LocalEmbeddings("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", tmp_path)

    embeddings.embed_documents(["Développeur React"])
    embeddings.embed_query("React")

    assert fake.received == ["Développeur React", "React"]


def test_model_is_loaded_lazily(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loads: list[str] = []
    monkeypatch.setattr(embeddings_module, "_load_model", lambda name, cache_dir: loads.append(name))

    LocalEmbeddings("intfloat/multilingual-e5-large", tmp_path)

    assert loads == []
