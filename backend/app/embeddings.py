"""Fabrique d'embeddings LangChain selon le fournisseur configuré (local, OpenAI, Mistral)."""

import threading
from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

from app.config import EmbeddingsProvider, Settings
from app.providers import MissingAPIKeyError, require_key

__all__ = ["DEFAULT_MODELS", "LocalEmbeddings", "MissingAPIKeyError", "embeddings_model", "get_embeddings"]

DEFAULT_MODELS: dict[EmbeddingsProvider, str] = {
    "local": "intfloat/multilingual-e5-large",
    "openai": "text-embedding-3-small",
    "mistral": "mistral-embed",
}

# Préfixes (requête, passage) exigés par certains modèles ; fastembed ne les ajoute pas.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


def embeddings_model(settings: Settings) -> str:
    return settings.embeddings_model or DEFAULT_MODELS[settings.embeddings_provider]


def get_embeddings(settings: Settings) -> Embeddings:
    model = embeddings_model(settings)
    match settings.embeddings_provider:
        case "local":
            return LocalEmbeddings(model, settings.embeddings_cache_dir.expanduser())
        case "openai":
            key = _require_key(settings.openai_api_key, "openai", "OPENAI_API_KEY")
            return OpenAIEmbeddings(model=model, api_key=key)
        case "mistral":
            key = _require_key(settings.mistral_api_key, "mistral", "MISTRAL_API_KEY")
            # Tokenizer fourni : sinon la classe télécharge celui de Mixtral depuis
            # Hugging Face à la construction.
            return MistralAIEmbeddings(model=model, api_key=key, tokenizer=_char_tokenizer())


def _require_key(key: SecretStr | None, provider: str, variable: str) -> SecretStr:
    return require_key(
        key,
        setting="EMBEDDINGS_PROVIDER",
        provider=provider,
        variable=variable,
        hint=" ou choisissez EMBEDDINGS_PROVIDER=local",
    )


@lru_cache(maxsize=1)
def _char_tokenizer() -> Tokenizer:
    """Un token par caractère : ne sert qu'à grouper les textes en lots sous la limite de
    tokens de Mistral, et surestime volontairement (comme le repli de la bibliothèque)."""
    tokenizer = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    return tokenizer


class LocalEmbeddings(Embeddings):
    """Embeddings calculés en local (fastembed, ONNX, sans PyTorch).

    Le modèle est téléchargé au premier usage dans `cache_dir`, puis chargé une seule fois
    par processus et partagé entre les instances.
    """

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.query_prefix, self.passage_prefix = _PREFIXES.get(model_name, ("", ""))

    def load(self) -> None:
        """Charge le modèle (et le télécharge au premier lancement) sans rien calculer :
        appelé au démarrage de l'application pour que la première analyse n'attende pas."""
        _load_model(self.model_name, str(self.cache_dir))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = _load_model(self.model_name, str(self.cache_dir))
        return [vector.tolist() for vector in model.embed(texts)]


_load_lock = threading.Lock()


def _load_model(model_name: str, cache_dir: str) -> TextEmbedding:
    # Verrou : deux premières requêtes simultanées ne chargent pas deux fois ~2 Go.
    with _load_lock:
        return _cached_model(model_name, cache_dir)


@lru_cache(maxsize=2)
def _cached_model(model_name: str, cache_dir: str) -> TextEmbedding:
    return TextEmbedding(model_name, cache_dir=cache_dir)
