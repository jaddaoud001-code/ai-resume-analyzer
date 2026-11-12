"""Éléments communs aux fabriques de modèles externes (embeddings, LLM)."""

from pydantic import SecretStr


class MissingAPIKeyError(RuntimeError):
    pass


def require_key(key: SecretStr | None, *, setting: str, provider: str, variable: str, hint: str = "") -> SecretStr:
    """Renvoie la clé du fournisseur choisi, ou lève une erreur qui dit quoi corriger."""
    if key is None or not key.get_secret_value():
        raise MissingAPIKeyError(f"{setting}={provider} mais {variable} est absente : renseignez-la dans .env{hint}.")
    return key
