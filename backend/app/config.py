"""Configuration de l'application, lue depuis l'environnement et le fichier .env."""

import difflib
import logging
import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# .env à la racine du repo, résolu indépendamment du répertoire de lancement.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

EmbeddingsProvider = Literal["local", "openai", "mistral"]
LLMProvider = Literal["mistral", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Seule la clé du fournisseur choisi est exigée, au moment où on l'utilise.
    openai_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:5173"]

    # Embeddings : "local" n'envoie aucune donnée à l'extérieur.
    embeddings_provider: EmbeddingsProvider = "local"
    # None : modèle par défaut du fournisseur (voir app.embeddings.DEFAULT_MODELS).
    embeddings_model: str | None = None
    embeddings_cache_dir: Path = Path("~/.cache/ai-resume-analyzer/fastembed")
    # LLM : extraction des exigences, puis évaluation de la correspondance.
    llm_provider: LLMProvider = "mistral"
    # None : modèle par défaut du fournisseur (voir app.llm.DEFAULT_LLM_MODELS).
    llm_model: str | None = None
    # Modèle d'évaluation des exigences (un appel par exigence) ; None : défaut du fournisseur.
    match_llm_model: str | None = None
    llm_timeout_s: float = 60
    # Graine transmise au fournisseur (random_seed chez Mistral, seed chez OpenAI) pour des
    # résultats reproductibles ; None : aucune graine.
    llm_seed: int | None = 42
    # Limiteur partagé par tous les appels au LLM du processus : départs espacés d'au moins
    # 1 / LLM_RPS seconde, au plus LLM_CONCURRENCY appels en cours. 0 : pas d'espacement.
    llm_rps: float = 0.5
    llm_concurrency: int = 4
    # Nouvelles tentatives sur 429 (attente croissante ou Retry-After) ; une seule sinon.
    llm_rate_limit_retries: int = 3
    llm_rate_limit_backoff_s: float = 2.0

    # Matching CV / exigences et score.
    match_k: int = 5
    # Places du top MATCH_K réservées aux sections d'expérience et de projets (les listes de
    # compétences, très proches des requêtes, occupent sinon tout le top k).
    match_reserved_k: int = 3
    # Mots-clés (majuscules, sans accents) qui identifient ces sections dans leur titre.
    match_reserved_sections: list[str] = ["EXPERIENCE", "PROJET", "STAGE"]
    match_level_points: dict[str, int] = {
        "non démontré": 0,
        "mentionné": 40,
        "démontré": 75,
        "solide": 100,
    }
    match_weights: dict[str, float] = {
        "obligatoire": 2.0,
        "souhaitée": 1.0,
        "savoir-être souhaité": 0.5,
    }

    # Analyses en flux (POST /analyze) : simultanées au plus (le quota du fournisseur est
    # partagé ; 429 au-delà), durée maximale d'une analyse, intervalle des pings SSE.
    analysis_concurrency: int = 2
    analysis_timeout_s: float = 600
    sse_ping_s: float = 15

    # Offre d'emploi.
    max_job_offer_chars: int = 15_000
    max_requirements: int = 15
    # Cache LRU en mémoire des exigences extraites (par offre, modèle et version du prompt) ;
    # 0 : désactivé.
    requirements_cache_size: int = 64

    # Découpage du CV et recherche.
    chunk_max_chars: int = 800
    chunk_overlap: int = 80
    search_k: int = 4

    # Limites d'upload et d'extraction des CV.
    max_pdf_size_bytes: int = 5 * 1024 * 1024
    max_pdf_pages: int = 10
    # En dessous, le PDF est considéré comme scanné / sans texte extractible.
    min_text_chars: int = 100

    enable_debug_routes: bool = True
    # DEBUG pour journaliser les empreintes des extractions de /debug/parse.
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --- Garde-fou sur le .env ---------------------------------------------------------
# extra="ignore" : une variable mal nommée (MATCH_MAX_RETRIES au lieu de
# LLM_RATE_LIMIT_RETRIES) serait ignorée sans bruit ; on la signale au démarrage.

# Nom de la variable en début de ligne ; seuls les noms sont lus, jamais les valeurs.
_ENV_NAME = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
_SUGGESTION_MIN_SCORE = 0.75


def env_file_names(path: Path) -> list[str]:
    """Noms des variables définies dans le fichier, en majuscules, dans l'ordre des lignes
    (doublons compris). Fichier absent : liste vide."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [m.group(1).upper() for line in lines if (m := _ENV_NAME.match(line))]


def _stem(token: str) -> str:
    return token.rstrip("S")  # EMBEDDING_ ~ EMBEDDINGS_, REQUIREMENT_ ~ REQUIREMENTS_


def _similarity(name: str, known: str) -> float:
    """Ressemblance des caractères, plus un bonus si le dernier mot (le sens : RETRIES,
    SEED…) ou le premier (le groupe : LLM, MATCH…) est le même."""
    a, b = name.split("_"), known.split("_")
    return (
        difflib.SequenceMatcher(None, name, known).ratio()
        + 0.3 * (_stem(a[-1]) == _stem(b[-1]))
        + 0.1 * (_stem(a[0]) == _stem(b[0]))
    )


def closest_setting(name: str, known: Iterable[str]) -> str | None:
    best = max(known, key=lambda k: _similarity(name, k), default=None)
    return best if best is not None and _similarity(name, best) >= _SUGGESTION_MIN_SCORE else None


def env_file_warnings(path: Path, settings_cls: type[BaseSettings] = Settings) -> list[str]:
    """Avertissements pour les variables du fichier qui ressemblent aux nôtres (même premier
    mot qu'un paramètre : LLM_, MATCH_, EMBEDDINGS_, MAX_…) mais qu'aucun paramètre ne
    connaît, avec le nom le plus proche ; et pour les variables définies plusieurs fois."""
    known = [name.upper() for name in settings_cls.model_fields]
    prefixes = {_stem(name.split("_")[0]) for name in known}
    names = env_file_names(path)
    warnings: list[str] = []
    for name in dict.fromkeys(names):
        if name not in known and _stem(name.split("_")[0]) in prefixes:
            suggestion = closest_setting(name, known)
            hint = f" ; vouliez-vous dire {suggestion} ?" if suggestion else ""
            warnings.append(f"Variable {name} du .env inconnue, ignorée{hint}")
    for name, count in Counter(names).items():
        if count > 1:
            warnings.append(f"Variable {name} définie {count} fois dans le .env : seule la dernière compte")
    return warnings


def warn_about_env_file(path: Path = ENV_FILE) -> None:
    for message in env_file_warnings(path):
        logger.warning(message)
