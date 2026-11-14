"""Fabrique du modèle de chat LangChain selon le fournisseur configuré (Groq, Mistral, OpenAI)."""

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from app.config import LLMProvider, Settings
from app.providers import require_key

DEFAULT_LLM_MODELS: dict[LLMProvider, str] = {
    "groq": "openai/gpt-oss-120b",
    "mistral": "mistral-small-latest",
    "openai": "gpt-4o-mini",
}
# Évaluation des exigences : un appel par exigence, d'où un modèle plus léger par défaut.
DEFAULT_MATCH_LLM_MODELS: dict[LLMProvider, str] = {
    "groq": "openai/gpt-oss-120b",
    "mistral": "ministral-14b-2512",
    "openai": "gpt-4o-mini",
}


def llm_model(settings: Settings) -> str:
    return settings.llm_model or DEFAULT_LLM_MODELS[settings.llm_provider]


def match_llm_model(settings: Settings) -> str:
    return settings.match_llm_model or DEFAULT_MATCH_LLM_MODELS[settings.llm_provider]


def match_settings(settings: Settings) -> Settings:
    """Settings dont LLM_MODEL et LLM_REASONING_EFFORT sont ceux de l'évaluation : à passer à get_llm."""
    return settings.model_copy(
        update={"llm_model": match_llm_model(settings), "llm_reasoning_effort": settings.match_llm_reasoning_effort}
    )


def get_llm(settings: Settings) -> BaseChatModel:
    """Température 0 et aucune nouvelle tentative interne : l'appelant décide seul des
    nouvelles tentatives (une au maximum, voir app.requirements)."""
    model = llm_model(settings)
    match settings.llm_provider:
        case "groq":
            key = require_key(settings.groq_api_key, setting="LLM_PROVIDER", provider="groq", variable="GROQ_API_KEY")
            return ChatGroq(
                model=model,
                api_key=key,
                temperature=0,
                max_retries=0,
                timeout=settings.llm_timeout_s,
                # Modèles qui raisonnent (gpt-oss) seulement : refusé par l'API pour les autres.
                reasoning_effort=settings.llm_reasoning_effort,
                model_kwargs={"seed": settings.llm_seed} if settings.llm_seed is not None else {},
            )
        case "mistral":
            key = require_key(
                settings.mistral_api_key, setting="LLM_PROVIDER", provider="mistral", variable="MISTRAL_API_KEY"
            )
            # max_retries est ici un nombre total de tentatives ; 0 ou 1 donnent un seul appel.
            return ChatMistralAI(
                model=model,
                api_key=key,
                temperature=0,
                max_retries=0,
                timeout=int(settings.llm_timeout_s),
                random_seed=settings.llm_seed,
            )
        case "openai":
            key = require_key(
                settings.openai_api_key, setting="LLM_PROVIDER", provider="openai", variable="OPENAI_API_KEY"
            )
            return ChatOpenAI(
                model=model,
                api_key=key,
                temperature=0,
                max_retries=0,
                timeout=settings.llm_timeout_s,
                seed=settings.llm_seed,
                reasoning_effort=settings.llm_reasoning_effort,
            )
