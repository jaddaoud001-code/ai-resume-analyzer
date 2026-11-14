import pytest
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from app.llm import get_llm, llm_model, match_llm_model, match_settings
from app.providers import MissingAPIKeyError
from tests.conftest import make_settings

PROVIDERS = [
    ("groq", "groq_api_key", ChatGroq, "openai/gpt-oss-120b"),
    ("mistral", "mistral_api_key", ChatMistralAI, "mistral-small-latest"),
    ("openai", "openai_api_key", ChatOpenAI, "gpt-4o-mini"),
]


@pytest.mark.parametrize(("provider", "key_field", "expected_type", "default_model"), PROVIDERS)
def test_provider_model_and_deterministic_settings(
    provider: str, key_field: str, expected_type: type, default_model: str
) -> None:
    llm = get_llm(make_settings(llm_provider=provider, **{key_field: "sk-test"}))

    assert isinstance(llm, expected_type)
    assert llm_model(make_settings(llm_provider=provider)) == default_model
    # Température 0 (ChatGroq la remplace par 1e-8, l'API refusant 0 pour certains modèles).
    assert llm.temperature < 1e-6  # type: ignore[attr-defined]
    # Aucune nouvelle tentative cachée : elles sont gérées par app.llm_calls.
    assert llm.max_retries == 0  # type: ignore[attr-defined]


def test_groq_is_the_default_provider() -> None:
    llm = get_llm(make_settings(groq_api_key="sk-test"))

    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "openai/gpt-oss-120b"


def test_seed_is_passed_to_each_provider() -> None:
    groq = get_llm(make_settings(groq_api_key="sk-test", llm_seed=7))
    mistral = get_llm(make_settings(llm_provider="mistral", mistral_api_key="sk-test", llm_seed=7))
    openai = get_llm(make_settings(llm_provider="openai", openai_api_key="sk-test", llm_seed=7))
    unseeded = get_llm(make_settings(llm_provider="mistral", mistral_api_key="sk-test", llm_seed=None))
    unseeded_groq = get_llm(make_settings(groq_api_key="sk-test", llm_seed=None))

    assert groq.model_kwargs == {"seed": 7}  # type: ignore[attr-defined]
    assert mistral.random_seed == 7  # type: ignore[attr-defined]
    assert openai.seed == 7  # type: ignore[attr-defined]
    assert unseeded.random_seed is None  # type: ignore[attr-defined]
    assert unseeded_groq.model_kwargs == {}  # type: ignore[attr-defined]


def test_seed_has_a_default() -> None:
    llm = get_llm(make_settings(llm_provider="mistral", mistral_api_key="sk-test"))

    assert llm.random_seed == 42  # type: ignore[attr-defined]


def test_model_and_timeout_are_configurable() -> None:
    llm = get_llm(
        make_settings(
            llm_provider="mistral", mistral_api_key="sk-test", llm_model="mistral-large-latest", llm_timeout_s=12
        )
    )

    assert llm.model == "mistral-large-latest"  # type: ignore[attr-defined]
    assert llm.timeout == 12  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("provider", "variable"), [("groq", "GROQ_API_KEY"), ("mistral", "MISTRAL_API_KEY"), ("openai", "OPENAI_API_KEY")]
)
@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_raises_clear_error(provider: str, variable: str, key: str | None) -> None:
    settings = make_settings(llm_provider=provider, groq_api_key=key, mistral_api_key=key, openai_api_key=key)

    with pytest.raises(MissingAPIKeyError, match=f"LLM_PROVIDER={provider} mais {variable} est absente"):
        get_llm(settings)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("groq", "openai/gpt-oss-120b"), ("mistral", "ministral-14b-2512"), ("openai", "gpt-4o-mini")],
)
def test_match_model_defaults_per_provider(provider: str, expected: str) -> None:
    settings = make_settings(llm_provider=provider, llm_model="some-extraction-model")

    assert match_llm_model(settings) == expected
    assert match_settings(settings).llm_model == expected
    # L'extraction garde LLM_MODEL.
    assert llm_model(settings) == "some-extraction-model"


def test_match_model_is_configurable() -> None:
    settings = make_settings(match_llm_model="openai/gpt-oss-20b", groq_api_key="sk-test")

    assert get_llm(match_settings(settings)).model_name == "openai/gpt-oss-20b"  # type: ignore[attr-defined]


def test_reasoning_effort_is_configurable_per_step() -> None:
    settings = make_settings(groq_api_key="sk-test", llm_reasoning_effort="medium", match_llm_reasoning_effort="low")

    assert get_llm(settings).reasoning_effort == "medium"  # type: ignore[attr-defined]
    assert get_llm(match_settings(settings)).reasoning_effort == "low"  # type: ignore[attr-defined]
    # Non défini : rien n'est envoyé (défaut du modèle ; les modèles qui ne raisonnent pas refusent le paramètre).
    assert get_llm(make_settings(groq_api_key="sk-test")).reasoning_effort is None  # type: ignore[attr-defined]


def test_reasoning_effort_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="llm_reasoning_effort"):
        make_settings(llm_reasoning_effort="extreme")


def test_key_of_other_provider_is_not_required() -> None:
    llm = get_llm(make_settings(llm_provider="openai", openai_api_key="sk-test", groq_api_key=None))

    assert isinstance(llm, ChatOpenAI)
