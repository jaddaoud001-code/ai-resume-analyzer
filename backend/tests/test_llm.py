import pytest
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from app.llm import get_llm, llm_model, match_llm_model, match_settings
from app.providers import MissingAPIKeyError
from tests.conftest import make_settings


@pytest.mark.parametrize(
    ("provider", "key_field", "expected_type", "default_model"),
    [
        ("mistral", "mistral_api_key", ChatMistralAI, "mistral-small-latest"),
        ("openai", "openai_api_key", ChatOpenAI, "gpt-4o-mini"),
    ],
)
def test_provider_model_and_deterministic_settings(
    provider: str, key_field: str, expected_type: type, default_model: str
) -> None:
    llm = get_llm(make_settings(llm_provider=provider, **{key_field: "sk-test"}))

    assert isinstance(llm, expected_type)
    assert llm_model(make_settings(llm_provider=provider)) == default_model
    assert llm.temperature == 0  # type: ignore[attr-defined]
    # Aucune nouvelle tentative cachée : la seule autorisée est gérée par app.requirements.
    assert llm.max_retries == 0  # type: ignore[attr-defined]


def test_seed_is_passed_to_each_provider() -> None:
    mistral = get_llm(make_settings(mistral_api_key="sk-test", llm_seed=7))
    openai = get_llm(make_settings(llm_provider="openai", openai_api_key="sk-test", llm_seed=7))
    unseeded = get_llm(make_settings(mistral_api_key="sk-test", llm_seed=None))

    assert mistral.random_seed == 7  # type: ignore[attr-defined]
    assert openai.seed == 7  # type: ignore[attr-defined]
    assert unseeded.random_seed is None  # type: ignore[attr-defined]


def test_seed_has_a_default() -> None:
    assert get_llm(make_settings(mistral_api_key="sk-test")).random_seed == 42  # type: ignore[attr-defined]


def test_mistral_is_the_default_provider() -> None:
    llm = get_llm(make_settings(mistral_api_key="sk-test"))

    assert isinstance(llm, ChatMistralAI)
    assert llm.model == "mistral-small-latest"


def test_model_and_timeout_are_configurable() -> None:
    llm = get_llm(make_settings(mistral_api_key="sk-test", llm_model="mistral-large-latest", llm_timeout_s=12))

    assert llm.model == "mistral-large-latest"  # type: ignore[attr-defined]
    assert llm.timeout == 12  # type: ignore[attr-defined]


@pytest.mark.parametrize(("provider", "variable"), [("mistral", "MISTRAL_API_KEY"), ("openai", "OPENAI_API_KEY")])
@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_raises_clear_error(provider: str, variable: str, key: str | None) -> None:
    settings = make_settings(llm_provider=provider, mistral_api_key=key, openai_api_key=key)

    with pytest.raises(MissingAPIKeyError, match=f"LLM_PROVIDER={provider} mais {variable} est absente"):
        get_llm(settings)


@pytest.mark.parametrize(("provider", "expected"), [("mistral", "ministral-14b-2512"), ("openai", "gpt-4o-mini")])
def test_match_model_defaults_per_provider(provider: str, expected: str) -> None:
    settings = make_settings(llm_provider=provider, llm_model="mistral-large-4-0")

    assert match_llm_model(settings) == expected
    assert match_settings(settings).llm_model == expected
    # L'extraction garde LLM_MODEL.
    assert llm_model(settings) == "mistral-large-4-0"


def test_match_model_is_configurable() -> None:
    settings = make_settings(match_llm_model="mistral-large-4-0", mistral_api_key="sk-test")

    assert get_llm(match_settings(settings)).model == "mistral-large-4-0"  # type: ignore[attr-defined]


def test_key_of_other_provider_is_not_required() -> None:
    llm = get_llm(make_settings(llm_provider="openai", openai_api_key="sk-test", mistral_api_key=None))

    assert isinstance(llm, ChatOpenAI)
