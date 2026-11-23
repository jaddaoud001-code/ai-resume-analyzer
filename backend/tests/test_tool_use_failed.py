"""Le modèle répond en texte au lieu d'appeler l'outil (400 tool_use_failed chez Groq) :
JSON valide accepté après validation Pydantic, sinon une seule nouvelle tentative avec un
rappel, jamais de nouvelle tentative identique. Aucun appel réseau."""

import asyncio
from typing import Literal

import groq
import httpx
import pytest
from pydantic import BaseModel

from app.llm_calls import (
    RateLimiter,
    ToolUseFailedError,
    describe_error,
    invoke_structured,
    parse_generation,
    tool_use_failure,
)
from tests.conftest import FakeStructuredChatModel, make_settings
from tests.test_provider_limits import groq_error

MESSAGE = "Tool choice is required, but model did not call a tool"


class Decision(BaseModel):
    decision: Literal["oui", "non"]
    reason: str = ""


def tool_use_failed(generation: str | None, message: str = MESSAGE) -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error: dict[str, str] = {"message": message, "type": "invalid_request_error", "code": "tool_use_failed"}
    if generation is not None:
        error["failed_generation"] = generation
    body = {"error": error}
    return groq.BadRequestError(f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body)


def other_bad_request() -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "context too long", "type": "invalid_request_error", "code": "context_length"}}
    return groq.BadRequestError(f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body)


def run(model: FakeStructuredChatModel) -> Decision:
    settings = make_settings()
    return asyncio.run(
        invoke_structured(model, Decision, [], RateLimiter.from_settings(settings), settings)  # type: ignore[arg-type]
    )


def test_valid_json_text_is_accepted_without_new_call() -> None:
    model = FakeStructuredChatModel(responses=[tool_use_failed('{\n  "decision": "non"\n}')])

    assert run(model) == Decision(decision="non")
    assert len(model.calls) == 1


def test_fenced_json_text_is_accepted() -> None:
    model = FakeStructuredChatModel(responses=[tool_use_failed('```json\n{"decision": "oui", "reason": "r"}\n```')])

    assert run(model) == Decision(decision="oui", reason="r")


def test_text_then_tool_call_after_one_reminder() -> None:
    model = FakeStructuredChatModel(responses=[tool_use_failed("**non**"), Decision(decision="non")])

    assert run(model) == Decision(decision="non")
    assert len(model.calls) == 2
    assert model.calls[0] == []  # requête d'origine
    reminder = model.calls[1][-1]
    assert "Réponds uniquement en appelant l'outil Decision" in str(reminder.content)


def test_json_invalid_for_the_schema_is_not_accepted() -> None:
    # JSON bien formé mais valeur hors du schéma : validation Pydantic refusée -> rappel.
    model = FakeStructuredChatModel(responses=[tool_use_failed('{"decision": "peut-être"}'), Decision(decision="oui")])

    assert run(model) == Decision(decision="oui")
    assert len(model.calls) == 2


def test_text_twice_raises_with_the_provider_message_and_no_identical_retry() -> None:
    model = FakeStructuredChatModel(responses=[tool_use_failed("non"), tool_use_failed(None)])

    with pytest.raises(ToolUseFailedError) as caught:
        run(model)

    assert str(caught.value) == f"400 tool_use_failed : {MESSAGE}"
    assert len(model.calls) == 2  # une seule nouvelle tentative, celle du rappel
    assert describe_error(caught.value) == f"ToolUseFailedError : 400 tool_use_failed : {MESSAGE}"


def test_provider_message_is_truncated() -> None:
    failure = tool_use_failure(tool_use_failed("x", message="m" * 500))

    assert failure is not None and len(str(failure)) == len("400 tool_use_failed : ") + 200


def test_other_errors_keep_their_retry_policy() -> None:
    # Un autre 400 : une nouvelle tentative identique (politique habituelle), aucun rappel.
    model = FakeStructuredChatModel(responses=[other_bad_request(), Decision(decision="oui")])

    assert run(model) == Decision(decision="oui")
    assert len(model.calls) == 2 and model.calls[1] == []
    assert tool_use_failure(other_bad_request()) is None
    assert tool_use_failure(groq_error(429, "Rate limit reached")) is None
    assert describe_error(other_bad_request()) == "BadRequestError 400"  # sans le message du fournisseur


@pytest.mark.parametrize("text", [None, "", "rien", "[1, 2]", '{"reason": "sans décision"}', "```\nnon\n```"])
def test_parse_generation_rejects_anything_but_a_valid_object(text: str | None) -> None:
    assert parse_generation(text, Decision) is None
