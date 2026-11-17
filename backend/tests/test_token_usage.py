"""Coût en tokens : usage renvoyé par le fournisseur à chaque réponse, additionné par
exigence et par analyse (faux LLM qui renvoie un usage ; aucun appel réseau)."""

import asyncio
from typing import Any

from pydantic import BaseModel

from app.llm_calls import RateLimiter, current_usage, invoke_structured
from app.matcher import analyze, evaluate_requirement
from app.requirements import _finalize
from app.schemas import Chunk, JobRequirements, MatchEvent, Requirement, SearchResult, TokenUsage
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_matcher import EXTRACTED, TODAY, _assess, _responder

USAGE = {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500, "output_token_details": {"reasoning": 120}}


class Answer(BaseModel):
    value: str


def _invoke(model: FakeStructuredChatModel, usage: TokenUsage | None) -> Answer:
    settings = make_settings()

    async def run() -> Answer:
        token = current_usage.set(usage)
        try:
            return await invoke_structured(model, Answer, [], RateLimiter.from_settings(settings), settings)
        finally:
            current_usage.reset(token)

    return asyncio.run(run())


class TestInvokeStructured:
    def test_usage_of_each_response_is_added_to_the_context_counter(self) -> None:
        usage = TokenUsage()

        answer = _invoke(FakeStructuredChatModel(responses=[Answer(value="ok")], usage=USAGE), usage)

        assert answer == Answer(value="ok")
        assert usage.model_dump() == {
            "calls": 1,
            "input_tokens": 1200,
            "output_tokens": 300,
            "reasoning_tokens": 120,
            "total_tokens": 1500,
        }

    def test_an_invalid_output_costs_tokens_too(self) -> None:
        # Première réponse sans sortie structurée valide : nouvelle tentative, deux réponses payées.
        usage = TokenUsage()
        model = FakeStructuredChatModel(responses=[None, Answer(value="ok")], usage=USAGE)

        _invoke(model, usage)

        assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 2400, 600)
        assert model.structured_kwargs["include_raw"] is True

    def test_provider_without_usage_counts_calls_only(self) -> None:
        usage = TokenUsage()

        _invoke(FakeStructuredChatModel(responses=[Answer(value="ok")]), usage)

        assert (usage.calls, usage.total_tokens) == (1, 0)

    def test_without_counter_nothing_is_recorded(self) -> None:
        assert _invoke(FakeStructuredChatModel(responses=[Answer(value="ok")], usage=USAGE), None).value == "ok"


def test_token_usage_sum_and_total() -> None:
    a = TokenUsage(calls=1, input_tokens=100, output_tokens=40, reasoning_tokens=10)
    b = TokenUsage(calls=2, input_tokens=300, output_tokens=60)

    total = TokenUsage.sum([a, b])

    assert (total.calls, total.input_tokens, total.output_tokens, total.reasoning_tokens) == (3, 400, 100, 10)
    assert total.total_tokens == 500


def test_requirement_usage_includes_the_second_chance() -> None:
    settings = make_settings()
    requirement = Requirement(
        id="R1", label="Kubernetes", category="compétence technique", importance="souhaitée", search_queries=["a", "b"]
    )
    chunk = Chunk(section="EXPÉRIENCE", index=0, text="Déploiement sur Kubernetes en production.")
    invented = _assess("démontré", (0, "Expert Kubernetes depuis dix ans"))  # rejetée : seconde chance
    model = FakeStructuredChatModel(responses=[invented, invented], usage=USAGE)

    result = asyncio.run(
        evaluate_requirement(
            requirement,
            [SearchResult(chunk=chunk, score=0.9)],
            model,
            RateLimiter.from_settings(settings),
            settings,
            TODAY,
        )
    )

    assert result.second_chance
    assert (result.usage.calls, result.usage.total_tokens, result.usage.reasoning_tokens) == (2, 3000, 240)


def _run_analysis(pdf: bytes, usage: dict[str, Any], requirements: JobRequirements | None = None) -> list[MatchEvent]:
    settings = make_settings()

    async def run() -> list[MatchEvent]:
        stream = analyze(
            pdf,
            "Offre : Python 3 ans, Kubernetes, React, Rust, Go, rigueur.",
            settings=settings,
            embeddings=KeywordEmbeddings(),
            extraction_llm=FakeStructuredChatModel(responses=[EXTRACTED], usage=usage),
            match_llm=FakeStructuredChatModel(responder=_responder, usage=usage),
            limiter=RateLimiter.from_settings(settings),
            requirements=requirements,
            today=TODAY,
        )
        return [event async for event in stream]

    return asyncio.run(run())


class TestReport:
    def test_report_adds_extraction_and_every_requirement(self, sectioned_cv_pdf: bytes) -> None:
        report = _run_analysis(sectioned_cv_pdf, USAGE)[-1].report  # type: ignore[union-attr]

        per_requirement = [r.usage for r in report.requirements]
        assert report.extraction_usage.calls == 1 and report.extraction_usage.total_tokens == 1500
        assert report.usage == TokenUsage.sum([report.extraction_usage, *per_requirement])
        # Une réponse par appel : 1 extraction + 1 par exigence + secondes chances + nouvelles
        # tentatives sur sortie invalide ou erreur (Rust échoue deux fois : aucune réponse).
        assert report.usage.calls == 1 + sum(u.calls for u in per_requirement)
        assert next(r for r in report.requirements if r.requirement.label == "Rust").usage.calls == 0

    def test_provided_requirements_cost_no_extraction(self, sectioned_cv_pdf: bytes) -> None:
        provided = JobRequirements(job_title="Liste figée", requirements=_finalize(EXTRACTED, 15).requirements[:2])

        report = _run_analysis(sectioned_cv_pdf, USAGE, provided)[-1].report  # type: ignore[union-attr]

        assert report.extraction_usage == TokenUsage()
        assert report.usage.calls == sum(r.usage.calls for r in report.requirements) > 0
