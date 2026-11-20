"""Agent de suggestions (graphe LangGraph, étape 3) avec de faux LLM scénarisés : aucun appel
réseau. Cas de gain (gain_cv.pdf, offre logistique) ; la réévaluation passe par le vrai
matcher (evaluate_requirement) branché sur un faux LLM d'évaluation."""

import asyncio
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from app.chunking import chunk_cv
from app.grounding import Fact
from app.llm_calls import RateLimiter
from app.matcher import RequirementAssessment, requirement_weight
from app.pdf_parser import extract_text
from app.schemas import Evidence, JobRequirements, MatchReport, MatchTimings, RequirementMatch, TokenUsage
from app.suggestions import (
    AgentDone,
    AgentError,
    AgentStarted,
    AgentStep,
    JudgeVerdict,
    Proposal,
    Refusal,
    SearchDecision,
    Suggestion,
    TargetStarted,
    run_suggestions,
    select_targets,
)
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_provider_limits import GROQ_HEADERS, TPD_MESSAGE, groq_error

EVAL = Path(__file__).resolve().parents[1] / "eval"
TODAY = date(2026, 11, 14)
JOB = JobRequirements.model_validate(json.loads((EVAL / "offers" / "logistique.requirements.json").read_text()))
CHUNKS = chunk_cv(extract_text((EVAL / "cvs" / "gain_cv.pdf").read_bytes(), make_settings()).text)
PIPELINE = "Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab."
HONEST_CI_CD = "Pipeline CI/CD sur GitLab : tests et déploiement lancés à chaque fusion sur la branche principale."
USAGE = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}


def _report(levels: dict[str, str]) -> MatchReport:
    """Rapport d'analyse : les exigences citées au niveau donné, les autres « démontré »."""
    settings = make_settings()
    results = []
    for requirement in JOB.requirements:
        level = levels.get(requirement.id, "démontré")
        results.append(
            RequirementMatch(
                requirement=requirement,
                status="ok",
                level=level,  # type: ignore[arg-type]
                llm_level=level,  # type: ignore[arg-type]
                points=settings.match_level_points[level],
                weight=requirement_weight(requirement, settings.match_weights),
            )
        )
    return MatchReport(
        job_title=JOB.job_title,
        score=None,
        requirements=results,
        missing_mandatory=[],
        rejected_evidence_count=0,
        errors=[],
        extraction_model="m",
        match_model="m",
        chunk_count=len(CHUNKS),
        timings=MatchTimings(),
    )


def _matcher(gain_term: str | None = "CI/CD") -> FakeStructuredChatModel:
    """Faux LLM d'évaluation : « démontré » si un chunk envoyé contient `gain_term` (la puce
    reformulée), « mentionné » sinon ; citations exactes, donc vérifiées par le vrai code."""

    def respond(messages: list[BaseMessage]) -> RequirementAssessment:
        blocks = re.findall(r'<chunk id="(\d+)" section="[^"]*">\n(.*?)\n</chunk>', str(messages[1].content), re.S)
        for chunk_id, text in blocks:
            if gain_term and gain_term in text:
                quote = next(line[2:] for line in text.splitlines() if gain_term in line)
                return _assessment("démontré", int(chunk_id), quote)
        chunk_id, text = blocks[0]
        return _assessment("mentionné", int(chunk_id), text.splitlines()[0])

    return FakeStructuredChatModel(responder=respond, usage=USAGE)


def _assessment(level: str, chunk_id: int, quote: str) -> RequirementAssessment:
    return RequirementAssessment(
        level=level,  # type: ignore[arg-type]
        evidence=[Evidence(chunk_id=chunk_id, quote=quote)],
        periods=[],
        gaps=[],
        justification="—",
    )


def _run(
    report: MatchReport,
    agent: FakeStructuredChatModel,
    matcher: FakeStructuredChatModel | None = None,
    **settings: Any,
) -> list[Any]:
    config = make_settings(**settings)
    return asyncio.run(
        run_suggestions(
            report,
            CHUNKS,
            settings=config,
            embeddings=KeywordEmbeddings(),
            agent_llm=agent,
            match_llm=matcher or _matcher(),
            limiter=RateLimiter.from_settings(config),
            today=TODAY,
        )
    )


def _types(events: list[Any]) -> list[str]:
    return [e.type for e in events]


CANDIDATE_CI_CD = SearchDecision(
    decision="candidat", chunk_id=2, bullet_index=1, reason="pratique décrite sans son nom"
)
HONEST = Proposal(
    new_text=HONEST_CI_CD,
    facts=[Fact(claim="tests et déploiement à chaque fusion", chunk_id=2, quote=PIPELINE)],
    justification="La puce décrit une intégration et un déploiement continus.",
)
CHEAT = Proposal(
    new_text="Pipeline CI/CD sur GitLab, déployé dans le cloud à chaque fusion sur la branche principale.",
    facts=[Fact(claim="déploiement à chaque fusion", chunk_id=2, quote=PIPELINE)],
)


def test_honest_suggestion_is_accepted() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)], usage=USAGE)
    matcher = _matcher()

    events = _run(_report({"R9": "mentionné"}), agent, matcher)

    assert _types(events) == ["agent_started", "target_started", "step", "step", "step", "step", "suggestion", "done"]
    assert [(e.kind, e.status) for e in events if isinstance(e, AgentStep)] == [
        ("search", "ok"),
        ("propose", "ok"),
        ("verify", "ok"),
        ("reevaluate", "ok"),
    ]
    [suggestion] = [e for e in events if isinstance(e, Suggestion)]
    assert (suggestion.before, suggestion.after) == (PIPELINE, HONEST_CI_CD)
    assert (suggestion.level_before, suggestion.level_after) == ("mentionné", "démontré")
    assert suggestion.item == "Développeuse backend · Transports Velmaris"
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.suggestions, done.refusals, done.hidden) == (1, 0, 0)
    # Tokens : 3 appels de l'agent (recherche, proposition, juge) + 1 réévaluation.
    assert (done.usage.calls, done.usage.total_tokens) == (4, 6000)
    assert len(agent.calls) == 3 and len(matcher.calls) == 1


def test_cheating_proposal_is_rejected_by_grounding_then_retried_then_refused() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, CHEAT, CHEAT])

    events = _run(_report({"R9": "mentionné"}), agent)

    verify = [e for e in events if isinstance(e, AgentStep) and e.kind == "verify"]
    assert [(e.status, e.attempt) for e in verify] == [("rejected", 1), ("rejected", 2)]
    assert "cloud" in (verify[0].detail or "")
    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.reason, refusal.shown) == ("aucune_reformulation_fidele", True)
    # Jamais de juge ni de réévaluation pour une proposition non ancrée ; la seconde
    # proposition reçoit le motif du rejet.
    assert len(agent.calls) == 3
    assert "Contrôle d'ancrage" in str(agent.calls[2][1].content)
    assert not any(isinstance(e, Suggestion) for e in events)


def test_judge_can_reject() -> None:
    agent = FakeStructuredChatModel(
        responses=[
            CANDIDATE_CI_CD,
            HONEST,
            JudgeVerdict(faithful=False, reason="verbe renforcé"),
            HONEST,
            JudgeVerdict(faithful=False, reason="portée élargie"),
        ]
    )
    matcher = _matcher()

    events = _run(_report({"R9": "mentionné"}), agent, matcher)

    verify = [e for e in events if isinstance(e, AgentStep) and e.kind == "verify"]
    assert [e.detail for e in verify] == ["Juge : verbe renforcé", "Juge : portée élargie"]
    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert refusal.reason == "aucune_reformulation_fidele"
    assert matcher.calls == []  # le juge ne peut que rejeter : aucune réévaluation


def test_nothing_in_the_cv_is_a_normal_refusal() -> None:
    agent = FakeStructuredChatModel(responses=[SearchDecision(decision="rien", reason="aucun déploiement cloud")])

    events = _run(_report({"R4": "non démontré"}), agent)

    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.reason, refusal.message, refusal.shown) == (
        "rien_dans_le_cv",
        "Rien dans le CV ne permet de le prouver.",
        True,
    )
    assert isinstance(events[-1], AgentDone) and events[-1].refusals == 1
    assert len(agent.calls) == 1


def test_one_extra_search_at_most() -> None:
    agent = FakeStructuredChatModel(
        responses=[
            SearchDecision(decision="rien", queries=["hébergement", "conteneurs en production"]),
            SearchDecision(decision="rien", queries=["encore"], reason="toujours rien"),
        ]
    )

    events = _run(_report({"R4": "non démontré"}), agent)

    assert len(agent.calls) == 2
    assert [e.reason for e in events if isinstance(e, Refusal)] == ["rien_dans_le_cv"]


def test_no_gain_is_retried_once_then_counted_but_not_shown() -> None:
    agent = FakeStructuredChatModel(
        responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True), HONEST, JudgeVerdict(faithful=True)]
    )
    matcher = _matcher(gain_term=None)  # le niveau ne monte jamais

    events = _run(_report({"R9": "mentionné"}), agent, matcher)

    reevaluate = [e for e in events if isinstance(e, AgentStep) and e.kind == "reevaluate"]
    assert [e.status for e in reevaluate] == ["rejected", "rejected"]
    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.reason, refusal.shown) == ("sans_effet", False)
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.hidden, done.refusals, done.suggestions) == (1, 0, 0)
    assert len(matcher.calls) == 2


def test_run_budget_leaves_remaining_targets_untreated() -> None:
    # 1 500 tokens par appel, budget du run 3 000 : recherche + proposition, puis le juge
    # n'est plus appelé ; la cible courante et les suivantes sont « non traitées ».
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST], usage=USAGE)
    report = _report({"R9": "mentionné", "R4": "non démontré", "R6": "non démontré"})

    events = _run(report, agent, suggest_max_tokens_per_run=3000)

    untreated = [e for e in events if isinstance(e, Refusal) and e.reason == "non_traitee"]
    assert len(untreated) == 3 and all(e.shown for e in untreated)
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.untreated, done.suggestions) == (3, 0)
    assert done.usage.total_tokens == 3000
    assert len(agent.calls) == 2


def test_requirement_budget_stops_only_that_requirement() -> None:
    # R4 (obligatoire) est traitée d'abord : recherche + proposition = 3 000 tokens, son budget ;
    # le juge n'est pas appelé, R4 est « non traitée » ; R9 est ensuite traitée normalement.
    agent = FakeStructuredChatModel(
        responses=[CANDIDATE_CI_CD, HONEST, SearchDecision(decision="rien", reason="—")], usage=USAGE
    )
    report = _report({"R9": "mentionné", "R4": "non démontré"})

    events = _run(report, agent, suggest_max_tokens_per_requirement=3000)

    assert [(e.requirement.id, e.reason) for e in events if isinstance(e, Refusal)] == [
        ("R4", "non_traitee"),
        ("R9", "rien_dans_le_cv"),
    ]
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.untreated, done.refusals) == (1, 1)
    assert len(agent.calls) == 3


def test_daily_quota_stops_cleanly() -> None:
    quota = groq_error(429, TPD_MESSAGE, GROQ_HEADERS | {"retry-after": "450"})
    agent = FakeStructuredChatModel(responses=[quota])

    events = _run(_report({"R9": "mentionné", "R4": "non démontré"}), agent)

    assert _types(events) == ["agent_started", "target_started", "error"]
    error = events[-1]
    assert isinstance(error, AgentError) and error.code == "quota_exceeded"
    assert error.message.startswith("Quota journalier du fournisseur LLM atteint")
    assert len(agent.calls) == 1  # jamais de nouvelle tentative, aucune autre cible


def test_already_proven_requirement_targeted_by_a_false_negative() -> None:
    # « Développement backend » est prouvé par le CV ; l'analyse l'a donné « mentionné »
    # (faux négatif du matching) : l'agent répond « aucune reformulation nécessaire ».
    agent = FakeStructuredChatModel(
        responses=[
            SearchDecision(decision="deja_prouve", chunk_id=2, bullet_index=0, reason="poste de développeuse backend")
        ]
    )

    events = _run(_report({"R1": "mentionné"}), agent)

    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.requirement.id, refusal.reason, refusal.shown) == ("R1", "deja_prouve", True)
    assert refusal.message == "Le CV le prouve déjà : aucune reformulation nécessaire."
    assert len(agent.calls) == 1 and not any(isinstance(e, Suggestion) for e in events)


class TestSelection:
    def test_soft_skills_and_errors_are_excluded_heaviest_first(self) -> None:
        report = _report({"R4": "non démontré", "R9": "mentionné", "R11": "non démontré", "R6": "non démontré"})
        report.requirements[0] = report.requirements[0].model_copy(update={"status": "error", "level": None})

        targets, skipped = select_targets(report, make_settings(suggest_max_requirements=2))

        # R4 obligatoire (poids 2, 0 point) ; puis R6 souhaitée non démontrée (poids 1, 0 point)
        # avant R9 souhaitée « mentionné » (40 points) ; R11 savoir-être exclu.
        assert [t.requirement.id for t in targets] == ["R4", "R6"]
        assert skipped == 1

    def test_no_gap_means_no_target(self) -> None:
        agent = FakeStructuredChatModel(responses=[])

        events = _run(_report({}), agent)

        assert _types(events) == ["agent_started", "done"]
        assert isinstance(events[0], AgentStarted) and events[0].targets == []
        assert not any(isinstance(e, TargetStarted) for e in events)


def test_events_are_typed_and_serializable() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)])

    events = _run(_report({"R9": "mentionné"}), agent)

    for event in events:
        assert json.loads(event.model_dump_json())["type"] == event.type
    assert isinstance(events[-1].usage, TokenUsage)
