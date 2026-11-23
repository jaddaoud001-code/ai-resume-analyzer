"""Agent de suggestions v2 (graphe LangGraph) avec de faux LLM scénarisés : aucun appel réseau.
Cas de gain (gain_cv.pdf, offre logistique) ; la réévaluation passe par le vrai matcher
(evaluate_requirement) branché sur un faux LLM d'évaluation.

Objectif v2 : rendre explicite une preuve implicite (compétence « démontré » ou « solide » dont
aucune preuve citée ne nomme le terme de l'offre) ; les lacunes reçoivent une carte « à
décrire », sans texte proposé ni appel au LLM."""

import asyncio
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import groq
import httpx
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
VELMARIS = 2  # Développeuse backend · Transports Velmaris
BRUMELLE = 3  # Développeuse · Atelier Brumelle
SKILLS = 5  # COMPÉTENCES : Langages : Python, SQL
PIPELINE = "Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab."
HONEST_CI_CD = "Pipeline CI/CD sur GitLab : tests et déploiement lancés à chaque fusion sur la branche principale."
USAGE = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}

# Preuve implicite : CI/CD prouvée par une pratique décrite sans son nom.
IMPLICIT_CI_CD = {"R9": ("démontré", VELMARIS, PIPELINE.rstrip("."))}


def _report(
    implicit: dict[str, tuple[str, int, str]] | None = None, levels: dict[str, str] | None = None
) -> MatchReport:
    """Rapport d'analyse. `implicit` : exigence -> (niveau, chunk, citation sans le terme) ;
    `levels` : exigence -> niveau bas, sans preuve. Les autres : « démontré », preuve qui nomme
    le libellé (explicite, donc jamais ciblée)."""
    settings = make_settings()
    implicit, levels = implicit or {}, levels or {}
    results = []
    for requirement in JOB.requirements:
        if requirement.id in implicit:
            level, chunk_id, quote = implicit[requirement.id]
            evidence = [Evidence(chunk_id=chunk_id, quote=quote)]
        elif requirement.id in levels:
            level, evidence = levels[requirement.id], []
        else:
            level, evidence = "démontré", [Evidence(chunk_id=VELMARIS, quote=requirement.label)]
        results.append(
            RequirementMatch(
                requirement=requirement,
                status="ok",
                level=level,  # type: ignore[arg-type]
                llm_level=level,  # type: ignore[arg-type]
                evidence=evidence,
                justification="—",
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


def _matcher(term: str | None = "CI/CD", level: str = "démontré") -> FakeStructuredChatModel:
    """Faux LLM d'évaluation : `level` si un chunk envoyé contient `term` (la puce reformulée),
    « mentionné » sinon ; citations exactes, donc vérifiées par le vrai code."""

    def respond(messages: list[BaseMessage]) -> RequirementAssessment:
        blocks = re.findall(r'<chunk id="(\d+)" section="[^"]*">\n(.*?)\n</chunk>', str(messages[1].content), re.S)
        for chunk_id, text in blocks:
            if term and term in text:
                quote = next(line[2:] for line in text.splitlines() if term in line)
                return _assessment(level, int(chunk_id), quote)
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


def _refusals(events: list[Any]) -> list[tuple[str, str]]:
    return [(e.requirement.id, e.reason) for e in events if isinstance(e, Refusal)]


CANDIDATE_CI_CD = SearchDecision(
    decision="candidat", chunk_id=VELMARIS, bullet_index=1, reason="pratique décrite sans son nom"
)
HONEST = Proposal(
    new_text=HONEST_CI_CD,
    facts=[Fact(claim="tests et déploiement à chaque fusion", chunk_id=VELMARIS, quote=PIPELINE)],
    justification="La puce décrit une intégration et un déploiement continus.",
)
CHEAT = Proposal(
    new_text="Pipeline CI/CD sur GitLab, déployé dans le cloud à chaque fusion sur la branche principale.",
    facts=[Fact(claim="déploiement à chaque fusion", chunk_id=VELMARIS, quote=PIPELINE)],
)
NO_TERM = Proposal(
    new_text="Pipeline automatisé de tests et de déploiement lancé à chaque fusion sur la branche principale.",
    facts=[Fact(claim="tests et déploiement à chaque fusion", chunk_id=VELMARIS, quote=PIPELINE)],
)


def test_honest_suggestion_names_the_offer_term() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)], usage=USAGE)
    matcher = _matcher()

    events = _run(_report(IMPLICIT_CI_CD), agent, matcher)

    assert _types(events) == ["agent_started", "target_started", "step", "step", "step", "step", "suggestion", "done"]
    assert [(e.kind, e.status) for e in events if isinstance(e, AgentStep)] == [
        ("search", "ok"),
        ("propose", "ok"),
        ("verify", "ok"),
        ("reevaluate", "ok"),
    ]
    [suggestion] = [e for e in events if isinstance(e, Suggestion)]
    assert (suggestion.before, suggestion.after) == (PIPELINE, HONEST_CI_CD)
    assert suggestion.terms == ["CI/CD"]
    # Gain v2 : le niveau ne baisse pas (il n'a pas à monter).
    assert (suggestion.level_before, suggestion.level_after) == ("démontré", "démontré")
    assert suggestion.item == "Développeuse backend · Transports Velmaris"
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.suggestions, done.refusals, done.hidden) == (1, 0, 0)
    # Tokens : 3 appels de l'agent (recherche, proposition, juge) + 1 réévaluation.
    assert (done.usage.calls, done.usage.total_tokens) == (4, 6000)
    assert len(agent.calls) == 3 and len(matcher.calls) == 1
    # La recherche reçoit les termes de l'offre, les preuves retenues et l'expérience qui les porte.
    search = str(agent.calls[0][1].content)
    assert "« CI/CD »" in search and f"[chunk {VELMARIS}]" in search and "Atelier Brumelle" not in search


def test_cheating_proposal_is_rejected_by_grounding_then_retried_then_refused() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, CHEAT, CHEAT])

    events = _run(_report(IMPLICIT_CI_CD), agent)

    verify = [e for e in events if isinstance(e, AgentStep) and e.kind == "verify"]
    assert [(e.status, e.attempt) for e in verify] == [("rejected", 1), ("rejected", 2)]
    assert "cloud" in (verify[0].detail or "")
    assert _refusals(events) == [("R9", "aucune_reformulation_fidele")]
    # Jamais de juge ni de réévaluation pour une proposition non ancrée ; la seconde
    # proposition reçoit le motif du rejet.
    assert len(agent.calls) == 3
    assert "Contrôle d'ancrage" in str(agent.calls[2][1].content)
    assert not any(isinstance(e, Suggestion) for e in events)


def test_proposal_without_the_offer_term_is_rejected_before_grounding_and_judge() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, NO_TERM, HONEST, JudgeVerdict(faithful=True)])

    events = _run(_report(IMPLICIT_CI_CD), agent)

    verify = [e for e in events if isinstance(e, AgentStep) and e.kind == "verify"]
    assert verify[0].status == "rejected" and (verify[0].detail or "").startswith("Terme absent")
    assert "« CI/CD »" in (verify[0].detail or "")
    assert [e.type for e in events if isinstance(e, Suggestion)] == ["suggestion"]
    assert len(agent.calls) == 4  # recherche, proposition sans terme, proposition, juge


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

    events = _run(_report(IMPLICIT_CI_CD), agent, matcher)

    verify = [e for e in events if isinstance(e, AgentStep) and e.kind == "verify"]
    assert [e.detail for e in verify] == ["Juge : verbe renforcé", "Juge : portée élargie"]
    assert _refusals(events) == [("R9", "aucune_reformulation_fidele")]
    assert matcher.calls == []  # le juge ne peut que rejeter : aucune réévaluation


def test_no_bullet_can_name_the_term_is_a_normal_refusal() -> None:
    agent = FakeStructuredChatModel(responses=[SearchDecision(decision="aucune_puce", reason="terme plus large")])

    events = _run(_report(IMPLICIT_CI_CD), agent)

    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.reason, refusal.shown, refusal.detail) == ("aucune_puce", True, "terme plus large")
    assert len(agent.calls) == 1


def test_evidence_outside_experiences_is_refused_without_calling_the_llm() -> None:
    agent = FakeStructuredChatModel(responses=[])

    events = _run(_report({"R9": ("démontré", SKILLS, "Python, SQL")}), agent)

    assert _refusals(events) == [("R9", "aucune_puce")]
    assert agent.calls == []


def test_candidate_outside_the_evidence_items_is_refused() -> None:
    # La preuve est chez Velmaris : une puce de l'Atelier Brumelle ne peut pas la porter.
    agent = FakeStructuredChatModel(responses=[SearchDecision(decision="candidat", chunk_id=BRUMELLE, bullet_index=0)])

    events = _run(_report(IMPLICIT_CI_CD), agent)

    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert refusal.reason == "aucune_puce" and "candidat invalide" in (refusal.detail or "")


def test_term_already_explicit_according_to_the_llm() -> None:
    agent = FakeStructuredChatModel(responses=[SearchDecision(decision="deja_explicite", reason="équivalent direct")])

    events = _run(_report(IMPLICIT_CI_CD), agent)

    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.reason, refusal.message) == (
        "deja_explicite",
        "Le terme de l'offre figure déjà dans le CV : aucune reformulation nécessaire.",
    )


def test_level_drop_is_retried_once_then_counted_but_not_shown() -> None:
    agent = FakeStructuredChatModel(
        responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True), HONEST, JudgeVerdict(faithful=True)]
    )
    matcher = _matcher(term=None)  # la reformulation fait tomber le niveau à « mentionné »

    events = _run(_report(IMPLICIT_CI_CD), agent, matcher)

    reevaluate = [e for e in events if isinstance(e, AgentStep) and e.kind == "reevaluate"]
    assert [e.status for e in reevaluate] == ["rejected", "rejected"]
    assert (reevaluate[0].detail or "").startswith("Baisse de niveau (démontré -> mentionné)")
    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.reason, refusal.shown) == ("baisse_de_niveau", False)
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.hidden, done.refusals, done.suggestions) == (1, 0, 0)
    assert len(matcher.calls) == 2
    assert "garde tous les faits" in str(agent.calls[3][1].content)


def test_gaps_get_a_describe_card_without_llm_call() -> None:
    agent = FakeStructuredChatModel(responses=[])

    levels = {"R4": "mentionné", "R5": "mentionné", "R6": "non démontré", "R11": "non démontré"}
    events = _run(_report(levels=levels), agent)

    started = events[0]
    assert isinstance(started, AgentStarted) and started.targets == []
    # R5 (langue) reçoit aussi sa carte ; R11 (savoir-être) est exclu.
    assert [t.id for t in started.to_describe] == ["R4", "R5", "R6"]
    cards = [e for e in events if isinstance(e, Refusal)]
    assert [(e.requirement.id, e.reason, e.shown) for e in cards] == [
        ("R4", "a_decrire", True),
        ("R5", "a_decrire", True),
        ("R6", "a_decrire", True),
    ]
    assert cards[0].message.startswith("Le CV ne le prouve pas encore : si vous l'avez réellement pratiqué")
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.refusals, done.to_describe, done.suggestions) == (3, 3, 0)
    assert agent.calls == [] and done.usage.total_tokens == 0


def test_run_budget_leaves_remaining_targets_untreated() -> None:
    # 1 500 tokens par appel, budget du run 3 000 : recherche + proposition, puis le juge
    # n'est plus appelé ; la cible courante et la suivante sont « non traitées ».
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST], usage=USAGE)
    report = _report({**IMPLICIT_CI_CD, "R10": ("démontré", VELMARIS, PIPELINE.rstrip("."))})

    events = _run(report, agent, suggest_max_tokens_per_run=3000)

    assert _refusals(events) == [("R9", "non_traitee"), ("R10", "non_traitee")]
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.untreated, done.suggestions) == (2, 0)
    assert done.usage.total_tokens == 3000
    assert len(agent.calls) == 2


def test_requirement_budget_stops_only_that_requirement() -> None:
    # R9 d'abord : recherche + proposition = 3 000 tokens, son budget ; le juge n'est pas
    # appelé, R9 est « non traitée » ; R10 est ensuite traitée normalement.
    agent = FakeStructuredChatModel(
        responses=[CANDIDATE_CI_CD, HONEST, SearchDecision(decision="aucune_puce", reason="—")], usage=USAGE
    )
    report = _report({**IMPLICIT_CI_CD, "R10": ("démontré", VELMARIS, PIPELINE.rstrip("."))})

    events = _run(report, agent, suggest_max_tokens_per_requirement=3000)

    assert _refusals(events) == [("R9", "non_traitee"), ("R10", "aucune_puce")]
    done = events[-1]
    assert isinstance(done, AgentDone) and (done.untreated, done.refusals) == (1, 1)
    assert len(agent.calls) == 3


def test_daily_quota_stops_cleanly() -> None:
    quota = groq_error(429, TPD_MESSAGE, GROQ_HEADERS | {"retry-after": "450"})
    agent = FakeStructuredChatModel(responses=[quota])

    events = _run(_report({**IMPLICIT_CI_CD, "R10": ("démontré", VELMARIS, "Pipeline de tests")}), agent)

    assert _types(events) == ["agent_started", "target_started", "error"]
    error = events[-1]
    assert isinstance(error, AgentError) and error.code == "quota_exceeded"
    assert error.message.startswith("Quota journalier du fournisseur LLM atteint")
    assert len(agent.calls) == 1  # jamais de nouvelle tentative, aucune autre cible


def _tool_use_failed(generation: str) -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {
        "error": {
            "message": "Tool choice is required, but model did not call a tool",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": generation,
        }
    }
    return groq.BadRequestError(f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body)


def test_text_answer_twice_is_an_error_with_the_provider_message() -> None:
    # Texte au lieu de l'outil, deux fois (rappel compris) : refus « erreur », message du
    # fournisseur dans le détail ; jamais de nouvelle tentative identique.
    agent = FakeStructuredChatModel(responses=[_tool_use_failed("**rien**"), _tool_use_failed("rien")])

    events = _run(_report(IMPLICIT_CI_CD), agent)

    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert refusal.reason == "erreur"
    assert refusal.detail == (
        "ToolUseFailedError : 400 tool_use_failed : Tool choice is required, but model did not call a tool"
    )
    assert len(agent.calls) == 2
    assert "appelant l'outil SearchDecision" in str(agent.calls[1][-1].content)


def test_text_answer_with_valid_json_is_accepted() -> None:
    agent = FakeStructuredChatModel(responses=[_tool_use_failed('{\n  "decision": "aucune_puce",\n  "reason": "—"\n}')])

    events = _run(_report(IMPLICIT_CI_CD), agent)

    assert _refusals(events) == [("R9", "aucune_puce")]
    assert len(agent.calls) == 1


class TestSelection:
    def test_only_implicit_technical_skills_heaviest_first(self) -> None:
        report = _report(
            {
                "R9": ("démontré", VELMARIS, PIPELINE),  # souhaitée, implicite
                "R2": ("solide", VELMARIS, "FastAPI"),  # obligatoire, implicite (implication)
                "R10": ("démontré", VELMARIS, "Pipeline de tests automatisés"),  # nomme le terme
                "R5": ("démontré", VELMARIS, "B2"),  # langue : hors compétences techniques
                "R11": ("démontré", VELMARIS, "seule"),  # savoir-être
            },
            levels={"R4": "mentionné"},
        )
        report.requirements[7] = report.requirements[7].model_copy(update={"status": "error", "level": None})

        selection = select_targets(report, make_settings(suggest_max_requirements=1))

        assert [t.requirement.id for t in selection.targets] == ["R2"]
        assert selection.skipped == 1  # R9
        assert [t.requirement.id for t in selection.to_describe] == ["R4"]

    def test_nothing_implicit_means_no_target(self) -> None:
        agent = FakeStructuredChatModel(responses=[])

        events = _run(_report(), agent)

        assert _types(events) == ["agent_started", "done"]
        assert isinstance(events[0], AgentStarted) and events[0].targets == []
        assert not any(isinstance(e, TargetStarted) for e in events)


def test_events_are_typed_and_serializable() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)])

    events = _run(_report(IMPLICIT_CI_CD, levels={"R4": "mentionné"}), agent)

    for event in events:
        assert json.loads(event.model_dump_json())["type"] == event.type
    assert isinstance(events[-1].usage, TokenUsage)


def test_reevaluation_failure_is_its_own_reason() -> None:
    # Le LLM d'évaluation échoue (deux fois : nouvelle tentative comprise) : le matcher rend
    # un résultat en erreur. Raison distincte, montrée, jamais « baisse de niveau », pas de
    # nouvelle proposition (la reformulation n'est pas en cause).
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)])
    matcher = FakeStructuredChatModel(responses=[ConnectionError("réseau"), ConnectionError("réseau")])

    events = _run(_report(IMPLICIT_CI_CD), agent, matcher)

    [refusal] = [e for e in events if isinstance(e, Refusal)]
    assert (refusal.reason, refusal.shown) == ("reevaluation_impossible", True)
    done = events[-1]
    assert isinstance(done, AgentDone)
    assert (done.reevaluation_failed, done.hidden, done.refusals, done.suggestions) == (1, 0, 0, 0)
    assert len(agent.calls) == 3


def test_agent_prompts_ask_for_the_tool_only() -> None:
    from app.suggestions import JUDGE_PROMPT, PROPOSE_PROMPT, SEARCH_PROMPT

    for prompt in (SEARCH_PROMPT, PROPOSE_PROMPT, JUDGE_PROMPT):
        assert "Réponds uniquement en appelant l'outil prévu" in prompt


def test_agent_prompt_change_requires_a_new_version() -> None:
    from app.suggestions import JUDGE_PROMPT, PROPOSE_PROMPT, SEARCH_PROMPT, SUGGEST_PROMPT_VERSION

    fingerprints = {
        1: "c4ee68813cf76cf255f45e675f586c75412e984c2f81a61dd35f1dd8793f23c5",
        2: "23a50d918852dcbcdd7fe8539c28fa97a4d0c2a956a31c8514f382dd79e998ee",
    }
    digest = hashlib.sha256((SEARCH_PROMPT + PROPOSE_PROMPT + JUDGE_PROMPT).encode()).hexdigest()
    assert digest == fingerprints.get(SUGGEST_PROMPT_VERSION), "prompts de l'agent modifiés : incrémenter la version"
