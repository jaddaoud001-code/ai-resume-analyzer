"""Contrat backend / frontend : un exemple valide de chaque événement des flux SSE, produit à
partir des modèles Pydantic : POST /analyze dans frontend/src/api/__fixtures__/events.sample.json,
POST /suggest (agent de suggestions, forme publique : public_json) dans
suggest_events.sample.json. Le frontend (Vitest) vérifie qu'il reconnaît chaque exemple avec
ses champs.

Régénérer le fichier après une modification des schémas :
    cd backend && UPDATE_CONTRACT_FIXTURES=1 .venv/bin/pytest tests/test_contract.py
"""

import json
import os
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

from app.grounding import ImpliedSkill
from app.matcher import rate_limit_status
from app.schemas import (
    AnalysisError,
    AnalysisStarted,
    CVIndexed,
    Evidence,
    MatchEvent,
    MatchReport,
    MatchTimings,
    Period,
    ReportReady,
    Requirement,
    RequirementEvaluated,
    RequirementMatch,
    RequirementsExtracted,
    TokenUsage,
)
from app.suggestions import (
    AgentDone,
    AgentError,
    AgentEvent,
    AgentStarted,
    AgentStep,
    AgentTarget,
    Refusal,
    SourceQuote,
    Suggestion,
    TargetStarted,
    public_json,
)

FIXTURES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "api" / "__fixtures__"
FIXTURE = FIXTURES / "events.sample.json"
SUGGEST_FIXTURE = FIXTURES / "suggest_events.sample.json"
UPDATE_ENV = "UPDATE_CONTRACT_FIXTURES"

# Données fictives (aucune donnée d'un vrai CV).
PYTHON = Requirement(
    id="R1",
    label="Python",
    category="compétence technique",
    importance="obligatoire",
    min_years=3,
    search_queries=["Python", "développement Python"],
)
GO = Requirement(
    id="R2",
    label="Go",
    category="compétence technique",
    importance="obligatoire",
    search_queries=["Go", "Golang"],
)
DOCKER = Requirement(
    id="R3",
    label="Docker",
    category="compétence technique",
    importance="souhaitée",
    search_queries=["Docker", "conteneurs"],
)
AUTONOMY = Requirement(
    id="R4",
    label="Autonomie",
    category="savoir-être",
    importance="souhaitée",
    search_queries=["autonomie", "travail autonome"],
)

PYTHON_MATCH = RequirementMatch(
    requirement=PYTHON,
    status="ok",
    level="mentionné",
    llm_level="solide",
    evidence=[Evidence(chunk_id=3, quote="Conception d'API REST en Python avec FastAPI")],
    rejected_evidence=1,
    rejected_evidence_items=[Evidence(chunk_id=7, quote="Expert Python depuis 10 ans")],
    periods=[Period(start="2023-09", end="en cours")],
    rejected_periods=0,
    total_months=14,
    capped_for_duration=True,
    gaps=["Durée prouvée inférieure aux 3 ans demandés"],
    justification="Python est utilisé en stage, mais sur 14 mois seulement.",
    retrieved_chunk_ids=[3, 1, 7, 2, 5],
    points=40,
    weight=2.0,
)
GO_MATCH = RequirementMatch(
    requirement=GO,
    status="ok",
    level="non démontré",
    llm_level="mentionné",
    downgraded=True,
    second_chance=True,
    rejected_evidence=1,
    rejected_evidence_items=[Evidence(chunk_id=4, quote="Développement de microservices en Go")],
    gaps=["Go n'apparaît dans aucun extrait"],
    justification="Aucune preuve vérifiable.",
    retrieved_chunk_ids=[1, 2, 4, 5, 6],
    points=0,
    weight=2.0,
)
DOCKER_MATCH = RequirementMatch(
    requirement=DOCKER,
    status="error",
    error="Évaluation impossible (HTTPStatusError 429).",
    retrieved_chunk_ids=[2, 4, 1, 3, 6],
    weight=1.0,
)
AUTONOMY_MATCH = RequirementMatch(
    requirement=AUTONOMY,
    status="ok",
    level="démontré",
    llm_level="solide",
    capped_soft_skill=True,
    evidence=[Evidence(chunk_id=4, quote="Projet mené seul de la conception au déploiement")],
    justification="Projet personnel conduit en autonomie.",
    retrieved_chunk_ids=[4, 2, 1, 3, 5],
    points=75,
    weight=0.5,
)
MATCHES = [PYTHON_MATCH, GO_MATCH, DOCKER_MATCH, AUTONOMY_MATCH]


def sample_events() -> list[BaseModel]:
    report = MatchReport(
        job_title="Développeuse backend",
        score=33,
        requirements=MATCHES,
        missing_mandatory=[GO],
        rejected_evidence_count=1,
        errors=["R3"],
        extraction_model="mistral-large-4-0",
        match_model="mistral-large-4-0",
        requirements_source="extracted",
        chunk_count=8,
        timings=MatchTimings(extraction_s=21.4, indexing_s=3.2, retrieval_s=0.4, evaluation_s=96.1),
    )
    return [
        AnalysisStarted(timeout_s=600),
        CVIndexed(chunk_count=8, seconds=3.2),
        RequirementsExtracted(
            job_title="Développeuse backend",
            requirements=[m.requirement for m in MATCHES],
            seconds=21.4,
            requirements_source="extracted",
        ),
        rate_limit_status(60),
        *[
            RequirementEvaluated(result=match, done=done, total=len(MATCHES))
            for done, match in enumerate(MATCHES, start=1)
        ],
        ReportReady(report=report),
        AnalysisError(code="timeout", message="L'analyse a dépassé la durée maximale de 600 s."),
    ]


def render(events: list[BaseModel]) -> str:
    return json.dumps([e.model_dump(mode="json") for e in events], ensure_ascii=False, indent=2) + "\n"


def _event_types() -> set[str]:
    """Tous les types possibles du flux : ceux de MatchEvent, plus « error »."""
    union = get_args(MatchEvent)[0]
    models: list[type[BaseModel]] = [*get_args(union), AnalysisError]
    return {model.model_fields["type"].default for model in models}


def test_samples_cover_every_event_type() -> None:
    assert {e.type for e in sample_events()} == _event_types()  # type: ignore[attr-defined]


def test_samples_are_valid_models() -> None:
    for event in sample_events():
        assert type(event).model_validate(event.model_dump(mode="json")) == event


def test_fixture_is_up_to_date() -> None:
    expected = render(sample_events())
    if os.environ.get(UPDATE_ENV) == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(expected, encoding="utf-8")
    current = FIXTURE.read_text(encoding="utf-8") if FIXTURE.exists() else ""
    assert current == expected, (
        f"{FIXTURE.name} n'est pas à jour avec les schémas : "
        f"cd backend && {UPDATE_ENV}=1 .venv/bin/pytest tests/test_contract.py"
    )


# --- POST /suggest : événements de l'agent (forme publique) -----------------------------------

CI_CD = Requirement(
    id="R5",
    label="CI/CD",
    category="compétence technique",
    importance="souhaitée",
    search_queries=["CI/CD", "intégration continue"],
)
USAGE = TokenUsage(calls=4, input_tokens=4200, output_tokens=1300, reasoning_tokens=700)


def sample_agent_events() -> list[BaseModel]:
    targets = [
        AgentTarget(id="R5", label="CI/CD", level="démontré", weight=1.0, terms=["CI/CD", "intégration continue"])
    ]
    return [
        AgentStarted(
            targets=targets,
            skipped=0,
            to_describe=[AgentTarget(id="R2", label="Go", level="non démontré", weight=2.0, terms=["Go"])],
            limits={"max_requirements": 3, "max_attempts": 2},
        ),
        Refusal(
            requirement=GO,
            reason="a_decrire",
            message="Le CV ne le prouve pas encore : si vous l'avez réellement pratiqué, décrivez-le.",
            shown=True,
            detail="niveau actuel : non démontré",
            usage=TokenUsage(),
        ),
        TargetStarted(requirement=CI_CD, level="démontré", rank=1),
        AgentStep(
            requirement_id="R5",
            kind="propose",
            status="ok",
            attempt=1,
            detail="Pipeline CI/CD de tests lancé à chaque fusion.",
            # Champs d'audit : absents de la forme publique (public_json).
            facts=[],
            implied_skills=[ImpliedSkill(skill="Git", chunk_id=3, quote="GitLab CI")],
        ),
        rate_limit_status(60),
        Suggestion(
            requirement=CI_CD,
            item="Développeuse backend · Entreprise fictive",
            chunk_id=3,
            bullet_index=1,
            before="Pipeline de tests lancé à chaque fusion.",
            after="Pipeline CI/CD de tests lancé à chaque fusion.",
            terms=["CI/CD"],
            facts=[SourceQuote(chunk_id=3, quote="Pipeline de tests lancé à chaque fusion.")],
            implied_skills=[],
            level_before="démontré",
            level_after="démontré",
            justification="",
            judge="Seul le terme est ajouté.",
            usage=USAGE,
        ),
        AgentDone(suggestions=1, refusals=1, to_describe=1, hidden=0, untreated=0, reevaluation_failed=0, usage=USAGE),
        AgentError(code="timeout", message="Les suggestions ont dépassé la durée maximale de 600 s."),
    ]


def render_public(events: list[BaseModel]) -> str:
    return json.dumps([json.loads(public_json(e)) for e in events], ensure_ascii=False, indent=2) + "\n"


def test_agent_samples_cover_every_agent_event_type() -> None:
    models = get_args(AgentEvent)
    assert {e.type for e in sample_agent_events()} == {m.model_fields["type"].default for m in models}  # type: ignore[attr-defined]


def test_agent_public_samples_never_expose_model_paraphrase() -> None:
    for payload in json.loads(render_public(sample_agent_events())):
        assert "claim" not in json.dumps(payload)
        if payload["type"] == "step":
            assert "facts" not in payload and "implied_skills" not in payload


def test_agent_fixture_is_up_to_date() -> None:
    expected = render_public(sample_agent_events())
    if os.environ.get(UPDATE_ENV) == "1":
        SUGGEST_FIXTURE.write_text(expected, encoding="utf-8")
    current = SUGGEST_FIXTURE.read_text(encoding="utf-8") if SUGGEST_FIXTURE.exists() else ""
    assert current == expected, (
        f"{SUGGEST_FIXTURE.name} n'est pas à jour avec les schémas : "
        f"cd backend && {UPDATE_ENV}=1 .venv/bin/pytest tests/test_contract.py"
    )
