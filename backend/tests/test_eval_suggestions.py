"""Évaluation de l'agent de suggestions (eval/eval_suggestions.py) : classification d'après les
annotations, choix des rapports, mesures ; de bout en bout avec de faux LLM. Aucun appel réseau."""

import asyncio
import json
from pathlib import Path

from app.llm_calls import RateLimiter
from app.schemas import TokenUsage
from app.suggestions import JudgeVerdict, Refusal, SearchDecision, Suggestion
from eval.eval_suggestions import (
    GAIN_CASE,
    classify,
    latest_reports,
    load_annotations,
    measure,
    results_path,
    run_case,
    summarize,
)
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_suggestions_agent import CANDIDATE_CI_CD, HONEST, JOB, _matcher, _report

ANNOTATIONS = load_annotations()
GAIN = ANNOTATIONS[GAIN_CASE]
REQUIREMENTS = {r.id: r for r in JOB.requirements}


def _suggestion(rid: str, before: str = "mentionné", after: str = "démontré", bullet: int = 1) -> Suggestion:
    return Suggestion(
        requirement=REQUIREMENTS[rid],
        item="Développeuse backend · Transports Velmaris",
        chunk_id=2,
        bullet_index=bullet,
        before="avant",
        after="après",
        facts=[],
        implied_skills=[],
        level_before=before,
        level_after=after,
        justification="—",
        usage=TokenUsage(),
    )


def _refusal(rid: str, reason: str) -> Refusal:
    return Refusal(
        requirement=REQUIREMENTS[rid], reason=reason, message="—", shown=reason != "sans_effet", usage=TokenUsage()
    )  # type: ignore[arg-type]


class TestClassify:
    def test_suggestions(self) -> None:
        assert classify(_suggestion("R9"), GAIN).verdict == "gain"  # lacune possible
        assert classify(_suggestion("R9"), GAIN).source_expected is True  # Velmaris, puce 1
        assert classify(_suggestion("R9", bullet=2), GAIN).source_expected is False
        assert classify(_suggestion("R4"), GAIN).verdict == "invention"  # lacune impossible
        assert classify(_suggestion("R9", after="solide"), GAIN).verdict == "invention"  # au-delà du maximum
        assert classify(_suggestion("R1"), GAIN).verdict == "deja_prouve_acceptable"

    def test_refusals(self) -> None:
        assert classify(_refusal("R4", "rien_dans_le_cv"), GAIN).verdict == "refus_juste"
        assert classify(_refusal("R4", "sans_effet"), GAIN).verdict == "refus_juste"
        assert classify(_refusal("R9", "rien_dans_le_cv"), GAIN).verdict == "faux_refus"
        assert classify(_refusal("R9", "aucune_reformulation_fidele"), GAIN).verdict == "faux_refus"
        # Déjà prouvé : jamais faux refus.
        assert classify(_refusal("R1", "deja_prouve"), GAIN).verdict == "deja_prouve_acceptable"
        assert classify(_refusal("R1", "rien_dans_le_cv"), GAIN).verdict == "deja_prouve_acceptable"
        assert classify(_refusal("R9", "reevaluation_impossible"), GAIN).verdict == "reevaluation_impossible"
        assert classify(_refusal("R9", "non_traitee"), GAIN).verdict == "non_concluant"


def test_measure_splits_grounding_and_judge_rejections() -> None:
    from app.suggestions import AgentStep

    events = [
        AgentStep(
            requirement_id="R9", kind="verify", status="rejected", attempt=1, detail="Contrôle d'ancrage : « cloud »"
        ),
        AgentStep(requirement_id="R9", kind="verify", status="rejected", attempt=2, detail="Juge : verbe renforcé"),
        _refusal("R9", "sans_effet"),
    ]

    result = measure(GAIN_CASE, "rapport", events, GAIN, 1.0)

    assert len(result.grounding_rejections) == 1 and len(result.judge_rejections) == 1
    assert result.hidden_no_gain == 1
    assert [o.verdict for o in result.outcomes] == ["faux_refus"]


def test_latest_report_per_case_ignores_suggestion_outputs(tmp_path: Path) -> None:
    real = json.loads((Path(__file__).resolve().parents[1] / "eval/results/2026-11-20_difficile_v4.json").read_text())
    report = next(iter(real.values()))
    for name, key in (
        ("2026-11-18_difficile_v3.json", "java_cv__logistique (run 1)"),
        ("2026-11-20_difficile_v4.json", "java_cv__logistique (run 1)"),
        ("2026-11-22_suggestions_v1.json", "java_cv__logistique"),
    ):
        (tmp_path / name).write_text(json.dumps({key: report}))

    found = latest_reports(tmp_path)

    assert list(found) == ["java_cv__logistique"]
    assert found["java_cv__logistique"][0].startswith("2026-11-20_difficile_v4.json")


def test_conserved_reports_are_found() -> None:
    found = latest_reports()

    assert {"backend_cv__logistique", "java_cv__logistique"} <= set(found)


def _evaluate(agent: FakeStructuredChatModel, levels: dict[str, str]) -> object:
    settings = make_settings()
    return asyncio.run(
        run_case(
            GAIN_CASE,
            "rapport construit pour le test",
            _report(levels),
            GAIN,
            settings=settings,
            embeddings=KeywordEmbeddings(),
            agent_llm=agent,
            match_llm=_matcher(),
            limiter=RateLimiter.from_settings(settings),
        )
    )


def test_end_to_end_gain_and_right_refusal() -> None:
    # R4 (obligatoire, impossible) d'abord : « rien » ; puis R9 (possible) : suggestion honnête.
    agent = FakeStructuredChatModel(
        responses=[SearchDecision(decision="rien", reason="—"), CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)]
    )

    result = _evaluate(agent, {"R4": "non démontré", "R9": "mentionné"})

    assert [(o.id, o.verdict) for o in result.outcomes] == [("R4", "refus_juste"), ("R9", "gain")]  # type: ignore[attr-defined]
    summary = summarize([result])  # type: ignore[list-item]
    assert (summary["affirmations inventées"], summary["refus justes"], summary["gains"]) == (0, 1, "1/1")
    assert summary["gains depuis la source annotée"] == 1
    assert result.suggestions[0]["avant"].startswith("Pipeline de tests")  # type: ignore[attr-defined]


def test_end_to_end_catches_an_invention_that_passed_grounding() -> None:
    # Une reformulation ancrée (CI/CD décrit par la puce) présentée pour « déploiement cloud » :
    # l'ancrage la laisse passer, le faux matcher y voit un gain ; l'annotation dit qu'aucune
    # suggestion honnête n'existe pour R4 : affirmation inventée, listée pour l'audit.
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)])

    result = _evaluate(agent, {"R4": "non démontré"})

    assert [(o.id, o.verdict) for o in result.outcomes] == [("R4", "invention")]  # type: ignore[attr-defined]
    assert summarize([result])["affirmations inventées"] == 1  # type: ignore[list-item]
    assert result.suggestions[0]["verdict"] == "invention"  # type: ignore[attr-defined]


def test_results_path() -> None:
    from datetime import date

    path = results_path(date(2026, 11, 22))

    assert path.name.startswith("2026-11-22_suggestions_v") and path.parent.name == "results"


def test_analyze_gain_case_keeps_its_report(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import eval.eval_suggestions as module

    matcher = _matcher()
    monkeypatch.setattr(module, "get_llm", lambda settings: matcher)  # faux LLM : aucun appel réseau
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    settings = make_settings()

    path = asyncio.run(module.analyze_gain_case(settings, KeywordEmbeddings(), RateLimiter.from_settings(settings)))

    assert path.parent == tmp_path and "_gain_v" in path.name
    found = latest_reports(tmp_path)
    assert list(found) == [GAIN_CASE]
    report = found[GAIN_CASE][1]
    assert report.requirements_source == "provided" and len(report.requirements) == len(JOB.requirements)
