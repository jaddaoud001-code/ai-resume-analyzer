"""Évaluation de l'agent de suggestions v2 (eval/eval_suggestions.py) : classification d'après
les annotations (cibles implicites, lacunes), choix des rapports, mesures ; de bout en bout
avec de faux LLM. Aucun appel réseau."""

import asyncio
import json
from pathlib import Path

from app.grounding import Fact
from app.llm_calls import RateLimiter
from app.schemas import TokenUsage
from app.suggestions import AgentStarted, AgentStep, JudgeVerdict, Proposal, Refusal, SearchDecision, Suggestion
from eval.eval_suggestions import (
    GAIN_CASE,
    classify,
    latest_reports,
    load_annotations,
    measure,
    report_by_source,
    results_path,
    run_case,
    summarize,
)
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_suggestions_agent import (
    CANDIDATE_CI_CD,
    HONEST,
    HONEST_CI_CD,
    IMPLICIT_CI_CD,
    JOB,
    VELMARIS,
    _matcher,
    _report,
)

ANNOTATIONS = load_annotations()
GAIN = ANNOTATIONS[GAIN_CASE]
REQUIREMENTS = {r.id: r for r in JOB.requirements}
SERVICE = "Service de calcul des tarifs de livraison, appelé par le site marchand et l'application des livreurs."


def _suggestion(rid: str, bullet: int = 1, after: str = HONEST_CI_CD) -> Suggestion:
    return Suggestion(
        requirement=REQUIREMENTS[rid],
        item="Développeuse backend · Transports Velmaris",
        chunk_id=VELMARIS,
        bullet_index=bullet,
        before="avant",
        after=after,
        terms=["CI/CD"],
        facts=[],
        implied_skills=[],
        level_before="démontré",
        level_after="démontré",
        justification="—",
        usage=TokenUsage(),
    )


def _refusal(rid: str, reason: str) -> Refusal:
    return Refusal(
        requirement=REQUIREMENTS[rid],
        reason=reason,  # type: ignore[arg-type]
        message="—",
        shown=reason != "baisse_de_niveau",
        usage=TokenUsage(),
    )


class TestAmbiguousTarget:
    SII = ANNOTATIONS["demo_cv__sii"]

    def _event(self, after: str | None) -> Suggestion | Refusal:
        from app.schemas import JobRequirements

        sii = JobRequirements.model_validate(
            json.loads((Path(__file__).resolve().parents[1] / "eval/offers/sii.requirements.json").read_text())
        )
        requirement = next(r for r in sii.requirements if r.id == "R8")
        if after is None:
            return Refusal(requirement=requirement, reason="aucune_puce", message="—", shown=True, usage=TokenUsage())
        return _suggestion("R9", bullet=0, after=after).model_copy(update={"requirement": requirement, "chunk_id": 13})

    def test_refusal_and_keeping_the_action_are_acceptable_replacing_it_is_an_invention(self) -> None:
        entry = next(e for e in self.SII["implicit"] if e["id"] == "R8")

        assert classify(self._event(None), self.SII).verdict == "refus_acceptable"
        assert classify(self._event(entry["expected_rewrite"]), self.SII).verdict == "gain"
        invented = classify(self._event(entry["tempting_rewrite"]), self.SII)
        assert invented.verdict == "invention" and "classement" in (invented.detail or "")


class TestClassify:
    def test_suggestions(self) -> None:
        gain = classify(_suggestion("R9"), GAIN)
        assert (gain.verdict, gain.source_expected, gain.term_expected) == ("gain", True, True)
        assert classify(_suggestion("R9", bullet=2), GAIN).source_expected is False
        assert classify(_suggestion("R10", after="Pipeline sur GitLab."), GAIN).term_expected is False
        assert classify(_suggestion("R8"), GAIN).verdict == "invention"  # cible annotée impossible
        assert classify(_suggestion("R1"), GAIN).verdict == "non_annotee"

    def test_refusals(self) -> None:
        assert classify(_refusal("R8", "aucune_puce"), GAIN).verdict == "refus_juste"
        assert classify(_refusal("R8", "aucune_reformulation_fidele"), GAIN).verdict == "refus_juste"
        assert classify(_refusal("R9", "aucune_puce"), GAIN).verdict == "faux_refus"
        assert classify(_refusal("R9", "deja_explicite"), GAIN).verdict == "faux_refus"
        assert classify(_refusal("R9", "baisse_de_niveau"), GAIN).verdict == "baisse_de_niveau"
        assert classify(_refusal("R9", "reevaluation_impossible"), GAIN).verdict == "reevaluation_impossible"
        assert classify(_refusal("R9", "non_traitee"), GAIN).verdict == "non_concluant"
        assert classify(_refusal("R9", "erreur"), GAIN).verdict == "non_concluant"
        card = classify(_refusal("R4", "a_decrire"), GAIN)
        assert (card.kind, card.verdict) == ("lacune", "carte")


def test_measure_splits_rejections_by_cause() -> None:
    events = [
        AgentStarted(targets=[], skipped=2, limits={}),
        AgentStep(requirement_id="R9", kind="verify", status="rejected", attempt=1, detail="Terme absent : …"),
        AgentStep(
            requirement_id="R9", kind="verify", status="rejected", attempt=1, detail="Contrôle d'ancrage : « cloud »"
        ),
        AgentStep(requirement_id="R9", kind="verify", status="rejected", attempt=2, detail="Juge : verbe renforcé"),
        AgentStep(requirement_id="R9", kind="reevaluate", status="rejected", attempt=2, detail="Baisse de niveau"),
        _refusal("R9", "baisse_de_niveau"),
    ]

    result = measure(GAIN_CASE, GAIN["implicit_report"], events, GAIN, 1.0)

    assert [len(x) for x in (result.term_rejections, result.grounding_rejections, result.judge_rejections)] == [1] * 3
    assert len(result.level_drops) == 1 and result.skipped == 2 and result.warning is None
    assert [o.verdict for o in result.outcomes] == ["baisse_de_niveau"]
    summary = summarize([result])
    assert (summary["baisses de niveau (échecs)"], summary["cibles au-delà de la limite"]) == (1, 2)


def test_first_word_rule_rejections_are_counted_apart_with_the_rejected_proposal() -> None:
    from app.grounding import ACTION_LOST
    from app.suggestions import GROUNDING_PREFIX

    only = f"{GROUNDING_PREFIX}« Classement » {ACTION_LOST} (même racine acceptée)."
    mixed = f"{GROUNDING_PREFIX}« Service » {ACTION_LOST} (même racine acceptée). ; « rest » n'apparaît pas"
    other = f"{GROUNDING_PREFIX}« cloud » n'apparaît pas dans cette expérience."
    events = [
        AgentStep(requirement_id="R9", kind="propose", status="ok", attempt=1, detail="Analyse des tickets…"),
        AgentStep(requirement_id="R9", kind="verify", status="rejected", attempt=1, detail=only),
        AgentStep(requirement_id="R8", kind="verify", status="rejected", attempt=1, detail=mixed),
        AgentStep(requirement_id="R4", kind="verify", status="rejected", attempt=1, detail=other),
    ]

    result = measure(GAIN_CASE, GAIN["implicit_report"], events, GAIN, 0.0)

    assert len(result.action_rejections) == 1 and "« Analyse des tickets… »" in result.action_rejections[0]
    assert result.action_with_other_issues == 1 and len(result.grounding_rejections) == 2
    summary = summarize([result])
    assert summary["rejets par la règle du premier mot (seul motif)"] == 1
    assert summary["rejets où la règle du premier mot s'ajoute à d'autres motifs"] == 1


def test_measure_warns_when_annotations_come_from_another_report() -> None:
    assert measure(GAIN_CASE, "autre.json : x (run 1)", [], GAIN, 0.0).warning is not None


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

    assert {"backend_cv__logistique", "java_cv__logistique", GAIN_CASE} <= set(found)
    assert report_by_source(found[GAIN_CASE][0]).job_title == found[GAIN_CASE][1].job_title


def _evaluate(agent: FakeStructuredChatModel, report: object) -> object:
    settings = make_settings()
    return asyncio.run(
        run_case(
            GAIN_CASE,
            GAIN["implicit_report"],
            report,  # type: ignore[arg-type]
            GAIN,
            settings=settings,
            embeddings=KeywordEmbeddings(),
            agent_llm=agent,
            match_llm=_matcher(),
            limiter=RateLimiter.from_settings(settings),
        )
    )


def test_end_to_end_gain_and_describe_card() -> None:
    agent = FakeStructuredChatModel(responses=[CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)])

    result = _evaluate(agent, _report(IMPLICIT_CI_CD, levels={"R4": "mentionné"}))

    assert [(o.id, o.verdict) for o in result.outcomes] == [("R4", "carte"), ("R9", "gain")]  # type: ignore[attr-defined]
    summary = summarize([result])  # type: ignore[list-item]
    assert (summary["affirmations inventées"], summary["cartes « à décrire »"], summary["gains"]) == (0, 1, "1/1")
    assert summary["gains depuis la source annotée"] == 1 and summary["gains avec un terme attendu"] == 1
    assert summary["cibles implicites traitées"] == 1
    assert result.suggestions[0]["après"] == HONEST_CI_CD and result.suggestions[0]["termes"] == ["CI/CD"]  # type: ignore[attr-defined]


def test_end_to_end_tempting_rewrite_is_stopped_by_grounding() -> None:
    # R8 « API REST » : preuve implicite (service FastAPI) ; nommer REST ajouterait le style
    # d'API, absent de l'expérience : rejeté deux fois par l'ancrage, refus juste.
    tempting = Proposal(
        new_text="API REST de calcul des tarifs de livraison, appelée par le site marchand et l'appli des livreurs.",
        facts=[Fact(claim="service appelé par deux applications", chunk_id=VELMARIS, quote=SERVICE)],
    )
    agent = FakeStructuredChatModel(
        responses=[SearchDecision(decision="candidat", chunk_id=VELMARIS, bullet_index=0), tempting, tempting]
    )

    result = _evaluate(agent, _report({"R8": ("démontré", VELMARIS, SERVICE.rstrip("."))}))

    assert [(o.id, o.verdict) for o in result.outcomes] == [("R8", "refus_juste")]  # type: ignore[attr-defined]
    assert len(result.grounding_rejections) == 2 and "REST" in result.grounding_rejections[0].upper()  # type: ignore[attr-defined]
    assert summarize([result])["affirmations inventées"] == 0  # type: ignore[list-item]


def test_results_path() -> None:
    from datetime import date

    path = results_path(date(2026, 11, 22))

    assert path.name == "2026-11-22_suggestions_v2.json" and path.parent.name == "results"


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


def test_save_results_keeps_the_other_cases_of_the_day(tmp_path: Path) -> None:
    from eval.eval_suggestions import CaseResult, save_results

    path = tmp_path / "2026-11-24_suggestions_v2.json"
    save_results(path, [CaseResult(case="a", report_source="r", seconds=1.0)])
    save_results(path, [CaseResult(case="b", report_source="r")])
    save_results(path, [CaseResult(case="a", report_source="r", seconds=2.0)])

    saved = json.loads(path.read_text())
    assert list(saved) == ["a", "b"] and saved["a"]["seconds"] == 2.0
