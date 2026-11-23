"""Annotations pour l'évaluation de l'agent de suggestions (eval/suggestions) : lacunes
(cartes « à décrire » en v2), exigences déjà prouvées par le CV et cibles implicites de
l'agent v2 (terme de l'offre à nommer). Cohérentes avec les listes figées, les corrigés du
matching, les rapports bruts conservés et le texte des CV. Aucun appel au LLM."""

import json
from pathlib import Path

from app.chunking import chunk_cv
from app.grounding import BulletRewrite, Fact, bullets, check_grounding
from app.pdf_parser import extract_text
from app.schemas import JobRequirements
from app.terms import is_implicit, mentions_term, names_offer_term
from eval.eval_matching import find_cases
from eval.eval_suggestions import report_by_source
from tests.conftest import make_settings

EVAL = Path(__file__).resolve().parents[1] / "eval"
SUGGESTIONS = EVAL / "suggestions"
LOW = {"non démontré", "mentionné"}
HIGH = {"démontré", "solide"}
LEVELS = LOW | HIGH


def _load(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((SUGGESTIONS / f"{name}.expected.json").read_text(encoding="utf-8"))


def _check_entries(data: dict, requirements: dict) -> None:  # type: ignore[type-arg]
    gaps = {g["id"]: g for g in data["gaps"]}
    proven = {p["id"]: p for p in data["already_proven"]}
    assert not set(gaps) & set(proven)
    for rid, entry in [*gaps.items(), *proven.items()]:
        assert entry["label"] == requirements[rid].label, rid
        assert requirements[rid].category != "savoir-être", rid  # exclus en v1
    for rid, gap in gaps.items():
        # Lacune certaine (niveaux bas seulement) ou probable (le matcher peut aussi juger
        # « démontré » : elle ne sera alors pas ciblée par l'agent).
        assert set(gap["current_levels"]) & LOW and gap["max_honest_level"] in LEVELS, rid
        assert gap["reason"] and gap["temptations"], rid
        # Une suggestion possible nomme sa source et l'esprit de la reformulation ; sinon rien.
        assert (gap["source"] is not None) == gap["possible"] == (gap["expected_rewrite"] is not None), rid
    for rid, entry in proven.items():
        assert set(entry["levels"]) <= HIGH, rid


def test_matching_cases_cover_their_technical_gaps_and_proven_requirements() -> None:
    for case in find_cases(series="toutes"):
        data = _load(case.name)
        requirements = {r.id: r for r in case.job.requirements}
        technical = {rid: e for rid, e in case.expected.items() if requirements[rid].category != "savoir-être"}
        _check_entries(data, requirements)
        assert {g["id"] for g in data["gaps"]} == {rid for rid, e in technical.items() if set(e.levels) <= LOW}
        assert {p["id"] for p in data["already_proven"]} == {
            rid for rid, e in technical.items() if set(e.levels) <= HIGH
        }
        for gap in data["gaps"]:
            assert gap["current_levels"] == case.expected[gap["id"]].levels
            assert not gap["possible"]  # les 27 lacunes validées sont de vraies absences


def test_gain_case_gaps_and_proven_follow_the_real_run() -> None:
    data = _load("gain_cv__logistique")
    job = JobRequirements.model_validate(json.loads((EVAL / "offers" / "logistique.requirements.json").read_text()))
    requirements = {r.id: r for r in job.requirements}
    report = report_by_source(data["implicit_report"])
    levels = {r.requirement.id: r.level for r in report.requirements}

    _check_entries(data, requirements)
    assert {g["id"] for g in data["gaps"]} == {"R4", "R5", "R6"}
    assert not any(g["possible"] for g in data["gaps"])  # vraies absences : cartes « à décrire »
    for gap in data["gaps"]:
        assert gap["current_levels"] == [levels[gap["id"]]]
    for entry in data["already_proven"]:
        assert levels[entry["id"]] in entry["levels"]


ALL_CASES = sorted(p.name.removesuffix(".expected.json") for p in SUGGESTIONS.glob("*.expected.json"))


def test_implicit_targets_match_the_code_detection_on_the_annotated_report() -> None:
    for case in ALL_CASES:
        data = _load(case)
        report = report_by_source(data["implicit_report"])
        detected = [r.requirement.id for r in report.requirements if is_implicit(r)]
        assert [e["id"] for e in data["implicit"]] == detected, case
        assert data["implicit_report"].endswith(f"{case} (run 1)")
    # Au moins un cas par issue : reformulation possible, arrêtée par l'ancrage, par le juge seul.
    entries = [e for case in ALL_CASES for e in _load(case)["implicit"]]
    assert {e.get("caught_by") for e in entries} == {None, "ancrage", "juge"}


def test_implicit_annotations_are_consistent_with_the_cvs_and_the_checks() -> None:
    settings = make_settings()
    for case in ALL_CASES:
        data = _load(case)
        report = report_by_source(data["implicit_report"])
        results = {r.requirement.id: r for r in report.requirements}
        chunks = chunk_cv(extract_text((EVAL / "cvs" / data["cv"]).read_bytes(), settings).text)
        by_id = {c.index: c for c in chunks}
        for entry in data["implicit"]:
            rid, source = entry["id"], entry["source"]
            requirement = results[rid].requirement
            assert entry["label"] == requirement.label and entry["level"] == results[rid].level, (case, rid)
            chunk = by_id[source["chunk_id"]]
            assert chunk.text.splitlines()[0] == source["item"], (case, rid)
            bullet = bullets(chunk.text)[source["bullet"]].text
            assert bullet == source["bullet_text"], (case, rid)
            assert not names_offer_term(bullet, requirement), (case, rid)  # le terme manque bien
            rewrite = entry.get("expected_rewrite") or entry["tempting_rewrite"]
            # Les termes attendus sont des termes de l'offre, et la reformulation en nomme un.
            assert all(names_offer_term(t, requirement) for t in entry["terms_expected"]), (case, rid)
            assert any(mentions_term(rewrite, t) for t in entry["terms_expected"]), (case, rid)
            grounding = check_grounding(
                BulletRewrite(
                    chunk_id=chunk.index,
                    bullet_index=source["bullet"],
                    new_text=rewrite,
                    facts=[Fact(claim="puce d'origine", chunk_id=chunk.index, quote=bullet)],
                ),
                chunks,
                settings.match_reserved_sections,
                settings.suggest_max_length_ratio,
            )
            if entry["possible"]:
                # Une reformulation honnête passe le contrôle d'ancrage.
                assert grounding.ok, (case, rid, grounding.codes())
            else:
                assert entry["caught_by"] == ("juge" if grounding.ok else "ancrage"), (case, rid)
                assert entry["reason"] and entry["temptations"], (case, rid)
