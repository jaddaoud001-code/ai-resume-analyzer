"""Annotations pour l'évaluation de l'agent de suggestions (eval/suggestions) : lacunes
(suggestion honnête possible ou non) et exigences déjà prouvées par le CV (faux négatifs
possibles du matching). Cohérentes avec les listes figées, les corrigés du matching et le
texte des CV. Aucun appel au LLM."""

import json
from pathlib import Path

from app.chunking import chunk_cv
from app.grounding import bullets
from app.pdf_parser import extract_text
from app.schemas import JobRequirements
from eval.eval_matching import find_cases
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


def test_gain_case_points_to_real_bullets() -> None:
    data = _load("gain_cv__logistique")
    job = JobRequirements.model_validate(json.loads((EVAL / "offers" / "logistique.requirements.json").read_text()))
    requirements = {r.id: r for r in job.requirements}
    chunks = chunk_cv(extract_text((EVAL / "cvs" / data["cv"]).read_bytes(), make_settings()).text)

    _check_entries(data, requirements)
    possible = [g for g in data["gaps"] if g["possible"]]
    assert {g["id"] for g in possible} == {"R2", "R3", "R7", "R9"}
    assert {g["id"] for g in data["gaps"] if not g["possible"]} == {"R4", "R5", "R6", "R8"}
    for gap in possible:
        source = gap["source"]
        item = [c for c in chunks if c.text.splitlines()[0] == source["item"]]
        assert item, gap["id"]
        assert bullets(item[0].text)[source["bullet"]].text == source["bullet_text"], gap["id"]
