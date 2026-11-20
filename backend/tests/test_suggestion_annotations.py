"""Annotations des lacunes pour l'évaluation de l'agent de suggestions (eval/suggestions) :
cohérentes avec les listes figées et avec les corrigés du matching. Aucun appel au LLM."""

import json
from pathlib import Path

from eval.eval_matching import find_cases

SUGGESTIONS = Path(__file__).resolve().parents[1] / "eval" / "suggestions"
LOW = {"non démontré", "mentionné"}
LEVELS = {"non démontré", "mentionné", "démontré", "solide"}


def test_annotations_cover_exactly_the_technical_gaps_of_the_matching_keys() -> None:
    cases = {c.name: c for c in find_cases(series="toutes")}
    files = {p.name.removesuffix(".expected.json"): p for p in SUGGESTIONS.glob("*.expected.json")}

    assert set(files) == set(cases)
    for name, case in cases.items():
        data = json.loads(files[name].read_text(encoding="utf-8"))
        requirements = {r.id: r for r in case.job.requirements}
        expected_gaps = {
            rid
            for rid, e in case.expected.items()
            if requirements[rid].category != "savoir-être" and set(e.levels) <= LOW
        }
        annotated = {gap["id"]: gap for gap in data["gaps"]}
        assert set(annotated) == expected_gaps, name  # savoir-être exclus en v1
        for rid, gap in annotated.items():
            assert gap["label"] == requirements[rid].label, (name, rid)
            assert gap["current_level"] in case.expected[rid].levels, (name, rid)
            assert gap["max_honest_level"] in LEVELS and gap["reason"] and gap["temptations"], (name, rid)
            # Une suggestion possible doit nommer sa source (item et puce) ; sinon aucune.
            assert (gap["source"] is not None) == gap["possible"], (name, rid)
