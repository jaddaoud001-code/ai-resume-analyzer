"""Script d'évaluation du matching (eval/eval_matching.py) : comparaison au corrigé,
mesures séparées technique / savoir-être, cohérence des fichiers. Aucun appel au LLM."""

import asyncio
from pathlib import Path

import pytest

from app.llm_calls import RateLimiter
from app.requirements import _finalize
from app.schemas import (
    Evidence,
    JobRequirements,
    MatchReport,
    MatchTimings,
    Requirement,
    RequirementMatch,
    TokenUsage,
)
from eval.eval_matching import (
    CASES_DIR,
    ERROR,
    Case,
    CaseMismatchError,
    ExpectedLevel,
    assess,
    case_run,
    check_case,
    confusion,
    expected_scores,
    find_cases,
    run_case,
    summarize,
)
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_matcher import EXTRACTED, TODAY, _responder


def _requirement(
    rid: str, label: str, category: str = "compétence technique", importance: str = "obligatoire"
) -> Requirement:
    return Requirement(id=rid, label=label, category=category, importance=importance, search_queries=[label, label])  # type: ignore[arg-type]


def _result(requirement: Requirement, level: str | None, **extra: object) -> RequirementMatch:
    return RequirementMatch(
        requirement=requirement,
        status="ok" if level else "error",
        level=level,  # type: ignore[arg-type]
        weight=1,
        **extra,  # type: ignore[arg-type]
    )


PYTHON = _requirement("R1", "Python")
RIGUEUR = _requirement("R2", "Rigueur", category="savoir-être", importance="souhaitée")


class TestAssess:
    @pytest.mark.parametrize(
        ("accepted", "obtained", "exact", "within_one"),
        [
            (["démontré"], "démontré", True, True),
            (["démontré", "solide"], "solide", True, True),  # second niveau accepté
            (["démontré"], "mentionné", False, True),
            (["démontré", "solide"], "mentionné", False, True),  # un niveau du plus proche
            (["solide"], "mentionné", False, False),
            (["non démontré"], "démontré", False, False),
        ],
    )
    def test_exact_and_within_one(self, accepted: list[str], obtained: str, exact: bool, within_one: bool) -> None:
        outcome = assess(ExpectedLevel("R1", "Python", accepted), _result(PYTHON, obtained))

        assert (outcome.exact, outcome.within_one) == (exact, within_one)

    @pytest.mark.parametrize(
        ("accepted", "obtained", "false_positive", "false_negative"),
        [
            (["non démontré"], "démontré", True, False),
            (["non démontré", "mentionné"], "solide", True, False),
            (["mentionné"], "mentionné", False, False),
            # Corrigé ambigu entre bas et haut : ni faux positif ni faux négatif possible.
            (["mentionné", "démontré"], "solide", False, False),
            (["mentionné", "démontré"], "non démontré", False, False),
            (["démontré", "solide"], "mentionné", False, True),
            (["solide"], "non démontré", False, True),
        ],
    )
    def test_false_positives_and_negatives(
        self, accepted: list[str], obtained: str, false_positive: bool, false_negative: bool
    ) -> None:
        outcome = assess(ExpectedLevel("R1", "Python", accepted), _result(PYTHON, obtained))

        assert (outcome.false_positive, outcome.false_negative) == (false_positive, false_negative)

    def test_error_is_neither_exact_nor_a_false_positive(self) -> None:
        outcome = assess(ExpectedLevel("R1", "Python", ["non démontré"]), _result(PYTHON, None, error="x"))

        assert outcome.obtained == ERROR
        assert not (outcome.exact or outcome.within_one or outcome.false_positive or outcome.false_negative)

    def test_group_and_second_chance(self) -> None:
        recovered = _result(
            RIGUEUR,
            "démontré",
            second_chance=True,
            evidence=[Evidence(chunk_id=1, quote="travail rigoureux et méthodique")],
        )
        lost = _result(PYTHON, "non démontré", second_chance=True, downgraded=True, rejected_evidence=2)

        soft = assess(ExpectedLevel("R2", "Rigueur", ["non démontré"]), recovered)
        technical = assess(ExpectedLevel("R1", "Python", ["démontré"]), lost)

        assert (soft.group, soft.second_chance, soft.recovered, soft.evidence) == (
            "savoir-être",
            True,
            True,
            ["travail rigoureux et méthodique"],
        )
        assert (technical.group, technical.recovered, technical.rejected) == ("technique", False, 2)


def _case(*items: tuple[Requirement, list[str]]) -> Case:
    job = JobRequirements(job_title="Stage", requirements=[r for r, _ in items])
    expected = {r.id: ExpectedLevel(r.id, r.label, levels) for r, levels in items}
    return Case("cv__offre", Path("cv.pdf"), Path("offre.txt"), job, expected)


def _report(*results: RequirementMatch, score: int | None) -> MatchReport:
    return MatchReport(
        job_title="Stage",
        score=score,
        requirements=list(results),
        missing_mandatory=[],
        rejected_evidence_count=sum(r.rejected_evidence for r in results),
        errors=[],
        extraction_model="m",
        match_model="m",
        chunk_count=3,
        timings=MatchTimings(),
    )


def test_certification_floor_is_reported_with_the_llm_level() -> None:
    english = _requirement("R3", "Anglais professionnel", category="langue")
    raised = _result(english, "démontré", llm_level="mentionné", certification_floor=True)
    case = _case((english, ["démontré"]))

    summary = summarize([case_run(case, _report(raised, score=75), make_settings(), run=1, seconds=1.0)])

    outcome = assess(ExpectedLevel("R3", "Anglais professionnel", ["démontré"]), raised)
    assert (outcome.exact, outcome.certification_floor, outcome.llm_level) == (True, True, "mentionné")
    assert summary["planchers de certification"] == ["cv__offre R3 Anglais professionnel (LLM : mentionné)"]


def test_tokens_are_reported_per_case_and_in_total() -> None:
    case = _case((PYTHON, ["démontré"]), (RIGUEUR, ["non démontré"]))
    usage = TokenUsage(calls=3, input_tokens=4000, output_tokens=800, reasoning_tokens=200)
    report = _report(_result(PYTHON, "démontré"), _result(RIGUEUR, "non démontré"), score=75).model_copy(
        update={"usage": usage}
    )
    runs = [
        case_run(case, report, make_settings(), run=1, seconds=10.0),
        case_run(case, report, make_settings(), run=2, seconds=10.0),
    ]

    tokens = summarize(runs)["tokens"]

    assert runs[0].usage["total_tokens"] == 4800
    assert (tokens["calls"], tokens["total_tokens"], tokens["reasoning_tokens"]) == (6, 9600, 400)
    assert (tokens["exigences"], tokens["moyenne par exigence"]) == (4, 2400)


class TestSummary:
    def test_soft_skill_false_positives_are_reported_apart(self) -> None:
        case = _case((PYTHON, ["non démontré"]), (RIGUEUR, ["non démontré"]))
        report = _report(
            _result(PYTHON, "mentionné"),
            _result(RIGUEUR, "démontré", evidence=[Evidence(chunk_id=2, quote="un travail rigoureux au quotidien")]),
            score=50,
        )

        summary = summarize([case_run(case, report, make_settings(), run=1, seconds=12.0)])

        technical, soft = summary["par groupe"]["technique"], summary["par groupe"]["savoir-être"]
        assert (technical["faux positifs"], technical["exactes"], technical["à un niveau près"]) == (0, 0, 1)
        assert (soft["faux positifs"], soft["exactitude"]) == (1, 0.0)
        assert summary["total"]["faux positifs"] == 1
        [false_positive] = summary["faux positifs"]
        assert (false_positive["groupe"], false_positive["citations"]) == (
            "savoir-être",
            ["un travail rigoureux au quotidien"],
        )

    def test_confusion_uses_the_main_expected_level(self) -> None:
        outcomes = [
            assess(ExpectedLevel("R1", "Python", ["démontré", "solide"]), _result(PYTHON, "solide")),
            assess(ExpectedLevel("R1", "Python", ["démontré"]), _result(PYTHON, None, error="x")),
        ]

        matrix = confusion(outcomes)

        assert matrix["démontré"]["solide"] == 1
        assert matrix["démontré"][ERROR] == 1
        assert sum(sum(row.values()) for row in matrix.values()) == 2

    def test_expected_score_and_range_use_the_app_points_and_weights(self) -> None:
        # Python obligatoire (poids 2), Rigueur savoir-être souhaité (poids 0,5).
        case = _case((PYTHON, ["démontré", "solide"]), (RIGUEUR, ["non démontré", "mentionné"]))

        main, low, high = expected_scores(case, make_settings())

        assert main == round(75 * 2 / 2.5)  # 60
        assert (low, high) == (60, round((100 * 2 + 40 * 0.5) / 2.5))  # (60, 88)

    def test_score_gap_and_unstable_levels_across_runs(self) -> None:
        case = _case(
            (PYTHON, ["démontré"]),
        )
        runs = [
            case_run(case, _report(_result(PYTHON, "démontré"), score=75), make_settings(), run=1, seconds=1.0),
            case_run(case, _report(_result(PYTHON, "solide"), score=100), make_settings(), run=2, seconds=1.0),
        ]

        summary = summarize(runs)

        assert [s["écart"] for s in summary["scores"]] == [0, 25]
        assert summary["écart moyen de score"] == 12.5
        assert summary["instables"] == {"cv__offre": ["R1 Python : {'démontré': 1, 'solide': 1}"]}


class TestCaseFiles:
    def test_mismatching_ids_or_labels_are_detected(self) -> None:
        job = JobRequirements(job_title="Stage", requirements=[PYTHON])

        with pytest.raises(CaseMismatchError, match="identifiants"):
            check_case(job, {"R9": ExpectedLevel("R9", "Python", ["démontré"])}, "cas")
        with pytest.raises(CaseMismatchError, match="« Java » dans le corrigé"):
            check_case(job, {"R1": ExpectedLevel("R1", "Java", ["démontré"])}, "cas")
        with pytest.raises(CaseMismatchError, match="niveaux invalides"):
            check_case(job, {"R1": ExpectedLevel("R1", "Python", ["expert"])}, "cas")

    def test_all_cases_match_their_frozen_lists(self) -> None:
        cases = find_cases(series="toutes")

        assert len(cases) == len(list(CASES_DIR.glob("*.expected.json"))) == 6
        for case in cases:
            assert case.cv.exists() and case.offer.exists()

    def test_series_are_kept_apart_by_default(self) -> None:
        reference = {"demo_cv__hager", "demo_cv__sii", "java_cv__hager", "java_cv__sii"}
        hard = {"backend_cv__logistique", "java_cv__logistique"}

        assert {c.name for c in find_cases()} == reference
        assert {c.name for c in find_cases(series="difficile")} == hard
        assert {c.name for c in find_cases(series="toutes")} == reference | hard
        # Un cas nommé est évalué quelle que soit sa série.
        assert [c.name for c in find_cases(["java_cv__logistique"])] == ["java_cv__logistique"]


def test_run_case_uses_the_frozen_requirements_through_the_real_pipeline(
    tmp_path: Path, sectioned_cv_pdf: bytes
) -> None:
    requirements = [r for r in _finalize(EXTRACTED, 15).requirements if r.label in {"Python", "Kubernetes", "Rigueur"}]
    cv, offer = tmp_path / "cv.pdf", tmp_path / "offre.txt"
    cv.write_bytes(sectioned_cv_pdf)
    offer.write_text("Offre : Python 3 ans, Kubernetes, rigueur.", encoding="utf-8")
    levels = {"Python": ["solide"], "Kubernetes": ["non démontré"], "Rigueur": ["non démontré", "mentionné"]}
    case = Case(
        "cv__offre",
        cv,
        offer,
        JobRequirements(job_title="Liste figée", requirements=requirements),
        {r.id: ExpectedLevel(r.id, r.label, levels[r.label]) for r in requirements},
    )
    extraction = FakeStructuredChatModel(responses=[])  # tout appel échouerait
    settings = make_settings()

    run = asyncio.run(
        run_case(
            case,
            settings=settings,
            embeddings=KeywordEmbeddings(),
            extraction_llm=extraction,
            match_llm=FakeStructuredChatModel(responder=_responder),
            limiter=RateLimiter.from_settings(settings),
            today=TODAY,
        )
    )

    assert extraction.calls == []
    by_label = {o.label: o for o in run.outcomes}
    assert all(o.exact for o in run.outcomes)
    # Kubernetes : citation inventée, seconde chance sans preuve retrouvée.
    assert (by_label["Kubernetes"].second_chance, by_label["Kubernetes"].recovered) == (True, False)
    assert run.score_gap == 0 and run.error is None
