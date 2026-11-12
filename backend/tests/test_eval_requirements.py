"""Script d'évaluation de l'extraction (eval/eval_requirements.py) : rapprochement, liste
« acceptable », agrégation de plusieurs exécutions. Aucun appel au LLM."""

import json
from pathlib import Path

from app.schemas import JobRequirements, Requirement
from eval.eval_requirements import Expected, Report, aggregate, compare, load_expected

OFFERS = Path(__file__).resolve().parents[1] / "eval" / "offers"

EXPECTED = [
    Expected("Python", "compétence technique", "obligatoire", [["python"]]),
    Expected("RAG", "compétence technique", "souhaitée", [["rag"]]),
]
FORBIDDEN = [Expected("Lieu", match=[["rennes"]])]
ACCEPTABLE = [Expected("Claude / API Anthropic", match=[["claude"], ["anthropic"]])]


def _job(*labels: str, importance: str = "souhaitée") -> JobRequirements:
    return JobRequirements(
        job_title="Stage",
        requirements=[
            Requirement(
                id=f"R{i}",
                label=label,
                category="compétence technique",
                importance="obligatoire" if label == "Python" else importance,
                search_queries=[label, f"{label} projet"],
            )
            for i, label in enumerate(labels, start=1)
        ],
    )


def _compare(job: JobRequirements) -> Report:
    report = Report(model="test", seconds=0.0)
    compare(job, EXPECTED, FORBIDDEN, report, ACCEPTABLE)
    return report


class TestAcceptable:
    def test_absent_acceptable_is_not_missing(self) -> None:
        report = _compare(_job("Python", "RAG"))

        assert report.missing == [] and report.acceptable == [] and report.invented == []

    def test_extracted_acceptable_is_not_invented_and_is_listed_apart(self) -> None:
        report = _compare(_job("Python", "RAG", "Utilisation de Claude"))

        assert report.invented == []
        assert [(e.label, [r.id for r in reqs]) for e, reqs in report.acceptable] == [
            ("Claude / API Anthropic", ["R3"])
        ]
        assert report.summary(len(EXPECTED))["acceptable"] == ["Claude / API Anthropic -> R3"]
        assert len(report.found) == 2  # ne compte pas dans les trouvées

    def test_unrelated_extraction_is_still_invented(self) -> None:
        report = _compare(_job("Python", "RAG", "Kubernetes"))

        assert [r.label for r in report.invented] == ["Kubernetes"]

    def test_forbidden_wins_over_acceptable(self) -> None:
        report = _compare(_job("Python", "RAG", "Claude à Rennes"))

        assert [r.label for r, _ in report.forbidden] == ["Claude à Rennes"]
        assert report.acceptable == [] and report.invented == []


class TestAggregate:
    def test_rates_means_and_missed_frequencies(self) -> None:
        runs = [
            _compare(_job("Python", "RAG")),
            _compare(_job("Python", "Kubernetes")),  # RAG manquée, 1 inventée
            _compare(_job("Python", "Claude", "Rennes")),  # RAG manquée, 1 interdite, 1 acceptable
        ]

        summary = aggregate(runs, total=2)

        assert summary["runs"] == 3 and summary["failed_runs"] == []
        assert summary["found_rate_mean"] == round((1 + 0.5 + 0.5) / 3, 3)
        assert summary["found_rate_min"] == 0.5
        assert summary["found_per_run"] == ["2/2", "1/2", "1/2"]
        assert summary["invented_mean"] == round(1 / 3, 2)
        assert summary["forbidden_mean"] == round(1 / 3, 2)
        assert summary["missed"] == ["RAG : manquée 2/3"]
        assert summary["invented"] == ["Kubernetes : inventée 1/3"]
        assert summary["acceptable"] == ["Claude / API Anthropic : trouvée 1/3"]

    def test_importance_errors_merges_and_splits_are_counted(self) -> None:
        wrong_importance = _job("Python", "RAG", importance="obligatoire")  # RAG attendue souhaitée
        merged = _compare(_job("Python et RAG"))
        split = _compare(_job("Python", "RAG", "RAG avancé"))

        summary = aggregate([_compare(wrong_importance), merged, split], total=2)

        # « Python et RAG » est extraite « souhaitée » : erreur d'importance pour Python aussi.
        assert summary["importance_errors"] == [
            "Python : attendu obligatoire, obtenu souhaitée : vue 1/3",
            "RAG : attendu souhaitée, obtenu obligatoire : vue 1/3",
        ]
        assert summary["merged"] == ["« Python et RAG » -> Python, RAG : vue 1/3"]
        assert summary["split"] == ["RAG -> 2 extraites : vue 1/3"]

    def test_failed_runs_are_reported_and_excluded_from_rates(self) -> None:
        failed = Report(model="test", seconds=1.0, error="HTTPStatusError: 429")

        summary = aggregate([_compare(_job("Python", "RAG")), failed], total=2)

        assert summary["failed_runs"] == ["HTTPStatusError: 429"]
        assert summary["found_rate_mean"] == 1.0 and summary["found_per_run"] == ["2/2"]


class TestReferenceCases:
    def test_sii_keeps_environment_tools_as_acceptable(self) -> None:
        expected, forbidden, acceptable = load_expected(OFFERS / "sii.txt")

        assert {e.label for e in acceptable} == {
            "Claude / API Anthropic",
            "LLM hébergés en interne / open source",
            "Restitution / reporting des résultats",
        }
        assert not any(e.label in {a.label for a in acceptable} for e in expected)
        assert len(expected) == 13 and len(forbidden) == 3

    def test_every_reference_case_loads(self) -> None:
        for path in sorted(OFFERS.glob("*.expected.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            offer = OFFERS / data["offer"]
            expected, _, _ = load_expected(offer)
            assert offer.exists() and expected, path.name
