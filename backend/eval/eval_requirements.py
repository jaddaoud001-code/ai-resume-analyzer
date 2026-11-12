"""Évaluation de l'extraction des exigences sur des offres de référence (hors pytest).

Appelle le vrai LLM (clé dans .env) et compare le résultat à `<offre>.expected.json` :
exigences trouvées, manquantes, inventées, interdites (must_not_extract), acceptables
(défendables mais non exigées), erreurs d'importance (et de catégorie). Avec --runs N,
chaque modèle est exécuté N fois et le rapport agrège les exécutions (taux moyen et
minimum, fréquence de chaque exigence manquée…).

Usage (depuis backend/) :
    .venv/bin/python eval/eval_requirements.py                      # modèle du .env
    .venv/bin/python eval/eval_requirements.py --model mistral-large-4-0 --model ministral-14b-2512
    .venv/bin/python eval/eval_requirements.py --offer eval/offers/hager.txt --json
    .venv/bin/python eval/eval_requirements.py --offer eval/offers/sii.txt --runs 3

Rapprochement (approximatif, documenté dans eval/README.md) : les libellés sont normalisés
(minuscules, sans accents, découpés en mots). Une exigence extraite correspond à une
exigence attendue si, pour l'une de ses alternatives `match`, tous les mots y figurent ;
à défaut de `match`, si la similarité difflib des libellés normalisés atteint 0,6.
"""

import argparse
import asyncio
import difflib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.llm import get_llm, llm_model  # noqa: E402
from app.llm_calls import RateLimiter  # noqa: E402
from app.requirements import extract_requirements  # noqa: E402
from app.schemas import JobRequirements, Requirement  # noqa: E402

DEFAULT_OFFER = BACKEND_DIR / "eval" / "offers" / "hager.txt"
SIMILARITY_THRESHOLD = 0.6


def normalize(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", plain)


@dataclass
class Expected:
    label: str
    # Vides pour une entrée de must_not_extract.
    category: str = ""
    importance: str = ""
    match: list[list[str]] = field(default_factory=list)

    def matches(self, label: str) -> bool:
        words = set(normalize(label))
        if self.match:
            return any(all(word in words for word in alternative) for alternative in self.match)
        ratio = difflib.SequenceMatcher(None, " ".join(normalize(self.label)), " ".join(normalize(label))).ratio()
        return ratio >= SIMILARITY_THRESHOLD


@dataclass
class Report:
    model: str
    seconds: float
    job_title: str = ""
    found: list[tuple[Expected, list[Requirement]]] = field(default_factory=list)
    missing: list[Expected] = field(default_factory=list)
    invented: list[Requirement] = field(default_factory=list)
    importance_errors: list[tuple[Expected, Requirement]] = field(default_factory=list)
    category_errors: list[tuple[Expected, Requirement]] = field(default_factory=list)
    merged: list[tuple[Requirement, list[Expected]]] = field(default_factory=list)
    forbidden: list[tuple[Requirement, Expected]] = field(default_factory=list)
    # Exigences « acceptables » extraites (ni manquantes si absentes, ni inventées si présentes).
    acceptable: list[tuple[Expected, list[Requirement]]] = field(default_factory=list)
    error: str | None = None

    def summary(self, total: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "seconds": round(self.seconds, 1),
            "error": self.error,
            "found": f"{len(self.found)}/{total}",
            "missing": [e.label for e in self.missing],
            "invented": [r.label for r in self.invented],
            "importance_errors": [
                f"{e.label} : attendu {e.importance}, obtenu {r.importance} ({r.id})" for e, r in self.importance_errors
            ],
            "category_errors": [
                f"{e.label} : attendu {e.category}, obtenu {r.category} ({r.id})" for e, r in self.category_errors
            ],
            "split": [f"{e.label} -> {', '.join(r.id for r in reqs)}" for e, reqs in self.found if len(reqs) > 1],
            "merged": [f"{r.id} « {r.label} » -> {', '.join(e.label for e in es)}" for r, es in self.merged],
            "forbidden": [f"{r.id} « {r.label} » ({f.label})" for r, f in self.forbidden],
            "acceptable": [f"{e.label} -> {', '.join(r.id for r in reqs)}" for e, reqs in self.acceptable],
        }


def compare(
    result: JobRequirements,
    expected: list[Expected],
    forbidden: list[Expected],
    report: Report,
    acceptable: list[Expected] | None = None,
) -> None:
    """Une exigence extraite peut correspondre à plusieurs attendues (fusion) et une
    attendue à plusieurs extraites (découpage) : ni l'un ni l'autre n'est « inventé ».
    Une extraite qui correspond à must_not_extract est une erreur à part (« interdite »).
    Une « acceptable » (lecture défendable de l'offre, non exigée) n'est jamais manquante,
    et une extraite qui lui correspond n'est pas inventée : elle est listée à part."""
    acceptable = acceptable or []
    report.job_title = result.job_title
    for requirement in result.requirements:
        for item in forbidden:
            if item.matches(requirement.label):
                report.forbidden.append((requirement, item))
                break
    forbidden_ids = {r.id for r, _ in report.forbidden}
    matched_by: dict[str, list[Expected]] = {
        r.id: [e for e in expected if e.matches(r.label)] for r in result.requirements
    }
    for item in expected:
        requirements = [r for r in result.requirements if item in matched_by[r.id]]
        if not requirements:
            report.missing.append(item)
            continue
        report.found.append((item, requirements))
        best = requirements[0]
        if best.importance != item.importance:
            report.importance_errors.append((item, best))
        if best.category != item.category:
            report.category_errors.append((item, best))
    for item in acceptable:
        requirements = [r for r in result.requirements if r.id not in forbidden_ids and item.matches(r.label)]
        if requirements:
            report.acceptable.append((item, requirements))
    acceptable_ids = {r.id for _, reqs in report.acceptable for r in reqs}
    for requirement in result.requirements:
        matches = matched_by[requirement.id]
        if not matches and requirement.id not in forbidden_ids and requirement.id not in acceptable_ids:
            report.invented.append(requirement)
        elif len(matches) > 1:
            report.merged.append((requirement, matches))


def load_expected(offer: Path) -> tuple[list[Expected], list[Expected], list[Expected]]:
    """(exigences attendues, exigences qui ne doivent pas être extraites, exigences
    acceptables : défendables mais non exigées)."""
    data = json.loads(offer.with_suffix(".expected.json").read_text(encoding="utf-8"))

    def parse(items: list[dict[str, Any]]) -> list[Expected]:
        return [
            Expected(**{k: item[k] for k in ("label", "category", "importance", "match") if k in item})
            for item in items
        ]

    return parse(data["requirements"]), parse(data.get("must_not_extract", [])), parse(data.get("acceptable", []))


async def evaluate(offer: Path, models: list[str | None], runs: int = 1) -> list[tuple[Report, JobRequirements | None]]:
    """Chaque modèle `runs` fois, l'un après l'autre ; tous les appels passent par le même
    limiteur (LLM_RPS), comme dans l'application."""
    text = offer.read_text(encoding="utf-8")
    expected, forbidden, acceptable = load_expected(offer)
    base = Settings()
    # Limiteur partagé par tous les appels (LLM_RPS), comme dans l'application.
    limiter = RateLimiter.from_settings(base)
    results: list[tuple[Report, JobRequirements | None]] = []
    for model in models:
        settings = base.model_copy(update={"llm_model": model}) if model else base
        for _ in range(runs):
            started = time.perf_counter()
            report = Report(model=llm_model(settings), seconds=0.0)
            result: JobRequirements | None = None
            try:
                result = await extract_requirements(text, get_llm(settings), settings, limiter)
                compare(result, expected, forbidden, report, acceptable)
            except Exception as error:  # le rapport doit rester lisible même si un modèle échoue
                report.error = f"{type(error).__name__}: {error}"
            report.seconds = time.perf_counter() - started
            results.append((report, result))
    return results


def _frequencies(counter: Counter[str], runs: int, verb: str) -> list[str]:
    return [
        f"{label} : {verb} {count}/{runs}" for label, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def aggregate(reports: list[Report], total: int) -> dict[str, Any]:
    """Synthèse de plusieurs exécutions d'un même modèle. Les taux et moyennes portent sur
    les exécutions réussies ; les fréquences sont données sur ce même nombre."""
    ok = [r for r in reports if r.error is None]
    n = len(ok)
    rates = [len(r.found) / total for r in ok] if total else []

    def mean(values: list[int]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    missed: Counter[str] = Counter(e.label for r in ok for e in r.missing)
    invented: Counter[str] = Counter(req.label for r in ok for req in r.invented)
    forbidden: Counter[str] = Counter(f"« {req.label} » ({f.label})" for r in ok for req, f in r.forbidden)
    importance: Counter[str] = Counter(
        f"{e.label} : attendu {e.importance}, obtenu {req.importance}" for r in ok for e, req in r.importance_errors
    )
    category: Counter[str] = Counter(
        f"{e.label} : attendu {e.category}, obtenu {req.category}" for r in ok for e, req in r.category_errors
    )
    merged: Counter[str] = Counter(
        f"« {req.label} » -> {', '.join(e.label for e in es)}" for r in ok for req, es in r.merged
    )
    split: Counter[str] = Counter(
        f"{e.label} -> {len(reqs)} extraites" for r in ok for e, reqs in r.found if len(reqs) > 1
    )
    acceptable: Counter[str] = Counter(e.label for r in ok for e, _ in r.acceptable)
    return {
        "model": reports[0].model if reports else "",
        "runs": len(reports),
        "failed_runs": [r.error for r in reports if r.error is not None],
        "mean_seconds": round(sum(r.seconds for r in ok) / n, 1) if n else None,
        "found_rate_mean": round(sum(rates) / n, 3) if n else None,
        "found_rate_min": round(min(rates), 3) if rates else None,
        "found_per_run": [f"{len(r.found)}/{total}" for r in ok],
        "invented_mean": mean([len(r.invented) for r in ok]),
        "forbidden_mean": mean([len(r.forbidden) for r in ok]),
        "missed": _frequencies(missed, n, "manquée"),
        "invented": _frequencies(invented, n, "inventée"),
        "forbidden": _frequencies(forbidden, n, "extraite"),
        "importance_errors": _frequencies(importance, n, "vue"),
        "category_errors": _frequencies(category, n, "vue"),
        "merged": _frequencies(merged, n, "vue"),
        "split": _frequencies(split, n, "vue"),
        "acceptable": _frequencies(acceptable, n, "trouvée"),
    }


def print_aggregate(summary: dict[str, Any], total: int) -> None:
    n = summary["runs"] - len(summary["failed_runs"])
    print(f"\n=== {summary['model']} — {summary['runs']} exécution(s), {summary['mean_seconds']} s en moyenne")
    for error in summary["failed_runs"]:
        print(f"  ERREUR : {error}")
    if not n:
        return
    print(
        f"  Trouvées : {summary['found_rate_mean']:.0%} en moyenne, {summary['found_rate_min']:.0%} au minimum "
        f"({', '.join(summary['found_per_run'])} sur {total} attendues)"
    )
    print(f"  Inventées : {summary['invented_mean']} en moyenne ; interdites : {summary['forbidden_mean']} en moyenne")
    for key, title in (
        ("missed", "Manquées au moins une fois"),
        ("invented", "Inventées (hors attendu)"),
        ("forbidden", "Interdites (must_not_extract)"),
        ("importance_errors", "Erreurs d'importance"),
        ("category_errors", "Erreurs de catégorie"),
        ("merged", "Fusionnées"),
        ("split", "Découpées"),
        ("acceptable", "Acceptables trouvées"),
    ):
        items = summary[key]
        print(f"  {title} : {'aucune' if not items else ''}")
        for item in items:
            print(f"    - {item}")


def print_report(report: Report, result: JobRequirements | None, total: int) -> None:
    summary = report.summary(total)
    print(f"\n=== {report.model} — {summary['seconds']} s")
    if report.error:
        print(f"  ERREUR : {report.error}")
        return
    print(f"  Poste : {report.job_title}")
    print(f"  Trouvées : {summary['found']}")
    for key, title in (
        ("missing", "Manquantes"),
        ("invented", "Inventées (hors attendu)"),
        ("importance_errors", "Erreurs d'importance"),
        ("category_errors", "Erreurs de catégorie"),
        ("split", "Découpées"),
        ("merged", "Fusionnées"),
        ("forbidden", "Interdites (must_not_extract)"),
        ("acceptable", "Acceptables trouvées"),
    ):
        print(f"  {title} : {summary[key] or 'aucune'}")
    assert result is not None
    print("  Extraites :")
    for r in result.requirements:
        print(f"    {r.id:>3} {r.importance:<11} {r.category:<20} {r.label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offer", type=Path, default=DEFAULT_OFFER)
    parser.add_argument("--model", action="append", help="modèle à évaluer (répétable) ; défaut : LLM_MODEL du .env")
    parser.add_argument("--runs", type=int, default=1, help="exécutions par modèle (agrégées si > 1)")
    parser.add_argument("--json", action="store_true", help="résumés au format JSON")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs doit être au moins 1")

    total = len(load_expected(args.offer)[0])
    results = asyncio.run(evaluate(args.offer, args.model or [None], args.runs))
    if args.runs == 1:
        if args.json:
            print(json.dumps([report.summary(total) for report, _ in results], ensure_ascii=False, indent=2))
            return
        for report, result in results:
            print_report(report, result, total)
        return
    by_model: dict[str, list[tuple[Report, JobRequirements | None]]] = {}
    for report, result in results:
        by_model.setdefault(report.model, []).append((report, result))
    summaries = [aggregate([r for r, _ in runs], total) for runs in by_model.values()]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return
    print(f"Offre : {args.offer.name} ({total} exigences attendues)")
    for runs, summary in zip(by_model.values(), summaries, strict=True):
        for index, (report, result) in enumerate(runs, start=1):
            print(f"\n--- exécution {index}/{len(runs)}", end="")
            print_report(report, result, total)
        print_aggregate(summary, total)


if __name__ == "__main__":
    main()
