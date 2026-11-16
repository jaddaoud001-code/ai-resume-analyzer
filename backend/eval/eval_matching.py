"""Évaluation du matching CV / exigences (hors pytest).

Pour chaque cas `matching/<cv>__<offre>.expected.json` : le vrai pipeline (`analyze`, vrai
LLM de l'évaluation, embeddings du .env, limiteur partagé) évalue le CV `cvs/<cv>.pdf` sur
la liste d'exigences FIGÉE `offers/<offre>.requirements.json` (aucune extraction : ses
variations ne se mélangent pas à celles du matching), puis compare chaque niveau au corrigé.

Mesures, séparées entre exigences techniques (compétences, formation, langues…) et
savoir-être (le corrigé exige qu'un savoir-être soit nommé ou décrit, règle que le prompt
n'impose pas) : exactitude (niveau parmi ceux acceptés), exactitude à un niveau près,
matrice de confusion, faux positifs (démontré / solide alors que le corrigé n'accepte que
non démontré / mentionné : la mesure la plus importante), faux négatifs (l'inverse), écart
entre le score obtenu et celui du corrigé, citations rejetées, secondes chances, durée.

Usage (depuis backend/) :
    .venv/bin/python eval/eval_matching.py                    # 4 cas, 1 exécution
    .venv/bin/python eval/eval_matching.py --case java_cv__sii --runs 2
    .venv/bin/python eval/eval_matching.py --json
Budget : un cas fait un appel par exigence (et une seconde chance au besoin), soit ~30 000
tokens ; les 4 cas, ~130 000 tokens, environ 65 % du quota journalier gratuit de Groq.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402

from app.config import Settings  # noqa: E402
from app.embeddings import get_embeddings  # noqa: E402
from app.llm import get_llm, match_llm_model, match_settings  # noqa: E402
from app.llm_calls import ProviderBlockedError, RateLimiter  # noqa: E402
from app.matcher import analyze, requirement_weight  # noqa: E402
from app.schemas import JobRequirements, MatchReport, ReportReady, RequirementMatch  # noqa: E402

EVAL_DIR = BACKEND_DIR / "eval"
CASES_DIR = EVAL_DIR / "matching"
# Date fixe : les périodes « en cours » donnent la même durée d'un jour à l'autre.
EVAL_TODAY = date(2026, 11, 14)

LEVELS = ["non démontré", "mentionné", "démontré", "solide"]
RANK = {level: rank for rank, level in enumerate(LEVELS)}
LOW = {"non démontré", "mentionné"}
HIGH = {"démontré", "solide"}
TECHNICAL, SOFT = "technique", "savoir-être"
GROUPS = (TECHNICAL, SOFT)
ERROR = "erreur"


class CaseMismatchError(ValueError):
    """Le corrigé ne correspond plus à la liste figée (identifiants ou libellés)."""


@dataclass
class ExpectedLevel:
    id: str
    label: str
    levels: list[str]  # niveaux acceptés ; le premier est le niveau principal
    justification: str = ""


@dataclass
class Case:
    name: str
    cv: Path
    offer: Path
    job: JobRequirements
    expected: dict[str, ExpectedLevel]


def load_case(path: Path, eval_dir: Path = EVAL_DIR) -> Case:
    """Corrigé + liste figée de l'offre + CV ; vérifie qu'ils se correspondent."""
    data = json.loads(path.read_text(encoding="utf-8"))
    frozen = json.loads((eval_dir / "offers" / f"{data['offer']}.requirements.json").read_text(encoding="utf-8"))
    job = JobRequirements.model_validate(frozen)
    expected = {item["id"]: ExpectedLevel(**item) for item in data["requirements"]}
    check_case(job, expected, path.name)
    return Case(
        name=path.name.removesuffix(".expected.json"),
        cv=eval_dir / "cvs" / data["cv"],
        offer=eval_dir / "offers" / f"{data['offer']}.txt",
        job=job,
        expected=expected,
    )


def check_case(job: JobRequirements, expected: dict[str, ExpectedLevel], name: str) -> None:
    frozen = {r.id: r.label for r in job.requirements}
    if set(frozen) != set(expected):
        missing, extra = sorted(set(frozen) - set(expected)), sorted(set(expected) - set(frozen))
        raise CaseMismatchError(
            f"{name} : identifiants différents de la liste figée (absents {missing}, en trop {extra})"
        )
    for requirement_id, item in expected.items():
        if item.label != frozen[requirement_id]:
            raise CaseMismatchError(
                f"{name} : {requirement_id} vaut « {frozen[requirement_id]} » dans la liste figée, "
                f"« {item.label} » dans le corrigé"
            )
        unknown = [level for level in item.levels if level not in RANK]
        if not item.levels or unknown:
            raise CaseMismatchError(f"{name} : {requirement_id} : niveaux invalides {item.levels}")


def find_cases(names: list[str] | None = None, cases_dir: Path = CASES_DIR) -> list[Case]:
    paths = sorted(cases_dir.glob("*.expected.json"))
    if names:
        wanted = set(names)
        paths = [p for p in paths if p.name.removesuffix(".expected.json") in wanted]
        if len(paths) != len(wanted):
            raise SystemExit(
                f"cas inconnu(s) : {sorted(wanted - {p.name.removesuffix('.expected.json') for p in paths})}"
            )
    return [load_case(p) for p in paths]


# --- Mesures (fonctions pures) -------------------------------------------------------


@dataclass
class Outcome:
    """Une exigence d'une exécution, comparée au corrigé."""

    id: str
    label: str
    group: str
    expected: list[str]
    obtained: str  # niveau, ou "erreur"
    exact: bool
    within_one: bool
    false_positive: bool
    false_negative: bool
    evidence: list[str] = field(default_factory=list)
    rejected: int = 0
    second_chance: bool = False
    recovered: bool = False  # seconde chance qui a retrouvé une preuve exacte


def assess(expected: ExpectedLevel, result: RequirementMatch) -> Outcome:
    group = SOFT if result.requirement.category == "savoir-être" else TECHNICAL
    obtained = result.level if result.status == "ok" and result.level is not None else ERROR
    accepted = set(expected.levels)
    scored = obtained != ERROR
    return Outcome(
        id=expected.id,
        label=expected.label,
        group=group,
        expected=expected.levels,
        obtained=obtained,
        exact=obtained in accepted,
        within_one=scored and min(abs(RANK[obtained] - RANK[level]) for level in accepted) <= 1,
        false_positive=obtained in HIGH and accepted <= LOW,
        false_negative=obtained in LOW and accepted <= HIGH,
        evidence=[e.quote for e in result.evidence],
        rejected=result.rejected_evidence,
        second_chance=result.second_chance,
        recovered=result.second_chance and bool(result.evidence) and not result.downgraded,
    )


def _rounded(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def expected_scores(case: Case, settings: Settings) -> tuple[int, int, int]:
    """Score du corrigé (niveaux principaux), puis fourchette (niveaux acceptés les plus bas
    et les plus hauts), avec les points et poids de l'application."""
    points = settings.match_level_points

    def score(pick: Any) -> int:
        weights = [
            (requirement_weight(r, settings.match_weights), case.expected[r.id].levels) for r in case.job.requirements
        ]
        return _rounded(sum(w * points[pick(levels)] for w, levels in weights) / sum(w for w, _ in weights))

    return (
        score(lambda levels: levels[0]),
        score(lambda levels: min(levels, key=RANK.__getitem__)),
        score(lambda levels: max(levels, key=RANK.__getitem__)),
    )


@dataclass
class CaseRun:
    case: str
    run: int
    outcomes: list[Outcome]
    score: int | None
    expected_score: int
    expected_range: tuple[int, int]
    seconds: float
    error: str | None = None

    @property
    def score_gap(self) -> int | None:
        return None if self.score is None else self.score - self.expected_score


def case_run(case: Case, report: MatchReport, settings: Settings, run: int, seconds: float) -> CaseRun:
    by_id = {r.requirement.id: r for r in report.requirements}
    main, low, high = expected_scores(case, settings)
    return CaseRun(
        case=case.name,
        run=run,
        outcomes=[assess(case.expected[r.id], by_id[r.id]) for r in case.job.requirements],
        score=report.score,
        expected_score=main,
        expected_range=(low, high),
        seconds=seconds,
    )


def group_metrics(outcomes: list[Outcome]) -> dict[str, Any]:
    scored = [o for o in outcomes if o.obtained != ERROR]
    n = len(outcomes)
    return {
        "exigences": n,
        "exactes": sum(o.exact for o in outcomes),
        "exactitude": round(sum(o.exact for o in outcomes) / n, 3) if n else None,
        "à un niveau près": sum(o.within_one for o in outcomes),
        "exactitude à un niveau près": round(sum(o.within_one for o in outcomes) / n, 3) if n else None,
        "faux positifs": sum(o.false_positive for o in outcomes),
        "faux négatifs": sum(o.false_negative for o in outcomes),
        "erreurs": n - len(scored),
    }


def confusion(outcomes: list[Outcome]) -> dict[str, dict[str, int]]:
    """Niveau principal du corrigé (lignes) × niveau obtenu (colonnes)."""
    matrix = {row: dict.fromkeys([*LEVELS, ERROR], 0) for row in LEVELS}
    for outcome in outcomes:
        matrix[outcome.expected[0]][outcome.obtained] += 1
    return matrix


def summarize(runs: list[CaseRun]) -> dict[str, Any]:
    outcomes = [o for run in runs for o in run.outcomes]
    gaps = [run.score_gap for run in runs if run.score_gap is not None]
    by_case: dict[str, list[CaseRun]] = {}
    for run in runs:
        by_case.setdefault(run.case, []).append(run)
    unstable: dict[str, list[str]] = {}
    for name, case_runs in by_case.items():
        levels: dict[str, Counter[str]] = {}
        for run in case_runs:
            for o in run.outcomes:
                levels.setdefault(f"{o.id} {o.label}", Counter())[o.obtained] += 1
        changing = [f"{label} : {dict(counter)}" for label, counter in levels.items() if len(counter) > 1]
        if changing:
            unstable[name] = changing
    return {
        "exécutions": len({run.run for run in runs}),
        "par groupe": {group: group_metrics([o for o in outcomes if o.group == group]) for group in GROUPS},
        "total": group_metrics(outcomes),
        "confusion": confusion(outcomes),
        "faux positifs": [
            {
                "cas": run.case,
                "run": run.run,
                "id": o.id,
                "exigence": o.label,
                "groupe": o.group,
                "attendu": o.expected,
                "obtenu": o.obtained,
                "citations": o.evidence,
            }
            for run in runs
            for o in run.outcomes
            if o.false_positive
        ],
        "faux négatifs": [
            {
                "cas": run.case,
                "run": run.run,
                "id": o.id,
                "exigence": o.label,
                "groupe": o.group,
                "attendu": o.expected,
                "obtenu": o.obtained,
            }
            for run in runs
            for o in run.outcomes
            if o.false_negative
        ],
        "scores": [
            {
                "cas": run.case,
                "run": run.run,
                "obtenu": run.score,
                "corrigé": run.expected_score,
                "fourchette": list(run.expected_range),
                "écart": run.score_gap,
            }
            for run in runs
        ],
        "écart moyen de score": round(sum(gaps) / len(gaps), 1) if gaps else None,
        "écart absolu moyen de score": round(sum(abs(g) for g in gaps) / len(gaps), 1) if gaps else None,
        "citations rejetées": sum(o.rejected for o in outcomes),
        "secondes chances": sum(o.second_chance for o in outcomes),
        "secondes chances avec preuve retrouvée": sum(o.recovered for o in outcomes),
        "durée totale (s)": round(sum(run.seconds for run in runs), 1),
        "instables": unstable,
        "cas en échec": [f"{run.case} (run {run.run}) : {run.error}" for run in runs if run.error],
    }


# --- Exécution -----------------------------------------------------------------------


async def run_case(
    case: Case,
    *,
    settings: Settings,
    embeddings: Embeddings,
    extraction_llm: BaseChatModel,
    match_llm: BaseChatModel,
    limiter: RateLimiter,
    run: int = 1,
    today: date = EVAL_TODAY,
) -> CaseRun:
    """Le vrai pipeline sur les exigences figées : l'extraction n'est jamais appelée."""
    started = time.perf_counter()
    report: MatchReport | None = None
    stream = analyze(
        case.cv.read_bytes(),
        case.offer.read_text(encoding="utf-8"),
        settings=settings,
        embeddings=embeddings,
        extraction_llm=extraction_llm,
        match_llm=match_llm,
        limiter=limiter,
        requirements=case.job,
        today=today,
    )
    async with aclosing(stream) as events:
        async for event in events:
            if isinstance(event, ReportReady):
                report = event.report
    assert report is not None and report.requirements_source == "provided"
    return case_run(case, report, settings, run, time.perf_counter() - started)


async def evaluate(cases: list[Case], runs: int, settings: Settings) -> list[CaseRun]:
    """Cas et exécutions l'un après l'autre, par le même limiteur que l'application. Un
    refus durable du fournisseur (quota journalier, modèle refusé) arrête tout."""
    limiter = RateLimiter.from_settings(settings)
    embeddings = get_embeddings(settings)
    extraction_llm, match_llm = get_llm(settings), get_llm(match_settings(settings))
    results: list[CaseRun] = []
    for run in range(1, runs + 1):
        for case in cases:
            print(f"… run {run}/{runs} : {case.name}", file=sys.stderr, flush=True)
            started = time.perf_counter()
            try:
                results.append(
                    await run_case(
                        case,
                        settings=settings,
                        embeddings=embeddings,
                        extraction_llm=extraction_llm,
                        match_llm=match_llm,
                        limiter=limiter,
                        run=run,
                    )
                )
            except ProviderBlockedError as error:
                print(f"Arrêt : {error}", file=sys.stderr)
                return results
            except Exception as error:  # le rapport reste lisible si un cas échoue
                main, low, high = expected_scores(case, settings)
                results.append(
                    CaseRun(
                        case.name,
                        run,
                        [],
                        None,
                        main,
                        (low, high),
                        time.perf_counter() - started,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
    return results


# --- Rapport -------------------------------------------------------------------------

_MARK = {"exact": "✓", "near": "≈", "far": "✗"}


def _mark(o: Outcome) -> str:
    if o.false_positive:
        return "FAUX POSITIF"
    if o.false_negative:
        return "faux négatif"
    if o.obtained == ERROR:
        return "erreur"
    return _MARK["exact"] if o.exact else _MARK["near"] if o.within_one else _MARK["far"]


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def print_report(runs: list[CaseRun], settings: Settings) -> None:
    print(
        f"Modèle d'évaluation : {match_llm_model(settings)} (effort {settings.match_llm_reasoning_effort or 'défaut'})"
    )
    for run in runs:
        print(f"\n=== {run.case} (run {run.run}) — {run.seconds:.0f} s")
        if run.error:
            print(f"  ÉCHEC : {run.error}")
            continue
        low, high = run.expected_range
        print(
            f"  Score : {run.score} (corrigé {run.expected_score}, fourchette {low}–{high}, écart {run.score_gap:+d})"
        )
        for o in run.outcomes:
            extra = f" ; {o.rejected} citation(s) rejetée(s)" if o.rejected else ""
            extra += " ; seconde chance" + (" (preuve retrouvée)" if o.recovered else "") if o.second_chance else ""
            print(
                f"  {o.id:>3} [{o.group[:4]}] {o.label[:44]:<44} attendu {' / '.join(o.expected):<28} "
                f"obtenu {o.obtained:<13} {_mark(o)}{extra}"
            )
    summary = summarize(runs)
    print("\n=== Synthèse")
    for group, metrics in [*summary["par groupe"].items(), ("total", summary["total"])]:
        print(
            f"  {group:<12} {metrics['exigences']:>3} exigences : exactes {metrics['exactes']} "
            f"({_pct(metrics['exactitude'])}), à un niveau près {metrics['à un niveau près']} "
            f"({_pct(metrics['exactitude à un niveau près'])}), faux positifs {metrics['faux positifs']}, "
            f"faux négatifs {metrics['faux négatifs']}, erreurs {metrics['erreurs']}"
        )
    print("\n  Matrice de confusion (lignes : niveau principal du corrigé ; colonnes : niveau obtenu)")
    columns = [*LEVELS, ERROR]
    print("  " + " " * 14 + "".join(f"{c[:12]:>14}" for c in columns))
    for row, counts in summary["confusion"].items():
        print(f"  {row:<14}" + "".join(f"{counts[c]:>14}" for c in columns))
    print("\n  Faux positifs (démontré / solide alors que le corrigé dit non démontré / mentionné) :")
    for fp in summary["faux positifs"] or [None]:
        if fp is None:
            print("    aucun")
            continue
        print(
            f"    [{fp['groupe']}] {fp['cas']} {fp['id']} {fp['exigence']} : attendu {' / '.join(fp['attendu'])}, "
            f"obtenu {fp['obtenu']}"
        )
        for quote in fp["citations"]:
            print(f"        « {quote} »")
    print("\n  Faux négatifs :")
    for fn in summary["faux négatifs"] or [None]:
        print(
            "    aucun"
            if fn is None
            else f"    [{fn['groupe']}] {fn['cas']} {fn['id']} {fn['exigence']} : attendu {' / '.join(fn['attendu'])}, "
            f"obtenu {fn['obtenu']}"
        )
    print(
        f"\n  Écart de score moyen : {summary['écart moyen de score']} "
        f"(absolu : {summary['écart absolu moyen de score']})"
    )
    print(
        f"  Citations rejetées : {summary['citations rejetées']} ; secondes chances : {summary['secondes chances']} "
        f"(preuve retrouvée : {summary['secondes chances avec preuve retrouvée']})"
    )
    print(f"  Durée totale : {summary['durée totale (s)']} s")
    if summary["instables"]:
        print("  Niveaux qui varient d'une exécution à l'autre :")
        for name, items in summary["instables"].items():
            for item in items:
                print(f"    {name} {item}")
    for failure in summary["cas en échec"]:
        print(f"  Cas en échec : {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--case", action="append", help="cas à évaluer, par exemple java_cv__sii (répétable) ; défaut : tous"
    )
    parser.add_argument("--runs", type=int, default=1, help="exécutions de chaque cas (agrégées)")
    parser.add_argument("--json", action="store_true", help="détail et synthèse au format JSON")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs doit être au moins 1")
    settings = Settings()
    cases = find_cases(args.case)
    runs = asyncio.run(evaluate(cases, args.runs, settings))
    if args.json:
        print(
            json.dumps({"runs": [asdict(r) for r in runs], "synthèse": summarize(runs)}, ensure_ascii=False, indent=2)
        )
        return
    print_report(runs, settings)


if __name__ == "__main__":
    main()
