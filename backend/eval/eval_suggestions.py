"""Évaluation de l'agent de suggestions (hors pytest).

Entrées : les rapports bruts du matching conservés dans eval/results/ (le plus récent pour
chaque cas, aucune nouvelle analyse) et les annotations eval/suggestions/<cas>.expected.json.
Pour chaque cas, l'agent (vrai LLM de l'agent, vrai matcher pour la réévaluation, limiteur
partagé) tourne sur le rapport et le CV fictif ; chaque exigence ciblée est classée d'après
son annotation :

- suggestion sur une lacune annotée impossible, ou au-delà du niveau maximal honnête :
  **affirmation inventée** (objectif 0 ; toutes les suggestions sont listées pour un audit
  humain, car l'annotation ne voit pas tout) ;
- suggestion sur une lacune possible : **gain** (niveau avant -> après, source attendue ou non) ;
- refus sur une lacune impossible : **refus juste** ; sur une lacune possible : **faux refus** ;
- exigence **déjà prouvée** (faux négatif du matching) : « aucune reformulation nécessaire » ou
  reformulation fidèle acceptables, jamais comptées comme invention ni comme faux refus ;
- « non traitée » (budget) et erreur du LLM : non concluant ; **réévaluation impossible** à part.
S'y ajoutent les rejets par le contrôle d'ancrage et par le juge (avec motifs), les refus
« sans effet » (non montrés à l'utilisateur), les tokens et la durée.

Usage (depuis backend/) :
    .venv/bin/python eval/eval_suggestions.py                     # cas qui ont un rapport
    .venv/bin/python eval/eval_suggestions.py --case gain_cv__logistique --analyze-gain
    .venv/bin/python eval/eval_suggestions.py --save --json
--analyze-gain lance d'abord l'analyse du cas de gain (vrai LLM) et conserve son rapport dans
eval/results/<date>_gain_v<prompt du matching>.json.
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
from pathlib import Path
from typing import Any, Literal

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402

from app.chunking import chunk_cv  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import get_embeddings  # noqa: E402
from app.llm import get_llm, llm_model, match_settings, suggest_settings  # noqa: E402
from app.llm_calls import RateLimiter  # noqa: E402
from app.matcher import MATCH_PROMPT_VERSION, analyze  # noqa: E402
from app.pdf_parser import extract_text  # noqa: E402
from app.schemas import Chunk, JobRequirements, MatchReport, ReportReady  # noqa: E402
from app.suggestions import (  # noqa: E402
    SUGGEST_PROMPT_VERSION,
    AgentDone,
    AgentError,
    AgentStep,
    Refusal,
    Suggestion,
    suggest,
)

EVAL_DIR = BACKEND_DIR / "eval"
ANNOTATIONS_DIR = EVAL_DIR / "suggestions"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_TODAY = date(2026, 11, 14)  # même date fixe que l'évaluation du matching
GAIN_CASE = "gain_cv__logistique"
LEVEL_RANK = {"non démontré": 0, "mentionné": 1, "démontré": 2, "solide": 3}

Kind = Literal["possible", "impossible", "deja_prouve", "non_annotee"]
Verdict = Literal[
    "invention",
    "gain",
    "faux_refus",
    "refus_juste",
    "deja_prouve_acceptable",
    "reevaluation_impossible",
    "non_concluant",
    "non_annotee",
]


# --- Données --------------------------------------------------------------------------------


def load_annotations(directory: Path = ANNOTATIONS_DIR) -> dict[str, dict[str, Any]]:
    return {p.name.removesuffix(".expected.json"): json.loads(p.read_text()) for p in directory.glob("*.expected.json")}


def latest_reports(directory: Path = RESULTS_DIR) -> dict[str, tuple[str, MatchReport]]:
    """Le rapport de matching le plus récent pour chaque cas (fichiers <date>_… triés par nom ;
    dans un fichier, la dernière exécution). Les sorties de cette évaluation sont ignorées."""
    found: dict[str, tuple[str, MatchReport]] = {}
    for path in sorted(directory.glob("*.json")):
        if "_suggestions_" in path.name:
            continue
        for key, report in json.loads(path.read_text(encoding="utf-8")).items():
            case = key.split(" (run")[0]
            found[case] = (f"{path.name} : {key}", MatchReport.model_validate(report))
    return found


def annotation_kind(annotation: dict[str, Any], requirement_id: str) -> tuple[Kind, dict[str, Any] | None]:
    for gap in annotation.get("gaps", []):
        if gap["id"] == requirement_id:
            return ("possible" if gap["possible"] else "impossible"), gap
    for entry in annotation.get("already_proven", []):
        if entry["id"] == requirement_id:
            return "deja_prouve", entry
    return "non_annotee", None


# --- Mesures (fonctions pures) ----------------------------------------------------------------


@dataclass
class TargetOutcome:
    id: str
    label: str
    kind: Kind
    verdict: Verdict
    reason: str | None = None  # raison du refus, ou None pour une suggestion
    level_before: str | None = None
    level_after: str | None = None
    source_expected: bool | None = None  # suggestion sur une lacune possible : source annotée ?
    detail: str | None = None


def classify(event: Suggestion | Refusal, annotation: dict[str, Any]) -> TargetOutcome:
    requirement = event.requirement
    kind, entry = annotation_kind(annotation, requirement.id)
    base = {"id": requirement.id, "label": requirement.label, "kind": kind}
    if isinstance(event, Suggestion):
        levels = {"level_before": event.level_before, "level_after": event.level_after}
        if kind == "impossible":
            return TargetOutcome(
                **base, verdict="invention", detail="suggestion sur une lacune sans suggestion honnête", **levels
            )  # type: ignore[arg-type]
        if kind == "possible":
            assert entry is not None
            if LEVEL_RANK.get(event.level_after, 0) > LEVEL_RANK.get(entry["max_honest_level"], 0):
                return TargetOutcome(**base, verdict="invention", detail="niveau au-delà du maximum honnête", **levels)  # type: ignore[arg-type]
            source = entry["source"] or {}
            expected = event.item == source.get("item") and event.bullet_index == source.get("bullet")
            return TargetOutcome(**base, verdict="gain", source_expected=expected, **levels)  # type: ignore[arg-type]
        if kind == "deja_prouve":
            return TargetOutcome(**base, verdict="deja_prouve_acceptable", detail="reformulation à auditer", **levels)  # type: ignore[arg-type]
        return TargetOutcome(**base, verdict="non_annotee", **levels)  # type: ignore[arg-type]
    reason = event.reason
    if reason == "reevaluation_impossible":
        verdict: Verdict = "reevaluation_impossible"
    elif reason in ("non_traitee", "erreur"):
        verdict = "non_concluant"
    elif kind == "deja_prouve":
        verdict = "deja_prouve_acceptable"
    elif kind == "impossible":
        verdict = "refus_juste"
    elif kind == "possible":
        verdict = "faux_refus"
    else:
        verdict = "non_annotee"
    return TargetOutcome(**base, verdict=verdict, reason=reason, detail=event.detail)  # type: ignore[arg-type]


@dataclass
class CaseResult:
    case: str
    report_source: str
    outcomes: list[TargetOutcome] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)  # audit humain
    grounding_rejections: list[str] = field(default_factory=list)
    judge_rejections: list[str] = field(default_factory=list)
    hidden_no_gain: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)


def measure(case: str, report_source: str, events: list[Any], annotation: dict[str, Any], seconds: float) -> CaseResult:
    result = CaseResult(case=case, report_source=report_source, seconds=seconds)
    for event in events:
        if isinstance(event, Suggestion | Refusal):
            outcome = classify(event, annotation)
            result.outcomes.append(outcome)
            if isinstance(event, Refusal) and event.reason == "sans_effet":
                result.hidden_no_gain += 1
            if isinstance(event, Suggestion):
                result.suggestions.append(
                    {
                        "id": event.requirement.id,
                        "exigence": event.requirement.label,
                        "verdict": outcome.verdict,
                        "item": event.item,
                        "puce": event.bullet_index,
                        "avant": event.before,
                        "après": event.after,
                        "faits": [f"[{f.chunk_id}] « {f.quote} » -> {f.claim}" for f in event.facts],
                        "implications": [f"{s.skill} <- « {s.quote} »" for s in event.implied_skills],
                        "niveaux": f"{event.level_before} -> {event.level_after}",
                        "justification": event.justification,
                    }
                )
        elif isinstance(event, AgentStep) and event.kind == "verify" and event.status == "rejected":
            detail = event.detail or ""
            target = result.judge_rejections if detail.startswith("Juge") else result.grounding_rejections
            target.append(f"{event.requirement_id} (tentative {event.attempt}) : {detail}")
        elif isinstance(event, AgentDone):
            result.usage = event.usage.model_dump()
        elif isinstance(event, AgentError):
            result.error = f"{event.code} : {event.message}"
    result.events = [event.model_dump(mode="json") for event in events]
    return result


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    outcomes = [o for r in results for o in r.outcomes]
    verdicts = Counter(o.verdict for o in outcomes)
    possible = [o for o in outcomes if o.kind == "possible"]
    gains = [o for o in possible if o.verdict == "gain"]
    tokens = sum(r.usage.get("total_tokens", 0) for r in results)
    return {
        "cibles": len(outcomes),
        "affirmations inventées": verdicts["invention"],
        "refus justes": verdicts["refus_juste"],
        "faux refus": verdicts["faux_refus"],
        "déjà prouvées (acceptable)": verdicts["deja_prouve_acceptable"],
        "gains": f"{len(gains)}/{len(possible)}" if possible else "–",
        "gains depuis la source annotée": sum(1 for o in gains if o.source_expected),
        "réévaluations impossibles": verdicts["reevaluation_impossible"],
        "non concluants": verdicts["non_concluant"],
        "non annotées": verdicts["non_annotee"],
        "sans effet (non montrés)": sum(r.hidden_no_gain for r in results),
        "rejets par l'ancrage": sum(len(r.grounding_rejections) for r in results),
        "rejets par le juge": sum(len(r.judge_rejections) for r in results),
        "tokens": tokens,
        "tokens par cible": round(tokens / len(outcomes)) if outcomes else None,
        "durée (s)": round(sum(r.seconds for r in results), 1),
        "arrêts": [f"{r.case} : {r.error}" for r in results if r.error],
    }


# --- Exécution --------------------------------------------------------------------------------


def cv_chunks(cv: Path, settings: Settings) -> list[Chunk]:
    """Même découpage que l'analyse : les identifiants de chunks du rapport y correspondent."""
    return chunk_cv(extract_text(cv.read_bytes(), settings).text, settings.chunk_max_chars, settings.chunk_overlap)


async def run_case(
    case: str,
    report_source: str,
    report: MatchReport,
    annotation: dict[str, Any],
    *,
    settings: Settings,
    embeddings: Embeddings,
    agent_llm: BaseChatModel,
    match_llm: BaseChatModel,
    limiter: RateLimiter,
    today: date = EVAL_TODAY,
) -> CaseResult:
    chunks = cv_chunks(EVAL_DIR / "cvs" / annotation["cv"], settings)
    started = time.perf_counter()
    events: list[Any] = []
    stream = suggest(
        report,
        chunks,
        settings=settings,
        embeddings=embeddings,
        agent_llm=agent_llm,
        match_llm=match_llm,
        limiter=limiter,
        today=today,
    )
    async with aclosing(stream) as items:
        async for event in items:
            events.append(event)
    return measure(case, report_source, events, annotation, time.perf_counter() - started)


async def analyze_gain_case(settings: Settings, embeddings: Embeddings, limiter: RateLimiter) -> Path:
    """Analyse du cas de gain (vrai LLM, exigences figées de l'offre logistique), rapport conservé."""
    job = JobRequirements.model_validate(json.loads((EVAL_DIR / "offers" / "logistique.requirements.json").read_text()))
    report: MatchReport | None = None
    stream = analyze(
        (EVAL_DIR / "cvs" / "gain_cv.pdf").read_bytes(),
        (EVAL_DIR / "offers" / "logistique.txt").read_text(encoding="utf-8"),
        settings=settings,
        embeddings=embeddings,
        extraction_llm=get_llm(settings),
        match_llm=get_llm(match_settings(settings)),
        limiter=limiter,
        requirements=job,
        today=EVAL_TODAY,
    )
    async with aclosing(stream) as events:
        async for event in events:
            if isinstance(event, ReportReady):
                report = event.report
    assert report is not None
    path = RESULTS_DIR / f"{date.today().isoformat()}_gain_v{MATCH_PROMPT_VERSION}.json"
    path.write_text(
        json.dumps({f"{GAIN_CASE} (run 1)": report.model_dump(mode="json")}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return path


async def evaluate(cases: list[str] | None, analyze_gain: bool, settings: Settings) -> list[CaseResult]:
    limiter = RateLimiter.from_settings(settings)
    embeddings = get_embeddings(settings)
    if analyze_gain:
        print(f"Analyse du cas de gain : {await analyze_gain_case(settings, embeddings, limiter)}", file=sys.stderr)
    annotations = load_annotations()
    reports = latest_reports()
    selected = [c for c in sorted(annotations) if (not cases or c in cases)]
    missing = [c for c in selected if c not in reports]
    if missing:
        print(f"Sans rapport de matching (ignorés) : {', '.join(missing)}", file=sys.stderr)
    agent_llm, match_llm = get_llm(suggest_settings(settings)), get_llm(match_settings(settings))
    results: list[CaseResult] = []
    for case in [c for c in selected if c in reports]:
        print(f"… {case}", file=sys.stderr, flush=True)
        source, report = reports[case]
        result = await run_case(
            case,
            source,
            report,
            annotations[case],
            settings=settings,
            embeddings=embeddings,
            agent_llm=agent_llm,
            match_llm=match_llm,
            limiter=limiter,
        )
        results.append(result)
        if result.error:  # refus durable du fournisseur : inutile de continuer
            break
    return results


# --- Rapport ----------------------------------------------------------------------------------


def results_path(day: date, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"{day.isoformat()}_suggestions_v{SUGGEST_PROMPT_VERSION}.json"


def print_report(results: list[CaseResult], settings: Settings) -> None:
    agent = suggest_settings(settings)
    print(
        f"Agent : {llm_model(agent)} (effort {agent.llm_reasoning_effort or 'défaut'}) ; prompts de l'agent "
        f"v{SUGGEST_PROMPT_VERSION} ; réévaluation : prompt du matching v{MATCH_PROMPT_VERSION}"
    )
    for result in results:
        print(f"\n=== {result.case} — {result.seconds:.0f} s ; {result.usage.get('total_tokens', 0)} tokens")
        print(f"  Rapport de matching : {result.report_source}")
        for o in result.outcomes:
            levels = f" ; {o.level_before} -> {o.level_after}" if o.level_after else ""
            source = (
                "" if o.source_expected is None else (" ; source annotée" if o.source_expected else " ; autre source")
            )
            print(f"  {o.id:>3} {o.label[:40]:<40} [{o.kind}] {o.verdict.upper()}"
                  f"{f' ({o.reason})' if o.reason else ''}{levels}{source}")  # fmt: skip
        for line in result.grounding_rejections:
            print(f"    rejet par l'ancrage : {line}")
        for line in result.judge_rejections:
            print(f"    rejet par le juge : {line}")
        if result.error:
            print(f"  ARRÊT : {result.error}")
    print("\n=== Suggestions à auditer (toutes)")
    for result in results:
        for s in result.suggestions:
            print(
                f"  [{s['verdict']}] {result.case} {s['id']} {s['exigence']} ({s['niveaux']})"
                f" — {s['item']}, puce {s['puce']}"
            )
            print(f"      avant : {s['avant']}\n      après : {s['après']}")
            for line in s["faits"] + s["implications"]:
                print(f"      {line}")
    print("\n=== Synthèse")
    for key, value in summarize(results).items():
        print(f"  {key} : {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--case", action="append", help="cas à évaluer (répétable) ; défaut : tous ceux qui ont un rapport"
    )
    parser.add_argument("--analyze-gain", action="store_true", help="analyser d'abord le cas de gain (vrai LLM)")
    parser.add_argument(
        "--save", type=Path, nargs="?", const=RESULTS_DIR, help="garder les événements bruts (défaut : eval/results/)"
    )
    parser.add_argument("--json", action="store_true", help="résultats et synthèse au format JSON")
    args = parser.parse_args()
    settings = Settings()
    results = asyncio.run(evaluate(args.case, args.analyze_gain, settings))
    if args.save:
        target = results_path(date.today()) if args.save == RESULTS_DIR else args.save
        target.write_text(
            json.dumps({r.case: asdict(r) for r in results}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"Événements bruts : {target}", file=sys.stderr)
    if args.json:
        print(
            json.dumps(
                {"résultats": [asdict(r) for r in results], "synthèse": summarize(results)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print_report(results, settings)


if __name__ == "__main__":
    main()
