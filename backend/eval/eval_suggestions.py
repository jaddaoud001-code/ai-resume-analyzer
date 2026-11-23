"""Évaluation de l'agent de suggestions v2 (hors pytest).

Entrées : les rapports bruts du matching conservés dans eval/results/ (le plus récent pour
chaque cas, aucune nouvelle analyse) et les annotations eval/suggestions/<cas>.expected.json.
Pour chaque cas, l'agent (vrai LLM de l'agent, vrai matcher pour la réévaluation, limiteur
partagé) tourne sur le rapport et le CV fictif. Objectif v2 : nommer le terme de l'offre dans
la puce qui porte une preuve implicite. Chaque cible implicite est classée d'après son
annotation (`implicit`) :

- suggestion sur une cible annotée possible : **gain** (terme nommé, ancrage et juge passés,
  niveau qui ne baisse pas ; source annotée ou non, terme attendu ou non) ;
- suggestion sur une cible annotée impossible : **affirmation inventée** (objectif 0 ; toutes
  les suggestions sont listées pour un audit humain) ;
- refus sur une cible impossible : **refus juste** ; sur une cible possible : **faux refus** ;
- reformulation qui fait baisser le niveau : **baisse de niveau** (échec, non montrée) ;
- « non traitée » (budget) et erreur du LLM : non concluant ; **réévaluation impossible** à part ;
- compétence « mentionné » ou « non démontré » : **carte « à décrire »** (aucun appel au LLM).
S'y ajoutent les rejets par le contrôle d'ancrage, par le juge et pour terme absent (avec
motifs), les tokens et la durée.

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
    AgentStarted,
    AgentStep,
    Refusal,
    Suggestion,
    suggest,
)
from app.terms import mentions_term  # noqa: E402

EVAL_DIR = BACKEND_DIR / "eval"
ANNOTATIONS_DIR = EVAL_DIR / "suggestions"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_TODAY = date(2026, 11, 14)  # même date fixe que l'évaluation du matching
GAIN_CASE = "gain_cv__logistique"

Kind = Literal["possible", "impossible", "lacune", "non_annotee"]
Verdict = Literal[
    "invention",
    "gain",
    "faux_refus",
    "refus_juste",
    "carte",
    "baisse_de_niveau",
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


def report_by_source(source: str, directory: Path = RESULTS_DIR) -> MatchReport:
    """Le rapport désigné par « <fichier> : <clé> » (champ implicit_report des annotations)."""
    name, key = source.split(" : ", 1)
    return MatchReport.model_validate(json.loads((directory / name).read_text(encoding="utf-8"))[key])


def annotation_kind(annotation: dict[str, Any], requirement_id: str) -> tuple[Kind, dict[str, Any] | None]:
    for entry in annotation.get("implicit", []):
        if entry["id"] == requirement_id:
            return ("possible" if entry["possible"] else "impossible"), entry
    for gap in annotation.get("gaps", []):
        if gap["id"] == requirement_id:
            return "lacune", gap
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
    source_expected: bool | None = None  # gain : puce source annotée ?
    term_expected: bool | None = None  # gain : un des termes attendus est nommé ?
    detail: str | None = None


def classify(event: Suggestion | Refusal, annotation: dict[str, Any]) -> TargetOutcome:
    requirement = event.requirement
    kind, entry = annotation_kind(annotation, requirement.id)
    base = {"id": requirement.id, "label": requirement.label, "kind": kind}
    if isinstance(event, Suggestion):
        levels = {"level_before": event.level_before, "level_after": event.level_after}
        if kind == "possible":
            assert entry is not None
            source = entry["source"]
            return TargetOutcome(
                **base,  # type: ignore[arg-type]
                verdict="gain",
                source_expected=event.chunk_id == source["chunk_id"] and event.bullet_index == source["bullet"],
                term_expected=any(mentions_term(event.after, t) for t in entry["terms_expected"]),
                **levels,
            )
        if kind == "impossible":
            return TargetOutcome(**base, verdict="invention", detail="aucune reformulation honnête annotée", **levels)  # type: ignore[arg-type]
        return TargetOutcome(**base, verdict="non_annotee", **levels)  # type: ignore[arg-type]
    reason = event.reason
    verdict: Verdict
    if reason == "a_decrire":
        verdict = "carte"
    elif reason == "baisse_de_niveau":
        verdict = "baisse_de_niveau"
    elif reason == "reevaluation_impossible":
        verdict = "reevaluation_impossible"
    elif reason in ("non_traitee", "erreur"):
        verdict = "non_concluant"
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
    term_rejections: list[str] = field(default_factory=list)
    level_drops: list[str] = field(default_factory=list)  # chaque réévaluation en baisse
    skipped: int = 0  # cibles implicites au-delà de la limite
    usage: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None
    warning: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)


def measure(case: str, report_source: str, events: list[Any], annotation: dict[str, Any], seconds: float) -> CaseResult:
    result = CaseResult(case=case, report_source=report_source, seconds=seconds)
    annotated = annotation.get("implicit_report")
    if annotated and annotated != report_source:
        result.warning = f"annotations établies sur {annotated} : cibles à revoir"
    for event in events:
        if isinstance(event, Suggestion | Refusal):
            outcome = classify(event, annotation)
            result.outcomes.append(outcome)
            if isinstance(event, Suggestion):
                result.suggestions.append(
                    {
                        "id": event.requirement.id,
                        "exigence": event.requirement.label,
                        "verdict": outcome.verdict,
                        "item": event.item,
                        "chunk": event.chunk_id,
                        "puce": event.bullet_index,
                        "avant": event.before,
                        "après": event.after,
                        "termes": event.terms,
                        "faits": [f"[{f.chunk_id}] « {f.quote} » -> {f.claim}" for f in event.facts],
                        "implications": [f"{s.skill} <- « {s.quote} »" for s in event.implied_skills],
                        "niveaux": f"{event.level_before} -> {event.level_after}",
                        "justification": event.justification,
                    }
                )
        elif isinstance(event, AgentStarted):
            result.skipped = event.skipped
        elif isinstance(event, AgentStep) and event.status == "rejected" and event.kind in ("verify", "reevaluate"):
            detail = event.detail or ""
            line = f"{event.requirement_id} (tentative {event.attempt}) : {detail}"
            if event.kind == "reevaluate":
                result.level_drops.append(line)
            elif detail.startswith("Juge"):
                result.judge_rejections.append(line)
            elif detail.startswith("Terme absent"):
                result.term_rejections.append(line)
            else:
                result.grounding_rejections.append(line)
        elif isinstance(event, AgentDone):
            result.usage = event.usage.model_dump()
        elif isinstance(event, AgentError):
            result.error = f"{event.code} : {event.message}"
    result.events = [event.model_dump(mode="json") for event in events]
    return result


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    outcomes = [o for r in results for o in r.outcomes]
    targets = [o for o in outcomes if o.verdict != "carte"]
    verdicts = Counter(o.verdict for o in outcomes)
    possible = [o for o in targets if o.kind == "possible"]
    gains = [o for o in possible if o.verdict == "gain"]
    tokens = sum(r.usage.get("total_tokens", 0) for r in results)
    return {
        "cibles implicites traitées": len(targets),
        "cibles au-delà de la limite": sum(r.skipped for r in results),
        "cartes « à décrire »": verdicts["carte"],
        "affirmations inventées": verdicts["invention"],
        "gains": f"{len(gains)}/{len(possible)}" if possible else "–",
        "gains depuis la source annotée": sum(1 for o in gains if o.source_expected),
        "gains avec un terme attendu": sum(1 for o in gains if o.term_expected),
        "refus justes": verdicts["refus_juste"],
        "faux refus": verdicts["faux_refus"],
        "baisses de niveau (échecs)": verdicts["baisse_de_niveau"],
        "réévaluations en baisse (tentatives)": sum(len(r.level_drops) for r in results),
        "réévaluations impossibles": verdicts["reevaluation_impossible"],
        "non concluants": verdicts["non_concluant"],
        "non annotées": verdicts["non_annotee"],
        "rejets par l'ancrage": sum(len(r.grounding_rejections) for r in results),
        "rejets par le juge": sum(len(r.judge_rejections) for r in results),
        "rejets pour terme absent": sum(len(r.term_rejections) for r in results),
        "tokens": tokens,
        "tokens par cible": round(tokens / len(targets)) if targets else None,
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
        if result.warning:
            print(f"  ATTENTION : {result.warning}")
        for o in result.outcomes:
            levels = f" ; {o.level_before} -> {o.level_after}" if o.level_after else ""
            source = (
                "" if o.source_expected is None else (" ; source annotée" if o.source_expected else " ; autre source")
            )
            detail = f" — {o.detail}" if o.detail and o.verdict not in ("carte", "gain") else ""
            print(f"  {o.id:>3} {o.label[:40]:<40} [{o.kind}] {o.verdict.upper()}"
                  f"{f' ({o.reason})' if o.reason else ''}{levels}{source}{detail}")  # fmt: skip
        if result.skipped:
            print(f"    cibles implicites au-delà de la limite : {result.skipped}")
        for label, lines in (
            ("terme absent", result.term_rejections),
            ("rejet par l'ancrage", result.grounding_rejections),
            ("rejet par le juge", result.judge_rejections),
            ("réévaluation en baisse", result.level_drops),
        ):
            for line in lines:
                print(f"    {label} : {line}")
        if result.error:
            print(f"  ARRÊT : {result.error}")
    print("\n=== Suggestions à auditer (toutes)")
    for result in results:
        for s in result.suggestions:
            print(
                f"  [{s['verdict']}] {result.case} {s['id']} {s['exigence']} ({s['niveaux']})"
                f" — {s['item']}, chunk {s['chunk']}, puce {s['puce']} ; termes : {', '.join(s['termes'])}"
            )
            print(f"      avant : {s['avant']}\n      après : {s['après']}")
            for line in s["faits"] + s["implications"]:
                print(f"      {line}")
            print(f"      justification : {s['justification']}")
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
