"""POST /suggest : flux SSE de l'agent de suggestions (faux LLM, faux embeddings, aucun
réseau). CV du cas de gain et son vrai rapport d'analyse conservé (eval/results), donc un
rapport cohérent avec le CV par construction."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import BaseMessage

from app.config import Settings
from app.llm_calls import RateLimiter
from app.main import REPORT_CV_MISMATCH, REPORT_OFFER_MISMATCH, TOO_MANY_ANALYSES, create_app
from app.requirements import requirements_cache_key
from app.schemas import JobRequirements, MatchReport
from app.suggestions import JudgeVerdict, SearchDecision
from tests.conftest import FakeStructuredChatModel, KeywordEmbeddings, make_settings
from tests.test_analyze_stream import SSE, AsyncFakeModel, parse_sse
from tests.test_provider_limits import GROQ_HEADERS, TPD_MESSAGE, groq_error
from tests.test_suggestions_agent import CANDIDATE_CI_CD, HONEST, _matcher

EVAL = Path(__file__).resolve().parents[1] / "eval"
GAIN_PDF = (EVAL / "cvs" / "gain_cv.pdf").read_bytes()
DEMO_PDF = (EVAL / "cvs" / "demo_cv.pdf").read_bytes()
OFFER = (EVAL / "offers" / "logistique.txt").read_text(encoding="utf-8")
REPORT = json.loads((EVAL / "results" / "2026-11-23_gain_v4.json").read_text())["gain_cv__logistique (run 1)"]
REPORT_JSON = json.dumps(REPORT, ensure_ascii=False)
MATCH_MODEL = "modele-du-matching"
NOTHING = SearchDecision(decision="aucune_puce", reason="le terme ajouterait une information")


def agent_script() -> FakeStructuredChatModel:
    """Cibles du rapport : R8 (API REST), R9 (CI/CD), R10 (tests automatisés), dans l'ordre."""
    return FakeStructuredChatModel(
        responses=[
            NOTHING,
            CANDIDATE_CI_CD,
            HONEST,
            JudgeVerdict(faithful=True, reason="terme ajouté seulement"),
            NOTHING,
        ]
    )


def make_app(agent: FakeStructuredChatModel, matcher: FakeStructuredChatModel | None = None, **overrides: Any) -> Any:
    settings = make_settings(sse_ping_s=0, match_llm_model=MATCH_MODEL, **overrides)
    match = matcher or _matcher()
    kwargs = {k: overrides.pop(k) for k in ("limiter", "sleep", "embeddings_factory") if k in overrides}

    def llm_factory(s: Settings) -> FakeStructuredChatModel:
        return match if s.llm_model == MATCH_MODEL else agent

    kwargs.setdefault("embeddings_factory", lambda _: KeywordEmbeddings())
    return create_app(settings, llm_factory=llm_factory, **kwargs)


def form(pdf: bytes = GAIN_PDF, offer: str = OFFER, report: str = REPORT_JSON) -> dict[str, Any]:
    return {"files": {"file": ("cv.pdf", pdf, "application/pdf")}, "data": {"job_offer": offer, "report": report}}


async def post(app: Any, **fields: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/suggest", **form(**fields)) as response,
    ):
        await response.aread()
        return response


def run_post(app: Any, **fields: Any) -> httpx.Response:
    return asyncio.run(post(app, **fields))


def events_of(response: httpx.Response) -> list[SSE]:
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


# --- Flux complet --------------------------------------------------------------------------


def test_full_stream_of_agent_events() -> None:
    agent = agent_script()
    app = make_app(agent)

    events = events_of(run_post(app))

    assert [e.id for e in events] == list(range(1, len(events) + 1))
    assert all(e.event == e.data["type"] for e in events)
    assert events[0].event == "agent_started"
    assert [t["id"] for t in events[0].data["targets"]] == ["R8", "R9", "R10"]
    cards = [e.data["requirement"]["id"] for e in events if e.event == "refusal" and e.data["reason"] == "a_decrire"]
    assert cards == ["R4", "R5", "R6"]
    [suggestion] = [e.data for e in events if e.event == "suggestion"]
    assert suggestion["requirement"]["id"] == "R9" and suggestion["terms"] == ["CI/CD"]
    assert suggestion["after"] == HONEST.new_text and suggestion["judge"] == "terme ajouté seulement"
    assert [e.data["reason"] for e in events if e.event == "refusal" and e.data["reason"] != "a_decrire"] == [
        "aucune_puce",
        "aucune_puce",
    ]
    assert events[-1].event == "done" and events[-1].data["suggestions"] == 1
    assert app.state.analysis_slots.active == 0


def test_suggestion_exposes_only_exact_source_quotes() -> None:
    events = events_of(run_post(make_app(agent_script())))

    [suggestion] = [e.data for e in events if e.event == "suggestion"]
    # Citations sources exactes seulement : jamais la paraphrase du modèle (« claim »).
    assert suggestion["facts"] == [{"chunk_id": 2, "quote": HONEST.facts[0].quote}]
    assert "claim" not in json.dumps(suggestion)
    # Étapes : ni faits paraphrasés, ni compétences déclarées par le modèle.
    for step in (e.data for e in events if e.event == "step"):
        assert "facts" not in step and "implied_skills" not in step


# --- Validations avant le flux ------------------------------------------------------------


def assert_http_error(response: httpx.Response, code: int, app: Any, message: str | None = None) -> None:
    assert response.status_code == code
    assert response.headers["content-type"].startswith("application/json")
    if message is not None:
        assert response.json()["detail"] == message
    assert app.state.analysis_slots.active == 0


def test_invalid_pdf_returns_400_without_stream() -> None:
    app = make_app(agent_script())
    assert_http_error(run_post(app, pdf=b"pas un pdf"), 400, app)


def test_invalid_report_returns_422_without_stream() -> None:
    app = make_app(agent_script())
    assert_http_error(run_post(app, report="{pas du json"), 422, app)
    assert_http_error(run_post(app, report=json.dumps({"score": 3})), 422, app)


def test_report_of_another_cv_returns_422() -> None:
    app = make_app(agent_script())
    assert_http_error(run_post(app, pdf=DEMO_PDF), 422, app, REPORT_CV_MISMATCH)


def test_report_with_a_fabricated_quote_returns_422() -> None:
    report = MatchReport.model_validate(REPORT)
    r9 = next(r for r in report.requirements if r.requirement.id == "R9")
    r9.evidence[0].quote = "Pipeline CI/CD sur Kubernetes managé par mes soins"
    app = make_app(agent_script())

    assert_http_error(run_post(app, report=report.model_dump_json()), 422, app, REPORT_CV_MISMATCH)


def test_incoherent_report_returns_422() -> None:
    report = MatchReport.model_validate(REPORT)
    r7 = next(r for r in report.requirements if r.requirement.id == "R7")
    assert r7.level == "démontré"
    r7.points = 100  # points de « solide » annoncés pour un « démontré »
    duplicated = MatchReport.model_validate(REPORT)
    duplicated.requirements.append(duplicated.requirements[0])
    app = make_app(agent_script())

    assert_http_error(run_post(app, report=report.model_dump_json()), 422, app)
    assert_http_error(run_post(app, report=duplicated.model_dump_json()), 422, app)


def test_report_must_match_the_offer_requirements_known_to_the_app() -> None:
    app = make_app(agent_script())
    settings = make_settings(match_llm_model=MATCH_MODEL)
    report = MatchReport.model_validate(REPORT)
    requirements = [r.requirement for r in report.requirements]
    cache = app.state.requirements_cache
    other = JobRequirements(
        job_title="Autre", requirements=[r.model_copy(update={"label": "Autre"}) for r in requirements]
    )
    cache.put(requirements_cache_key(OFFER, settings), other)

    assert_http_error(run_post(app), 422, app, REPORT_OFFER_MISMATCH)

    cache.put(
        requirements_cache_key("Autre offre", settings), JobRequirements(job_title="x", requirements=requirements)
    )
    assert run_post(app, offer="Autre offre").status_code == 200  # exigences identiques : accepté


def test_too_large_report_and_offer_return_413() -> None:
    app = make_app(agent_script(), max_report_chars=1000)
    assert_http_error(run_post(app), 413, app)
    app = make_app(agent_script(), max_job_offer_chars=100)
    assert_http_error(run_post(app), 413, app)


def test_missing_field_returns_422() -> None:
    app = make_app(agent_script())
    assert_http_error(run_post(app, report=""), 422, app)


def test_shared_concurrency_limit_with_analyses() -> None:
    app = make_app(agent_script())
    app.state.analysis_slots.active = app.state.analysis_slots.limit  # analyses en cours

    response = run_post(app)

    assert response.status_code == 429 and response.json()["detail"] == TOO_MANY_ANALYSES
    app.state.analysis_slots.active = 0


# --- Erreurs fatales et délai --------------------------------------------------------------


def test_daily_quota_ends_the_stream_and_blocks_the_next_requests_before_the_stream() -> None:
    quota = groq_error(429, TPD_MESSAGE, GROQ_HEADERS | {"retry-after": "450"})
    app = make_app(FakeStructuredChatModel(responses=[quota]))

    events = events_of(run_post(app))

    assert events[-1].event == "error" and events[-1].data["code"] == "quota_exceeded"
    second = run_post(app)
    assert second.status_code == 429 and second.json()["detail"].startswith("Quota journalier")
    assert app.state.analysis_slots.active == 0


class BrokenEmbeddings(KeywordEmbeddings):
    """Embeddings qui échouent au premier calcul (réévaluation) : erreur inattendue."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("détail interne secret")


def test_unexpected_error_becomes_error_event_without_details() -> None:
    agent = FakeStructuredChatModel(responses=[NOTHING, CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True)])
    app = make_app(agent, embeddings_factory=lambda _: BrokenEmbeddings())

    events = events_of(run_post(app))

    assert events[-1].event == "error" and events[-1].data["code"] == "internal_error"
    assert "secret" not in events[-1].data["message"]
    assert app.state.analysis_slots.active == 0


def test_timeout_cancels_the_agent_sends_error_and_frees_slot() -> None:
    cancelled: list[str] = []
    call_started = asyncio.Event()

    async def hang(messages: list[BaseMessage]) -> Any:
        call_started.set()
        try:
            await asyncio.Event().wait()  # le fournisseur ne répond jamais
        except asyncio.CancelledError:
            cancelled.append("appel")
            raise

    async def simulated_deadline(seconds: float) -> None:
        assert seconds == 600  # SUGGEST_TIMEOUT_S par défaut
        await call_started.wait()

    app = make_app(AsyncFakeModel(async_responder=hang), sleep=simulated_deadline)

    events = events_of(run_post(app))

    assert events[-1].event == "error" and events[-1].data["code"] == "timeout"
    assert "600" in events[-1].data["message"]
    assert "suggestion" not in [e.event for e in events]
    assert cancelled and app.state.analysis_slots.active == 0


# --- Déconnexion ---------------------------------------------------------------------------


async def drive_until(app: Any, marker: bytes) -> bytes:
    """Pilote l'application ASGI directement : le client se déconnecte dès que `marker`
    apparaît dans le flux."""
    request = httpx.Request("POST", "http://test/suggest", **form())
    body = request.read()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/suggest",
        "raw_path": b"/suggest",
        "query_string": b"",
        "root_path": "",
        "client": ("test", 1234),
        "server": ("test", 80),
        "headers": [(k.lower().encode(), v.encode()) for k, v in request.headers.items()],
    }
    received = bytearray()
    disconnected = asyncio.Event()
    body_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            received.extend(message.get("body", b""))
            if marker in received:
                disconnected.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return bytes(received)


def test_disconnect_stops_the_agent_and_no_new_llm_call() -> None:
    script = [NOTHING, CANDIDATE_CI_CD, HONEST, JudgeVerdict(faithful=True), NOTHING]

    async def slow(messages: list[BaseMessage]) -> Any:
        await asyncio.sleep(0.02)
        return script.pop(0)

    agent = AsyncFakeModel(async_responder=slow)
    app = make_app(agent, limiter=RateLimiter(0, 1))
    counts: list[int] = []

    async def main() -> bytes:
        raw = await drive_until(app, b'"reason":"aucune_puce"')  # première cible (R8) traitée
        counts.append(len(agent.calls))
        await asyncio.sleep(0.2)  # plus rien ne doit partir
        counts.append(len(agent.calls))
        return raw

    raw = asyncio.run(main())

    events = parse_sse(raw.decode())
    assert "done" not in [e.event for e in events]
    assert counts[0] == counts[1] and counts[0] < 5  # aucun nouvel appel, des appels évités
    assert app.state.analysis_slots.active == 0
