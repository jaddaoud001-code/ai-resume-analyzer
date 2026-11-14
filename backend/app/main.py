"""Point d'entrée FastAPI : création de l'app et déclaration des routes."""

import asyncio
import hashlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.background import BackgroundTask

from app.chunking import chunk_cv
from app.config import ENV_FILE, Settings, get_settings, warn_about_env_file
from app.embeddings import embeddings_model, get_embeddings
from app.llm import get_llm, match_settings
from app.llm_calls import ProviderAccessDeniedError, ProviderBlockedError, ProviderQuotaExceededError, RateLimiter
from app.matcher import AnalysisTimeoutError, EmbeddingsError, analyze
from app.pdf_parser import PDFParseError, PDFTooLargeError, TooManyPagesError, extract_text
from app.providers import MissingAPIKeyError
from app.requirements import (
    JobOfferTooLongError,
    LLMProviderError,
    NoRequirementsError,
    RequirementsCache,
    check_job_offer,
    extract_requirements_cached,
)
from app.schemas import (
    AnalysisError,
    AnalysisErrorCode,
    EventTiming,
    JobOfferRequest,
    JobRequirements,
    MatchDebugResponse,
    MatchEvent,
    MatchReport,
    ParsedCV,
    ReportReady,
    SearchDebugResponse,
)
from app.vectorstore import CVIndex

logger = logging.getLogger(__name__)

EmbeddingsFactory = Callable[[Settings], Embeddings]
LLMFactory = Callable[[Settings], BaseChatModel]


class HealthResponse(BaseModel):
    status: str
    # Modèle d'embeddings prêt (chargé) ; sinon une analyse attendra la fin du chargement.
    embeddings_ready: bool


class PublicConfig(BaseModel):
    """Limites publiques, lues par le frontend pour valider avant l'envoi."""

    max_pdf_size_bytes: int
    max_pdf_pages: int
    max_job_offer_chars: int


class EmbeddingsUnavailableError(RuntimeError):
    """Le modèle d'embeddings n'a pas pu être chargé (au démarrage ni à la reprise)."""


def _http_status(error: PDFParseError) -> int:
    if isinstance(error, PDFTooLargeError | TooManyPagesError):
        return status.HTTP_413_CONTENT_TOO_LARGE
    return status.HTTP_400_BAD_REQUEST


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _configure_logging(level: str) -> None:
    # uvicorn ne configure que ses propres loggers : on donne un handler à ceux de l'app.
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level.upper())
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
        app_logger.addHandler(handler)


class AnalysisSlot:
    """Créneau d'analyse simultanée ; `release` est idempotent (appelé à la fin du flux et
    en tâche de fond de la réponse, pour couvrir une déconnexion avant le premier événement)."""

    def __init__(self, slots: "AnalysisSlots") -> None:
        self._slots = slots
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._slots.active -= 1


class AnalysisSlots:
    """Au plus `limit` analyses en même temps (quota du fournisseur partagé), sans file
    d'attente : au-delà, l'appelant répond 429."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def try_acquire(self) -> AnalysisSlot | None:
        # Pas d'await entre le test et l'incrément : atomique dans la boucle asyncio.
        if self.active >= self.limit:
            return None
        self.active += 1
        return AnalysisSlot(self)


@dataclass
class _EmbeddingsState:
    ready: bool = False


@dataclass
class PreparedAnalysis:
    parsed: ParsedCV
    embeddings: Embeddings
    extraction_llm: BaseChatModel
    match_llm: BaseChatModel


def _preparation_http_error(error: Exception) -> HTTPException | None:
    """Erreurs de validation détectables avant toute analyse -> réponse HTTP classique."""
    if isinstance(error, JobOfferTooLongError):
        return HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(error))
    if isinstance(error, PDFParseError):
        return HTTPException(_http_status(error), str(error))
    if isinstance(error, MissingAPIKeyError):
        return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(error))
    if isinstance(error, EmbeddingsUnavailableError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error))
    return None


# Quota journalier épuisé sans délai de reprise annoncé : analyses refusées pendant 1 h.
_QUOTA_BLOCK_S = 3600.0


class ProviderBlock:
    """Dernier refus durable du fournisseur (quota journalier, clé ou modèle refusés) constaté
    par une analyse : les suivantes reçoivent la même réponse avant d'ouvrir le flux, sans
    appeler le fournisseur. Quota : jusqu'à la reprise annoncée ; accès refusé : jusqu'au
    redémarrage (le .env n'est lu qu'au démarrage)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._error: ProviderBlockedError | None = None
        self._until: float | None = None

    def record(self, error: ProviderBlockedError) -> None:
        self._error = error
        if isinstance(error, ProviderQuotaExceededError):
            self._until = self._clock() + max(error.retry_after or _QUOTA_BLOCK_S, 60.0)
        else:
            self._until = None

    def current(self) -> ProviderBlockedError | None:
        if self._error is not None and self._until is not None and self._clock() >= self._until:
            self._error = self._until = None
        return self._error


def _blocked_http_error(error: ProviderBlockedError) -> HTTPException:
    if isinstance(error, ProviderQuotaExceededError):
        return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(error))
    return HTTPException(status.HTTP_502_BAD_GATEWAY, str(error))


def _stream_error(error: Exception) -> AnalysisError:
    """Erreur fatale pendant le flux -> événement « error » (jamais le détail d'une exception
    inattendue, qui pourrait contenir des données)."""
    code: AnalysisErrorCode
    if isinstance(error, AnalysisTimeoutError):
        code, message = "timeout", str(error)
    elif isinstance(error, ProviderQuotaExceededError):
        code, message = "quota_exceeded", str(error)
    elif isinstance(error, ProviderAccessDeniedError):
        code, message = "provider_access_denied", str(error)
    elif isinstance(error, LLMProviderError):
        code, message = "llm_provider_error", str(error)
    elif isinstance(error, NoRequirementsError):
        code, message = "no_requirements", str(error)
    elif isinstance(error, EmbeddingsError):
        code, message = "embeddings_error", str(error)
    else:
        code, message = "internal_error", "Erreur interne pendant l'analyse."
    return AnalysisError(code=code, message=message)


TOO_MANY_ANALYSES = "Trop d'analyses en cours, réessayez dans quelques instants."


def create_app(
    settings: Settings | None = None,
    embeddings_factory: EmbeddingsFactory = get_embeddings,
    llm_factory: LLMFactory = get_llm,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FastAPI:
    """`limiter` et `sleep` (délai maximal d'une analyse) : injectables pour les tests."""
    from_env_file = settings is None
    settings = settings or get_settings()
    _configure_logging(settings.log_level)
    if from_env_file:
        # Settings lus depuis le .env : signaler les variables mal nommées ou en double.
        warn_about_env_file(ENV_FILE)
    embeddings_state = _EmbeddingsState()
    provider_block = ProviderBlock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Chargement du modèle local en tâche de fond : le serveur répond tout de suite
        # (/health compris) même pendant un premier téléchargement de ~2 Go ; une analyse
        # arrivée entre-temps attend la fin du chargement (verrou de app.embeddings).
        if settings.embeddings_provider == "local":
            app.state.embeddings_warmup = asyncio.create_task(asyncio.to_thread(warm_embeddings))
        yield

    app = FastAPI(title="AI Resume Analyzer", lifespan=lifespan)
    # Un seul limiteur pour tous les appels au LLM du processus (extraction et évaluation).
    limiter = limiter or RateLimiter.from_settings(settings)
    # Exigences déjà extraites, par offre : en mémoire seulement, propre à cette application.
    requirements_cache = RequirementsCache(settings.requirements_cache_size)
    slots = AnalysisSlots(settings.analysis_concurrency)
    app.state.analysis_slots = slots
    app.state.provider_block = provider_block

    # Embeddings construits une seule fois par application, au premier usage (le modèle
    # local, ~2 Go, est en plus partagé par le processus : voir app.embeddings).
    embeddings_lock = threading.Lock()
    embeddings_instance: list[Embeddings] = []

    def app_embeddings() -> Embeddings:
        with embeddings_lock:
            if not embeddings_instance:
                embeddings_instance.append(embeddings_factory(settings))
            return embeddings_instance[0]

    def load_embeddings() -> Embeddings:
        """Embeddings de l'app, modèle chargé (attend un chargement en cours, réessaie après
        un échec). Lève EmbeddingsUnavailableError si le chargement échoue encore."""
        embeddings = app_embeddings()
        load = getattr(embeddings, "load", None)
        if callable(load):
            try:
                load()
            except Exception as error:
                raise EmbeddingsUnavailableError(
                    f"Modèle d'embeddings indisponible ({type(error).__name__}). Réessayez plus tard."
                ) from error
        embeddings_state.ready = True
        return embeddings

    def warm_embeddings() -> None:
        started = time.perf_counter()
        try:
            load_embeddings()
        except Exception as error:
            # L'application démarre quand même ; la première analyse réessaiera.
            cause = error.__cause__ or error
            logger.error(
                "Chargement du modèle d'embeddings impossible (%s) : la première analyse réessaiera",
                type(cause).__name__,
            )
            return
        logger.info("Modèle d'embeddings chargé en %.1f s", time.perf_counter() - started)

    def embeddings_ready() -> bool:
        return settings.embeddings_provider != "local" or embeddings_state.ready

    def prepare(data: bytes, job_offer: str) -> PreparedAnalysis:
        """Validations avant analyse (offre, PDF, clés, modèle) : exécutée dans un thread."""
        check_job_offer(job_offer, settings)
        parsed = extract_text(data, settings)
        return PreparedAnalysis(
            parsed=parsed,
            embeddings=load_embeddings(),
            extraction_llm=llm_factory(settings),
            match_llm=llm_factory(match_settings(settings)),
        )

    def run_analysis(prepared: PreparedAnalysis, job_offer: str) -> AsyncIterator[MatchEvent]:
        return analyze(
            prepared.parsed,
            job_offer,
            settings=settings,
            embeddings=prepared.embeddings,
            extraction_llm=prepared.extraction_llm,
            match_llm=prepared.match_llm,
            limiter=limiter,
            requirements_cache=requirements_cache,
            sleep=sleep,
        )

    async def sse_events(
        prepared: PreparedAnalysis, job_offer: str, slot: AnalysisSlot
    ) -> AsyncIterator[ServerSentEvent]:
        event_id = 0

        def sse(payload: BaseModel, kind: str) -> ServerSentEvent:
            nonlocal event_id
            event_id += 1
            return ServerSentEvent(data=payload.model_dump_json(), event=kind, id=str(event_id))

        try:
            # aclosing : si ce flux est fermé (client parti), l'analyse l'est aussi.
            async with aclosing(run_analysis(prepared, job_offer)) as events:
                async for event in events:
                    yield sse(event, event.type)
        except Exception as error:
            if isinstance(error, ProviderBlockedError):
                provider_block.record(error)
            elif not isinstance(error, AnalysisTimeoutError | LLMProviderError | NoRequirementsError | EmbeddingsError):
                logger.error("Analyse : erreur inattendue %s", type(error).__name__)
            payload = _stream_error(error)
            yield sse(payload, payload.type)
        finally:
            slot.release()

    @app.post(
        "/analyze",
        response_class=EventSourceResponse,
        responses={
            200: {"content": {"text/event-stream": {}}, "description": "Flux SSE des événements de l'analyse."},
            400: {"description": "PDF invalide ou sans texte."},
            413: {"description": "PDF ou offre trop volumineux."},
            422: {"description": "Champ manquant ou vide."},
            429: {"description": TOO_MANY_ANALYSES + " Ou quota journalier du fournisseur LLM atteint."},
            500: {"description": "Clé du fournisseur absente."},
            502: {"description": "Clé ou modèle refusés par le fournisseur LLM (constaté par une analyse précédente)."},
            503: {"description": "Modèle d'embeddings indisponible (chargement échoué)."},
        },
    )
    async def analyze_stream(file: UploadFile, job_offer: Annotated[str, Form(min_length=1)]) -> EventSourceResponse:
        # Refus durable déjà constaté (quota journalier, clé ou modèle) : réponse avant le flux.
        if (blocked := provider_block.current()) is not None:
            raise _blocked_http_error(blocked)
        slot = slots.try_acquire()
        if slot is None:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, TOO_MANY_ANALYSES)
        try:
            data = await file.read(settings.max_pdf_size_bytes + 1)
            prepared = await run_in_threadpool(prepare, data, job_offer)
        except Exception as error:
            slot.release()
            if (http_error := _preparation_http_error(error)) is not None:
                raise http_error from error
            raise
        return EventSourceResponse(
            sse_events(prepared, job_offer, slot),
            ping=settings.sse_ping_s,
            background=BackgroundTask(slot.release),
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", embeddings_ready=embeddings_ready())

    @app.get("/config", response_model=PublicConfig)
    def public_config() -> PublicConfig:
        return PublicConfig(
            max_pdf_size_bytes=settings.max_pdf_size_bytes,
            max_pdf_pages=settings.max_pdf_pages,
            max_job_offer_chars=settings.max_job_offer_chars,
        )

    if settings.enable_debug_routes:

        @app.post("/debug/parse", response_model=ParsedCV)
        async def debug_parse(request: Request, file: UploadFile) -> ParsedCV:
            # +1 octet suffit pour détecter un dépassement sans tout charger en mémoire.
            data = await file.read(settings.max_pdf_size_bytes + 1)
            # Empreintes uniquement : aucun contenu du CV dans les logs.
            context = (
                f"pdf_sha256={_fingerprint(data)} pdf_bytes={len(data)} "
                f"host={request.headers.get('host')} "
                f"client={request.client.host if request.client else None}"
            )
            try:
                parsed = await run_in_threadpool(extract_text, data, settings)
            except PDFParseError as error:
                logger.debug("debug/parse error=%s %s", type(error).__name__, context)
                raise HTTPException(status_code=_http_status(error), detail=str(error)) from error
            logger.debug(
                "debug/parse text_sha256=%s chars=%d pages=%d %s",
                _fingerprint(parsed.text.encode()),
                parsed.char_count,
                parsed.page_count,
                context,
            )
            return parsed

        @app.post("/debug/search", response_model=SearchDebugResponse)
        async def debug_search(
            file: UploadFile,
            query: Annotated[str, Form(min_length=1)],
            k: Annotated[int | None, Form(ge=1, le=20)] = None,
        ) -> SearchDebugResponse:
            data = await file.read(settings.max_pdf_size_bytes + 1)
            try:
                embeddings = app_embeddings()
            except MissingAPIKeyError as error:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(error)) from error

            def run() -> SearchDebugResponse:
                parsed = extract_text(data, settings)
                chunks = chunk_cv(parsed.text, settings.chunk_max_chars, settings.chunk_overlap)
                started = time.perf_counter()
                index = CVIndex(chunks, embeddings)
                indexed = time.perf_counter()
                results = index.search(query, k or settings.search_k)
                logger.debug(
                    "debug/search provider=%s chunks=%d index_ms=%.0f search_ms=%.0f",
                    settings.embeddings_provider,
                    len(chunks),
                    (indexed - started) * 1000,
                    (time.perf_counter() - indexed) * 1000,
                )
                return SearchDebugResponse(
                    provider=settings.embeddings_provider,
                    model=embeddings_model(settings),
                    chunks=chunks,
                    results=results,
                )

            try:
                return await run_in_threadpool(run)
            except PDFParseError as error:
                raise HTTPException(status_code=_http_status(error), detail=str(error)) from error
            except Exception as error:
                if settings.embeddings_provider == "local":
                    raise
                # Erreur réseau ou d'API du fournisseur externe ; le message peut contenir
                # des détails de requête, on n'en renvoie que le type.
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    f"Erreur du fournisseur d'embeddings {settings.embeddings_provider} : {type(error).__name__}.",
                ) from error

        @app.post("/debug/requirements", response_model=JobRequirements)
        async def debug_requirements(body: JobOfferRequest) -> JobRequirements:
            try:
                # Taille vérifiée avant de construire le LLM : 413 même sans clé.
                check_job_offer(body.job_offer, settings)
                llm = llm_factory(settings)
                job, _ = await extract_requirements_cached(body.job_offer, llm, settings, limiter, requirements_cache)
                return job
            except JobOfferTooLongError as error:
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(error)) from error
            except MissingAPIKeyError as error:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(error)) from error
            except NoRequirementsError as error:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
            except LLMProviderError as error:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
            except ProviderBlockedError as error:
                provider_block.record(error)
                raise _blocked_http_error(error) from error

        @app.post("/debug/match", response_model=MatchDebugResponse)
        async def debug_match(file: UploadFile, job_offer: Annotated[str, Form(min_length=1)]) -> MatchDebugResponse:
            data = await file.read(settings.max_pdf_size_bytes + 1)
            started = time.perf_counter()
            events: list[EventTiming] = []
            report: MatchReport | None = None
            try:
                prepared = await run_in_threadpool(prepare, data, job_offer)
            except Exception as error:
                if (http_error := _preparation_http_error(error)) is not None:
                    raise http_error from error
                raise
            try:
                async with aclosing(run_analysis(prepared, job_offer)) as stream:
                    async for event in stream:
                        events.append(EventTiming(type=event.type, t_s=round(time.perf_counter() - started, 2)))
                        if isinstance(event, ReportReady):
                            report = event.report
            except NoRequirementsError as error:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
            except LLMProviderError as error:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
            except ProviderBlockedError as error:
                provider_block.record(error)
                raise _blocked_http_error(error) from error
            except AnalysisTimeoutError as error:
                raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, str(error)) from error
            except EmbeddingsError as error:
                if settings.embeddings_provider == "local":
                    raise
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
            assert report is not None
            return MatchDebugResponse(
                report=report, events=events, total_seconds=round(time.perf_counter() - started, 2)
            )

    return app


app = create_app()
