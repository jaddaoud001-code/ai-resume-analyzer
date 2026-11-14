"""Appels au LLM : limiteur de débit partagé, nouvelles tentatives, sortie structurée.

Tous les appels au LLM d'un processus (extraction et évaluation) passent par le même
`RateLimiter` : les offres gratuites limitent le compte entier (requêtes et tokens par
minute, quota journalier).
"""

import asyncio
import email.utils
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)

_MAX_RETRY_AFTER_S = 60.0
# Fenêtre des limites « par minute » ; attendre sa durée entière garantit qu'elle est
# écoulée, qu'elle soit fixe ou glissante.
_TOKEN_WINDOW_S = 60.0
# Quota journalier épuisé : aucun en-tête chez Groq (seulement les limites par minute et le
# nombre de requêtes du jour) ; le message du 429 nomme la limite atteinte, par exemple
# « … on tokens per day (TPD): Limit 200000 … ».
_DAILY_LIMIT = re.compile(r"per day|\((?:TPD|RPD)\)|daily", re.IGNORECASE)
# 429 sur les tokens par minute (Groq : « … on tokens per minute (TPM) … »).
_MINUTE_TOKENS_LIMIT = re.compile(r"tokens per minute|\(TPM\)", re.IGNORECASE)
# Durée Groq des en-têtes x-ratelimit-reset-* : « 7.66s », « 1m26.4s », « 2h3m4s », « 120ms ».
_GO_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?$")


class ProviderBlockedError(RuntimeError):
    """Le fournisseur refuse durablement les appels : réessayer ne servirait à rien.
    L'analyse s'arrête (erreur fatale) au lieu de marquer les exigences en erreur."""


class ProviderQuotaExceededError(ProviderBlockedError):
    """Quota journalier du fournisseur épuisé (429 « per day »)."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        hint = ""
        if retry_after is not None and retry_after > 0:
            hint = f" (le fournisseur indique une reprise possible dans environ {_human_duration(retry_after)})"
        super().__init__(f"Quota journalier du fournisseur LLM atteint : réessayez demain{hint}.")


class ProviderAccessDeniedError(ProviderBlockedError):
    """Clé refusée (401) ou modèle non autorisé par l'abonnement (403)."""

    def __init__(self, status: int, model: str | None = None) -> None:
        self.status = status
        if status == 401:
            message = "Le fournisseur LLM refuse la clé API (401) : vérifiez la clé du fournisseur choisi dans .env."
        else:
            target = f"au modèle {model}" if model else "au modèle demandé"
            message = (
                f"Le fournisseur LLM refuse l'accès {target} ({status} : abonnement ou modèle non autorisé). "
                "Vérifiez LLM_PROVIDER, LLM_MODEL et MATCH_LLM_MODEL dans .env."
            )
        super().__init__(message)


def _human_duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return f"{max(1, round(seconds))} s"
    if minutes < 90:
        return f"{minutes} min"
    return f"{minutes / 60:.0f} h"


class AnalysisMonitor:
    """Suivi des appels au LLM d'une analyse, indépendant de l'API web.

    L'analyse le place dans `current_monitor` (contexte de ses tâches) : le limiteur compte
    alors ses appels en attente et en cours, et `on_wait(secondes)` est appelé chaque fois
    qu'un 429 impose une attente à l'un de ses appels, y compris quand le limiteur partagé
    se met en pause à cause du 429 d'une autre analyse."""

    def __init__(self, on_wait: Callable[[float], None] | None = None) -> None:
        self.on_wait = on_wait
        self.waiting = 0  # appels qui attendent leur départ dans le limiteur
        self.in_flight = 0  # appels envoyés au fournisseur, sans réponse encore
        self.started = 0  # appels envoyés au total (nouvelles tentatives comprises)

    def notify_wait(self, seconds: float) -> None:
        if self.on_wait is not None:
            self.on_wait(seconds)


current_monitor: ContextVar[AnalysisMonitor | None] = ContextVar("current_monitor", default=None)


class RateLimiter:
    """Départs espacés d'au moins 1 / rps seconde et au plus `max_concurrency` appels en
    cours. Horloge et attente injectables (tests déterministes)."""

    def __init__(
        self,
        rps: float,
        max_concurrency: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.base_interval = 1.0 / rps if rps > 0 else 0.0
        self.interval = self.base_interval
        self.max_concurrency = max_concurrency
        self._clock = clock
        self.sleep = sleep
        # Analyses dont des appels attendent leur départ : prévenues des pauses (429).
        self._waiting: dict[AnalysisMonitor, int] = {}
        self.active_analyses = 0
        self._next_start = float("-inf")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "RateLimiter":
        return cls(settings.llm_rps, settings.llm_concurrency)

    def _bind_to_running_loop(self) -> None:
        # Les primitives asyncio sont liées à une boucle : on les recrée si elle change
        # (un TestClient par test, par exemple).
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        self._bind_to_running_loop()
        monitor = current_monitor.get()
        self._enter_waiting(monitor)
        waiting = True
        try:
            async with self._semaphore:
                async with self._lock:
                    wait = self._next_start - self._clock()
                    if wait > 0:
                        await self.sleep(wait)
                    self._next_start = self._clock() + self.interval
                self._leave_waiting(monitor)
                waiting = False
                if monitor is not None:
                    monitor.in_flight += 1
                    monitor.started += 1
                try:
                    yield
                finally:
                    if monitor is not None:
                        monitor.in_flight -= 1
        finally:
            if waiting:  # annulé avant le départ : aucun appel envoyé
                self._leave_waiting(monitor)

    def _enter_waiting(self, monitor: AnalysisMonitor | None) -> None:
        if monitor is not None:
            monitor.waiting += 1
            self._waiting[monitor] = self._waiting.get(monitor, 0) + 1

    def _leave_waiting(self, monitor: AnalysisMonitor | None) -> None:
        if monitor is not None:
            monitor.waiting -= 1
            remaining = self._waiting[monitor] - 1
            if remaining:
                self._waiting[monitor] = remaining
            else:
                del self._waiting[monitor]

    def waiting_monitors(self) -> set[AnalysisMonitor]:
        """Analyses qui ont au moins un appel en attente de départ."""
        return set(self._waiting)

    def slow_down(self, interval: float) -> None:
        """Le fournisseur impose un débit plus faible : au moins `interval` secondes entre
        deux départs à partir de maintenant, jusqu'à `restore()`."""
        if interval > self.interval:
            logger.warning("Limiteur LLM ralenti : %.1f s entre deux requêtes", interval)
            self.interval = interval
        self._next_start = max(self._next_start, self._clock() + interval)

    def hold(self, seconds: float) -> None:
        """Aucun départ pendant `seconds` secondes (quota épuisé pour la fenêtre en cours),
        sans changer le débit ensuite."""
        self._next_start = max(self._next_start, self._clock() + seconds)

    def restore(self) -> None:
        """Revient au débit configuré (LLM_RPS)."""
        self.interval = self.base_interval

    def begin_analysis(self) -> None:
        self.active_analyses += 1

    def end_analysis(self) -> None:
        """Fin d'une analyse : le débit normal n'est rétabli que lorsque plus aucune analyse
        n'est active, car un ralentissement imposé par le fournisseur vaut pour le compte
        entier, donc aussi pour les analyses encore en cours."""
        self.active_analyses = max(0, self.active_analyses - 1)
        if self.active_analyses == 0:
            self.restore()


async def call_with_retry(
    call: Callable[[], Awaitable[T]],
    limiter: RateLimiter,
    *,
    rate_limit_retries: int,
    backoff_s: float,
    other_retries: int = 1,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    model: str | None = None,
) -> T:
    """Chaque tentative passe par le limiteur. Sur 429 : jusqu'à `rate_limit_retries`
    nouvelles tentatives, après une attente de :
    - si le 429 porte sur les tokens de la minute (x-ratelimit-remaining-tokens-minute chez
      Mistral ou x-ratelimit-remaining-tokens chez Groq à 0, ou message « tokens per
      minute ») : Retry-After, sinon x-ratelimit-reset-tokens (Groq), sinon 60 s (fenêtre
      entière, quand le fournisseur n'indique pas sa fin) ; pendant cette attente, le
      limiteur ne lance aucun appel (ils échoueraient aussi) ;
    - sinon Retry-After, s'il est fourni ;
    - sinon 60 / x-ratelimit-limit-req-minute secondes, et le limiteur est ralenti à ce
      débit (jusqu'à `RateLimiter.restore()`) ;
    - sinon backoff_s, 2×, 4×…
    Jamais de nouvelle tentative (ProviderBlockedError) : quota journalier épuisé (429 dont
    le message nomme une limite « per day ») ; clé refusée (401) ; modèle ou abonnement
    refusé (403, `model` nommé dans le message).
    Sur toute autre erreur : `other_retries` nouvelle(s) tentative(s).
    Les attentes passent par `sleep`, à défaut par celle du limiteur (horloge simulée en test)."""
    rate_limited = other = 0
    while True:
        try:
            async with limiter.slot():
                return await call()
        except Exception as error:
            status, retry_after = rate_limit_info(error)
            if status in (401, 403):
                logger.warning("%d : accès refusé par le fournisseur LLM (modèle %s)", status, model)
                raise ProviderAccessDeniedError(status, model) from error
            if status == 429:
                limits = rate_limit_headers(error)
                if _DAILY_LIMIT.search(_error_text(error)):
                    logger.warning("429 : quota journalier du fournisseur épuisé ; limites %s", limits)
                    raise ProviderQuotaExceededError(retry_after) from error
                if rate_limited >= rate_limit_retries:
                    logger.warning("429 : abandon après %d nouvelles tentatives ; limites %s", rate_limited, limits)
                    raise
                per_minute = _requests_per_minute(limits)
                pauses_limiter = False
                minute_tokens = _tokens_exhausted(limits) or bool(_MINUTE_TOKENS_LIMIT.search(_error_text(error)))
                if minute_tokens:
                    # Quota de tokens de la minute : aucun appel ne passerait avant sa reprise.
                    delay = retry_after if retry_after is not None else _reset_tokens(limits) or _TOKEN_WINDOW_S
                    limiter.hold(min(delay, _MAX_RETRY_AFTER_S))
                    pauses_limiter = True
                elif retry_after is not None:
                    delay = retry_after
                elif per_minute:
                    delay = 60.0 / per_minute
                    limiter.slow_down(delay)
                    pauses_limiter = True
                else:
                    delay = backoff_s * 2**rate_limited
                delay = min(delay, _MAX_RETRY_AFTER_S)
                rate_limited += 1
                logger.warning(
                    "429 : nouvelle tentative %d/%d dans %.1f s ; limites %s",
                    rate_limited,
                    rate_limit_retries,
                    delay,
                    limits,
                )
                # L'analyse de cet appel attend ; si le limiteur partagé se met en pause,
                # toutes celles qui ont des appels en attente attendent aussi.
                to_notify = limiter.waiting_monitors() if pauses_limiter else set()
                if (own := current_monitor.get()) is not None:
                    to_notify.add(own)
                for monitor in to_notify:
                    monitor.notify_wait(delay)
                await (sleep or limiter.sleep)(delay)
            elif other < other_retries:
                other += 1
            else:
                raise


def _http_response(error: BaseException) -> tuple[int | None, Any]:
    """(code HTTP, réponse) trouvés dans la chaîne d'exceptions : httpx.HTTPStatusError
    (Mistral) ou openai.APIStatusError exposent `.response`."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None) or getattr(current, "status_code", None)
        if isinstance(status, int):
            return status, response
        current = current.__cause__ or current.__context__
    return None, None


def rate_limit_info(error: BaseException) -> tuple[int | None, float | None]:
    """(code HTTP, Retry-After en secondes) de l'erreur, s'ils existent."""
    status, response = _http_response(error)
    headers = getattr(response, "headers", None) or {}
    return status, _parse_retry_after(headers.get("retry-after"))


def _header_int(limits: dict[str, str], header: str) -> int | None:
    for name, value in limits.items():
        if name.lower() == header:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _requests_per_minute(limits: dict[str, str]) -> int | None:
    value = _header_int(limits, "x-ratelimit-limit-req-minute")
    return value if value and value > 0 else None


def _tokens_exhausted(limits: dict[str, str]) -> bool:
    """Tokens de la minute épuisés : en-tête Mistral, ou Groq (x-ratelimit-*-tokens y est par minute)."""
    return 0 in (
        _header_int(limits, "x-ratelimit-remaining-tokens-minute"),
        _header_int(limits, "x-ratelimit-remaining-tokens"),
    )


def _reset_tokens(limits: dict[str, str]) -> float | None:
    """x-ratelimit-reset-tokens (Groq) : délai avant la reprise du quota de tokens."""
    for name, value in limits.items():
        if name.lower() == "x-ratelimit-reset-tokens":
            return parse_go_duration(value)
    return None


def parse_go_duration(value: str) -> float | None:
    """« 7.66s », « 1m26.4s », « 2h3m4s », « 120ms » -> secondes ; None si illisible."""
    match = _GO_DURATION.match(value.strip())
    if not value.strip() or match is None or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0) + float(millis or 0) / 1000


def _error_text(error: BaseException) -> str:
    """Messages de la chaîne d'exceptions (et corps des erreurs d'API) : pour reconnaître la
    limite atteinte. Jamais journalisé (le message du fournisseur peut nommer le compte)."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        body = getattr(current, "body", None)
        if body is not None:
            parts.append(str(body))
        response = getattr(current, "response", None)
        text = getattr(response, "text", None)
        if isinstance(text, str):
            parts.append(text)
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


def rate_limit_headers(error: BaseException) -> dict[str, str]:
    """En-têtes x-ratelimit-* et retry-after (jamais d'en-tête d'authentification)."""
    headers = getattr(_http_response(error)[1], "headers", None) or {}
    return {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}


def describe_error(error: BaseException) -> str:
    """Type de l'erreur et code HTTP éventuel, sans le message du fournisseur."""
    status = _http_response(error)[0]
    return f"{type(error).__name__} {status}" if status else type(error).__name__


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:  # forme date HTTP : "Wed, 21 Oct 2026 07:28:00 GMT"
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, parsed.timestamp() - time.time())


async def invoke_structured(
    llm: BaseChatModel,
    schema: type[ModelT],
    messages: list[BaseMessage],
    limiter: RateLimiter,
    settings: Settings,
) -> ModelT:
    """Sortie structurée (function calling) validée par `schema`, avec la politique de
    nouvelles tentatives ci-dessus. Une réponse sans sortie structurée compte comme invalide."""

    def require_parsed(output: Any) -> ModelT:
        if not isinstance(output, schema):
            raise OutputParserException(f"Sortie structurée absente ou invalide : {type(output).__name__}")
        return output

    chain = llm.with_structured_output(schema, method="function_calling") | RunnableLambda(require_parsed)
    return await call_with_retry(
        lambda: chain.ainvoke(messages),
        limiter,
        rate_limit_retries=settings.llm_rate_limit_retries,
        backoff_s=settings.llm_rate_limit_backoff_s,
        model=_model_name(llm),
    )


def _model_name(llm: BaseChatModel) -> str | None:
    for attribute in ("model_name", "model"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str):
            return value
    return None
