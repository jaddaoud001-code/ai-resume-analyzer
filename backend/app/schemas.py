"""Modèles Pydantic partagés."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

RequirementCategory = Literal["compétence technique", "expérience", "formation", "langue", "savoir-être", "autre"]
RequirementImportance = Literal["obligatoire", "souhaitée"]


class Requirement(BaseModel):
    id: str = Field(pattern=r"^R\d+$", description="Identifiant stable : R1, R2…")
    label: str = Field(min_length=1, max_length=120, description="Formulation courte d'un seul critère.")
    category: RequirementCategory
    importance: RequirementImportance
    min_years: int | None = Field(default=None, ge=0, le=50, description="Durée exigée par l'offre, en années.")
    search_queries: list[str] = Field(
        min_length=2, max_length=3, description="Reformulations pour la recherche dans le CV."
    )


class JobRequirements(BaseModel):
    job_title: str
    requirements: list[Requirement]


class JobOfferRequest(BaseModel):
    job_offer: str = Field(min_length=1)


# --- Matching CV / exigences --------------------------------------------------

EvidenceLevel = Literal["non démontré", "mentionné", "démontré", "solide"]


class Evidence(BaseModel):
    chunk_id: int = Field(description="Identifiant du chunk du CV cité.")
    quote: str = Field(description="Extrait exact du chunk (morceaux séparés par « … » autorisés).")


class Period(BaseModel):
    start: str = Field(description="Début, au format AAAA-MM.")
    end: str = Field(description="Fin, au format AAAA-MM, ou « en cours ».")


class RequirementMatch(BaseModel):
    requirement: Requirement
    status: Literal["ok", "error"]
    # Niveau final, après les vérifications du code ; None si l'évaluation a échoué.
    level: EvidenceLevel | None = None
    # Niveau proposé par le LLM, avant vérifications.
    llm_level: EvidenceLevel | None = None
    evidence: list[Evidence] = []
    rejected_evidence: int = 0
    # Citations rejetées (absentes du chunk cité), pour comprendre une rétrogradation.
    rejected_evidence_items: list[Evidence] = []
    periods: list[Period] = []
    rejected_periods: int = 0
    # Durée cumulée des périodes retenues (chevauchements fusionnés), si min_years.
    total_months: int | None = None
    # Niveau ramené à « non démontré » faute de preuve valide.
    downgraded: bool = False
    # Niveau plafonné à « mentionné » : durée inférieure à min_years.
    capped_for_duration: bool = False
    # Savoir-être plafonné à « démontré » : un CV ne peut pas le prouver solidement.
    capped_soft_skill: bool = False
    # Toutes les citations rejetées : un second appel au LLM a demandé de les recopier mot
    # pour mot (ou de baisser le niveau) ; le résultat est celui de ce second appel.
    second_chance: bool = False
    gaps: list[str] = []
    justification: str = ""
    retrieved_chunk_ids: list[int] = []
    points: int | None = None
    weight: float = 0.0
    error: str | None = None


class MatchTimings(BaseModel):
    extraction_s: float = 0.0
    indexing_s: float = 0.0
    retrieval_s: float = 0.0
    evaluation_s: float = 0.0


# Origine des exigences : extraites par le LLM, servies par le cache (même offre, même
# modèle, même version du prompt) ou fournies par l'appelant (évaluation, liste figée).
RequirementsSource = Literal["extracted", "cache", "provided"]


class MatchReport(BaseModel):
    job_title: str
    score: int | None = Field(description="Moyenne pondérée 0-100 ; None si aucune exigence évaluée.")
    requirements: list[RequirementMatch]
    # Exigences obligatoires « non démontrées » : l'information la plus utile au candidat.
    missing_mandatory: list[Requirement]
    rejected_evidence_count: int
    errors: list[str] = Field(description="Identifiants des exigences dont l'évaluation a échoué.")
    extraction_model: str
    match_model: str
    requirements_source: RequirementsSource = "extracted"
    chunk_count: int
    timings: MatchTimings


class RequirementsExtracted(BaseModel):
    type: Literal["requirements_extracted"] = "requirements_extracted"
    job_title: str
    requirements: list[Requirement]
    seconds: float
    requirements_source: RequirementsSource = "extracted"


class CVIndexed(BaseModel):
    type: Literal["cv_indexed"] = "cv_indexed"
    chunk_count: int
    seconds: float


class RequirementEvaluated(BaseModel):
    type: Literal["requirement_evaluated"] = "requirement_evaluated"
    result: RequirementMatch
    done: int
    total: int


class ReportReady(BaseModel):
    type: Literal["report"] = "report"
    report: MatchReport


AnalysisPhase = Literal["indexing", "extraction", "evaluation"]


class AnalysisStarted(BaseModel):
    """Premier événement, dès l'ouverture du flux : les phases prévues, dans l'ordre."""

    type: Literal["started"] = "started"
    phases: list[AnalysisPhase] = ["indexing", "extraction", "evaluation"]
    timeout_s: float


class AnalysisStatus(BaseModel):
    """Attente imposée par le fournisseur (429) : l'analyse continue après `wait_s` secondes."""

    type: Literal["status"] = "status"
    reason: Literal["rate_limit"] = "rate_limit"
    wait_s: float
    message: str


MatchEvent = Annotated[
    AnalysisStarted | RequirementsExtracted | CVIndexed | RequirementEvaluated | AnalysisStatus | ReportReady,
    Field(discriminator="type"),
]

AnalysisErrorCode = Literal[
    "llm_provider_error",
    "quota_exceeded",
    "provider_access_denied",
    "no_requirements",
    "embeddings_error",
    "timeout",
    "internal_error",
]


class AnalysisError(BaseModel):
    """Erreur fatale pendant le flux : dernier événement, puis le flux se ferme."""

    type: Literal["error"] = "error"
    code: AnalysisErrorCode
    message: str


class EventTiming(BaseModel):
    type: str
    t_s: float


class MatchDebugResponse(BaseModel):
    report: MatchReport
    events: list[EventTiming]
    total_seconds: float


class ParsedCV(BaseModel):
    text: str
    page_count: int
    char_count: int


class Chunk(BaseModel):
    section: str
    index: int
    text: str


class SearchResult(BaseModel):
    chunk: Chunk
    score: float = Field(description="Similarité cosinus dans [-1, 1] : plus élevée = plus pertinent.")


class SearchDebugResponse(BaseModel):
    provider: str
    model: str
    chunks: list[Chunk]
    results: list[SearchResult]
