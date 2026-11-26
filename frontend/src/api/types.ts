// Types alignés sur les modèles Pydantic de backend/app/schemas.py.
// Le test de contrat (events.test.ts + __fixtures__/events.sample.json, généré par le
// backend) vérifie que chaque événement du flux est reconnu avec ces champs.

export type RequirementCategory =
  | 'compétence technique'
  | 'expérience'
  | 'formation'
  | 'langue'
  | 'savoir-être'
  | 'autre'

export type RequirementImportance = 'obligatoire' | 'souhaitée'

export type EvidenceLevel = 'non démontré' | 'mentionné' | 'démontré' | 'solide'

export interface Requirement {
  id: string
  label: string
  category: RequirementCategory
  importance: RequirementImportance
  min_years: number | null
  search_queries: string[]
}

export interface Evidence {
  chunk_id: number
  quote: string
}

export interface Period {
  start: string
  end: string
}

/** Tokens consommés (usage renvoyé par le fournisseur) ; la sortie inclut le raisonnement. */
export interface TokenUsage {
  calls: number
  input_tokens: number
  output_tokens: number
  reasoning_tokens: number
  total_tokens: number
}

export interface RequirementMatch {
  requirement: Requirement
  status: 'ok' | 'error'
  level: EvidenceLevel | null
  llm_level: EvidenceLevel | null
  evidence: Evidence[]
  rejected_evidence: number
  rejected_evidence_items: Evidence[]
  periods: Period[]
  rejected_periods: number
  total_months: number | null
  downgraded: boolean
  capped_for_duration: boolean
  capped_soft_skill: boolean
  /** Langue : niveau relevé à « démontré » par une certification ou un score officiel cité. */
  certification_floor: boolean
  /** « solide » ramené à « démontré » : preuves dans moins de deux expériences ou projets distincts. */
  capped_single_item: boolean
  /** Toutes les citations rejetées : le LLM a été invité une fois à les recopier mot pour mot. */
  second_chance: boolean
  /** Déclencheur de la seconde chance ; null sans seconde chance. */
  second_chance_reason: 'sans_preuve' | 'durée' | 'solide' | null
  gaps: string[]
  justification: string
  retrieved_chunk_ids: number[]
  points: number | null
  weight: number
  error: string | null
  usage: TokenUsage
}

export interface MatchTimings {
  extraction_s: number
  indexing_s: number
  retrieval_s: number
  evaluation_s: number
}

export interface MatchReport {
  job_title: string
  score: number | null
  requirements: RequirementMatch[]
  missing_mandatory: Requirement[]
  rejected_evidence_count: number
  errors: string[]
  extraction_model: string
  match_model: string
  requirements_source: RequirementsSource
  chunk_count: number
  timings: MatchTimings
  usage: TokenUsage
  extraction_usage: TokenUsage
}

/** Origine des exigences : extraites par le LLM, servies par le cache, ou fournies (évaluation). */
export type RequirementsSource = 'extracted' | 'cache' | 'provided'

export type AnalysisPhase = 'indexing' | 'extraction' | 'evaluation'

export type AnalysisErrorCode =
  | 'llm_provider_error'
  | 'quota_exceeded'
  | 'provider_access_denied'
  | 'no_requirements'
  | 'embeddings_error'
  | 'timeout'
  | 'internal_error'

export interface StartedEvent {
  type: 'started'
  phases: AnalysisPhase[]
  timeout_s: number
}

export interface CVIndexedEvent {
  type: 'cv_indexed'
  chunk_count: number
  seconds: number
}

export interface RequirementsExtractedEvent {
  type: 'requirements_extracted'
  job_title: string
  requirements: Requirement[]
  seconds: number
  requirements_source: RequirementsSource
}

export interface StatusEvent {
  type: 'status'
  reason: 'rate_limit'
  wait_s: number
  message: string
}

export interface RequirementEvaluatedEvent {
  type: 'requirement_evaluated'
  result: RequirementMatch
  done: number
  total: number
}

export interface ReportEvent {
  type: 'report'
  report: MatchReport
}

export interface ErrorEvent {
  type: 'error'
  code: AnalysisErrorCode
  message: string
}

export type AnalysisEvent =
  | StartedEvent
  | CVIndexedEvent
  | RequirementsExtractedEvent
  | StatusEvent
  | RequirementEvaluatedEvent
  | ReportEvent
  | ErrorEvent

export type AnalysisEventType = AnalysisEvent['type']

// --- POST /suggest : événements de l'agent de suggestions (backend/app/suggestions.py, forme
// publique public_json : jamais la paraphrase du modèle, seulement les citations sources). ---

export interface AgentTarget {
  id: string
  label: string
  level: string
  weight: number
  /** Termes de l'offre à nommer. */
  terms: string[]
}

export interface AgentStartedEvent {
  type: 'agent_started'
  targets: AgentTarget[]
  /** Cibles implicites au-delà de la limite d'exigences traitées. */
  skipped: number
  /** Compétences « mentionné » ou « non démontré » : cartes « décrivez-le ». */
  to_describe: AgentTarget[]
  limits: Record<string, number>
}

export interface TargetStartedEvent {
  type: 'target_started'
  requirement: Requirement
  level: string
  rank: number
}

export type AgentStepKind = 'search' | 'propose' | 'verify' | 'reevaluate'

export interface AgentStepEvent {
  type: 'step'
  requirement_id: string
  kind: AgentStepKind
  status: 'ok' | 'rejected'
  attempt: number
  detail: string | null
}

/** Citation source exacte d'un fait de la puce reformulée (à afficher telle quelle). */
export interface SourceQuote {
  chunk_id: number
  quote: string
}

export interface ImpliedSkill {
  skill: string
  chunk_id: number
  quote: string
}

export interface SuggestionEvent {
  type: 'suggestion'
  requirement: Requirement
  /** Expérience ou projet (intitulé). */
  item: string
  chunk_id: number
  bullet_index: number
  before: string
  after: string
  /** Termes de l'offre nommés par la nouvelle puce. */
  terms: string[]
  /** Citations sources exactes, jamais la paraphrase du modèle. */
  facts: SourceQuote[]
  /** Implications de la table réellement utilisées. */
  implied_skills: ImpliedSkill[]
  level_before: string
  level_after: string
  justification: string
  /** Raison donnée par le juge. */
  judge: string
  usage: TokenUsage
}

export type RefusalReason =
  | 'a_decrire'
  | 'deja_explicite'
  | 'aucune_puce'
  | 'aucune_reformulation_fidele'
  | 'baisse_de_niveau'
  | 'reevaluation_impossible'
  | 'non_traitee'
  | 'erreur'

export interface RefusalEvent {
  type: 'refusal'
  requirement: Requirement
  reason: RefusalReason
  message: string
  /** false : compté, pas montré à l'utilisateur (baisse de niveau). */
  shown: boolean
  detail: string | null
  usage: TokenUsage
}

export interface AgentDoneEvent {
  type: 'done'
  suggestions: number
  refusals: number
  to_describe: number
  hidden: number
  untreated: number
  reevaluation_failed: number
  usage: TokenUsage
}

export type AgentErrorCode = 'quota_exceeded' | 'provider_access_denied' | 'timeout' | 'internal_error'

export interface AgentErrorEvent {
  type: 'error'
  code: AgentErrorCode
  message: string
}

export type SuggestEvent =
  | AgentStartedEvent
  | TargetStartedEvent
  | AgentStepEvent
  | SuggestionEvent
  | RefusalEvent
  | StatusEvent
  | AgentDoneEvent
  | AgentErrorEvent

export type SuggestEventType = SuggestEvent['type']

/** Limites publiques du backend (GET /config). */
export interface PublicConfig {
  max_pdf_size_bytes: number
  max_pdf_pages: number
  max_job_offer_chars: number
}
