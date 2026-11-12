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
  /** Toutes les citations rejetées : le LLM a été invité une fois à les recopier mot pour mot. */
  second_chance: boolean
  gaps: string[]
  justification: string
  retrieved_chunk_ids: number[]
  points: number | null
  weight: number
  error: string | null
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
  requirements_from_cache: boolean
  chunk_count: number
  timings: MatchTimings
}

export type AnalysisPhase = 'indexing' | 'extraction' | 'evaluation'

export type AnalysisErrorCode =
  | 'llm_provider_error'
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
  from_cache: boolean
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

/** Limites publiques du backend (GET /config). */
export interface PublicConfig {
  max_pdf_size_bytes: number
  max_pdf_pages: number
  max_job_offer_chars: number
}
