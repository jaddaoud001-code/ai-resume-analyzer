// État de l'interface construit à partir de la suite d'événements de l'analyse.
// Fonction pure : testable sans navigateur.

import type {
  AnalysisErrorCode,
  AnalysisEvent,
  AnalysisPhase,
  MatchReport,
  Requirement,
  RequirementMatch,
} from '../api/types'

export type AnalysisStatus = 'idle' | 'running' | 'done' | 'error' | 'cancelled'
export type PhaseState = 'pending' | 'active' | 'done'

export interface RequirementEntry {
  requirement: Requirement
  /** null tant que l'évaluation n'est pas arrivée. */
  result: RequirementMatch | null
}

export interface AnalysisFailure {
  /** http : refus avant le flux ; stream : événement « error » ; connection : coupure ;
   * protocol : réponse inattendue ; unknown : autre. */
  kind: 'http' | 'stream' | 'connection' | 'protocol' | 'unknown'
  message: string
  status?: number
  code?: AnalysisErrorCode
}

export interface AnalysisState {
  status: AnalysisStatus
  phases: Record<AnalysisPhase, PhaseState>
  jobTitle: string | null
  fromCache: boolean
  requirements: RequirementEntry[]
  evaluated: number
  total: number | null
  /** Dernier message d'attente (quota du fournisseur) ; effacé au prochain événement. */
  waitMessage: string | null
  report: MatchReport | null
  error: AnalysisFailure | null
}

export type AnalysisAction =
  | { type: 'submit' }
  | { type: 'event'; event: AnalysisEvent }
  | { type: 'failed'; error: AnalysisFailure }
  | { type: 'cancelled' }
  | { type: 'reset' }

const PENDING_PHASES: Record<AnalysisPhase, PhaseState> = {
  indexing: 'pending',
  extraction: 'pending',
  evaluation: 'pending',
}

export const initialAnalysisState: AnalysisState = {
  status: 'idle',
  phases: PENDING_PHASES,
  jobTitle: null,
  fromCache: false,
  requirements: [],
  evaluated: 0,
  total: null,
  waitMessage: null,
  report: null,
  error: null,
}

/** Obligatoires d'abord ; ordre du backend conservé dans chaque groupe. */
export function mandatoryFirst<T extends { requirement: Requirement }>(entries: T[]): T[] {
  const rank = (entry: T): number => (entry.requirement.importance === 'obligatoire' ? 0 : 1)
  return [...entries].sort((a, b) => rank(a) - rank(b))
}

function withEvaluationPhase(phases: Record<AnalysisPhase, PhaseState>): Record<AnalysisPhase, PhaseState> {
  // L'évaluation commence quand l'indexation et l'extraction sont toutes deux terminées.
  const ready = phases.indexing === 'done' && phases.extraction === 'done'
  return ready && phases.evaluation === 'pending' ? { ...phases, evaluation: 'active' } : phases
}

function applyEvent(state: AnalysisState, event: AnalysisEvent): AnalysisState {
  // Tout événement autre que « status » signifie que l'analyse a repris.
  const base = event.type === 'status' ? state : { ...state, waitMessage: null }
  switch (event.type) {
    case 'started':
      return { ...base, phases: { indexing: 'active', extraction: 'active', evaluation: 'pending' } }
    case 'cv_indexed':
      return { ...base, phases: withEvaluationPhase({ ...base.phases, indexing: 'done' }) }
    case 'requirements_extracted':
      return {
        ...base,
        phases: withEvaluationPhase({ ...base.phases, extraction: 'done' }),
        jobTitle: event.job_title,
        fromCache: event.from_cache,
        requirements: mandatoryFirst(event.requirements.map((requirement) => ({ requirement, result: null }))),
        total: event.requirements.length,
      }
    case 'status':
      return { ...base, waitMessage: event.message }
    case 'requirement_evaluated': {
      const id = event.result.requirement.id
      const known = base.requirements.some((entry) => entry.requirement.id === id)
      const requirements = known
        ? base.requirements.map((entry) => (entry.requirement.id === id ? { ...entry, result: event.result } : entry))
        : mandatoryFirst([...base.requirements, { requirement: event.result.requirement, result: event.result }])
      return { ...base, requirements, evaluated: event.done, total: event.total }
    }
    case 'report':
      return {
        ...base,
        status: 'done',
        phases: { indexing: 'done', extraction: 'done', evaluation: 'done' },
        jobTitle: event.report.job_title,
        fromCache: event.report.requirements_from_cache,
        requirements: mandatoryFirst(
          event.report.requirements.map((result) => ({ requirement: result.requirement, result })),
        ),
        evaluated: event.report.requirements.length,
        total: event.report.requirements.length,
        report: event.report,
      }
    case 'error':
      return {
        ...base,
        status: 'error',
        error: { kind: 'stream', code: event.code, message: event.message },
      }
  }
}

export function analysisReducer(state: AnalysisState, action: AnalysisAction): AnalysisState {
  switch (action.type) {
    case 'submit':
      return { ...initialAnalysisState, status: 'running' }
    case 'event':
      // Événements arrivés après la fin (annulation, erreur) : ignorés.
      return state.status === 'running' ? applyEvent(state, action.event) : state
    case 'failed':
      return state.status === 'running'
        ? { ...state, status: 'error', waitMessage: null, error: action.error }
        : state
    case 'cancelled':
      // Les exigences déjà évaluées restent affichées.
      return state.status === 'running' ? { ...state, status: 'cancelled', waitMessage: null } : state
    case 'reset':
      return initialAnalysisState
  }
}
