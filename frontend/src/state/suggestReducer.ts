// État de l'interface de l'agent de suggestions, construit à partir de ses événements
// (POST /suggest). Fonction pure : testable sans navigateur. Les décisions Accepter /
// Refuser restent locales : rien n'est envoyé ni conservé.

import type {
  AgentDoneEvent,
  AgentErrorCode,
  AgentStepEvent,
  AgentStepKind,
  RefusalEvent,
  Requirement,
  SuggestEvent,
  SuggestionEvent,
} from '../api/types'

export type SuggestStatus = 'idle' | 'running' | 'done' | 'error' | 'cancelled'
export type StepState = 'pending' | 'active' | 'done' | 'rejected'
export type Decision = 'accepted' | 'refused'

export const STEP_ORDER: AgentStepKind[] = ['search', 'propose', 'verify', 'reevaluate']

/** Une exigence ciblée par l'agent et la frise de ses étapes. */
export interface TargetPanel {
  id: string
  label: string
  level: string
  terms: string[]
  requirement: Requirement | null
  steps: Record<AgentStepKind, StepState>
  /** Dernière proposition écartée par le contrôle (« Proposition écartée : … »), ou null. */
  rejection: string | null
  /** Propositions écartées pour cette exigence (prudence rendue visible). */
  rejections: number
  /** Étape qui a écarté la dernière proposition (vérification ou réévaluation), ou null. */
  lastRejected: AgentStepKind | null
  outcome: 'waiting' | 'running' | 'suggestion' | 'refusal' | 'hidden'
}

export interface SuggestFailure {
  kind: 'http' | 'stream' | 'connection' | 'protocol' | 'unknown'
  message: string
  status?: number
  code?: AgentErrorCode
}

export interface SuggestState {
  status: SuggestStatus
  targets: TargetPanel[]
  /** Cibles au-delà de la limite d'exigences traitées. */
  skipped: number
  suggestions: SuggestionEvent[]
  /** Refus montrés à l'utilisateur (cartes « décrivez-le » à part). */
  refusals: RefusalEvent[]
  /** Cartes « décrivez-le » : compétences que le CV ne prouve pas encore. */
  toDescribe: RefusalEvent[]
  /** Accepter / Refuser, par exigence : état local seulement. */
  decisions: Record<string, Decision>
  /** Dernier message d'attente (quota du fournisseur) ; effacé au prochain événement. */
  waitMessage: string | null
  done: AgentDoneEvent | null
  error: SuggestFailure | null
}

export type SuggestAction =
  | { type: 'submit' }
  | { type: 'event'; event: SuggestEvent }
  | { type: 'failed'; error: SuggestFailure }
  | { type: 'cancelled' }
  | { type: 'decide'; requirementId: string; decision: Decision | null }
  | { type: 'reset' }

export const initialSuggestState: SuggestState = {
  status: 'idle',
  targets: [],
  skipped: 0,
  suggestions: [],
  refusals: [],
  toDescribe: [],
  decisions: {},
  waitMessage: null,
  done: null,
  error: null,
}

function pendingSteps(): Record<AgentStepKind, StepState> {
  return { search: 'pending', propose: 'pending', verify: 'pending', reevaluate: 'pending' }
}

/** Raison lisible d'une proposition écartée (détail d'une étape « rejected » du backend). */
export function rejectionReason(detail: string | null): string {
  // Flèches du backend (« solide -> démontré ») affichées en typographie française.
  const text = (detail ?? '').trim().replaceAll('->', '→')
  if (!text) return 'motif non précisé'
  if (text.startsWith('Contrôle d’ancrage : ') || text.startsWith("Contrôle d'ancrage : ")) {
    return `contrôle du CV — ${text.slice('Contrôle d’ancrage : '.length)}`
  }
  if (text.startsWith('Juge : ')) return `relecture — ${text.slice('Juge : '.length)}`
  if (text.startsWith('Baisse de niveau')) {
    // « Baisse de niveau (démontré -> mentionné). Consigne pour la tentative suivante… »
    const first = text.split('. ')[0] ?? text
    return first.endsWith('.') ? first : `${first}.`
  }
  return text
}

function updateTarget(state: SuggestState, id: string, update: (panel: TargetPanel) => TargetPanel): SuggestState {
  return { ...state, targets: state.targets.map((panel) => (panel.id === id ? update(panel) : panel)) }
}

function applyStep(panel: TargetPanel, event: AgentStepEvent): TargetPanel {
  const steps = { ...panel.steps }
  const index = STEP_ORDER.indexOf(event.kind)
  if (event.status === 'ok') {
    steps[event.kind] = 'done'
    const next = STEP_ORDER[index + 1]
    if (next !== undefined) steps[next] = 'active'
    // Une nouvelle proposition passe la vérification : l'écart précédent n'est plus d'actualité.
    const rejection = event.kind === 'verify' || event.kind === 'reevaluate' ? null : panel.rejection
    return { ...panel, steps, rejection }
  }
  if (event.kind === 'search') {
    steps.search = 'rejected'
    return { ...panel, steps }
  }
  // Vérification ou réévaluation refusée : la proposition est écartée, une nouvelle
  // tentative repasse par la proposition (si le backend en fait une).
  steps.propose = 'active'
  steps.verify = 'pending'
  steps.reevaluate = 'pending'
  return {
    ...panel,
    steps,
    rejection: `Proposition écartée : ${rejectionReason(event.detail)}`,
    rejections: panel.rejections + 1,
    lastRejected: event.kind,
  }
}

function finishSteps(panel: TargetPanel): Record<AgentStepKind, StepState> {
  // Fin d'une cible sans suggestion, après une proposition écartée : l'étape qui l'a écartée
  // passe en rouge (Vérification, ou Réévaluation pour une baisse de niveau), celles d'avant
  // sont terminées.
  if (panel.lastRejected === 'verify' || panel.lastRejected === 'reevaluate') {
    const rejectedAt = STEP_ORDER.indexOf(panel.lastRejected)
    const finished = { ...panel.steps }
    STEP_ORDER.forEach((kind, index) => {
      finished[kind] = index < rejectedAt ? 'done' : index === rejectedAt ? 'rejected' : 'pending'
    })
    return finished
  }
  // Sinon, rien ne reste « en cours ».
  const finished = { ...panel.steps }
  for (const kind of STEP_ORDER) if (finished[kind] === 'active') finished[kind] = 'pending'
  return finished
}

function applyEvent(state: SuggestState, event: SuggestEvent): SuggestState {
  const base = event.type === 'status' ? state : { ...state, waitMessage: null }
  switch (event.type) {
    case 'agent_started':
      return {
        ...base,
        skipped: event.skipped,
        targets: event.targets.map((target) => ({
          id: target.id,
          label: target.label,
          level: target.level,
          terms: target.terms,
          requirement: null,
          steps: pendingSteps(),
          rejection: null,
          rejections: 0,
          lastRejected: null,
          outcome: 'waiting',
        })),
      }
    case 'target_started':
      return updateTarget(base, event.requirement.id, (panel) => ({
        ...panel,
        requirement: event.requirement,
        steps: { ...pendingSteps(), search: 'active' },
        outcome: 'running',
      }))
    case 'step':
      return updateTarget(base, event.requirement_id, (panel) => applyStep(panel, event))
    case 'suggestion':
      return {
        ...updateTarget(base, event.requirement.id, (panel) => ({
          ...panel,
          requirement: event.requirement,
          steps: { search: 'done', propose: 'done', verify: 'done', reevaluate: 'done' },
          rejection: null,
          outcome: 'suggestion',
        })),
        suggestions: [...base.suggestions, event],
      }
    case 'refusal': {
      if (event.reason === 'a_decrire') return { ...base, toDescribe: [...base.toDescribe, event] }
      const outcome = event.shown ? 'refusal' : 'hidden'
      const updated = updateTarget(base, event.requirement.id, (panel) => ({
        ...panel,
        requirement: event.requirement,
        steps: finishSteps(panel),
        outcome,
      }))
      return event.shown ? { ...updated, refusals: [...updated.refusals, event] } : updated
    }
    case 'status':
      return { ...base, waitMessage: event.message }
    case 'done':
      return { ...base, status: 'done', done: event }
    case 'error':
      return { ...base, status: 'error', error: { kind: 'stream', code: event.code, message: event.message } }
  }
}

export function suggestReducer(state: SuggestState, action: SuggestAction): SuggestState {
  switch (action.type) {
    case 'submit':
      return { ...initialSuggestState, status: 'running' }
    case 'event':
      // Événements arrivés après la fin (annulation, erreur) : ignorés.
      return state.status === 'running' ? applyEvent(state, action.event) : state
    case 'failed':
      return state.status === 'running' ? { ...state, status: 'error', waitMessage: null, error: action.error } : state
    case 'cancelled':
      // Les suggestions déjà reçues restent affichées (et décidables).
      return state.status === 'running' ? { ...state, status: 'cancelled', waitMessage: null } : state
    case 'decide': {
      if (!state.suggestions.some((s) => s.requirement.id === action.requirementId)) return state
      const decisions = { ...state.decisions }
      if (action.decision === null) delete decisions[action.requirementId]
      else decisions[action.requirementId] = action.decision
      return { ...state, decisions }
    }
    case 'reset':
      return initialSuggestState
  }
}

/** Puces acceptées, une par ligne, prêtes à coller dans le CV (ordre des suggestions). */
export function acceptedBullets(state: SuggestState): string {
  return state.suggestions
    .filter((s) => state.decisions[s.requirement.id] === 'accepted')
    .map((s) => s.after)
    .join('\n')
}
