// Flux simulés de l'agent de suggestions (POST /suggest), construits à partir des exemples
// du contrat (générés par le backend) : aucun appel réseau.

import suggestSamples from '../api/__fixtures__/suggest_events.sample.json'
import type {
  AgentStepEvent,
  AgentStepKind,
  RefusalEvent,
  RefusalReason,
  Requirement,
  SuggestEvent,
  SuggestEventType,
  SuggestionEvent,
} from '../api/types'

export const SUGGEST_SAMPLES = suggestSamples as unknown as SuggestEvent[]

export function suggestSample<T extends SuggestEventType>(type: T): Extract<SuggestEvent, { type: T }> {
  const event = SUGGEST_SAMPLES.find((e) => e.type === type)
  if (!event) throw new Error(`exemple absent : ${type}`)
  return structuredClone(event) as Extract<SuggestEvent, { type: T }>
}

export function requirement(id: string, label: string, category: Requirement['category'] = 'compétence technique'): Requirement {
  return { id, label, category, importance: 'souhaitée', min_years: null, search_queries: [label, label] }
}

export const CI_CD = requirement('R5', 'CI/CD')
export const REST = requirement('R7', 'API REST')
export const GO = requirement('R2', 'Go')
export const ENGLISH = requirement('R3', 'Anglais professionnel', 'langue')

export const BEFORE = 'Pipeline de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.'
export const AFTER = 'Pipeline CI/CD de tests et de déploiement lancé à chaque fusion sur la branche principale, sur GitLab.'

export function started(targets: Requirement[], toDescribe: Requirement[] = [], skipped = 0): SuggestEvent {
  const event = suggestSample('agent_started')
  const target = (r: Requirement) => ({ id: r.id, label: r.label, level: 'démontré', weight: 1, terms: [r.label] })
  return { ...event, targets: targets.map(target), to_describe: toDescribe.map(target), skipped }
}

export function targetStarted(r: Requirement, rank = 1): SuggestEvent {
  return { ...suggestSample('target_started'), requirement: r, level: 'démontré', rank }
}

export function step(
  r: Requirement,
  kind: AgentStepKind,
  status: AgentStepEvent['status'] = 'ok',
  detail: string | null = null,
  attempt = 1,
): SuggestEvent {
  return { type: 'step', requirement_id: r.id, kind, status, attempt, detail }
}

export function suggestion(r: Requirement, overrides: Partial<SuggestionEvent> = {}): SuggestionEvent {
  return {
    ...suggestSample('suggestion'),
    requirement: r,
    before: BEFORE,
    after: AFTER,
    terms: [r.label],
    facts: [{ chunk_id: 2, quote: BEFORE }],
    implied_skills: [],
    ...overrides,
  }
}

export function refusal(r: Requirement, reason: RefusalReason, shown = true): RefusalEvent {
  const messages: Record<RefusalReason, string> = {
    a_decrire: 'Le CV ne le prouve pas encore : si vous l’avez réellement pratiqué, décrivez-le.',
    deja_explicite: 'Le terme de l’offre figure déjà dans le CV : aucune reformulation nécessaire.',
    aucune_puce: 'Aucune puce d’expérience ou de projet ne permet de nommer ce terme sans rien inventer.',
    aucune_reformulation_fidele: 'Aucune reformulation fidèle au CV n’a été trouvée.',
    baisse_de_niveau: 'Reformulation écartée : elle affaiblissait la preuve.',
    reevaluation_impossible: 'Réévaluation impossible.',
    non_traitee: 'Non traitée : budget de tokens atteint.',
    erreur: 'Le fournisseur LLM n’a pas répondu correctement pour cette exigence.',
  }
  return { ...suggestSample('refusal'), requirement: r, reason, message: messages[reason], shown, detail: 'détail du modèle' }
}

export function done(suggestions: number, refusals: number): SuggestEvent {
  return { ...suggestSample('done'), suggestions, refusals }
}

/** Flux complet : une carte « décrivez-le », une suggestion après un écart, un refus. */
export function fullFlow(): SuggestEvent[] {
  return [
    started([CI_CD, REST], [GO]),
    refusal(GO, 'a_decrire'),
    targetStarted(CI_CD, 1),
    step(CI_CD, 'search', 'ok', 'chunk 2, puce 1'),
    step(CI_CD, 'propose', 'ok', 'Mise en place d’un pipeline CI/CD…'),
    step(CI_CD, 'verify', 'rejected', 'Contrôle d’ancrage : « mise en place » ajoute une responsabilité que cette expérience ne mentionne pas.'),
    step(CI_CD, 'propose', 'ok', AFTER, 2),
    step(CI_CD, 'verify', 'ok', 'Juge : fidèle — seul le terme est ajouté', 2),
    step(CI_CD, 'reevaluate', 'ok', 'démontré -> démontré', 2),
    suggestion(CI_CD),
    targetStarted(REST, 2),
    step(REST, 'search', 'rejected', 'le style d’API n’est pas dit'),
    refusal(REST, 'aucune_puce'),
    done(1, 2),
  ]
}
