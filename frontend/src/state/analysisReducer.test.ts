import { describe, expect, it } from 'vitest'
import type { AnalysisEvent } from '../api/types'
import { SAMPLES, sample } from '../test/stream'
import { type AnalysisAction, type AnalysisState, analysisReducer, initialAnalysisState } from './analysisReducer'

function run(actions: AnalysisAction[], from: AnalysisState = initialAnalysisState): AnalysisState {
  return actions.reduce(analysisReducer, from)
}

const events = (...list: AnalysisEvent[]): AnalysisAction[] => list.map((event) => ({ type: 'event', event }))

describe('analysisReducer', () => {
  it('suit une analyse complète jusqu’au rapport', () => {
    const extracted = sample('requirements_extracted')
    const evaluated = SAMPLES.filter((e) => e.type === 'requirement_evaluated')

    let state = run([{ type: 'submit' }, ...events(sample('started'))])
    expect(state.status).toBe('running')
    expect(state.phases).toEqual({ indexing: 'active', extraction: 'active', evaluation: 'pending' })

    state = run(events(sample('cv_indexed')), state)
    expect(state.phases.indexing).toBe('done')
    expect(state.phases.evaluation).toBe('pending') // l'extraction n'est pas finie

    state = run(events(extracted), state)
    expect(state.phases).toEqual({ indexing: 'done', extraction: 'done', evaluation: 'active' })
    expect(state.total).toBe(extracted.requirements.length)
    expect(state.requirements.every((entry) => entry.result === null)).toBe(true)
    // Obligatoires d'abord.
    const importances = state.requirements.map((e) => e.requirement.importance)
    expect(importances.lastIndexOf('obligatoire')).toBeLessThan(importances.indexOf('souhaitée'))

    state = run(events(evaluated[0] as AnalysisEvent), state)
    expect(state.evaluated).toBe(1)
    expect(state.requirements.filter((e) => e.result !== null)).toHaveLength(1)

    state = run(events(...(evaluated.slice(1) as AnalysisEvent[]), sample('report')), state)
    expect(state.status).toBe('done')
    expect(state.phases.evaluation).toBe('done')
    expect(state.report?.score).toBe(sample('report').report.score)
    expect(state.requirements.every((entry) => entry.result !== null)).toBe(true)
  })

  it('affiche le message d’attente puis l’efface quand l’analyse reprend', () => {
    let state = run([{ type: 'submit' }, ...events(sample('started'), sample('status'))])
    expect(state.waitMessage).toBe('En attente du quota du fournisseur, reprise dans 60 s')

    state = run(events(sample('cv_indexed')), state)
    expect(state.waitMessage).toBeNull()
  })

  it('un événement error termine l’analyse avec son code et son message', () => {
    const state = run([{ type: 'submit' }, ...events(sample('started'), sample('status'), sample('error'))])

    expect(state.status).toBe('error')
    expect(state.error).toEqual({ kind: 'stream', code: 'timeout', message: sample('error').message })
    expect(state.waitMessage).toBeNull()
  })

  it('une erreur HTTP avant le flux est conservée avec son statut', () => {
    const state = run([
      { type: 'submit' },
      { type: 'failed', error: { kind: 'http', status: 429, message: 'Trop d’analyses en cours.' } },
    ])

    expect(state.status).toBe('error')
    expect(state.error?.status).toBe(429)
  })

  it('l’annulation garde les exigences déjà évaluées et ignore les événements suivants', () => {
    const evaluated = SAMPLES.find((e) => e.type === 'requirement_evaluated') as AnalysisEvent
    let state = run([
      { type: 'submit' },
      ...events(sample('started'), sample('cv_indexed'), sample('requirements_extracted'), evaluated),
      { type: 'cancelled' },
    ])
    expect(state.status).toBe('cancelled')
    expect(state.requirements.filter((e) => e.result !== null)).toHaveLength(1)

    state = run(events(sample('report')), state)
    expect(state.status).toBe('cancelled')
    expect(state.report).toBeNull()
  })

  it('une nouvelle analyse repart d’un état vierge', () => {
    const done = run([{ type: 'submit' }, ...events(...SAMPLES.filter((e) => e.type !== 'error'))])
    const state = analysisReducer(done, { type: 'submit' })

    expect(state).toEqual({ ...initialAnalysisState, status: 'running' })
  })
})
