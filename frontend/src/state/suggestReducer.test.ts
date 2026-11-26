import { describe, expect, it } from 'vitest'
import type { SuggestEvent } from '../api/types'
import {
  AFTER,
  CI_CD,
  ENGLISH,
  GO,
  REST,
  done,
  fullFlow,
  refusal,
  started,
  step,
  suggestSample,
  suggestion,
  targetStarted,
} from '../test/suggestFlows'
import {
  type SuggestAction,
  type SuggestState,
  acceptedBullets,
  initialSuggestState,
  rejectionReason,
  suggestReducer,
} from './suggestReducer'

function run(events: SuggestEvent[], extra: SuggestAction[] = []): SuggestState {
  const actions: SuggestAction[] = [{ type: 'submit' }, ...events.map((event) => ({ type: 'event' as const, event })), ...extra]
  return actions.reduce(suggestReducer, initialSuggestState)
}

describe('suggestReducer : flux complet', () => {
  it('construit panneaux, suggestion, refus et cartes « décrivez-le »', () => {
    const state = run(fullFlow())

    expect(state.status).toBe('done')
    expect(state.targets.map((t) => [t.id, t.outcome])).toEqual([
      ['R5', 'suggestion'],
      ['R7', 'refusal'],
    ])
    const [ciCd, rest] = state.targets
    expect(ciCd?.steps).toEqual({ search: 'done', propose: 'done', verify: 'done', reevaluate: 'done' })
    expect(ciCd?.rejections).toBe(1)
    expect(ciCd?.rejection).toBeNull() // effacée quand la nouvelle proposition passe
    expect(rest?.steps.search).toBe('rejected')
    expect(state.suggestions.map((s) => s.after)).toEqual([AFTER])
    expect(state.refusals.map((r) => [r.requirement.id, r.reason])).toEqual([['R7', 'aucune_puce']])
    expect(state.toDescribe.map((r) => r.requirement.id)).toEqual(['R2'])
    expect(state.done?.suggestions).toBe(1)
  })

  it('allume la frise au fil des étapes et montre la proposition écartée', () => {
    const events = fullFlow()
    const afterSearch = run(events.slice(0, 4))
    expect(afterSearch.targets[0]?.steps).toEqual({ search: 'done', propose: 'active', verify: 'pending', reevaluate: 'pending' })

    const afterRejection = run(events.slice(0, 6))
    const panel = afterRejection.targets[0]
    expect(panel?.rejection).toBe(
      'Proposition écartée : contrôle du CV — « mise en place » ajoute une responsabilité que cette expérience ne mentionne pas.',
    )
    // Nouvelle tentative : retour à la proposition.
    expect(panel?.steps).toEqual({ search: 'done', propose: 'active', verify: 'pending', reevaluate: 'pending' })
    expect(afterRejection.targets[1]?.outcome).toBe('waiting')
  })

  it('les langues reçoivent aussi une carte, sans panneau', () => {
    const state = run([started([], [ENGLISH]), refusal(ENGLISH, 'a_decrire'), done(0, 1)])

    expect(state.targets).toEqual([])
    expect(state.toDescribe.map((r) => r.requirement.label)).toEqual(['Anglais professionnel'])
    expect(state.refusals).toEqual([])
  })
})

describe('suggestReducer : refus', () => {
  it('une baisse de niveau est comptée mais pas montrée', () => {
    const state = run([
      started([CI_CD]),
      targetStarted(CI_CD),
      step(CI_CD, 'search'),
      step(CI_CD, 'propose'),
      step(CI_CD, 'verify'),
      step(CI_CD, 'reevaluate', 'rejected', 'Baisse de niveau (démontré -> mentionné). La reformulation précédente a fait baisser le niveau.'),
      refusal(CI_CD, 'baisse_de_niveau', false),
      done(0, 0),
    ])

    expect(state.targets[0]?.outcome).toBe('hidden')
    expect(state.refusals).toEqual([])
    expect(state.targets[0]?.steps.propose).toBe('rejected') // plus rien « en cours »
  })

  it('une cible non traitée (budget) devient un refus montré', () => {
    const state = run([started([CI_CD, REST]), refusal(CI_CD, 'non_traitee'), refusal(REST, 'non_traitee'), done(0, 2)])

    expect(state.targets.map((t) => t.outcome)).toEqual(['refusal', 'refusal'])
    expect(state.refusals).toHaveLength(2)
  })
})

describe('suggestReducer : quota, erreur, annulation', () => {
  it('affiche l’attente de quota puis l’efface au prochain événement', () => {
    const waiting = run([started([CI_CD]), suggestSample('status')])
    expect(waiting.waitMessage).toBe('En attente du quota du fournisseur, reprise dans 60 s')

    const resumed = run([started([CI_CD]), suggestSample('status'), targetStarted(CI_CD)])
    expect(resumed.waitMessage).toBeNull()
  })

  it('une erreur arrête le flux ; les événements suivants sont ignorés', () => {
    const state = run([
      started([CI_CD, REST]),
      targetStarted(CI_CD),
      { type: 'error', code: 'quota_exceeded', message: 'Quota journalier du fournisseur LLM atteint : réessayez demain.' },
      targetStarted(REST),
    ])

    expect(state.status).toBe('error')
    expect(state.error).toEqual({ kind: 'stream', code: 'quota_exceeded', message: 'Quota journalier du fournisseur LLM atteint : réessayez demain.' })
    expect(state.targets[1]?.outcome).toBe('waiting')
  })

  it('l’annulation garde les suggestions reçues, toujours décidables', () => {
    const state = run(
      [started([CI_CD, REST]), targetStarted(CI_CD), suggestion(CI_CD), targetStarted(REST)],
      [{ type: 'cancelled' }, { type: 'event', event: refusal(REST, 'aucune_puce') }, { type: 'decide', requirementId: 'R5', decision: 'accepted' }],
    )

    expect(state.status).toBe('cancelled')
    expect(state.refusals).toEqual([]) // arrivé après l'annulation : ignoré
    expect(acceptedBullets(state)).toBe(AFTER)
  })

  it('une erreur HTTP avant le flux', () => {
    const state = run([], [{ type: 'failed', error: { kind: 'http', status: 422, message: 'Le rapport ne correspond pas.' } }])

    expect(state.status).toBe('error')
    expect(state.error?.status).toBe(422)
  })
})

describe('Accepter / Refuser et copie', () => {
  it('décisions locales, annulables ; seules les puces acceptées sont copiées', () => {
    const second = suggestion(GO, { after: 'Services Go et gRPC.' })
    const base = [started([CI_CD, GO]), suggestion(CI_CD), second, done(2, 0)]

    const accepted = run(base, [
      { type: 'decide', requirementId: 'R5', decision: 'accepted' },
      { type: 'decide', requirementId: 'R2', decision: 'accepted' },
      { type: 'decide', requirementId: 'R2', decision: 'refused' },
    ])
    expect(accepted.decisions).toEqual({ R5: 'accepted', R2: 'refused' })
    expect(acceptedBullets(accepted)).toBe(AFTER)

    const undone = suggestReducer(accepted, { type: 'decide', requirementId: 'R5', decision: null })
    expect(acceptedBullets(undone)).toBe('')
    // Exigence sans suggestion : ignorée.
    expect(suggestReducer(accepted, { type: 'decide', requirementId: 'R9', decision: 'accepted' })).toBe(accepted)
  })
})

describe('rejectionReason', () => {
  it('rend lisibles les motifs du backend', () => {
    expect(rejectionReason('Contrôle d’ancrage : « cloud » n’apparaît pas dans cette expérience.')).toBe(
      'contrôle du CV — « cloud » n’apparaît pas dans cette expérience.',
    )
    expect(rejectionReason('Juge : verbe renforcé')).toBe('relecture — verbe renforcé')
    expect(rejectionReason('Baisse de niveau (solide -> démontré). Garde tous les faits.')).toBe('Baisse de niveau (solide -> démontré).')
    expect(rejectionReason('Terme absent : la puce doit nommer « CI/CD ».')).toBe('Terme absent : la puce doit nommer « CI/CD ».')
    expect(rejectionReason(null)).toBe('motif non précisé')
  })
})
