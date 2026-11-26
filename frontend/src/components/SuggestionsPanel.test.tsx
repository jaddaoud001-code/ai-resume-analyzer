import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useReducer } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { SuggestEvent, SuggestionEvent } from '../api/types'
import { type SuggestState, initialSuggestState, suggestReducer } from '../state/suggestReducer'
import { AFTER, BEFORE, CI_CD, ENGLISH, GO, done, fullFlow, refusal, started, suggestion } from '../test/suggestFlows'
import { SUGGESTIONS_NOTICE, SuggestionsPanel } from './SuggestionsPanel'

function stateFrom(events: SuggestEvent[]): SuggestState {
  return [{ type: 'submit' as const }, ...events.map((event) => ({ type: 'event' as const, event }))].reduce(
    suggestReducer,
    initialSuggestState,
  )
}

/** Panneau branché sur le vrai reducer (Accepter / Refuser modifient l'état). */
function Harness({ initial, onStart = () => undefined }: { initial: SuggestState; onStart?: () => void }) {
  const [state, dispatch] = useReducer(suggestReducer, initial)
  return (
    <SuggestionsPanel
      state={state}
      elapsedSeconds={185}
      onStart={onStart}
      onCancel={() => dispatch({ type: 'cancelled' })}
      onDecide={(requirementId, decision) => dispatch({ type: 'decide', requirementId, decision })}
    />
  )
}

describe('déclenchement', () => {
  it('explique l’agent avant de le lancer', async () => {
    const onStart = vi.fn()
    render(<Harness initial={initialSuggestState} onStart={onStart} />)

    expect(screen.getByText(/rendre explicites/)).toHaveTextContent('sans rien inventer')
    expect(screen.getByText(/quelques minutes/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Proposer des améliorations du CV' }))
    expect(onStart).toHaveBeenCalledOnce()
  })

  it('pendant l’exécution : frise, proposition écartée, zone aria-live et Annuler', async () => {
    const running = stateFrom(fullFlow().slice(0, 6)) // juste après l'écart par le contrôle
    render(<Harness initial={running} />)

    const panel = screen.getByRole('listitem', { name: 'Exigence ciblée : CI/CD' })
    expect(within(panel).getByText(/^Proposition écartée : contrôle du CV — « mise en place »/)).toBeInTheDocument()
    expect(panel.querySelector('[data-step="search"]')).toHaveAttribute('data-state', 'done')
    expect(panel.querySelector('[data-step="propose"]')).toHaveAttribute('data-state', 'active')
    expect(screen.getByText(/^CI\/CD : Proposition écartée/)).toHaveAttribute('aria-live', 'polite')
    expect(screen.queryByRole('button', { name: 'Proposer des améliorations du CV' })).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Annuler' }))
    expect(screen.getAllByText('Suggestions annulées.')).not.toHaveLength(0)
  })
})

describe('résultats', () => {
  it('carte de suggestion : diff surligné, citations exactes, niveau indicatif', () => {
    render(<Harness initial={stateFrom(fullFlow())} />)

    const card = screen.getByRole('article', { name: 'Suggestion : CI/CD' })
    expect(within(card).getByText(BEFORE)).toBeInTheDocument()
    const proposed = within(card).getByTestId('proposed-bullet')
    expect(proposed).toHaveTextContent(AFTER)
    expect([...proposed.querySelectorAll('mark')].map((m) => m.textContent)).toEqual(['CI/CD'])
    expect(within(card).getByText(`« ${BEFORE} »`)).toBeInTheDocument()
    expect(within(card).getByText('indicatif')).toBeInTheDocument()
    expect(within(card).getByText(/démontré → démontré/)).toBeInTheDocument()
  })

  it('montre l’implication de la table utilisée, s’il y en a une', () => {
    const withImplication = suggestion(CI_CD, { implied_skills: [{ skill: 'Git', chunk_id: 2, quote: 'GitLab CI' }] })
    render(<Harness initial={stateFrom([started([CI_CD]), withImplication, done(1, 0)])} />)

    expect(screen.getByText(/Implication utilisée/)).toHaveTextContent('Git découle de « GitLab CI »')
  })

  it('n’affiche jamais la paraphrase ni la justification du modèle, seulement les citations exactes', () => {
    const fromModel = {
      ...suggestion(CI_CD),
      justification: 'JUSTIFICATION DU MODÈLE',
      judge: 'AVIS DU JUGE',
      // Un backend plus ancien qui enverrait encore la paraphrase : jamais affichée.
      facts: [{ chunk_id: 2, quote: BEFORE, claim: 'PARAPHRASE : mise en place d’un pipeline' }],
    } as unknown as SuggestionEvent
    const withDetail = { ...refusal(GO, 'aucune_puce'), detail: 'RAISON RÉDIGÉE PAR LE MODÈLE' }
    render(<Harness initial={stateFrom([started([CI_CD, GO]), fromModel, withDetail, done(1, 1)])} />)

    const page = document.body.textContent ?? ''
    for (const text of ['JUSTIFICATION DU MODÈLE', 'AVIS DU JUGE', 'PARAPHRASE', 'RAISON RÉDIGÉE PAR LE MODÈLE']) {
      expect(page).not.toContain(text)
    }
    expect(screen.getByText(`« ${BEFORE} »`)).toBeInTheDocument()
  })

  it('cartes de refus avec leur raison, et cartes « décrivez-le » sans texte proposé', () => {
    render(<Harness initial={stateFrom([...fullFlow().slice(0, -1), refusal(ENGLISH, 'a_decrire'), done(1, 3)])} />)

    const rest = screen.getByRole('article', { name: 'Refus : API REST' })
    expect(rest).toHaveTextContent('Aucune reformulation possible')
    expect(rest).toHaveTextContent('Aucune puce d’expérience ou de projet ne permet de nommer ce terme sans rien inventer.')
    for (const label of ['Go', 'Anglais professionnel']) {
      const card = screen.getByRole('article', { name: `À décrire : ${label}` })
      expect(card).toHaveTextContent('décrivez vous-même une expérience réelle')
      expect(card).toHaveTextContent('Aucun texte n’est proposé')
    }
  })

  it('mention de relecture, tokens et durée', () => {
    render(<Harness initial={stateFrom(fullFlow())} />)

    expect(screen.getByText(SUGGESTIONS_NOTICE)).toBeInTheDocument()
    expect(screen.getByText(/tokens · durée 3:05/)).toHaveTextContent(/^5[\s ]?500 tokens · durée 3:05$/)
  })
})

describe('Accepter / Refuser et copie', () => {
  it('copie seulement les puces acceptées, rien n’est envoyé', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn(() => Promise.resolve())
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const second = suggestion(GO, { before: 'Services en Gin.', after: 'Services Go en Gin.' })
    render(<Harness initial={stateFrom([started([CI_CD, GO]), suggestion(CI_CD), second, done(2, 0)])} />)

    const copy = screen.getByRole('button', { name: 'Copier les puces acceptées' })
    expect(copy).toBeDisabled()

    const ciCd = screen.getByRole('group', { name: 'Décision pour CI/CD' })
    await user.click(within(ciCd).getByRole('button', { name: 'Accepter' }))
    expect(within(ciCd).getByRole('button', { name: 'Accepter' })).toHaveAttribute('aria-pressed', 'true')
    const go = screen.getByRole('group', { name: 'Décision pour Go' })
    await user.click(within(go).getByRole('button', { name: 'Refuser' }))
    expect(within(go).getByText(/Refusée/)).toBeInTheDocument()

    await user.click(copy)
    expect(writeText).toHaveBeenCalledWith(AFTER)
    expect(await screen.findByText('Puces copiées.')).toBeInTheDocument()

    // Accepter de nouveau : décision annulée, plus rien à copier.
    await user.click(within(ciCd).getByRole('button', { name: 'Accepter' }))
    expect(copy).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('si le presse-papiers est refusé, le texte reste sélectionnable', async () => {
    const user = userEvent.setup()
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: () => Promise.reject(new Error('refusé')) },
      configurable: true,
    })
    render(<Harness initial={stateFrom([started([CI_CD]), suggestion(CI_CD), done(1, 0)])} />)

    await user.click(within(screen.getByRole('group', { name: 'Décision pour CI/CD' })).getByRole('button', { name: 'Accepter' }))
    await user.click(screen.getByRole('button', { name: 'Copier les puces acceptées' }))

    expect(await screen.findByRole('textbox', { name: 'Puces acceptées' })).toHaveValue(AFTER)
  })
})
