import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { RequirementMatch } from '../api/types'
import { SAMPLES } from '../test/stream'
import { RequirementCard } from './RequirementCard'

const RESULTS = SAMPLES.flatMap((e) => (e.type === 'requirement_evaluated' ? [e.result] : []))

function result(label: string): RequirementMatch {
  const found = RESULTS.find((r) => r.requirement.label === label)
  if (!found) throw new Error(label)
  return found
}

describe('RequirementCard', () => {
  it('passe de « en attente » à évaluée quand le résultat arrive', () => {
    const python = result('Python')
    const { rerender } = render(<RequirementCard entry={{ requirement: python.requirement, result: null }} />)
    const card = screen.getByRole('article', { name: 'Python' })

    expect(within(card).getByText('En attente…')).toBeInTheDocument()
    expect(card).toHaveAttribute('aria-busy', 'true')
    expect(within(card).getByText(/Obligatoire/)).toBeInTheDocument()

    rerender(<RequirementCard entry={{ requirement: python.requirement, result: python }} />)

    expect(within(card).queryByText('En attente…')).not.toBeInTheDocument()
    expect(within(card).getByText('Mentionné')).toBeInTheDocument()
    expect(within(card).getByText("« Conception d'API REST en Python avec FastAPI »")).toBeInTheDocument()
    expect(within(card).getByText('Durée prouvée inférieure aux 3 ans demandés')).toBeInTheDocument()
  })

  it('explique les drapeaux en mots simples', () => {
    render(
      <>
        <RequirementCard entry={{ requirement: result('Python').requirement, result: result('Python') }} />
        <RequirementCard entry={{ requirement: result('Go').requirement, result: result('Go') }} />
        <RequirementCard entry={{ requirement: result('Autonomie').requirement, result: result('Autonomie') }} />
        <RequirementCard entry={{ requirement: result('Docker').requirement, result: result('Docker') }} />
      </>,
    )

    expect(screen.getAllByText(/1 citation écartée : introuvable mot pour mot dans le CV/)).toHaveLength(2)
    expect(screen.getByText(/Durée insuffisante : 14 mois prouvés pour 3 ans demandés/)).toBeInTheDocument()
    expect(screen.getByText(/Niveau ramené à « non démontré » : aucune preuve vérifiable/)).toBeInTheDocument()
    expect(screen.getByText(/Savoir-être limité à « démontré »/)).toBeInTheDocument()
    expect(screen.getByText(/Évaluation impossible pour cette exigence/)).toBeInTheDocument()
    expect(within(screen.getByRole('article', { name: 'Docker' })).getByText('Erreur')).toBeInTheDocument()
  })

  it('affiche sur demande le texte des citations écartées', async () => {
    const python = result('Python')
    render(<RequirementCard entry={{ requirement: python.requirement, result: python }} />)
    const summary = screen.getByText(/1 citation écartée/)
    const details = summary.closest('details') as HTMLDetailsElement

    expect(details.open).toBe(false) // replié par défaut
    await userEvent.click(summary)

    expect(details.open).toBe(true)
    expect(within(details).getByText('Expert Python depuis 10 ans')).toBeVisible()
  })

  it('explique la seconde chance quand elle n’a pas suffi', () => {
    const go = result('Go')
    render(<RequirementCard entry={{ requirement: go.requirement, result: go }} />)

    expect(screen.getByText(/Même après une seconde demande de citations exactes/)).toBeInTheDocument()
  })

  it('explique la seconde chance quand elle a permis de retrouver la preuve', () => {
    const python = { ...result('Python'), second_chance: true }
    render(<RequirementCard entry={{ requirement: python.requirement, result: python }} />)

    expect(screen.getByText(/le modèle a recopié ses preuves mot pour mot/)).toBeInTheDocument()
  })

  it('affiche « Non évaluée » quand l’analyse s’est arrêtée avant', () => {
    const go = result('Go')
    render(<RequirementCard entry={{ requirement: go.requirement, result: null }} interrupted />)

    expect(screen.getByText('Non évaluée')).toBeInTheDocument()
  })
})
