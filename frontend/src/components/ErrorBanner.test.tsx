import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ErrorBanner } from './ErrorBanner'

describe('ErrorBanner', () => {
  it('quota journalier atteint pendant l’analyse : titre et message du backend', () => {
    const message = 'Quota journalier du fournisseur LLM atteint : réessayez demain.'
    render(<ErrorBanner error={{ kind: 'stream', code: 'quota_exceeded', message }} />)

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('Quota journalier du fournisseur d’IA atteint')
    expect(alert).toHaveTextContent(message)
  })

  it('modèle refusé par le fournisseur : titre explicite', () => {
    render(
      <ErrorBanner
        error={{ kind: 'stream', code: 'provider_access_denied', message: 'Abonnement ou modèle non autorisé.' }}
      />,
    )

    expect(screen.getByRole('alert')).toHaveTextContent('Le fournisseur d’IA refuse l’accès au modèle')
  })

  it('quota journalier constaté avant le flux (HTTP 429) : le message précise la cause', () => {
    const message = 'Quota journalier du fournisseur LLM atteint : réessayez demain.'
    render(<ErrorBanner error={{ kind: 'http', status: 429, message }} />)

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('Analyse impossible pour le moment')
    expect(alert).toHaveTextContent(message)
  })
})
