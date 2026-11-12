import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { sample } from '../test/stream'
import { ResultSummary } from './ResultSummary'

describe('ResultSummary', () => {
  it('met en évidence les exigences obligatoires non démontrées', () => {
    const { report } = sample('report')
    render(<ResultSummary report={report} />)

    const missing = screen.getByRole('region', { name: /Exigences obligatoires non démontrées \(1\)/ })
    expect(within(missing).getByRole('listitem')).toHaveTextContent('Go')
    expect(screen.getByText(String(report.score))).toBeInTheDocument()
    expect(screen.getByText(/varie d’environ ±5 points/)).toBeInTheDocument()
    expect(screen.getByText(/1 exigence n’a pas pu être évaluée/)).toBeInTheDocument()
  })

  it('affiche la couverture des obligatoires à côté du score', () => {
    const { report } = sample('report')
    const demonstrated = report.requirements.map((r) =>
      r.requirement.label === 'Python' ? { ...r, level: 'solide' as const } : r,
    )
    render(<ResultSummary report={{ ...report, requirements: demonstrated }} />)

    expect(screen.getByLabelText('Obligatoires : 1 sur 2 démontrées')).toHaveTextContent('1/2 démontrées')
    expect(screen.getByText(/une simple mention ne suffit pas/)).toBeInTheDocument()
  })

  it('le dit clairement quand toutes les obligatoires sont couvertes', () => {
    const { report } = sample('report')
    render(<ResultSummary report={{ ...report, missing_mandatory: [], errors: [] }} />)

    expect(screen.queryByRole('region', { name: /non démontrées/ })).not.toBeInTheDocument()
    expect(screen.getByText(/Toutes les exigences obligatoires sont au moins mentionnées/)).toBeInTheDocument()
  })
})
