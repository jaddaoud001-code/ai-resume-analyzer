import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { sample } from '../test/stream'
import { ResultSummary } from './ResultSummary'

describe('ResultSummary', () => {
  it('met en évidence les exigences obligatoires non démontrées', () => {
    const { report } = sample('report')
    render(<ResultSummary report={report} />)

    // Même compte que « 0/2 démontrées » : Go non démontrée, Python seulement mentionnée.
    const missing = screen.getByRole('region', { name: /Exigences obligatoires non démontrées \(2\)/ })
    expect(within(within(missing).getByRole('list', { name: 'Obligatoires non démontrées' })).getByRole('listitem')).toHaveTextContent('Go')
    expect(
      within(within(missing).getByRole('list', { name: 'Obligatoires seulement mentionnées' })).getByRole('listitem'),
    ).toHaveTextContent('Python')
    expect(screen.getByLabelText('Obligatoires : 0 sur 2 démontrées')).toBeInTheDocument()
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

  it('la case liste aussi les obligatoires seulement mentionnées, sans liste vide', () => {
    const { report } = sample('report')
    const onlyMentioned = report.requirements.map((r) =>
      r.requirement.label === 'Go' ? { ...r, level: 'démontré' as const } : r,
    )
    render(<ResultSummary report={{ ...report, requirements: onlyMentioned, missing_mandatory: [] }} />)

    const missing = screen.getByRole('region', { name: /Exigences obligatoires non démontrées \(1\)/ })
    expect(within(missing).getByText('Python')).toBeInTheDocument()
    expect(within(missing).queryByRole('list', { name: 'Obligatoires non démontrées' })).not.toBeInTheDocument()
  })

  it('le dit clairement quand toutes les obligatoires sont démontrées', () => {
    const { report } = sample('report')
    const proven = report.requirements.map((r) =>
      r.requirement.importance === 'obligatoire' ? { ...r, level: 'solide' as const } : r,
    )
    render(<ResultSummary report={{ ...report, requirements: proven, missing_mandatory: [], errors: [] }} />)

    expect(screen.queryByRole('region', { name: /non démontrées/ })).not.toBeInTheDocument()
    expect(screen.getByText('Toutes les exigences obligatoires sont démontrées par le CV.')).toBeInTheDocument()
  })
})
