import { describe, expect, it } from 'vitest'
import type { EvidenceLevel, RequirementMatch } from '../api/types'
import { sample } from '../test/stream'
import { mandatoryCoverage } from './coverage'

function withLevels(levels: Array<EvidenceLevel | 'error'>, importance: 'obligatoire' | 'souhaitée' = 'obligatoire') {
  const template = sample('report').report.requirements[0] as RequirementMatch
  return levels.map((level, index) => ({
    ...template,
    requirement: { ...template.requirement, id: `R${index + 1}`, importance },
    status: level === 'error' ? ('error' as const) : ('ok' as const),
    level: level === 'error' ? null : level,
  }))
}

describe('mandatoryCoverage', () => {
  it('compte « démontré » et « solide », pas « mentionné » ni « non démontré »', () => {
    const report = {
      ...sample('report').report,
      requirements: [
        ...withLevels(['solide', 'démontré', 'mentionné', 'non démontré']),
        ...withLevels(['solide', 'solide'], 'souhaitée'), // ignorées : pas obligatoires
      ],
    }

    const coverage = mandatoryCoverage(report)
    expect(coverage).toMatchObject({ demonstrated: 2, total: 4, notEvaluated: 0 })
    // Les deux non prouvées, chacune dans sa liste : la case du résultat les montre toutes.
    expect(coverage.notDemonstrated.map((r) => r.id)).toEqual(['R4'])
    expect(coverage.onlyMentioned.map((r) => r.id)).toEqual(['R3'])
  })

  it('une obligatoire en erreur est comptée comme non démontrée et signalée', () => {
    const report = { ...sample('report').report, requirements: withLevels(['démontré', 'error']) }

    const coverage = mandatoryCoverage(report)
    expect(coverage).toMatchObject({ demonstrated: 1, total: 2, notEvaluated: 1 })
    expect([...coverage.notDemonstrated, ...coverage.onlyMentioned]).toEqual([])
  })

  it('exemple du contrat : Python « mentionné » et Go « non démontré »', () => {
    const coverage = mandatoryCoverage(sample('report').report)
    expect(coverage).toMatchObject({ demonstrated: 0, total: 2, notEvaluated: 0 })
    expect(coverage.notDemonstrated.map((r) => r.label)).toEqual(['Go'])
    expect(coverage.onlyMentioned.map((r) => r.label)).toEqual(['Python'])
  })
})
