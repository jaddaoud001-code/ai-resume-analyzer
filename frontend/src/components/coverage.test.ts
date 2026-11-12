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

    expect(mandatoryCoverage(report)).toEqual({ demonstrated: 2, total: 4, notEvaluated: 0 })
  })

  it('une obligatoire en erreur est comptée comme non démontrée et signalée', () => {
    const report = { ...sample('report').report, requirements: withLevels(['démontré', 'error']) }

    expect(mandatoryCoverage(report)).toEqual({ demonstrated: 1, total: 2, notEvaluated: 1 })
  })

  it('exemple du contrat : Python « mentionné » et Go « non démontré »', () => {
    expect(mandatoryCoverage(sample('report').report)).toEqual({ demonstrated: 0, total: 2, notEvaluated: 0 })
  })
})
