import type { EvidenceLevel, MatchReport } from '../api/types'

/** Un niveau qui prouve l'exigence : « mentionné » (simple citation) ne suffit pas. */
const DEMONSTRATED: ReadonlySet<EvidenceLevel> = new Set<EvidenceLevel>(['démontré', 'solide'])

export interface MandatoryCoverage {
  demonstrated: number
  total: number
  /** Obligatoires dont l'évaluation a échoué (comptées comme non démontrées). */
  notEvaluated: number
}

/** Couverture des exigences obligatoires : celles prouvées (« démontré » ou « solide »). */
export function mandatoryCoverage(report: MatchReport): MandatoryCoverage {
  const mandatory = report.requirements.filter((r) => r.requirement.importance === 'obligatoire')
  return {
    demonstrated: mandatory.filter((r) => r.status === 'ok' && r.level !== null && DEMONSTRATED.has(r.level)).length,
    total: mandatory.length,
    notEvaluated: mandatory.filter((r) => r.status === 'error').length,
  }
}
