import type { EvidenceLevel, MatchReport, Requirement } from '../api/types'

/** Un niveau qui prouve l'exigence : « mentionné » (simple citation) ne suffit pas. */
const DEMONSTRATED: ReadonlySet<EvidenceLevel> = new Set<EvidenceLevel>(['démontré', 'solide'])

export interface MandatoryCoverage {
  demonstrated: number
  total: number
  /** Obligatoires dont l'évaluation a échoué (comptées comme non démontrées). */
  notEvaluated: number
  /** Obligatoires non prouvées, dans l'ordre du rapport : aucune preuve, ou simple mention. */
  notDemonstrated: Requirement[]
  onlyMentioned: Requirement[]
}

/** Couverture des exigences obligatoires : celles prouvées (« démontré » ou « solide »). */
export function mandatoryCoverage(report: MatchReport): MandatoryCoverage {
  const mandatory = report.requirements.filter((r) => r.requirement.importance === 'obligatoire')
  return {
    demonstrated: mandatory.filter((r) => r.status === 'ok' && r.level !== null && DEMONSTRATED.has(r.level)).length,
    total: mandatory.length,
    notEvaluated: mandatory.filter((r) => r.status === 'error').length,
    notDemonstrated: mandatory.filter((r) => r.status === 'ok' && r.level === 'non démontré').map((r) => r.requirement),
    onlyMentioned: mandatory.filter((r) => r.status === 'ok' && r.level === 'mentionné').map((r) => r.requirement),
  }
}
