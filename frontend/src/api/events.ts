// Conversion d'un message SSE en événement d'analyse typé, avec vérification des champs :
// une réponse mal formée devient une erreur explicite plutôt qu'un plantage de l'écran.

import type { AnalysisEvent, AnalysisEventType } from './types'

export class ProtocolError extends Error {
  constructor(message = 'Réponse inattendue du serveur.') {
    super(message)
    this.name = 'ProtocolError'
  }
}

type Check = 'string' | 'number' | 'boolean' | 'array' | 'object' | 'number|null'

const REQUIREMENT_FIELDS: Record<string, Check> = {
  id: 'string',
  label: 'string',
  category: 'string',
  importance: 'string',
  min_years: 'number|null',
  search_queries: 'array',
}

const MATCH_FIELDS: Record<string, Check> = {
  requirement: 'object',
  status: 'string',
  evidence: 'array',
  rejected_evidence: 'number',
  gaps: 'array',
  downgraded: 'boolean',
  capped_for_duration: 'boolean',
  capped_soft_skill: 'boolean',
  justification: 'string',
}

const REPORT_FIELDS: Record<string, Check> = {
  job_title: 'string',
  score: 'number|null',
  requirements: 'array',
  missing_mandatory: 'array',
  errors: 'array',
  requirements_from_cache: 'boolean',
}

/** Champs exigés pour chaque type d'événement (les autres sont acceptés tels quels). */
export const EVENT_FIELDS: Record<AnalysisEventType, Record<string, Check>> = {
  started: { phases: 'array', timeout_s: 'number' },
  cv_indexed: { chunk_count: 'number', seconds: 'number' },
  requirements_extracted: {
    job_title: 'string',
    requirements: 'array',
    seconds: 'number',
    from_cache: 'boolean',
  },
  status: { reason: 'string', wait_s: 'number', message: 'string' },
  requirement_evaluated: { result: 'object', done: 'number', total: 'number' },
  report: { report: 'object' },
  error: { code: 'string', message: 'string' },
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function matches(value: unknown, check: Check): boolean {
  switch (check) {
    case 'array':
      return Array.isArray(value)
    case 'object':
      return isObject(value)
    case 'number|null':
      return value === null || typeof value === 'number'
    default:
      return typeof value === check
  }
}

function hasFields(value: unknown, fields: Record<string, Check>): boolean {
  return isObject(value) && Object.entries(fields).every(([name, check]) => matches(value[name], check))
}

function isKnownType(type: string): type is AnalysisEventType {
  return Object.hasOwn(EVENT_FIELDS, type)
}

/**
 * `eventName` : champ « event » du message SSE ; `data` : son JSON.
 * Renvoie null pour un type inconnu (ignoré, pour qu'un backend plus récent ne casse pas
 * l'écran) ; lève ProtocolError pour un événement connu mal formé.
 */
export function toAnalysisEvent(eventName: string, data: string): AnalysisEvent | null {
  if (!isKnownType(eventName)) return null
  let payload: unknown
  try {
    payload = JSON.parse(data)
  } catch {
    throw new ProtocolError()
  }
  if (!isObject(payload) || payload.type !== eventName || !hasFields(payload, EVENT_FIELDS[eventName])) {
    throw new ProtocolError()
  }
  if (eventName === 'requirements_extracted') {
    const requirements = payload.requirements as unknown[]
    if (!requirements.every((r) => hasFields(r, REQUIREMENT_FIELDS))) throw new ProtocolError()
  }
  if (eventName === 'requirement_evaluated') {
    const result = payload.result
    if (!hasFields(result, MATCH_FIELDS) || !hasFields((result as Record<string, unknown>).requirement, REQUIREMENT_FIELDS)) {
      throw new ProtocolError()
    }
  }
  if (eventName === 'report') {
    const report = payload.report
    if (!hasFields(report, REPORT_FIELDS)) throw new ProtocolError()
    const results = (report as Record<string, unknown>).requirements as unknown[]
    if (!results.every((r) => hasFields(r, MATCH_FIELDS))) throw new ProtocolError()
  }
  return payload as unknown as AnalysisEvent
}
