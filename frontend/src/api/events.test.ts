// Test de contrat : chaque exemple de __fixtures__/events.sample.json (généré par le backend
// à partir des modèles Pydantic, voir backend/tests/test_contract.py) doit être reconnu.

import { describe, expect, it } from 'vitest'
import samples from './__fixtures__/events.sample.json'
import { EVENT_FIELDS, ProtocolError, toAnalysisEvent } from './events'
import type { AnalysisEventType } from './types'

const EXAMPLES = samples as unknown as Array<{ type: string } & Record<string, unknown>>

describe('contrat des événements', () => {
  it('le fichier contient un exemple de chaque type connu du frontend, et rien d’autre', () => {
    const types = new Set(EXAMPLES.map((e) => e.type))
    expect([...types].sort()).toEqual(Object.keys(EVENT_FIELDS).sort())
  })

  it.each(EXAMPLES.map((e) => [e.type, e] as const))('reconnaît l’événement %s avec ses champs', (type, example) => {
    const event = toAnalysisEvent(type, JSON.stringify(example))

    expect(event).not.toBeNull()
    expect(event?.type).toBe(type)
    for (const field of Object.keys(EVENT_FIELDS[type as AnalysisEventType])) {
      expect(event).toHaveProperty(field)
    }
  })

  it('les champs utilisés par l’écran sont présents dans les exemples', () => {
    const evaluated = EXAMPLES.filter((e) => e.type === 'requirement_evaluated')
    const flags = new Set<string>()
    for (const example of evaluated) {
      const result = example.result as Record<string, unknown>
      for (const name of ['level', 'evidence', 'gaps', 'rejected_evidence', 'downgraded', 'capped_for_duration', 'capped_soft_skill', 'total_months', 'status', 'error']) {
        expect(result).toHaveProperty(name)
      }
      for (const flag of ['downgraded', 'capped_for_duration', 'capped_soft_skill']) {
        if (result[flag] === true) flags.add(flag)
      }
      expect((result.requirement as Record<string, unknown>).min_years).not.toBeUndefined()
    }
    // Les exemples couvrent chaque drapeau affiché.
    expect([...flags].sort()).toEqual(['capped_for_duration', 'capped_soft_skill', 'downgraded'])
  })
})

describe('toAnalysisEvent', () => {
  it('ignore un type inconnu (backend plus récent)', () => {
    expect(toAnalysisEvent('nouveau_type', '{"type":"nouveau_type"}')).toBeNull()
  })

  it('rejette un événement connu mal formé', () => {
    expect(() => toAnalysisEvent('cv_indexed', 'pas du JSON')).toThrow(ProtocolError)
    expect(() => toAnalysisEvent('cv_indexed', '{"type":"cv_indexed"}')).toThrow(ProtocolError)
    expect(() => toAnalysisEvent('cv_indexed', '{"type":"report","chunk_count":1,"seconds":1}')).toThrow(ProtocolError)
  })
})
