// Test de contrat : chaque exemple de __fixtures__/events.sample.json (généré par le backend
// à partir des modèles Pydantic, voir backend/tests/test_contract.py) doit être reconnu.

import { describe, expect, it } from 'vitest'
import samples from './__fixtures__/events.sample.json'
import suggestSamples from './__fixtures__/suggest_events.sample.json'
import { EVENT_FIELDS, ProtocolError, SUGGEST_EVENT_FIELDS, toAnalysisEvent, toSuggestEvent } from './events'
import type { AnalysisEventType, SuggestEventType } from './types'

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

const SUGGEST_EXAMPLES = suggestSamples as unknown as Array<{ type: string } & Record<string, unknown>>

describe('contrat des événements de l’agent (POST /suggest)', () => {
  it('le fichier contient un exemple de chaque type connu du frontend, et rien d’autre', () => {
    const types = new Set(SUGGEST_EXAMPLES.map((e) => e.type))
    expect([...types].sort()).toEqual(Object.keys(SUGGEST_EVENT_FIELDS).sort())
  })

  it.each(SUGGEST_EXAMPLES.map((e) => [e.type, e] as const))('reconnaît l’événement %s avec ses champs', (type, example) => {
    const event = toSuggestEvent(type, JSON.stringify(example))

    expect(event).not.toBeNull()
    expect(event?.type).toBe(type)
    for (const field of Object.keys(SUGGEST_EVENT_FIELDS[type as SuggestEventType])) {
      expect(event).toHaveProperty(field)
    }
  })

  it('une suggestion n’expose que des citations sources exactes, jamais la paraphrase du modèle', () => {
    const suggestion = SUGGEST_EXAMPLES.find((e) => e.type === 'suggestion') as Record<string, unknown>
    expect(JSON.stringify(suggestion)).not.toContain('claim')
    const withParaphrase = {
      ...suggestion,
      facts: [{ chunk_id: 3, quote: 'Pipeline de tests', claim: 'Mise en place d’un pipeline CI/CD' }],
    }
    expect(() => toSuggestEvent('suggestion', JSON.stringify(withParaphrase))).toThrow(ProtocolError)
  })

  it('une étape publique ne porte ni faits ni compétences déclarées', () => {
    const step = SUGGEST_EXAMPLES.find((e) => e.type === 'step') as Record<string, unknown>
    expect(step).not.toHaveProperty('facts')
    expect(step).not.toHaveProperty('implied_skills')
  })

  it('ignore un type inconnu et rejette un événement connu mal formé', () => {
    expect(toSuggestEvent('nouveau_type', '{"type":"nouveau_type"}')).toBeNull()
    expect(() => toSuggestEvent('done', '{"type":"done"}')).toThrow(ProtocolError)
  })
})
