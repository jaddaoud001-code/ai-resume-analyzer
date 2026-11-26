import { describe, expect, it, vi } from 'vitest'
import { jsonResponse, pdfFile, sample, splitEvery, streamResponse, toSSE } from '../test/stream'
import { fullFlow } from '../test/suggestFlows'
import { HttpError, ProtocolError } from './analyze'
import { suggest } from './suggest'
import type { SuggestEvent } from './types'

async function collect(iterable: AsyncIterable<SuggestEvent>): Promise<SuggestEvent[]> {
  const events: SuggestEvent[] = []
  for await (const event of iterable) events.push(event)
  return events
}

const REPORT = sample('report').report

describe('suggest', () => {
  it('renvoie le CV, l’offre et le rapport, puis les événements jusqu’à « done »', async () => {
    const flow = fullFlow()
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) =>
      Promise.resolve(streamResponse(splitEvery(toSSE([...flow, flow[0]!], { ping: true }), 11))),
    )
    vi.stubGlobal('fetch', fetchMock)

    const events = await collect(suggest(pdfFile(), 'Offre', REPORT, new AbortController().signal))

    expect(events.map((e) => e.type)).toEqual(flow.map((e) => e.type)) // rien après « done »
    const [url, init] = fetchMock.mock.calls[0] ?? []
    expect(url).toMatch(/\/suggest$/)
    expect(init?.method).toBe('POST')
    const body = init?.body as FormData
    expect((body.get('file') as File).name).toBe('cv.pdf')
    expect(body.get('job_offer')).toBe('Offre')
    expect(JSON.parse(body.get('report') as string)).toEqual(REPORT)
  })

  it('donne le message du backend pour un rapport qui ne correspond pas (422)', async () => {
    const detail = 'Le rapport d’analyse ne correspond pas à ce CV : relancez l’analyse.'
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse({ detail }, 422))))

    await expect(collect(suggest(pdfFile(), 'Offre', REPORT))).rejects.toEqual(new HttpError(422, detail))
  })

  it('rejette une suggestion qui porterait la paraphrase du modèle', async () => {
    const flow = fullFlow()
    const withClaim = flow.map((e) =>
      e.type === 'suggestion' ? { ...e, facts: [{ chunk_id: 2, quote: 'x', claim: 'paraphrase' }] } : e,
    )
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(streamResponse([toSSE(withClaim)]))))

    await expect(collect(suggest(pdfFile(), 'Offre', REPORT))).rejects.toBeInstanceOf(ProtocolError)
  })
})
