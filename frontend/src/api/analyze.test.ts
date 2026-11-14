import { describe, expect, it, vi } from 'vitest'
import { SAMPLES, jsonResponse, pdfFile, sample, splitEvery, streamResponse, toSSE } from '../test/stream'
import { ConnectionLostError, HttpError, ProtocolError, analyze } from './analyze'
import type { AnalysisEvent } from './types'

async function collect(iterable: AsyncIterable<AnalysisEvent>): Promise<AnalysisEvent[]> {
  const events: AnalysisEvent[] = []
  for await (const event of iterable) events.push(event)
  return events
}

const SUCCESS = SAMPLES.filter((e) => e.type !== 'error')

describe('analyze', () => {
  it('envoie un POST multipart puis renvoie les événements typés jusqu’au rapport', async () => {
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) =>
      Promise.resolve(streamResponse(splitEvery(toSSE(SUCCESS, { ping: true }), 7))),
    )
    vi.stubGlobal('fetch', fetchMock)

    const events = await collect(analyze(pdfFile(), 'Offre', new AbortController().signal))

    expect(events.map((e) => e.type)).toEqual(SUCCESS.map((e) => e.type))
    const [url, init] = fetchMock.mock.calls[0] ?? []
    expect(url).toMatch(/\/analyze$/)
    expect(init?.method).toBe('POST')
    const body = init?.body as FormData
    expect(body.get('job_offer')).toBe('Offre')
    expect((body.get('file') as File).name).toBe('cv.pdf')
  })

  it('s’arrête après un événement error', async () => {
    const events = [sample('started'), sample('error'), sample('cv_indexed')]
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(streamResponse([toSSE(events)]))))

    const received = await collect(analyze(pdfFile(), 'Offre'))

    expect(received.map((e) => e.type)).toEqual(['started', 'error'])
  })

  it.each([
    [429, 'Trop d\'analyses en cours, réessayez dans quelques instants.'],
    [413, 'L\'offre dépasse la taille maximale de 15000 caractères (16000 reçus).'],
    [400, 'Le fichier n\'est pas un PDF valide.'],
    [500, 'Clé GROQ_API_KEY manquante.'],
  ])('lit le message du backend sur une erreur HTTP %i avant le flux', async (status, detail) => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse({ detail }, status))))

    const error = await collect(analyze(pdfFile(), 'Offre')).catch((e: unknown) => e)

    expect(error).toBeInstanceOf(HttpError)
    expect(error).toMatchObject({ status, message: detail })
  })

  it('donne un message en français pour une erreur 422 de validation FastAPI', async () => {
    const detail = [{ type: 'missing', loc: ['body', 'file'], msg: 'Field required' }]
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse({ detail }, 422))))

    const error = await collect(analyze(pdfFile(), 'Offre')).catch((e: unknown) => e)

    expect(error).toMatchObject({ status: 422, message: 'Le formulaire est incomplet : ajoutez un CV et une offre.' })
  })

  it('transforme un serveur injoignable en ConnectionLostError', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))

    await expect(collect(analyze(pdfFile(), 'Offre'))).rejects.toBeInstanceOf(ConnectionLostError)
  })

  it('une coupure réseau en cours de flux devient une erreur, sans nouvelle requête', async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(streamResponse([toSSE([sample('started'), sample('cv_indexed')])], { end: 'network-error' })),
    )
    vi.stubGlobal('fetch', fetchMock)
    const received: AnalysisEvent[] = []

    const error = await (async () => {
      for await (const event of analyze(pdfFile(), 'Offre')) received.push(event)
    })().catch((e: unknown) => e)

    expect(received.map((e) => e.type)).toEqual(['started', 'cv_indexed'])
    expect(error).toBeInstanceOf(ConnectionLostError)
    expect(fetchMock).toHaveBeenCalledTimes(1) // aucune reconnexion
  })

  it('un flux fermé sans rapport ni erreur est une connexion perdue', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(streamResponse([toSSE([sample('started')])]))))

    await expect(collect(analyze(pdfFile(), 'Offre'))).rejects.toBeInstanceOf(ConnectionLostError)
  })

  it('l’annulation interrompt la lecture et ferme le flux', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn((_url: string, init?: RequestInit) =>
      Promise.resolve(streamResponse([toSSE([sample('started')])], { end: 'hang', signal: init?.signal })),
    )
    vi.stubGlobal('fetch', fetchMock)
    const received: AnalysisEvent[] = []

    const error = await (async () => {
      for await (const event of analyze(pdfFile(), 'Offre', controller.signal)) {
        received.push(event)
        controller.abort()
      }
    })().catch((e: unknown) => e)

    expect(received.map((e) => e.type)).toEqual(['started'])
    expect((error as Error).name).toBe('AbortError')
    // Le signal est celui de fetch : l'annuler ferme la connexion, donc l'analyse côté backend.
    expect(fetchMock.mock.calls[0]?.[1]?.signal).toBe(controller.signal)
  })

  it('rejette une réponse qui n’est pas un flux SSE', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse({ ok: true }))))

    await expect(collect(analyze(pdfFile(), 'Offre'))).rejects.toBeInstanceOf(ProtocolError)
  })
})
