// Outils de test : flux SSE simulés avec un ReadableStream (aucun appel réseau).

import samples from '../api/__fixtures__/events.sample.json'
import type { AnalysisEvent, AnalysisEventType } from '../api/types'

/** Exemples du contrat (générés par le backend), un par type d'événement. */
export const SAMPLES = samples as unknown as AnalysisEvent[]

export function sample<T extends AnalysisEventType>(type: T): Extract<AnalysisEvent, { type: T }> {
  const event = SAMPLES.find((e) => e.type === type)
  if (!event) throw new Error(`exemple absent : ${type}`)
  return structuredClone(event) as Extract<AnalysisEvent, { type: T }>
}

/** Texte SSE tel qu'envoyé par sse-starlette (lignes en \r\n). */
export function toSSE(events: AnalysisEvent[], { ping = false } = {}): string {
  return events
    .map((event, index) => {
      const block = `event: ${event.type}\r\nid: ${index + 1}\r\ndata: ${JSON.stringify(event)}\r\n\r\n`
      return ping ? `: ping - 2026-11-10 10:00:00\r\n\r\n${block}` : block
    })
    .join('')
}

/** Découpe un texte en morceaux de `size` octets (un événement coupé en plusieurs morceaux). */
export function splitEvery(text: string, size: number): string[] {
  const chunks: string[] = []
  for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size))
  return chunks
}

interface StreamOptions {
  /** Après les morceaux : fermer (défaut), couper (erreur réseau) ou rester ouvert. */
  end?: 'close' | 'network-error' | 'hang'
  signal?: AbortSignal | null
}

export function streamResponse(chunks: string[], { end = 'close', signal }: StreamOptions = {}): Response {
  const encoder = new TextEncoder()
  let next = 0
  // Mode « pull » : un morceau par lecture, puis la fin (une erreur immédiate dans `start`
  // effacerait les morceaux encore en file).
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      // Comme fetch : l'annulation interrompt la lecture avec une AbortError.
      signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')))
    },
    pull(controller) {
      const chunk = chunks[next++]
      if (chunk !== undefined) {
        controller.enqueue(encoder.encode(chunk))
        return
      }
      if (end === 'close') controller.close()
      if (end === 'network-error') controller.error(new TypeError('network error'))
      // 'hang' : aucune donnée, le flux reste ouvert
      return end === 'hang' ? new Promise<void>(() => undefined) : undefined
    },
  })
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } })
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

export function pdfFile(name = 'cv.pdf', size = 1024, type = 'application/pdf'): File {
  return new File([new Uint8Array(size)], name, { type })
}
