// Lecture commune des flux SSE du backend (POST /analyze, POST /suggest) : fetch avec un
// corps FormData, puis ReadableStream (EventSource ne permet pas de POST). Aucune
// reconnexion automatique : une coupure devient une erreur, jamais une nouvelle exécution.

import { API_BASE_URL } from './client'
import { ProtocolError } from './events'
import { SSEParser } from './sse'

/** Réponse HTTP en erreur avant l'ouverture du flux (validation, limite, clé…). */
export class HttpError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'HttpError'
    this.status = status
  }
}

/** Serveur injoignable, ou flux interrompu avant le dernier événement. */
export class ConnectionLostError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ConnectionLostError'
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

async function readErrorMessage(response: Response, defaults: Record<number, string>): Promise<string> {
  const fallback = defaults[response.status] ?? `Erreur du serveur (HTTP ${response.status}).`
  try {
    const body: unknown = await response.json()
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      // Message clair du backend ; les erreurs de validation de FastAPI (422, liste en
      // anglais) gardent le message générique.
      if (typeof body.detail === 'string' && body.detail.trim()) return body.detail
    }
  } catch {
    // Corps absent ou non JSON : message générique.
  }
  return fallback
}

export interface EventStreamOptions<E> {
  path: string
  body: FormData
  signal?: AbortSignal
  /** Conversion d'un message SSE en événement typé (null : type inconnu, ignoré). */
  toEvent: (eventName: string, data: string) => E | null
  /** Dernier événement du flux : la lecture s'arrête après lui. */
  isLast: (event: E) => boolean
  /** Messages des erreurs HTTP sans `detail` lisible. */
  httpMessages: Record<number, string>
  /** Message d'un flux coupé avant son dernier événement. */
  lostMessage: string
}

/**
 * Événements du flux, dans l'ordre reçu ; se termine après le dernier (isLast).
 * Lève HttpError, ConnectionLostError, ProtocolError ou l'AbortError du signal. Annuler le
 * signal, ou arrêter l'itération, ferme la connexion : le backend arrête alors son travail.
 */
export async function* postEventStream<E>(options: EventStreamOptions<E>): AsyncGenerator<E, void, undefined> {
  const { path, body, signal, toEvent, isLast, httpMessages, lostMessage } = options
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      body,
      signal,
      headers: { Accept: 'text/event-stream' },
    })
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) throw error
    throw new ConnectionLostError('Impossible de joindre le serveur. Vérifiez qu’il est démarré.')
  }

  if (!response.ok) throw new HttpError(response.status, await readErrorMessage(response, httpMessages))
  if (!response.body || !(response.headers.get('content-type') ?? '').startsWith('text/event-stream')) {
    throw new ProtocolError()
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = new SSEParser()
  try {
    for (;;) {
      let chunk: ReadableStreamReadResult<Uint8Array>
      try {
        chunk = await reader.read()
      } catch (error) {
        if (signal?.aborted || isAbortError(error)) throw error
        throw new ConnectionLostError(lostMessage)
      }
      const messages = chunk.done
        ? [...parser.feed(decoder.decode()), ...parser.end()]
        : parser.feed(decoder.decode(chunk.value, { stream: true }))
      for (const message of messages) {
        const event = toEvent(message.event, message.data)
        if (event === null) continue
        yield event
        if (isLast(event)) return
      }
      if (chunk.done) break
    }
  } finally {
    // Fin normale, erreur, annulation ou consommateur parti : on ferme la connexion.
    reader.cancel().catch(() => undefined)
  }
  if (signal?.aborted) throw signal.reason instanceof Error ? signal.reason : new DOMException('Annulé', 'AbortError')
  throw new ConnectionLostError(lostMessage)
}
