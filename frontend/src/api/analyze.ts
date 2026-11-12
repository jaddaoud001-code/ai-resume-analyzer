// POST /analyze : envoie le CV et l'offre, puis lit le flux SSE avec fetch et un
// ReadableStream (EventSource ne permet pas de POST). Aucune reconnexion automatique :
// une coupure devient une erreur, jamais une nouvelle analyse.

import { API_BASE_URL } from './client'
import { ProtocolError, toAnalysisEvent } from './events'
import { SSEParser } from './sse'
import type { AnalysisEvent } from './types'

export { ProtocolError }

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

const DEFAULT_HTTP_MESSAGES: Record<number, string> = {
  400: 'Le fichier envoyé n’est pas un PDF lisible.',
  413: 'Le fichier ou l’offre est trop volumineux.',
  422: 'Le formulaire est incomplet : ajoutez un CV et une offre.',
  429: 'Trop d’analyses en cours, réessayez dans quelques instants.',
  500: 'Erreur du serveur.',
  503: 'Service momentanément indisponible, réessayez plus tard.',
}

async function readErrorMessage(response: Response): Promise<string> {
  const fallback = DEFAULT_HTTP_MESSAGES[response.status] ?? `Erreur du serveur (HTTP ${response.status}).`
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

const LOST = 'La connexion au serveur a été interrompue avant la fin de l’analyse.'

/**
 * Événements de l'analyse, dans l'ordre reçu ; se termine après « report » ou « error ».
 * Lève HttpError (réponse en erreur), ConnectionLostError (réseau, flux coupé),
 * ProtocolError (réponse mal formée) ou l'AbortError du signal (annulation). Annuler le
 * signal, ou arrêter l'itération, ferme la connexion : le backend annule alors l'analyse.
 */
export async function* analyze(
  file: File,
  jobOffer: string,
  signal?: AbortSignal,
): AsyncGenerator<AnalysisEvent, void, undefined> {
  const body = new FormData()
  body.append('file', file)
  body.append('job_offer', jobOffer)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body,
      signal,
      headers: { Accept: 'text/event-stream' },
    })
  } catch (error) {
    if (signal?.aborted || isAbortError(error)) throw error
    throw new ConnectionLostError('Impossible de joindre le serveur. Vérifiez qu’il est démarré.')
  }

  if (!response.ok) throw new HttpError(response.status, await readErrorMessage(response))
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
        throw new ConnectionLostError(LOST)
      }
      const messages = chunk.done
        ? [...parser.feed(decoder.decode()), ...parser.end()]
        : parser.feed(decoder.decode(chunk.value, { stream: true }))
      for (const message of messages) {
        const event = toAnalysisEvent(message.event, message.data)
        if (event === null) continue
        yield event
        if (event.type === 'report' || event.type === 'error') return
      }
      if (chunk.done) break
    }
  } finally {
    // Fin normale, erreur, annulation ou consommateur parti : on ferme la connexion.
    reader.cancel().catch(() => undefined)
  }
  if (signal?.aborted) throw signal.reason instanceof Error ? signal.reason : new DOMException('Annulé', 'AbortError')
  throw new ConnectionLostError(LOST)
}
