// POST /analyze : envoie le CV et l'offre, puis lit le flux SSE (api/stream.ts).

import { ProtocolError, toAnalysisEvent } from './events'
import { ConnectionLostError, HttpError, isAbortError, postEventStream } from './stream'
import type { AnalysisEvent } from './types'

export { ConnectionLostError, HttpError, ProtocolError, isAbortError }

const DEFAULT_HTTP_MESSAGES: Record<number, string> = {
  400: 'Le fichier envoyé n’est pas un PDF lisible.',
  413: 'Le fichier ou l’offre est trop volumineux.',
  422: 'Le formulaire est incomplet : ajoutez un CV et une offre.',
  429: 'Trop d’analyses en cours, réessayez dans quelques instants.',
  500: 'Erreur du serveur.',
  503: 'Service momentanément indisponible, réessayez plus tard.',
}

/**
 * Événements de l'analyse, dans l'ordre reçu ; se termine après « report » ou « error ».
 * Lève HttpError (réponse en erreur), ConnectionLostError (réseau, flux coupé),
 * ProtocolError (réponse mal formée) ou l'AbortError du signal (annulation). Annuler le
 * signal, ou arrêter l'itération, ferme la connexion : le backend annule alors l'analyse.
 */
export function analyze(file: File, jobOffer: string, signal?: AbortSignal): AsyncGenerator<AnalysisEvent, void, undefined> {
  const body = new FormData()
  body.append('file', file)
  body.append('job_offer', jobOffer)
  return postEventStream({
    path: '/analyze',
    body,
    signal,
    toEvent: toAnalysisEvent,
    isLast: (event) => event.type === 'report' || event.type === 'error',
    httpMessages: DEFAULT_HTTP_MESSAGES,
    lostMessage: 'La connexion au serveur a été interrompue avant la fin de l’analyse.',
  })
}
