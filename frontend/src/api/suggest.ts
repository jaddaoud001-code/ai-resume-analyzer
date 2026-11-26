// POST /suggest : envoie le même CV, l'offre et le rapport reçu de /analyze (rien n'est
// conservé par le serveur entre les deux appels), puis lit le flux des événements de l'agent.

import { toSuggestEvent } from './events'
import { postEventStream } from './stream'
import type { MatchReport, SuggestEvent } from './types'

const DEFAULT_HTTP_MESSAGES: Record<number, string> = {
  400: 'Le fichier envoyé n’est pas un PDF lisible.',
  413: 'Le CV, l’offre ou le rapport est trop volumineux.',
  422: 'Le rapport d’analyse ne correspond pas à ce CV ou à cette offre : relancez l’analyse.',
  429: 'Trop d’exécutions en cours, réessayez dans quelques instants.',
  500: 'Erreur du serveur.',
  503: 'Service momentanément indisponible, réessayez plus tard.',
}

/**
 * Événements de l'agent de suggestions, dans l'ordre reçu ; se termine après « done » ou
 * « error ». Mêmes erreurs que analyze() ; annuler le signal ferme la connexion, et le
 * backend arrête l'agent (aucun nouvel appel au LLM).
 */
export function suggest(
  file: File,
  jobOffer: string,
  report: MatchReport,
  signal?: AbortSignal,
): AsyncGenerator<SuggestEvent, void, undefined> {
  const body = new FormData()
  body.append('file', file)
  body.append('job_offer', jobOffer)
  body.append('report', JSON.stringify(report))
  return postEventStream({
    path: '/suggest',
    body,
    signal,
    toEvent: toSuggestEvent,
    isLast: (event) => event.type === 'done' || event.type === 'error',
    httpMessages: DEFAULT_HTTP_MESSAGES,
    lostMessage: 'La connexion au serveur a été interrompue avant la fin des suggestions.',
  })
}
