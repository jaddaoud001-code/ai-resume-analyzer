import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { suggest } from '../api/suggest'
import type { MatchReport, SuggestEvent } from '../api/types'
import { type Decision, initialSuggestState, suggestReducer } from '../state/suggestReducer'
import { describeFailure } from './useAnalysis'

export type SuggestFn = (
  file: File,
  jobOffer: string,
  report: MatchReport,
  signal: AbortSignal,
) => AsyncIterable<SuggestEvent>

/** Lance l'agent de suggestions, le suit événement par événement et permet de l'annuler. */
export function useSuggestions(run: SuggestFn = suggest) {
  const [state, dispatch] = useReducer(suggestReducer, initialSuggestState)
  const controllerRef = useRef<AbortController | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const running = state.status === 'running'

  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => {
      clearInterval(timer)
      setNow(Date.now()) // fige la durée à la fin
    }
  }, [running])

  // Page quittée pendant l'exécution : la connexion est fermée, le serveur arrête l'agent.
  useEffect(() => () => controllerRef.current?.abort(), [])

  const start = useCallback(
    async (file: File, jobOffer: string, report: MatchReport) => {
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller
      dispatch({ type: 'submit' })
      setStartedAt(Date.now())
      setNow(Date.now())
      try {
        for await (const event of run(file, jobOffer, report, controller.signal)) {
          dispatch({ type: 'event', event })
        }
      } catch (error) {
        if (controller.signal.aborted) dispatch({ type: 'cancelled' })
        else {
          // Erreurs avant ou hors du flux (HTTP, connexion, protocole) : jamais de code d'agent.
          const { kind, message, status } = describeFailure(error)
          dispatch({
            type: 'failed',
            error: { kind, status, message: kind === 'unknown' ? 'Erreur inattendue pendant les suggestions.' : message },
          })
        }
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null
      }
    },
    [run],
  )

  const cancel = useCallback(() => controllerRef.current?.abort(), [])

  /** Nouvelle analyse : l'exécution en cours est arrêtée et les suggestions oubliées. */
  const reset = useCallback(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
    dispatch({ type: 'reset' })
    setStartedAt(null)
  }, [])

  const decide = useCallback(
    (requirementId: string, decision: Decision | null) => dispatch({ type: 'decide', requirementId, decision }),
    [],
  )

  const elapsedSeconds = startedAt === null ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000))
  return { state, start, cancel, reset, decide, elapsedSeconds }
}
