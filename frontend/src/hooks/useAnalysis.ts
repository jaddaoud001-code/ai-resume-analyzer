import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { ConnectionLostError, HttpError, ProtocolError, analyze } from '../api/analyze'
import type { AnalysisEvent } from '../api/types'
import { type AnalysisFailure, analysisReducer, initialAnalysisState } from '../state/analysisReducer'

export type AnalyzeFn = (file: File, jobOffer: string, signal: AbortSignal) => AsyncIterable<AnalysisEvent>

export function describeFailure(error: unknown): AnalysisFailure {
  if (error instanceof HttpError) return { kind: 'http', status: error.status, message: error.message }
  if (error instanceof ConnectionLostError) return { kind: 'connection', message: error.message }
  if (error instanceof ProtocolError) return { kind: 'protocol', message: error.message }
  return { kind: 'unknown', message: 'Erreur inattendue pendant l’analyse.' }
}

/** Lance une analyse, la suit événement par événement et permet de l'annuler. */
export function useAnalysis(run: AnalyzeFn = analyze) {
  const [state, dispatch] = useReducer(analysisReducer, initialAnalysisState)
  const controllerRef = useRef<AbortController | null>(null)
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const running = state.status === 'running'

  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => {
      clearInterval(timer)
      setNow(Date.now()) // fige le temps écoulé à la fin de l'analyse
    }
  }, [running])

  // Page quittée pendant une analyse : la connexion est fermée.
  useEffect(() => () => controllerRef.current?.abort(), [])

  const start = useCallback(
    async (file: File, jobOffer: string) => {
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller
      dispatch({ type: 'submit' })
      setStartedAt(Date.now())
      setNow(Date.now())
      try {
        for await (const event of run(file, jobOffer, controller.signal)) {
          dispatch({ type: 'event', event })
        }
      } catch (error) {
        if (controller.signal.aborted) dispatch({ type: 'cancelled' })
        else dispatch({ type: 'failed', error: describeFailure(error) })
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null
      }
    },
    [run],
  )

  const cancel = useCallback(() => controllerRef.current?.abort(), [])

  const elapsedSeconds = startedAt === null ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000))
  return { state, start, cancel, elapsedSeconds }
}
