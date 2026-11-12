import { useCallback, useEffect, useState } from 'react'
import { checkHealth } from '../api/client'

export type BackendHealth = 'checking' | 'ready' | 'loading-model' | 'unreachable'

/** Délai entre deux vérifications tant que le modèle se charge. */
export const MODEL_POLL_MS = 2000

/** État du backend ; pendant le chargement du modèle, revérifie jusqu'à ce qu'il soit prêt. */
export function useBackendHealth(pollMs: number = MODEL_POLL_MS) {
  const [health, setHealth] = useState<BackendHealth>('checking')
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined

    const check = (): void => {
      checkHealth(controller.signal)
        .then(({ embeddingsReady }) => {
          if (controller.signal.aborted) return
          setHealth(embeddingsReady ? 'ready' : 'loading-model')
          if (!embeddingsReady) timer = setTimeout(check, pollMs)
        })
        .catch(() => {
          if (!controller.signal.aborted) setHealth('unreachable')
        })
    }
    check()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [attempt, pollMs])

  const retry = useCallback(() => {
    setHealth('checking')
    setAttempt((n) => n + 1)
  }, [])

  return { health, retry }
}
