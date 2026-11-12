import { useEffect, useState } from 'react'
import { FALLBACK_CONFIG, fetchPublicConfig } from '../api/client'
import type { PublicConfig } from '../api/types'

/** Limites du backend (GET /config), avec les valeurs de repli tant qu'elles manquent. */
export function usePublicConfig(): PublicConfig {
  const [config, setConfig] = useState<PublicConfig>(FALLBACK_CONFIG)

  useEffect(() => {
    const controller = new AbortController()
    fetchPublicConfig(controller.signal)
      .then(({ config: loaded }) => {
        if (!controller.signal.aborted) setConfig(loaded)
      })
      .catch(() => undefined) // annulé au démontage
    return () => controller.abort()
  }, [])

  return config
}
