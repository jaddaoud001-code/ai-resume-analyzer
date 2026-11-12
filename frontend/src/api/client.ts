import type { PublicConfig } from './types'

export const API_BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export interface HealthResponse {
  status: 'ok'
  /** Modèle d'embeddings chargé ; sinon une analyse attendra la fin du chargement. */
  embeddingsReady: boolean
}

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, { signal })
  if (!response.ok) {
    throw new Error(`Health check failed: HTTP ${response.status}`)
  }
  const body: unknown = await response.json()
  if (typeof body !== 'object' || body === null || !('status' in body) || body.status !== 'ok') {
    throw new Error('Health check failed: unexpected response')
  }
  // Absent (ancien backend) : considéré comme prêt.
  const ready = 'embeddings_ready' in body ? body.embeddings_ready !== false : true
  return { status: 'ok', embeddingsReady: ready }
}

/** Valeurs de repli si GET /config ne répond pas (défauts du backend). */
export const FALLBACK_CONFIG: PublicConfig = {
  max_pdf_size_bytes: 5 * 1024 * 1024,
  max_pdf_pages: 10,
  max_job_offer_chars: 15_000,
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
}

/** Limites publiques du backend ; FALLBACK_CONFIG si la route ne répond pas correctement. */
export async function fetchPublicConfig(signal?: AbortSignal): Promise<{ config: PublicConfig; fromServer: boolean }> {
  try {
    const response = await fetch(`${API_BASE_URL}/config`, { signal })
    if (response.ok) {
      const body: unknown = await response.json()
      if (typeof body === 'object' && body !== null) {
        const values = body as Record<string, unknown>
        const { max_pdf_size_bytes, max_pdf_pages, max_job_offer_chars } = values
        if (
          isPositiveInteger(max_pdf_size_bytes) &&
          isPositiveInteger(max_pdf_pages) &&
          isPositiveInteger(max_job_offer_chars)
        ) {
          return { config: { max_pdf_size_bytes, max_pdf_pages, max_job_offer_chars }, fromServer: true }
        }
      }
    }
  } catch (error) {
    if (signal?.aborted) throw error
  }
  return { config: FALLBACK_CONFIG, fromServer: false }
}
