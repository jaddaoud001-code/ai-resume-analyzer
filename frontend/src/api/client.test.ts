import { describe, expect, it, vi } from 'vitest'
import { jsonResponse } from '../test/stream'
import { FALLBACK_CONFIG, checkHealth, fetchPublicConfig } from './client'

describe('fetchPublicConfig', () => {
  it('lit les limites publiques du backend', async () => {
    const limits = { max_pdf_size_bytes: 2_000_000, max_pdf_pages: 3, max_job_offer_chars: 500 }
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse(limits))))

    await expect(fetchPublicConfig()).resolves.toEqual({ config: limits, fromServer: true })
  })

  it.each([
    ['serveur injoignable', () => Promise.reject(new TypeError('Failed to fetch'))],
    ['route absente', () => Promise.resolve(jsonResponse({ detail: 'Not Found' }, 404))],
    ['réponse invalide', () => Promise.resolve(jsonResponse({ max_pdf_size_bytes: 'beaucoup' }))],
  ])('garde les valeurs de repli : %s', async (_name, respond) => {
    vi.stubGlobal('fetch', vi.fn(respond))

    await expect(fetchPublicConfig()).resolves.toEqual({ config: FALLBACK_CONFIG, fromServer: false })
  })
})

describe('checkHealth', () => {
  it('indique si le modèle d’embeddings est prêt', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse({ status: 'ok', embeddings_ready: false }))))

    await expect(checkHealth()).resolves.toEqual({ status: 'ok', embeddingsReady: false })
  })
})
