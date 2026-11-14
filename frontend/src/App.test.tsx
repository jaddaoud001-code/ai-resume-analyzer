import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'
import { SAMPLES, jsonResponse, pdfFile, sample, splitEvery, streamResponse, toSSE } from './test/stream'

type Route = (init?: RequestInit) => Promise<Response>

/** fetch simulé par route ; aucune requête réseau. */
function stubBackend(routes: { health?: Route; config?: Route; analyze?: Route }) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    const path = new URL(url).pathname
    const route = path === '/health' ? routes.health : path === '/config' ? routes.config : path === '/analyze' ? routes.analyze : undefined
    return route ? route(init) : Promise.reject(new TypeError('Failed to fetch'))
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const healthy: Route = () => Promise.resolve(jsonResponse({ status: 'ok', embeddings_ready: true }))
const limits: Route = () =>
  Promise.resolve(jsonResponse({ max_pdf_size_bytes: 5_242_880, max_pdf_pages: 10, max_job_offer_chars: 15_000 }))

async function submit(): Promise<void> {
  const user = userEvent.setup()
  await user.upload(screen.getByLabelText('CV (PDF)'), pdfFile())
  await user.type(screen.getByLabelText('Offre d’emploi'), 'Développeuse backend Python')
  await user.click(screen.getByRole('button', { name: 'Analyser' }))
}

describe('indicateur du serveur', () => {
  it('affiche « Serveur prêt » quand /health répond avec le modèle chargé', async () => {
    stubBackend({ health: healthy, config: limits })
    render(<App />)

    expect(await screen.findByText('Serveur prêt')).toBeInTheDocument()
  })

  it('affiche « Modèle en cours de chargement » et laisse le formulaire utilisable', async () => {
    stubBackend({ health: () => Promise.resolve(jsonResponse({ status: 'ok', embeddings_ready: false })), config: limits })
    render(<App />)

    expect(await screen.findByText('Modèle en cours de chargement')).toBeInTheDocument()
    expect(screen.getByLabelText('Offre d’emploi')).toBeEnabled()
  })

  it('propose de réessayer quand le serveur est injoignable', async () => {
    const fetchMock = stubBackend({ config: limits })
    render(<App />)

    expect(await screen.findByText('Serveur injoignable')).toBeInTheDocument()
    fetchMock.mockImplementation((url: string) =>
      new URL(url).pathname === '/health' ? healthy() : limits(),
    )
    await userEvent.click(screen.getByRole('button', { name: 'Réessayer' }))
    expect(await screen.findByText('Serveur prêt')).toBeInTheDocument()
  })
})

describe('limites partagées', () => {
  it('utilise les limites de GET /config', async () => {
    stubBackend({
      health: healthy,
      config: () => Promise.resolve(jsonResponse({ max_pdf_size_bytes: 1_048_576, max_pdf_pages: 2, max_job_offer_chars: 300 })),
    })
    render(<App />)

    expect(await screen.findByText('0 / 300 caractères')).toBeInTheDocument()
    expect(screen.getByText('PDF de 1 Mo et 2 pages au plus.')).toBeInTheDocument()
  })

  it('garde les valeurs de repli si /config ne répond pas', async () => {
    const fetchMock = stubBackend({ health: healthy }) // /config : échec réseau
    render(<App />)

    await screen.findByText('Serveur prêt')
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/config$/), expect.anything()))
    expect(screen.getByText('0 / 15 000 caractères')).toBeInTheDocument()
    expect(screen.getByText('PDF de 5 Mo et 10 pages au plus.')).toBeInTheDocument()
  })
})

describe('analyse', () => {
  it('affiche le message du backend sur une erreur HTTP 429', async () => {
    stubBackend({
      health: healthy,
      config: limits,
      analyze: () =>
        Promise.resolve(jsonResponse({ detail: 'Trop d\'analyses en cours, réessayez dans quelques instants.' }, 429)),
    })
    render(<App />)

    await submit()

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Analyse impossible pour le moment')
    expect(alert).toHaveTextContent('Trop d\'analyses en cours, réessayez dans quelques instants.')
    expect(screen.getByRole('button', { name: 'Analyser' })).toBeEnabled()
  })

  it('remplit les cartes puis affiche le rapport à partir du flux', async () => {
    const events = SAMPLES.filter((e) => e.type !== 'error')
    stubBackend({
      health: healthy,
      config: limits,
      analyze: () => Promise.resolve(streamResponse(splitEvery(toSSE(events, { ping: true }), 50))),
    })
    render(<App />)

    await submit()

    const missing = await screen.findByRole('region', { name: /Exigences obligatoires non démontrées/ })
    expect(within(missing).getByText('Go')).toBeInTheDocument()
    expect(screen.getByText(String(sample('report').report.score))).toBeInTheDocument()
    expect(screen.getAllByRole('article')).toHaveLength(4)
    expect(screen.queryByText('En attente…')).not.toBeInTheDocument()
    expect(screen.getByText('Analyse terminée.')).toBeInTheDocument() // zone aria-live
  })

  it('annule l’analyse en cours avec le bouton Annuler', async () => {
    stubBackend({
      health: healthy,
      config: limits,
      analyze: (init) =>
        Promise.resolve(
          streamResponse([toSSE([sample('started'), sample('cv_indexed'), sample('requirements_extracted'), sample('status')])], {
            end: 'hang',
            signal: init?.signal,
          }),
        ),
    })
    render(<App />)
    await submit()
    // Message visible, et répété dans la zone aria-live pour les lecteurs d'écran.
    expect(await screen.findAllByText('En attente du quota du fournisseur, reprise dans 60 s')).toHaveLength(2)
    expect(screen.getAllByText('En attente…')).toHaveLength(4)

    await userEvent.click(screen.getByRole('button', { name: 'Annuler' }))

    expect(await screen.findAllByText('Analyse annulée.')).not.toHaveLength(0)
    expect(screen.getAllByText('Non évaluée')).toHaveLength(4)
    expect(screen.getByRole('button', { name: 'Analyser' })).toBeEnabled()
  })
})
