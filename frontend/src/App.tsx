import { useState } from 'react'
import { AnalysisForm } from './components/AnalysisForm'
import { ErrorBanner } from './components/ErrorBanner'
import { Header } from './components/Header'
import { ProgressPanel } from './components/ProgressPanel'
import { RequirementList } from './components/RequirementList'
import { ResultSummary } from './components/ResultSummary'
import { SuggestionsPanel } from './components/SuggestionsPanel'
import { type AnalyzeFn, useAnalysis } from './hooks/useAnalysis'
import { useBackendHealth } from './hooks/useBackendHealth'
import { usePublicConfig } from './hooks/usePublicConfig'
import { type SuggestFn, useSuggestions } from './hooks/useSuggestions'

interface AppProps {
  /** Injectables pour les tests ; par défaut, POST /analyze et POST /suggest. */
  analyzeFn?: AnalyzeFn
  suggestFn?: SuggestFn
}

/** CV et offre de la dernière analyse, renvoyés à /suggest (gardés en mémoire seulement). */
interface Submission {
  file: File
  jobOffer: string
}

function App({ analyzeFn, suggestFn }: AppProps) {
  const { health, retry } = useBackendHealth()
  const config = usePublicConfig()
  const { state, start, cancel, elapsedSeconds } = useAnalysis(analyzeFn)
  const suggestions = useSuggestions(suggestFn)
  const [submission, setSubmission] = useState<Submission | null>(null)
  const running = state.status === 'running'
  const showProgress = state.status !== 'idle' && !(state.error?.kind === 'http')
  const report = state.report
  const canSuggest = state.status === 'done' && report !== null && submission !== null

  function submit(file: File, jobOffer: string): void {
    // Nouvelle analyse : les suggestions de la précédente sont arrêtées et oubliées.
    suggestions.reset()
    setSubmission({ file, jobOffer })
    void start(file, jobOffer)
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <Header health={health} onRetry={retry} />
      <main className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
          <section aria-label="Nouvelle analyse" className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div>
              <h2 className="text-base font-semibold">Analyser un CV</h2>
              <p className="text-sm text-slate-600">
                Déposez votre CV et collez l’offre : chaque exigence est vérifiée dans le CV, preuves à l’appui.
              </p>
            </div>
            <AnalysisForm config={config} busy={running || suggestions.state.status === 'running'} onSubmit={submit} />
            <p className="text-xs text-slate-500">
              Le CV n’est pas conservé ; les coordonnées sont masquées avant l’envoi au fournisseur d’IA.
            </p>
          </section>

          <div className="flex flex-col gap-4">
            {state.error && <ErrorBanner error={state.error} />}
            {showProgress && <ProgressPanel state={state} elapsedSeconds={elapsedSeconds} onCancel={cancel} />}
            {state.report && <ResultSummary report={state.report} />}
          </div>
        </div>

        {canSuggest && (
          <SuggestionsPanel
            state={suggestions.state}
            elapsedSeconds={suggestions.elapsedSeconds}
            onStart={() => void suggestions.start(submission.file, submission.jobOffer, report)}
            onCancel={suggestions.cancel}
            onDecide={suggestions.decide}
          />
        )}

        <RequirementList entries={state.requirements} interrupted={state.status === 'cancelled' || state.status === 'error'} />
      </main>
    </div>
  )
}

export default App
