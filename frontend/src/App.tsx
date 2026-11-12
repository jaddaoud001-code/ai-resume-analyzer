import { AnalysisForm } from './components/AnalysisForm'
import { ErrorBanner } from './components/ErrorBanner'
import { Header } from './components/Header'
import { ProgressPanel } from './components/ProgressPanel'
import { RequirementList } from './components/RequirementList'
import { ResultSummary } from './components/ResultSummary'
import { type AnalyzeFn, useAnalysis } from './hooks/useAnalysis'
import { useBackendHealth } from './hooks/useBackendHealth'
import { usePublicConfig } from './hooks/usePublicConfig'

interface AppProps {
  /** Injectable pour les tests ; par défaut, POST /analyze. */
  analyzeFn?: AnalyzeFn
}

function App({ analyzeFn }: AppProps) {
  const { health, retry } = useBackendHealth()
  const config = usePublicConfig()
  const { state, start, cancel, elapsedSeconds } = useAnalysis(analyzeFn)
  const running = state.status === 'running'
  const showProgress = state.status !== 'idle' && !(state.error?.kind === 'http')

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
            <AnalysisForm config={config} busy={running} onSubmit={(file, offer) => void start(file, offer)} />
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

        <RequirementList entries={state.requirements} interrupted={state.status === 'cancelled' || state.status === 'error'} />
      </main>
    </div>
  )
}

export default App
