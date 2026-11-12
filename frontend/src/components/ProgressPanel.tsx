import type { AnalysisPhase } from '../api/types'
import type { AnalysisState, PhaseState } from '../state/analysisReducer'
import { formatDuration } from './labels'

const PHASE_LABELS: Record<AnalysisPhase, string> = {
  indexing: 'Indexation du CV',
  extraction: 'Extraction des exigences',
  evaluation: 'Évaluation des exigences',
}

const PHASE_ICONS: Record<PhaseState, { symbol: string; className: string; label: string }> = {
  pending: { symbol: '○', className: 'text-slate-400', label: 'à venir' },
  active: { symbol: '◐', className: 'text-sky-600 animate-pulse', label: 'en cours' },
  done: { symbol: '●', className: 'text-emerald-600', label: 'terminé' },
}

function liveSummary(state: AnalysisState): string {
  if (state.status === 'done') return 'Analyse terminée.'
  if (state.status === 'cancelled') return 'Analyse annulée.'
  if (state.status === 'error') return 'Analyse interrompue par une erreur.'
  if (state.waitMessage) return state.waitMessage
  if (state.phases.evaluation !== 'pending' && state.total !== null) {
    return `Évaluation : ${state.evaluated} exigence${state.evaluated > 1 ? 's' : ''} sur ${state.total}.`
  }
  return 'Analyse en cours : indexation du CV et extraction des exigences.'
}

interface ProgressPanelProps {
  state: AnalysisState
  elapsedSeconds: number
  onCancel: () => void
}

export function ProgressPanel({ state, elapsedSeconds, onCancel }: ProgressPanelProps) {
  const running = state.status === 'running'
  return (
    <section aria-labelledby="progress-title" className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h2 id="progress-title" className="text-sm font-semibold text-slate-900">
          Progression
        </h2>
        <span className="text-xs tabular-nums text-slate-500" aria-label={`Temps écoulé ${formatDuration(elapsedSeconds)}`}>
          {formatDuration(elapsedSeconds)}
        </span>
      </div>

      <ol className="mt-3 flex flex-col gap-2 text-sm">
        {(Object.keys(PHASE_LABELS) as AnalysisPhase[]).map((phase) => {
          const icon = PHASE_ICONS[state.phases[phase]]
          const counter =
            phase === 'evaluation' && state.total !== null ? ` ${state.evaluated}/${state.total}` : ''
          return (
            <li key={phase} className="flex items-center gap-2">
              <span aria-hidden="true" className={icon.className}>
                {icon.symbol}
              </span>
              <span className={state.phases[phase] === 'pending' ? 'text-slate-500' : 'text-slate-800'}>
                {PHASE_LABELS[phase]}
                <span className="tabular-nums">{counter}</span>
              </span>
              <span className="sr-only">({icon.label})</span>
            </li>
          )
        })}
      </ol>

      {state.total !== null && state.total > 0 && (
        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
          <div
            className="h-full rounded-full bg-sky-500 transition-[width] duration-500"
            style={{ width: `${Math.round((100 * state.evaluated) / state.total)}%` }}
          />
        </div>
      )}

      {state.waitMessage && (
        <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-900 ring-1 ring-amber-200">
          {state.waitMessage}
        </p>
      )}

      <p aria-live="polite" className="sr-only">
        {liveSummary(state)}
      </p>

      {running && (
        <button
          type="button"
          onClick={onCancel}
          className="mt-4 w-full rounded-lg bg-white px-4 py-2 text-sm font-medium text-slate-800 ring-1 ring-slate-300 hover:bg-slate-50"
        >
          Annuler
        </button>
      )}
      {state.status === 'cancelled' && <p className="mt-3 text-sm text-slate-600">Analyse annulée.</p>}
    </section>
  )
}
