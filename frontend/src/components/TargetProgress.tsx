import type { AgentStepKind } from '../api/types'
import { STEP_ORDER, type StepState, type TargetPanel } from '../state/suggestReducer'
import { distinctTerm } from './labels'

const STEP_LABELS: Record<AgentStepKind, string> = {
  search: 'Recherche',
  propose: 'Proposition',
  verify: 'Vérification',
  reevaluate: 'Réévaluation',
}

const STEP_STYLES: Record<StepState, { className: string; label: string }> = {
  pending: { className: 'bg-slate-100 text-slate-500', label: 'à venir' },
  active: { className: 'bg-sky-100 text-sky-800 animate-pulse', label: 'en cours' },
  done: { className: 'bg-emerald-100 text-emerald-800', label: 'terminée' },
  rejected: { className: 'bg-red-50 text-red-700', label: 'sans suite' },
}

const OUTCOME_TEXT: Record<TargetPanel['outcome'], string | null> = {
  waiting: 'En attente',
  running: null,
  suggestion: 'Suggestion proposée',
  refusal: 'Aucune suggestion',
  hidden: 'Aucune suggestion retenue',
}

/** Une exigence ciblée : frise Recherche → Proposition → Vérification → Réévaluation. */
export function TargetProgress({ panel }: { panel: TargetPanel }) {
  const outcome = OUTCOME_TEXT[panel.outcome]
  const term = distinctTerm(panel)
  return (
    <li className="flex flex-col gap-2 rounded-lg border border-slate-200 p-3" aria-label={`Exigence ciblée : ${panel.label}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium text-slate-900">
          {panel.label}
          {term && <span className="font-normal text-slate-500"> · « {term} »</span>}
        </p>
        {outcome && <p className="text-xs text-slate-500">{outcome}</p>}
      </div>
      <ol className="flex flex-wrap items-center gap-1 text-xs">
        {STEP_ORDER.map((kind, index) => {
          const state = STEP_STYLES[panel.steps[kind]]
          return (
            <li key={kind} className="flex items-center gap-1">
              {index > 0 && (
                <span aria-hidden="true" className="text-slate-300">
                  →
                </span>
              )}
              <span className={`rounded-full px-2 py-0.5 ${state.className}`} data-step={kind} data-state={panel.steps[kind]}>
                {STEP_LABELS[kind]}
                <span className="sr-only"> ({state.label})</span>
              </span>
            </li>
          )
        })}
      </ol>
      {panel.rejection && (
        <p className="rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-900 ring-1 ring-amber-200 transition-opacity">
          {panel.rejection}
        </p>
      )}
      {panel.rejections > 0 && !panel.rejection && (
        <p className="text-xs text-slate-500">
          {panel.rejections} proposition{panel.rejections > 1 ? 's' : ''} écartée{panel.rejections > 1 ? 's' : ''} par le
          contrôle.
        </p>
      )}
    </li>
  )
}
