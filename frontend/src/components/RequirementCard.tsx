import type { RequirementEntry } from '../state/analysisReducer'
import { LEVEL_LABELS, LEVEL_STYLES, flagExplanations, rejectedSummary } from './labels'

interface RequirementCardProps {
  entry: RequirementEntry
  /** Analyse arrêtée avant cette évaluation (annulation, erreur). */
  interrupted?: boolean
}

export function RequirementCard({ entry, interrupted = false }: RequirementCardProps) {
  const { requirement, result } = entry
  const mandatory = requirement.importance === 'obligatoire'
  const flags = result ? flagExplanations(result) : []
  const rejected = result ? rejectedSummary(result) : null

  return (
    <article
      aria-label={requirement.label}
      aria-busy={result === null && !interrupted}
      className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex min-w-0 flex-col gap-1">
          <h3 className="font-medium text-slate-900">{requirement.label}</h3>
          <p className="text-xs text-slate-500">
            <span className={mandatory ? 'font-medium text-slate-700' : ''}>
              {mandatory ? 'Obligatoire' : 'Souhaitée'}
            </span>
            {' · '}
            {requirement.category}
            {requirement.min_years !== null && ` · ${requirement.min_years} an${requirement.min_years > 1 ? 's' : ''} minimum`}
          </p>
        </div>
        {result === null ? (
          <span className="rounded-full bg-slate-50 px-2.5 py-0.5 text-xs text-slate-500 ring-1 ring-slate-200">
            {interrupted ? 'Non évaluée' : 'En attente…'}
          </span>
        ) : result.status === 'error' || result.level === null ? (
          <span className="rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700 ring-1 ring-red-200">
            Erreur
          </span>
        ) : (
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${LEVEL_STYLES[result.level]}`}>
            {LEVEL_LABELS[result.level]}
          </span>
        )}
      </div>

      {result === null && !interrupted && (
        <div className="flex flex-col gap-2" aria-hidden="true">
          <div className="h-2.5 w-3/4 animate-pulse rounded bg-slate-100" />
          <div className="h-2.5 w-1/2 animate-pulse rounded bg-slate-100" />
        </div>
      )}

      {result !== null && (
        <>
          {result.evidence.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Preuves citées</h4>
              <ul className="flex flex-col gap-1.5">
                {result.evidence.map((evidence) => (
                  <li key={`${evidence.chunk_id}-${evidence.quote}`}>
                    <blockquote className="border-l-2 border-slate-300 pl-3 text-sm text-slate-700">
                      « {evidence.quote} »
                    </blockquote>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.gaps.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Ce qui manque</h4>
              <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
                {result.gaps.map((gap) => (
                  <li key={gap}>{gap}</li>
                ))}
              </ul>
            </div>
          )}

          {(flags.length > 0 || rejected) && (
            <div className="flex flex-col gap-1 rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-700">
              {rejected && (
                <details>
                  <summary className="cursor-pointer select-none">ⓘ {rejected} Voir le texte</summary>
                  <ul className="mt-1.5 flex flex-col gap-1 pl-4">
                    {result.rejected_evidence_items.map((evidence) => (
                      <li key={`${evidence.chunk_id}-${evidence.quote}`} className="text-slate-600">
                        <q className="italic">{evidence.quote}</q>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {flags.length > 0 && (
                <ul className="flex flex-col gap-1">
                  {flags.map((flag) => (
                    <li key={flag}>ⓘ {flag}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </>
      )}
    </article>
  )
}
