import type { MatchReport } from '../api/types'
import { mandatoryCoverage } from './coverage'

interface ResultSummaryProps {
  report: MatchReport
}

function scoreColor(score: number): string {
  if (score >= 70) return 'text-emerald-700'
  if (score >= 40) return 'text-amber-700'
  return 'text-slate-800'
}

export function ResultSummary({ report }: ResultSummaryProps) {
  const missing = report.missing_mandatory
  const coverage = mandatoryCoverage(report)
  return (
    <section aria-labelledby="result-title" className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="result-title" className="text-sm font-semibold text-slate-900">
            Résultat
          </h2>
          <p className="text-xs text-slate-500">{report.job_title}</p>
        </div>
        <div className="flex flex-wrap items-end gap-x-5 gap-y-2">
          {coverage.total > 0 && (
            <p className="flex flex-col items-end" aria-label={`Obligatoires : ${coverage.demonstrated} sur ${coverage.total} démontrées`}>
              <span className="text-xs text-slate-500">Obligatoires</span>
              <span
                className={`text-lg font-semibold tabular-nums ${
                  coverage.demonstrated === coverage.total ? 'text-emerald-700' : 'text-slate-900'
                }`}
              >
                {coverage.demonstrated}/{coverage.total} démontrées
              </span>
            </p>
          )}
          <p className="flex items-baseline gap-1">
            {report.score === null ? (
              <span className="text-sm text-slate-600">Score indisponible</span>
            ) : (
              <>
                <span className={`text-4xl font-semibold tabular-nums ${scoreColor(report.score)}`}>{report.score}</span>
                <span className="text-sm text-slate-500">/ 100</span>
              </>
            )}
          </p>
        </div>
      </div>
      {coverage.total > 0 && (
        <p className="-mt-2 text-xs text-slate-500">
          Une obligatoire est démontrée quand le CV la prouve par une expérience, un projet, une formation ou une
          certification (niveau « démontré » ou « solide ») ; une simple mention ne suffit pas.
          {coverage.notEvaluated > 0 &&
            ` ${coverage.notEvaluated} ${coverage.notEvaluated > 1 ? 'n’ont pas pu être évaluées' : 'n’a pas pu être évaluée'}.`}
        </p>
      )}
      <p className="text-xs text-slate-500">
        Le score varie d’environ ±5 points d’une analyse à l’autre : comparez surtout les exigences.
      </p>

      {missing.length > 0 ? (
        <div role="region" aria-labelledby="missing-title" className="rounded-lg border border-red-200 bg-red-50 p-4">
          <h3 id="missing-title" className="text-sm font-semibold text-red-900">
            Exigences obligatoires non démontrées ({missing.length})
          </h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-red-900">
            {missing.map((requirement) => (
              <li key={requirement.id}>{requirement.label}</li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-red-800">
            Si vous les maîtrisez, décrivez-les dans une expérience ou un projet de votre CV.
          </p>
        </div>
      ) : (
        <p className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
          Toutes les exigences obligatoires sont au moins mentionnées dans le CV.
        </p>
      )}

      {report.errors.length > 0 && (
        <p className="text-xs text-slate-600">
          {report.errors.length} exigence{report.errors.length > 1 ? 's n’ont' : ' n’a'} pas pu être évaluée
          {report.errors.length > 1 ? 's' : ''} et {report.errors.length > 1 ? 'sont exclues' : 'est exclue'} du score.
        </p>
      )}
    </section>
  )
}
