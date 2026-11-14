import type { AnalysisFailure } from '../state/analysisReducer'
import { ERROR_CODE_TITLES } from './labels'

function title(error: AnalysisFailure): string {
  if (error.kind === 'stream' && error.code) return ERROR_CODE_TITLES[error.code]
  // 429 : trop d'analyses en cours, ou quota journalier du fournisseur ; le message précise.
  if (error.kind === 'http' && error.status === 429) return 'Analyse impossible pour le moment'
  if (error.kind === 'http') return 'Analyse refusée'
  if (error.kind === 'connection') return 'Connexion perdue'
  return 'Erreur'
}

export function ErrorBanner({ error }: { error: AnalysisFailure }) {
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
      <p className="font-semibold">{title(error)}</p>
      <p className="mt-1">{error.message}</p>
      {error.kind === 'connection' && (
        <p className="mt-1 text-xs text-red-800">L’analyse n’est pas relancée automatiquement : relancez-la si besoin.</p>
      )}
    </div>
  )
}
