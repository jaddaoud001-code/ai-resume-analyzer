import { API_BASE_URL } from '../api/client'
import type { BackendHealth } from '../hooks/useBackendHealth'

const INDICATOR: Record<BackendHealth, { label: string; dot: string }> = {
  checking: { label: 'Vérification du serveur…', dot: 'bg-slate-400' },
  ready: { label: 'Serveur prêt', dot: 'bg-emerald-500' },
  'loading-model': { label: 'Modèle en cours de chargement', dot: 'bg-amber-500 animate-pulse' },
  unreachable: { label: 'Serveur injoignable', dot: 'bg-red-500' },
}

interface HeaderProps {
  health: BackendHealth
  onRetry: () => void
}

export function Header({ health, onRetry }: HeaderProps) {
  const indicator = INDICATOR[health]
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <h1 className="text-lg font-semibold tracking-tight text-slate-900">AI Resume Analyzer</h1>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <span role="status" className="flex items-center gap-1.5" title={`${API_BASE_URL}/health`}>
            <span aria-hidden="true" className={`size-2 rounded-full ${indicator.dot}`} />
            {indicator.label}
          </span>
          {health === 'unreachable' && (
            <button
              type="button"
              onClick={onRetry}
              className="rounded px-1.5 py-0.5 font-medium text-slate-700 underline-offset-2 hover:underline"
            >
              Réessayer
            </button>
          )}
        </div>
      </div>
    </header>
  )
}
