import { useState } from 'react'
import type { AgentErrorCode } from '../api/types'
import { type Decision, type SuggestFailure, type SuggestState, acceptedBullets } from '../state/suggestReducer'
import { formatDuration } from './labels'
import { DescribeCard, RefusalCard, SuggestionCard } from './SuggestionCard'
import { TargetProgress } from './TargetProgress'

export const SUGGESTIONS_EFFECT =
  'Ces reformulations rendent visibles, pour un recruteur ou un tri par mots-clés, des compétences que votre CV prouve déjà. Elles ne changent pas le score : l’analyse reconnaissait déjà ces compétences.'

export const SUGGESTIONS_NOTICE =
  'Les suggestions reformulent uniquement ce que votre CV contient déjà. Relisez chacune avant de l’utiliser.'

const ERROR_TITLES: Record<AgentErrorCode, string> = {
  quota_exceeded: 'Quota journalier du fournisseur d’IA atteint',
  provider_access_denied: 'Le fournisseur d’IA refuse l’accès au modèle',
  timeout: 'Les suggestions ont pris trop de temps',
  internal_error: 'Erreur interne',
}

function errorTitle(error: SuggestFailure): string {
  if (error.kind === 'stream' && error.code) return ERROR_TITLES[error.code]
  if (error.kind === 'http' && error.status === 429) return 'Suggestions impossibles pour le moment'
  if (error.kind === 'http') return 'Suggestions refusées'
  if (error.kind === 'connection') return 'Connexion perdue'
  return 'Erreur'
}

function liveSummary(state: SuggestState): string {
  if (state.status === 'done') {
    const n = state.suggestions.length
    return `Suggestions terminées : ${n} suggestion${n > 1 ? 's' : ''}.`
  }
  if (state.status === 'cancelled') return 'Suggestions annulées.'
  if (state.status === 'error') return 'Suggestions interrompues par une erreur.'
  if (state.waitMessage) return state.waitMessage
  const current = state.targets.find((t) => t.outcome === 'running')
  if (current) {
    const rank = state.targets.indexOf(current) + 1
    return current.rejection
      ? `${current.label} : ${current.rejection}`
      : `Exigence ${rank} sur ${state.targets.length} : ${current.label}.`
  }
  return 'Suggestions en cours.'
}

type CopyState = 'idle' | 'copied' | 'failed'

interface SuggestionsPanelProps {
  state: SuggestState
  elapsedSeconds: number
  onStart: () => void
  onCancel: () => void
  onDecide: (requirementId: string, decision: Decision | null) => void
}

/** Agent de suggestions : déclenchement après l'analyse, progression, résultats. */
export function SuggestionsPanel({ state, elapsedSeconds, onStart, onCancel, onDecide }: SuggestionsPanelProps) {
  const [copy, setCopy] = useState<CopyState>('idle')
  const running = state.status === 'running'
  const started = state.status !== 'idle'
  const accepted = acceptedBullets(state)
  const finished = state.status === 'done' || state.status === 'cancelled' || state.status === 'error'

  async function copyAccepted(): Promise<void> {
    try {
      await navigator.clipboard.writeText(accepted)
      setCopy('copied')
    } catch {
      setCopy('failed')
    }
  }

  return (
    <section aria-labelledby="suggestions-title" className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-2">
        <h2 id="suggestions-title" className="text-base font-semibold">
          Améliorer le CV
        </h2>
        <p className="text-sm text-slate-600">
          L’agent repère les compétences que votre CV prouve déjà sans les nommer, et propose de les rendre explicites
          dans la puce qui les prouve, sans rien inventer. Cela prend quelques minutes.
        </p>
        {!running && (
          <button
            type="button"
            onClick={() => {
              setCopy('idle')
              onStart()
            }}
            className="self-start rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-700"
          >
            Proposer des améliorations du CV
          </button>
        )}
      </div>

      {started && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-900">Exigences ciblées</h3>
            <span className="text-xs tabular-nums text-slate-500" aria-label={`Temps écoulé ${formatDuration(elapsedSeconds)}`}>
              {formatDuration(elapsedSeconds)}
            </span>
          </div>
          {state.targets.length > 0 ? (
            <ol className="flex flex-col gap-2">
              {state.targets.map((panel) => (
                <TargetProgress key={panel.id} panel={panel} />
              ))}
            </ol>
          ) : (
            !running && <p className="text-sm text-slate-600">Aucune compétence prouvée sans être nommée : rien à reformuler.</p>
          )}
          {state.skipped > 0 && (
            <p className="text-xs text-slate-500">
              {state.skipped} autre{state.skipped > 1 ? 's' : ''} exigence{state.skipped > 1 ? 's' : ''} non traitée
              {state.skipped > 1 ? 's' : ''} (limite par exécution).
            </p>
          )}
          {state.waitMessage && (
            <p className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-900 ring-1 ring-amber-200">{state.waitMessage}</p>
          )}
          <p aria-live="polite" className="sr-only">
            {liveSummary(state)}
          </p>
          {running && (
            <button
              type="button"
              onClick={onCancel}
              className="w-full rounded-lg bg-white px-4 py-2 text-sm font-medium text-slate-800 ring-1 ring-slate-300 hover:bg-slate-50"
            >
              Annuler
            </button>
          )}
          {state.status === 'cancelled' && <p className="text-sm text-slate-600">Suggestions annulées.</p>}
        </div>
      )}

      {state.error && (
        <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
          <p className="font-semibold">{errorTitle(state.error)}</p>
          <p className="mt-1">{state.error.message}</p>
        </div>
      )}

      {state.suggestions.length > 0 && (
        <div className="flex flex-col gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Suggestions</h3>
            <p className="text-sm text-slate-600">{SUGGESTIONS_EFFECT}</p>
          </div>
          {state.suggestions.map((suggestion) => (
            <SuggestionCard
              key={suggestion.requirement.id}
              suggestion={suggestion}
              decision={state.decisions[suggestion.requirement.id]}
              onDecide={(decision) => {
                setCopy('idle')
                onDecide(suggestion.requirement.id, decision)
              }}
            />
          ))}
        </div>
      )}

      {state.refusals.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold text-slate-900">Sans suggestion</h3>
          <div className="grid gap-2 sm:grid-cols-2">
            {state.refusals.map((refusal) => (
              <RefusalCard key={`${refusal.requirement.id}:${refusal.reason}`} refusal={refusal} />
            ))}
          </div>
        </div>
      )}

      {state.toDescribe.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold text-slate-900">À décrire vous-même</h3>
          <div className="grid gap-2 sm:grid-cols-2">
            {state.toDescribe.map((refusal) => (
              <DescribeCard key={refusal.requirement.id} refusal={refusal} />
            ))}
          </div>
        </div>
      )}

      {started && (
        <div className="flex flex-col gap-2 border-t border-slate-100 pt-3">
          {state.suggestions.length > 0 && (
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                disabled={accepted === ''}
                onClick={() => void copyAccepted()}
                className="rounded-lg bg-white px-3 py-1.5 text-sm font-medium text-slate-800 ring-1 ring-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Copier les puces acceptées
              </button>
              {copy === 'copied' && <span className="text-xs text-emerald-700">Puces copiées.</span>}
            </div>
          )}
          {copy === 'failed' && (
            <div className="flex flex-col gap-1">
              <p className="text-xs text-red-700">Copie impossible : sélectionnez le texte ci-dessous.</p>
              <textarea readOnly value={accepted} aria-label="Puces acceptées" className="h-24 rounded border border-slate-300 p-2 text-sm" />
            </div>
          )}
          {finished && (
            <p className="text-xs text-slate-500">
              {state.done ? `${state.done.usage.total_tokens.toLocaleString('fr-FR')} tokens · ` : ''}
              durée {formatDuration(elapsedSeconds)}
            </p>
          )}
          <p className="text-xs font-medium text-slate-700">{SUGGESTIONS_NOTICE}</p>
        </div>
      )}
    </section>
  )
}
