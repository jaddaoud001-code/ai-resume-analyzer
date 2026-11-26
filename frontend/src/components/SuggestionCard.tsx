import type { RefusalEvent, SuggestionEvent } from '../api/types'
import type { Decision } from '../state/suggestReducer'
import { wordDiff } from './diff'
import { REFUSAL_TITLES } from './labels'

const DECISION_STYLES: Record<Decision, string> = {
  accepted: 'border-emerald-300 ring-1 ring-emerald-200',
  refused: 'border-slate-200 opacity-70',
}

interface SuggestionCardProps {
  suggestion: SuggestionEvent
  decision: Decision | undefined
  onDecide: (decision: Decision | null) => void
}

/**
 * Une suggestion : puce d'origine, puce proposée (mots ajoutés surlignés), citations sources
 * exactes, implication utilisée, niveau indicatif. Jamais la paraphrase ni la justification
 * du modèle : seules les citations du CV appuient la proposition.
 */
export function SuggestionCard({ suggestion, decision, onDecide }: SuggestionCardProps) {
  const { requirement } = suggestion
  const parts = wordDiff(suggestion.before, suggestion.after)
  const toggle = (value: Decision) => onDecide(decision === value ? null : value)

  return (
    <article
      aria-label={`Suggestion : ${requirement.label}`}
      className={`flex flex-col gap-3 rounded-xl border bg-white p-4 shadow-sm ${decision ? DECISION_STYLES[decision] : 'border-slate-200'}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 className="font-medium text-slate-900">{requirement.label}</h4>
          <p className="text-xs text-slate-500">
            {suggestion.item}
            {suggestion.terms.length > 0 && <> · terme de l’offre : {suggestion.terms.join(', ')}</>}
          </p>
        </div>
        <p className="text-xs text-slate-600">
          Niveau {suggestion.level_before} → {suggestion.level_after}{' '}
          <span className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-600">indicatif</span>
        </p>
      </div>

      <div className="grid gap-2 text-sm sm:grid-cols-2">
        <div className="rounded-lg bg-slate-50 p-3">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Puce d’origine</p>
          <p className="text-slate-700">{suggestion.before}</p>
        </div>
        <div className="rounded-lg bg-sky-50 p-3">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-sky-700">Puce proposée</p>
          <p className="text-slate-900" data-testid="proposed-bullet">
            {parts.map((part, index) =>
              part.added ? (
                <mark key={index} className="rounded bg-amber-200 px-0.5 text-slate-900">
                  {part.text}
                </mark>
              ) : (
                <span key={index}>{part.text}</span>
              ),
            )}
          </p>
        </div>
      </div>

      {suggestion.facts.length > 0 && (
        <div className="flex flex-col gap-1">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Ce que dit déjà votre CV</p>
          <ul className="flex flex-col gap-1">
            {suggestion.facts.map((fact) => (
              <li key={`${fact.chunk_id}:${fact.quote}`}>
                <blockquote className="border-l-2 border-slate-300 pl-2 text-sm text-slate-700">« {fact.quote} »</blockquote>
              </li>
            ))}
          </ul>
        </div>
      )}

      {suggestion.implied_skills.length > 0 && (
        <p className="text-xs text-slate-600">
          Implication utilisée :{' '}
          {suggestion.implied_skills.map((skill, index) => (
            <span key={`${skill.skill}:${skill.quote}`}>
              {index > 0 && ' ; '}
              {skill.skill} découle de « {skill.quote} »
            </span>
          ))}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2" role="group" aria-label={`Décision pour ${requirement.label}`}>
        <button
          type="button"
          aria-pressed={decision === 'accepted'}
          onClick={() => toggle('accepted')}
          className={`rounded-lg px-3 py-1.5 text-sm font-medium ring-1 ${decision === 'accepted' ? 'bg-emerald-600 text-white ring-emerald-600' : 'bg-white text-emerald-800 ring-emerald-300 hover:bg-emerald-50'}`}
        >
          Accepter
        </button>
        <button
          type="button"
          aria-pressed={decision === 'refused'}
          onClick={() => toggle('refused')}
          className={`rounded-lg px-3 py-1.5 text-sm font-medium ring-1 ${decision === 'refused' ? 'bg-slate-700 text-white ring-slate-700' : 'bg-white text-slate-700 ring-slate-300 hover:bg-slate-50'}`}
        >
          Refuser
        </button>
        {decision && (
          <span className="text-xs text-slate-500">{decision === 'accepted' ? 'Acceptée' : 'Refusée'} (rien n’est envoyé)</span>
        )}
      </div>
    </article>
  )
}

/** Refus avec sa raison (message fixe du serveur, jamais le texte du modèle). */
export function RefusalCard({ refusal }: { refusal: RefusalEvent }) {
  const title = refusal.reason === 'a_decrire' ? 'À décrire' : REFUSAL_TITLES[refusal.reason]
  return (
    <article
      aria-label={`Refus : ${refusal.requirement.label}`}
      className="flex flex-col gap-1 rounded-xl border border-slate-200 bg-white p-4 text-sm shadow-sm"
    >
      <h4 className="font-medium text-slate-900">{refusal.requirement.label}</h4>
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</p>
      <p className="text-slate-700">{refusal.message}</p>
    </article>
  )
}

/** Compétence que le CV ne prouve pas encore : invitation à décrire une expérience réelle. */
export function DescribeCard({ refusal }: { refusal: RefusalEvent }) {
  return (
    <article
      aria-label={`À décrire : ${refusal.requirement.label}`}
      className="flex flex-col gap-1 rounded-xl border border-dashed border-slate-300 bg-white p-4 text-sm"
    >
      <h4 className="font-medium text-slate-900">{refusal.requirement.label}</h4>
      <p className="text-slate-700">
        Votre CV ne le prouve pas encore. Si vous l’avez réellement pratiqué, décrivez vous-même une expérience
        réelle : le contexte, ce que vous avez fait, les outils, le résultat.
      </p>
      <p className="text-xs text-slate-500">Aucun texte n’est proposé : rien ne peut être inventé à votre place.</p>
    </article>
  )
}
