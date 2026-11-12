import type { RequirementEntry } from '../state/analysisReducer'
import { RequirementCard } from './RequirementCard'

interface RequirementListProps {
  entries: RequirementEntry[]
  interrupted: boolean
}

function Group({ title, entries, interrupted }: { title: string; entries: RequirementEntry[]; interrupted: boolean }) {
  if (entries.length === 0) return null
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-sm font-semibold text-slate-700">
        {title} <span className="font-normal text-slate-500">({entries.length})</span>
      </h3>
      <div className="grid gap-3 md:grid-cols-2">
        {entries.map((entry) => (
          <RequirementCard key={entry.requirement.id} entry={entry} interrupted={interrupted} />
        ))}
      </div>
    </section>
  )
}

/** Cartes des exigences, obligatoires d'abord. */
export function RequirementList({ entries, interrupted }: RequirementListProps) {
  if (entries.length === 0) return null
  return (
    <section aria-labelledby="requirements-title" className="flex flex-col gap-5">
      <h2 id="requirements-title" className="text-base font-semibold text-slate-900">
        Exigences de l’offre
      </h2>
      <Group
        title="Obligatoires"
        entries={entries.filter((e) => e.requirement.importance === 'obligatoire')}
        interrupted={interrupted}
      />
      <Group
        title="Souhaitées"
        entries={entries.filter((e) => e.requirement.importance !== 'obligatoire')}
        interrupted={interrupted}
      />
    </section>
  )
}
