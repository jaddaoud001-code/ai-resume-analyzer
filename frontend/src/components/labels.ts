// Libellés et styles partagés par les composants.

import type { AnalysisErrorCode, EvidenceLevel, RequirementMatch } from '../api/types'

export const LEVEL_STYLES: Record<EvidenceLevel, string> = {
  'non démontré': 'bg-slate-100 text-slate-700 ring-slate-300',
  mentionné: 'bg-amber-50 text-amber-800 ring-amber-300',
  démontré: 'bg-sky-50 text-sky-800 ring-sky-300',
  solide: 'bg-emerald-50 text-emerald-800 ring-emerald-300',
}

export const LEVEL_LABELS: Record<EvidenceLevel, string> = {
  'non démontré': 'Non démontré',
  mentionné: 'Mentionné',
  démontré: 'Démontré',
  solide: 'Solide',
}

export const ERROR_CODE_TITLES: Record<AnalysisErrorCode, string> = {
  llm_provider_error: 'Le fournisseur d’IA n’a pas répondu',
  quota_exceeded: 'Quota journalier du fournisseur d’IA atteint',
  provider_access_denied: 'Le fournisseur d’IA refuse l’accès au modèle',
  no_requirements: 'Aucune exigence trouvée dans l’offre',
  embeddings_error: 'Le CV n’a pas pu être indexé',
  timeout: 'L’analyse a pris trop de temps',
  internal_error: 'Erreur interne',
}

function plural(count: number, singular: string, pluralForm: string): string {
  return count > 1 ? pluralForm : singular
}

/** Résumé des citations écartées (affichées sur demande dans la carte), ou null. */
export function rejectedSummary(result: RequirementMatch): string | null {
  const n = result.rejected_evidence
  if (result.status === 'error' || n <= 0) return null
  return `${n} ${plural(n, 'citation écartée', 'citations écartées')} : ${plural(n, 'introuvable', 'introuvables')} mot pour mot dans le CV.`
}

/** Autres drapeaux des vérifications du code, expliqués en mots simples. */
export function flagExplanations(result: RequirementMatch): string[] {
  const flags: string[] = []
  if (result.status === 'error') {
    flags.push('Évaluation impossible pour cette exigence : elle n’est pas comptée dans le score.')
    return flags
  }
  if (result.second_chance) {
    flags.push(
      result.evidence.length > 0
        ? 'Seconde chance : les premières citations étaient inexactes, le modèle a recopié ses preuves mot pour mot.'
        : 'Même après une seconde demande de citations exactes, aucune preuve n’a été trouvée mot pour mot dans le CV.',
    )
  }
  if (result.downgraded) {
    flags.push('Niveau ramené à « non démontré » : aucune preuve vérifiable dans le CV.')
  }
  if (result.capped_for_duration) {
    const years = result.requirement.min_years ?? 0
    const months = result.total_months ?? 0
    flags.push(
      `Durée insuffisante : ${months} mois prouvés pour ${years} ${plural(years, 'an demandé', 'ans demandés')} ; niveau limité à « mentionné ».`,
    )
  }
  if (result.capped_soft_skill) {
    flags.push('Savoir-être limité à « démontré » : un CV ne peut pas le prouver davantage.')
  }
  return flags
}

export function formatDuration(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} Mo`
  return `${Math.ceil(bytes / 1024).toLocaleString('fr-FR')} Ko`
}
