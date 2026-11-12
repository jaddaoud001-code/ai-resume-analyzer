import type { PublicConfig } from '../api/types'
import { formatBytes } from './labels'

/** Vérifications faites avant l'envoi (le backend revérifie tout). */
export function validateFile(file: File, config: PublicConfig): string | null {
  const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
  if (!isPdf) return 'Le CV doit être un fichier PDF.'
  if (file.size === 0) return 'Le fichier est vide.'
  if (file.size > config.max_pdf_size_bytes) {
    return `Le fichier fait ${formatBytes(file.size)} : la limite est de ${formatBytes(config.max_pdf_size_bytes)}.`
  }
  return null
}
