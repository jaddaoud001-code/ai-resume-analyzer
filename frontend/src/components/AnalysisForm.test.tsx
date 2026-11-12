import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { FALLBACK_CONFIG } from '../api/client'
import { pdfFile } from '../test/stream'
import { AnalysisForm } from './AnalysisForm'
import { validateFile } from './validation'

const CONFIG = { ...FALLBACK_CONFIG, max_pdf_size_bytes: 2048, max_job_offer_chars: 20 }

function setup(busy = false) {
  const onSubmit = vi.fn()
  render(<AnalysisForm config={CONFIG} busy={busy} onSubmit={onSubmit} />)
  return { onSubmit, user: userEvent.setup({ applyAccept: false }) }
}

describe('validateFile', () => {
  it('accepte un PDF dans la limite, refuse les autres', () => {
    expect(validateFile(pdfFile('cv.pdf', 100), CONFIG)).toBeNull()
    expect(validateFile(pdfFile('cv.PDF', 100, ''), CONFIG)).toBeNull() // type absent, extension suffit
    expect(validateFile(pdfFile('cv.docx', 100, 'application/msword'), CONFIG)).toBe('Le CV doit être un fichier PDF.')
    expect(validateFile(pdfFile('cv.pdf', 0), CONFIG)).toBe('Le fichier est vide.')
    expect(validateFile(pdfFile('cv.pdf', 4096), CONFIG)).toMatch(/limite est de 2 Ko/)
  })
})

describe('AnalysisForm', () => {
  it('refuse un fichier qui n’est pas un PDF et garde le bouton désactivé', async () => {
    const { user } = setup()

    await user.upload(screen.getByLabelText('CV (PDF)'), pdfFile('photo.png', 100, 'image/png'))
    await user.type(screen.getByLabelText('Offre d’emploi'), 'Une offre')

    expect(screen.getByRole('alert')).toHaveTextContent('Le CV doit être un fichier PDF.')
    expect(screen.getByRole('button', { name: 'Analyser' })).toBeDisabled()
  })

  it('refuse un PDF trop lourd selon les limites du backend', async () => {
    const { user } = setup()

    await user.upload(screen.getByLabelText('CV (PDF)'), pdfFile('cv.pdf', 4096))

    expect(screen.getByRole('alert')).toHaveTextContent('Le fichier fait 4 Ko : la limite est de 2 Ko.')
  })

  it('accepte un PDF déposé par glisser-déposer', () => {
    setup()
    const zone = screen.getByText('Glissez-déposez votre CV ici').parentElement as HTMLElement

    fireEvent.drop(zone, { dataTransfer: { files: [pdfFile('depose.pdf', 100)] } })

    expect(screen.getByText('depose.pdf')).toBeInTheDocument()
  })

  it('compte les caractères de l’offre et bloque au-delà de la limite', async () => {
    const { user } = setup()
    await user.upload(screen.getByLabelText('CV (PDF)'), pdfFile())

    await user.type(screen.getByLabelText('Offre d’emploi'), 'x'.repeat(21))

    expect(screen.getByText(/21 \/ 20 caractères : offre trop longue/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Analyser' })).toBeDisabled()
  })

  it('envoie le fichier et l’offre quand tout est valide', async () => {
    const { user, onSubmit } = setup()
    const file = pdfFile()
    await user.upload(screen.getByLabelText('CV (PDF)'), file)
    await user.type(screen.getByLabelText('Offre d’emploi'), 'Une offre')

    await user.click(screen.getByRole('button', { name: 'Analyser' }))

    expect(onSubmit).toHaveBeenCalledWith(file, 'Une offre')
  })

  it('désactive le formulaire pendant une analyse', () => {
    setup(true)

    expect(screen.getByRole('button', { name: 'Analyse en cours…' })).toBeDisabled()
    expect(screen.getByLabelText('Offre d’emploi')).toBeDisabled()
  })
})
