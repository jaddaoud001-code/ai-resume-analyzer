import { type DragEvent, type FormEvent, useId, useRef, useState } from 'react'
import type { PublicConfig } from '../api/types'
import { formatBytes } from './labels'
import { validateFile } from './validation'

interface AnalysisFormProps {
  config: PublicConfig
  busy: boolean
  onSubmit: (file: File, jobOffer: string) => void
}

export function AnalysisForm({ config, busy, onSubmit }: AnalysisFormProps) {
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [offer, setOffer] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const ids = { file: useId(), fileHelp: useId(), offer: useId(), offerHelp: useId() }

  const offerLength = offer.length
  const offerTooLong = offerLength > config.max_job_offer_chars
  const canSubmit = !busy && file !== null && fileError === null && offer.trim() !== '' && !offerTooLong

  const choose = (candidate: File | undefined): void => {
    if (!candidate) return
    const error = validateFile(candidate, config)
    setFileError(error)
    setFile(error ? null : candidate)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault()
    setDragging(false)
    if (!busy) choose(event.dataTransfer.files[0])
  }

  const submit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    if (canSubmit && file) onSubmit(file, offer)
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-5" noValidate>
      <div className="flex flex-col gap-1.5">
        <label htmlFor={ids.file} className="text-sm font-medium text-slate-800">
          CV (PDF)
        </label>
        <div
          onDragOver={(event) => {
            event.preventDefault()
            if (!busy) setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex flex-col items-center gap-2 rounded-lg border-2 border-dashed px-4 py-6 text-center text-sm transition-colors ${
            dragging ? 'border-sky-500 bg-sky-50' : 'border-slate-300 bg-white'
          } ${busy ? 'opacity-60' : ''}`}
        >
          <p className="text-slate-700">
            {file ? (
              <>
                <span className="font-medium text-slate-900">{file.name}</span>{' '}
                <span className="text-slate-500">({formatBytes(file.size)})</span>
              </>
            ) : (
              'Glissez-déposez votre CV ici'
            )}
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
            className="rounded-md bg-white px-3 py-1.5 font-medium text-slate-800 ring-1 ring-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed"
          >
            {file ? 'Changer de fichier' : 'Choisir un fichier'}
          </button>
          <input
            ref={inputRef}
            id={ids.file}
            type="file"
            accept="application/pdf,.pdf"
            className="sr-only"
            disabled={busy}
            aria-describedby={ids.fileHelp}
            aria-invalid={fileError !== null}
            onChange={(event) => {
              choose(event.target.files?.[0])
              event.target.value = '' // permet de re-choisir le même fichier
            }}
          />
        </div>
        <p id={ids.fileHelp} className={`text-xs ${fileError ? 'text-red-700' : 'text-slate-500'}`} role={fileError ? 'alert' : undefined}>
          {fileError ?? `PDF de ${formatBytes(config.max_pdf_size_bytes)} et ${config.max_pdf_pages} pages au plus.`}
        </p>
      </div>

      <div className="flex flex-col gap-1.5">
        <label htmlFor={ids.offer} className="text-sm font-medium text-slate-800">
          Offre d’emploi
        </label>
        <textarea
          id={ids.offer}
          value={offer}
          onChange={(event) => setOffer(event.target.value)}
          disabled={busy}
          rows={10}
          placeholder="Collez ici le texte de l’offre…"
          aria-describedby={ids.offerHelp}
          aria-invalid={offerTooLong}
          className={`w-full resize-y rounded-lg border bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:outline-none focus:ring-2 disabled:opacity-60 ${
            offerTooLong ? 'border-red-400 focus:ring-red-300' : 'border-slate-300 focus:ring-sky-300'
          }`}
        />
        <p id={ids.offerHelp} className={`text-right text-xs tabular-nums ${offerTooLong ? 'text-red-700' : 'text-slate-500'}`}>
          {offerLength.toLocaleString('fr-FR')} / {config.max_job_offer_chars.toLocaleString('fr-FR')} caractères
          {offerTooLong && ' : offre trop longue'}
        </p>
      </div>

      <button
        type="submit"
        disabled={!canSubmit}
        className="rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {busy ? 'Analyse en cours…' : 'Analyser'}
      </button>
    </form>
  )
}
