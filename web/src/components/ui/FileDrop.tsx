import { useState, type DragEvent } from 'react'
import { formatFileSize, UPLOAD_LIMITS } from '../../lib/upload'

/** Drag & drop / browse for PDFs, with the upload limits from lib/upload. */
export default function FileDrop({
  label,
  hint = `PDF · up to ${UPLOAD_LIMITS.maxFileSizeMb} MB`,
  files,
  onChange,
  onReject,
  multiple = false,
}: {
  label: string
  hint?: string
  files: File[]
  onChange: (files: File[]) => void
  onReject: (message: string) => void
  multiple?: boolean
}) {
  const [over, setOver] = useState(false)

  function add(list: FileList | null) {
    if (!list) return
    const ok: File[] = []
    for (const f of Array.from(list)) {
      if (!UPLOAD_LIMITS.allowedTypes.includes(f.type)) onReject(`${f.name} is not a PDF`)
      else if (f.size > UPLOAD_LIMITS.maxFileSizeMb * 1048576) onReject(`${f.name} is over ${UPLOAD_LIMITS.maxFileSizeMb} MB`)
      else ok.push(f)
    }
    if (ok.length) onChange(multiple ? [...files, ...ok.filter((f) => !files.some((p) => p.name === f.name))] : ok.slice(0, 1))
  }

  const drag = (e: DragEvent, state: boolean) => {
    e.preventDefault()
    setOver(state)
  }

  return (
    <div>
      <label
        className={`ga-drop${over ? ' is-over' : ''}`}
        onDragOver={(e) => drag(e, true)}
        onDragLeave={(e) => drag(e, false)}
        onDrop={(e) => {
          drag(e, false)
          add(e.dataTransfer.files)
        }}
      >
        <strong>⬆ {label}</strong>
        <span className="ga-muted">{hint} · or click to browse</span>
        <input type="file" accept="application/pdf" multiple={multiple} onChange={(e) => add(e.target.files)} />
      </label>
      {files.length > 0 && (
        <div className="ga-list" style={{ marginTop: 12 }}>
          {files.map((f) => (
            <div key={f.name} className="ga-item">
              <div>
                <div className="ga-item-title">📄 {f.name}</div>
                <div className="ga-item-meta">{formatFileSize(f.size)}</div>
              </div>
              <button type="button" className="ga-btn ga-btn-ghost ga-btn-sm" onClick={() => onChange(files.filter((x) => x !== f))}>
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
