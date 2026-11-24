import { useCallback, useState } from 'react'
import MathPreview from '../MathPreview'

/** Per-card editing state. */
export interface Draft {
  text: string
  latex: boolean
  saving: boolean
  message: string
  messageKind: '' | 'saved' | 'error'
}

const newDraft = (text: string, latex: boolean): Draft => ({ text, latex, saving: false, message: '', messageKind: '' })

/**
 * Draft/save state for a list of OCR'd texts being corrected. `save` persists
 * one item and resolves to the status message to show; it should throw on failure.
 */
export function useCorrectionDrafts<T>(opts: {
  getText: (item: T) => string
  getLatex: (item: T) => boolean
  save: (item: T, draft: Draft) => Promise<string>
  emptyMessage: string
  savingMessage: string
}) {
  const { getText, getLatex, save, emptyMessage, savingMessage } = opts
  const [items, setItemsState] = useState<T[]>([])
  const [drafts, setDrafts] = useState<Draft[]>([])

  const setItems = useCallback(
    (list: T[]) => {
      setItemsState(list)
      setDrafts(list.map((i) => newDraft(getText(i), getLatex(i))))
    },
    [getText, getLatex],
  )

  const updateDraft = useCallback(
    (i: number, patch: Partial<Draft>) => setDrafts((ds) => ds.map((d, j) => (j === i ? { ...d, ...patch } : d))),
    [],
  )

  /** Saves one card; `apply` updates the stored item after a successful save. */
  async function saveOne(i: number, apply: (item: T, draft: Draft) => T): Promise<boolean> {
    const draft = drafts[i]
    if (!draft.text.trim()) {
      updateDraft(i, { message: emptyMessage, messageKind: 'error' })
      return false
    }
    updateDraft(i, { saving: true, message: savingMessage, messageKind: '' })
    try {
      const message = await save(items[i], draft)
      setItemsState((list) => list.map((it, j) => (j === i ? apply(it, draft) : it)))
      updateDraft(i, { saving: false, message, messageKind: 'saved' })
      return true
    } catch (err) {
      updateDraft(i, { saving: false, message: `Failed to save: ${(err as Error).message}`, messageKind: 'error' })
      return false
    }
  }

  return { items, drafts, setItems, updateDraft, saveOne }
}

interface CardProps {
  label: string
  done: boolean
  doneLabel: string
  maxMarks: unknown
  /** Read-only question shown above the editor (answer correction only). */
  question?: string
  editorLabel: string
  saveLabel: string
  draft: Draft
  onChange: (patch: Partial<Draft>) => void
  onSave: () => void
}

/** Plain/LaTeX editor with live preview — one OCR'd answer or question. */
export function EditorCard({ label, done, doneLabel, maxMarks, question, editorLabel, saveLabel, draft, onChange, onSave }: CardProps) {
  return (
    <div className="question-item answer-editor-card">
      <div className="question-header">
        <h3>{label}</h3>
        <span className={`answer-status-tag ${done ? 'status-corrected' : 'status-pending'}`}>
          {done ? `✓ ${doneLabel}` : 'Needs your review'}
        </span>
      </div>
      <div className="question-meta">
        <span className="marks">
          Max marks: <span className="marks-display">{String(maxMarks || 0)}</span>
        </span>
      </div>
      {question !== undefined && <p className="question-text">{question}</p>}

      <div className="answer-mode-toggle" role="tablist">
        <button type="button" className={`answer-mode-btn${draft.latex ? '' : ' active'}`} onClick={() => onChange({ latex: false })}>
          Plain text
        </button>
        <button type="button" className={`answer-mode-btn${draft.latex ? ' active' : ''}`} onClick={() => onChange({ latex: true })}>
          LaTeX
        </button>
      </div>

      <div className="answer-editor-grid">
        <div className="answer-editor-col">
          <h5>{editorLabel}</h5>
          <textarea
            className="answer-textarea"
            spellCheck={false}
            value={draft.text}
            onChange={(e) => onChange({ text: e.target.value })}
          />
        </div>
        <div className="answer-preview-col">
          <h5>Preview</h5>
          <MathPreview className="answer-preview-pane" text={draft.text} latex={draft.latex} />
        </div>
      </div>

      <div className="answer-editor-actions">
        <span className={`answer-save-status${draft.messageKind ? ` ${draft.messageKind}` : ''}`}>{draft.message}</span>
        <button type="button" className="btn btn-secondary save-one-btn" disabled={draft.saving} onClick={onSave}>
          <i className="fas fa-floppy-disk" /> {saveLabel}
        </button>
      </div>
    </div>
  )
}
