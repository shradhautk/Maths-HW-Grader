import { useEffect, useRef, useState, type FormEvent } from 'react'
import { askAssistant, type AssistantMessage } from '../../lib/workflow'
import '../../styles/ui.css'

interface Props {
  /** What the assistant is looking at: one paper (docked) or a whole batch (full page). */
  context: { assignment_id?: string; evaluation_id?: string; student_id?: string }
  subtitle?: string
  greeting: string
  suggestions: string[]
  placeholder?: string
  onClose?: () => void
}

/** §04 GradeFlow Assistant chat — the docked panel and the full page share it. */
export default function AssistantPanel({ context, subtitle, greeting, suggestions, placeholder = 'Ask the assistant anything…', onClose }: Props) {
  const [messages, setMessages] = useState<AssistantMessage[]>([{ role: 'assistant', content: greeting }])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [messages, busy])

  async function send(text: string) {
    const content = text.trim()
    if (!content || busy) return
    const next: AssistantMessage[] = [...messages, { role: 'user', content }]
    setMessages(next)
    setInput('')
    setBusy(true)
    try {
      // The greeting is UI copy, not part of the conversation.
      const reply = await askAssistant(context, next.slice(1))
      setMessages((m) => [...m, { role: 'assistant', content: reply || 'I don’t have an answer for that yet.' }])
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', content: `Sorry — I couldn’t reach GradeFlow (${(err as Error).message}).` }])
    } finally {
      setBusy(false)
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault()
    void send(input)
  }

  return (
    <div className="ga-assistant">
      <div className="ga-assistant-head">
        <span aria-hidden>✦</span> GradeFlow Assistant
        <small>{subtitle}</small>
        {onClose && (
          <button type="button" className="ga-btn ga-btn-sm" style={{ color: 'inherit', background: 'none' }} onClick={onClose} aria-label="Close assistant">
            ✕
          </button>
        )}
      </div>
      <div className="ga-assistant-log" ref={logRef} aria-live="polite">
        {messages.map((m, i) => (
          <div key={i} className={`ga-msg is-${m.role}`}>
            {m.content}
          </div>
        ))}
        {busy && <div className="ga-msg is-assistant">…</div>}
      </div>
      {messages.length <= 2 && (
        <div className="ga-assistant-chips">
          {suggestions.map((s) => (
            <button key={s} type="button" className="ga-tab" onClick={() => send(s)} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
      )}
      <form className="ga-assistant-form" onSubmit={submit}>
        <input className="ga-input" value={input} placeholder={placeholder} onChange={(e) => setInput(e.target.value)} aria-label="Message" />
        <button type="submit" className="ga-btn ga-btn-primary" disabled={busy || !input.trim()}>
          →
        </button>
      </form>
    </div>
  )
}

/** Floating "Ask GradeFlow" button that opens the docked panel (§04 A). */
export function AssistantDock(props: Omit<Props, 'onClose'>) {
  const [open, setOpen] = useState(false)
  if (!open) {
    return (
      <button type="button" className="ga-btn ga-btn-primary ga-assistant-fab" onClick={() => setOpen(true)}>
        ✦ Ask GradeFlow
      </button>
    )
  }
  return (
    <div className="ga-assistant-dock">
      <AssistantPanel {...props} onClose={() => setOpen(false)} />
    </div>
  )
}
