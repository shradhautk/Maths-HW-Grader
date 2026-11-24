import { useState, type FormEvent } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useAlerts } from '../../hooks/useAlerts'
import { useAsync } from '../../hooks/useAsync'
import { formatWhen } from '../../lib/dates'
import { deleteAnnouncement, listAnnouncements, postAnnouncement } from '../../lib/workflow'
import { isRouteMissing } from '../../lib/api'
import AlertStack from '../AlertStack'
import { Pill } from '../ui'
import { Panel } from './AppKit'

/** Post a notice to a class and see/delete your recent ones (shown on the teacher dashboard). */
export default function TeacherAnnouncements({ classes }: { classes: string[] }) {
  const list = useAsync((force) => listAnnouncements(force), [])
  const { alerts, show, dismiss } = useAlerts(5000)
  const [cls, setCls] = useState('')
  const [title, setTitle] = useState('')
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const selected = cls || classes[0] || ''
  const unavailable = !!list.error && isRouteMissing(list.error)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!selected || !title.trim() || !message.trim()) return show('Pick a class and write a title and message.', 'danger')
    setBusy(true)
    try {
      await postAnnouncement({ class_name: selected, title: title.trim(), message: message.trim() })
      setTitle('')
      setMessage('')
      show(`Posted to ${selected}.`, 'success')
      list.reload()
    } catch (err) {
      show((err as Error).message, 'danger')
    } finally {
      setBusy(false)
    }
  }

  async function remove(id: string) {
    try {
      await deleteAnnouncement(id)
      list.reload()
    } catch (err) {
      show((err as Error).message, 'danger')
    }
  }

  return (
    <Panel title="Announcements">
      {unavailable ? (
        <p className="ga-muted">Announcements aren't available until the workflow API is deployed.</p>
      ) : classes.length === 0 ? (
        <p className="ga-muted">Create an assignment for a class, then you can post notices to it.</p>
      ) : (
        <form onSubmit={submit}>
          <div className="ga-grid-2">
            <div className="ga-field">
              <label htmlFor="an-class">Class</label>
              <select id="an-class" className="ga-select" value={selected} onChange={(e) => setCls(e.target.value)}>
                {classes.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </div>
            <div className="ga-field">
              <label htmlFor="an-title">Title</label>
              <input id="an-title" className="ga-input" maxLength={150} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Quiz on Friday" />
            </div>
          </div>
          <div className="ga-field">
            <label htmlFor="an-msg">Message</label>
            <textarea id="an-msg" className="ga-textarea" maxLength={2000} value={message} onChange={(e) => setMessage(e.target.value)} placeholder="What should your class know?" />
          </div>
          <AlertStack alerts={alerts} onDismiss={dismiss} />
          <button type="submit" className="ga-btn ga-btn-primary" disabled={busy}>
            {busy ? 'Posting…' : 'Post to class'}
          </button>
        </form>
      )}
      <div className="app-rows" style={{ marginTop: 18 }}>
        <AnimatePresence initial={false}>
          {(list.data ?? []).slice(0, 3).map((n) => (
            <motion.article key={n.announcement_id} className="app-note" layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: 24 }}>
              <div className="app-note-head">
                <h3>{n.title}</h3>
                <span className="ga-row">
                  <Pill tone="brand">{n.class_name}</Pill>
                  <button type="button" className="ga-btn ga-btn-ghost ga-btn-sm" onClick={() => remove(n.announcement_id)} aria-label={`Delete ${n.title}`}>
                    <span className="material-icons">delete</span>
                  </button>
                </span>
              </div>
              <p>{n.message}</p>
              <small>{formatWhen(n.created_at)}</small>
            </motion.article>
          ))}
        </AnimatePresence>
      </div>
    </Panel>
  )
}
