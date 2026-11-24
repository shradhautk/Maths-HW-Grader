import { useState } from 'react'
import { Link } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import WorkflowPage from '../../components/WorkflowPage'
import { AppHero, Panel, Stagger } from '../../components/app/AppKit'
import { LoadError, Spinner } from '../../components/ui'
import { useAlerts } from '../../hooks/useAlerts'
import { useAsync } from '../../hooks/useAsync'
import { getTeacherSettings, updateTeacherSettings, type TeacherSettings } from '../../lib/workflow'
import type { Persona } from '../../lib/workflow'

const PERSONAS: { key: Persona; label: string; text: string }[] = [
  { key: 'balanced', label: 'Balanced', text: 'Fair marks with clear, constructive notes.' },
  { key: 'strict', label: 'Strict', text: 'Marks only what the rubric asks for.' },
  { key: 'lenient', label: 'Lenient', text: 'Rewards effort and partly-right reasoning.' },
]

function Switch({ checked, onChange, title, text }: { checked: boolean; onChange: (v: boolean) => void; title: string; text: string }) {
  return (
    <label className="app-switch">
      <span>
        <b>{title}</b>
        <small>{text}</small>
      </span>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="app-switch-track" aria-hidden />
    </label>
  )
}

/** §02 Settings: grading defaults and notifications. */
export default function Settings() {
  const loaded = useAsync((force) => getTeacherSettings(force), [])
  const { alerts, show, dismiss } = useAlerts(5000)
  // `saved` is what the server last confirmed; `edits` are unsaved changes on top of it.
  const [saved, setSaved] = useState<TeacherSettings | null>(null)
  const [edits, setEdits] = useState<Partial<TeacherSettings>>({})
  const [saving, setSaving] = useState(false)

  const baseline = saved ?? loaded.data ?? null
  const draft = baseline ? { ...baseline, ...edits } : null
  const dirty = !!baseline && (Object.keys(edits) as (keyof TeacherSettings)[]).some((k) => edits[k] !== baseline[k])
  const set = <K extends keyof TeacherSettings>(k: K, v: TeacherSettings[K]) => setEdits((e) => ({ ...e, [k]: v }))

  async function save() {
    if (!draft) return
    setSaving(true)
    try {
      setSaved(await updateTeacherSettings(draft))
      setEdits({})
      show('Settings saved.', 'success')
    } catch (err) {
      show(`Couldn't save: ${(err as Error).message}`, 'danger')
    } finally {
      setSaving(false)
    }
  }

  return (
    <WorkflowPage title="Settings">
      <AppHero eyebrow="Settings" title="Grading defaults & notifications" text="These apply to every new assignment you create. You can still change them per assignment." />
      <AlertStack alerts={alerts} onDismiss={dismiss} />
      {loaded.error && <LoadError error={loaded.error} onRetry={() => loaded.reload()} />}
      {loaded.loading && !draft && <Spinner label="Loading your settings…" />}
      {draft && (
        <Stagger className="app-grid">
          <div>
            <Panel title="AI grading defaults">
              <div className="ga-field">
                <label>Grading persona</label>
                <div className="app-segment" role="radiogroup" aria-label="Grading persona">
                  {PERSONAS.map((p) => (
                    <button key={p.key} type="button" role="radio" aria-checked={draft.default_persona === p.key} className={draft.default_persona === p.key ? 'is-active' : ''} onClick={() => set('default_persona', p.key)}>
                      <span>{p.label}</span>
                    </button>
                  ))}
                </div>
                <small className="ga-muted">{PERSONAS.find((p) => p.key === draft.default_persona)?.text}</small>
              </div>
              <Switch checked={draft.partial_credit} onChange={(v) => set('partial_credit', v)} title="Partial credit" text="Award method marks for correct working even when the final answer is wrong." />
            </Panel>
            <Panel title="Notifications">
              <Switch checked={draft.notify_submissions} onChange={(v) => set('notify_submissions', v)} title="New submissions" text="Tell me when students submit their answers." />
              <Switch checked={draft.notify_flagged} onChange={(v) => set('notify_flagged', v)} title="Answers flagged for review" text="Tell me when grading finishes and some answers need a human check." />
            </Panel>
            <div className="ga-row" style={{ justifyContent: 'flex-end' }}>
              <button type="button" className="ga-btn ga-btn-secondary" disabled={!dirty || saving} onClick={() => setEdits({})}>
                Reset
              </button>
              <button type="button" className="ga-btn ga-btn-primary" disabled={!dirty || saving} onClick={save}>
                {saving ? 'Saving…' : 'Save settings'}
              </button>
            </div>
          </div>
          <Panel title="Account">
            <div className="app-rows">
              <Link to="/teacher/profile" className="app-row app-tone-brand">
                <span className="app-row-icon material-icons" aria-hidden>
                  person
                </span>
                <span className="app-row-main">
                  <span className="app-row-title">My profile</span>
                  <span className="app-row-meta">Photo, email and phone</span>
                </span>
              </Link>
              <Link to="/teacher/change-password" className="app-row app-tone-sky">
                <span className="app-row-icon material-icons" aria-hidden>
                  lock
                </span>
                <span className="app-row-main">
                  <span className="app-row-title">Change password</span>
                  <span className="app-row-meta">Requires your current password</span>
                </span>
              </Link>
            </div>
          </Panel>
        </Stagger>
      )}
    </WorkflowPage>
  )
}
