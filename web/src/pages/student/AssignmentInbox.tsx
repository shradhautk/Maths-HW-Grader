import { useState } from 'react'
import { Link } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { Card, EmptyState, LoadError, PageHeader, Pill, Spinner, StatusPill } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { isPublished, reached, studentTimeline } from '../../lib/status'
import { listStudentAssignments, type StudentAssignment } from '../../lib/workflow'

type Tab = 'todo' | 'submitted' | 'graded'

function tabOf(a: StudentAssignment): Tab {
  if (isPublished(a.submission_status)) return 'graded'
  if (reached(a.submission_status, 'answer_uploaded')) return 'submitted'
  return 'todo'
}

function dueLabel(due?: string) {
  if (!due) return null
  const days = Math.ceil((new Date(due).getTime() - Date.now()) / 86_400_000)
  const tone = days < 0 ? 'danger' : days <= 2 ? 'warning' : 'neutral'
  const text = days < 0 ? 'Overdue' : days === 0 ? 'Due today' : `Due in ${days} day${days === 1 ? '' : 's'}`
  return (
    <Pill tone={tone}>
      {text} · {new Date(due).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}
    </Pill>
  )
}

/** §14 Student submission flow: receive → submit → track → feedback → re-check. */
export default function AssignmentInbox() {
  const list = useAsync((force) => listStudentAssignments(force), [])
  const [tab, setTab] = useState<Tab>('todo')
  const all = list.data ?? []
  const shown = all.filter((a) => tabOf(a) === tab)
  const labels: Record<Tab, string> = { todo: 'To do', submitted: 'Submitted', graded: 'Graded' }

  return (
    <WorkflowPage title="My Assignments">
      <PageHeader eyebrow="Student" title="My assignments" subtitle="Published work from your teachers lands here. Submit, track grading and read your feedback." />
      <div className="ga-tabs">
        {(Object.keys(labels) as Tab[]).map((t) => (
          <button key={t} type="button" className={`ga-tab${tab === t ? ' is-active' : ''}`} onClick={() => setTab(t)}>
            {labels[t]} · {all.filter((a) => tabOf(a) === t).length}
          </button>
        ))}
      </div>
      {list.error && <LoadError error={list.error} onRetry={() => list.reload()} />}
      {list.loading && !list.data && <Spinner label="Loading your assignments…" />}
      {list.data && shown.length === 0 && (
        <Card>
          <EmptyState title={tab === 'todo' ? "You're all caught up" : 'Nothing here yet'} />
        </Card>
      )}
      <div className="ga-list">
        {shown.map((a) => (
          <AssignmentRow key={a.assignment_id} a={a} />
        ))}
      </div>
    </WorkflowPage>
  )
}

function AssignmentRow({ a }: { a: StudentAssignment }) {
  const q = new URLSearchParams({ assignment_id: a.assignment_id, qp_id: a.qp_id })
  const tab = tabOf(a)
  const meta = [a.class_name, a.question_count !== undefined && `${a.question_count} questions`, a.total_marks !== undefined && `${a.total_marks} marks`]
    .filter(Boolean)
    .join(' · ')

  return (
    <div className="ga-item" style={{ alignItems: 'flex-start' }}>
      <div style={{ flex: 1, minWidth: 240 }}>
        <div className="ga-row">
          <span className="ga-item-title">{a.title}</span>
          {a.submission_status && <StatusPill status={a.submission_status} />}
        </div>
        <div className="ga-item-meta">{meta}</div>
        {tab === 'submitted' && (
          <ul className="ga-checklist" style={{ marginTop: 8 }}>
            {studentTimeline(a.submission_status).map((s, i, arr) => {
              const active = !s.done && (i === 0 || arr[i - 1].done)
              return (
                <li key={s.label} className={s.done ? 'is-ok' : active ? 'is-active' : ''} style={{ padding: '6px 0' }}>
                  <span className="ga-check-dot">{s.done ? '✓' : i + 1}</span>
                  <span>
                    {s.label}
                    {active && i === arr.length - 1 && <span className="ga-muted"> — you'll be notified when grades are published</span>}
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </div>
      <div className="ga-row">
        {tab === 'todo' && dueLabel(a.due_at)}
        {tab === 'todo' && (
          <Link to={`/student/upload?${q}`} className="ga-btn ga-btn-primary ga-btn-sm">
            Start submission →
          </Link>
        )}
        {tab === 'submitted' && a.submission_status === 'ocr_done' && (
          <Link to={`/student/answer-correction?${q}`} className="ga-btn ga-btn-primary ga-btn-sm">
            Confirm your answers →
          </Link>
        )}
        {tab === 'graded' && (
          <>
            <Link to={`/student/grades?${q}`} className="ga-btn ga-btn-secondary ga-btn-sm">
              View feedback
            </Link>
            <Link to={`/student/assignments/${encodeURIComponent(a.assignment_id)}/recheck`} className="ga-btn ga-btn-ghost ga-btn-sm">
              Request a re-check
            </Link>
          </>
        )}
      </div>
    </div>
  )
}
