import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import WorkflowPage from '../../components/WorkflowPage'
import { Card, EmptyState, LoadError, PageHeader, Pill, Spinner, StatTile, StatusPill } from '../../components/ui'
import { useAlerts } from '../../hooks/useAlerts'
import { useAsync } from '../../hooks/useAsync'
import { reached } from '../../lib/status'
import { getAssignment, listSubmissions, remindStudent, type Submission } from '../../lib/workflow'

type Filter = 'all' | 'submitted' | 'pending' | 'late'
const isSubmitted = (s: Submission) => reached(s.status, 'answer_uploaded')

/** §19 step 1: who uploaded — submitted vs pending at a glance. */
export default function Submissions() {
  const { id = '' } = useParams()
  const { alerts, show, dismiss } = useAlerts()
  const assignment = useAsync((force) => getAssignment(id, force), [id])
  const roster = useAsync((force) => listSubmissions(id, force), [id])
  const [filter, setFilter] = useState<Filter>('all')
  const [reminded, setReminded] = useState<Set<string>>(new Set())

  const subs = roster.data?.submissions ?? []
  const enrolled = roster.data?.enrolled ?? subs.length
  const submitted = subs.filter(isSubmitted)
  const late = subs.filter((s) => s.late)
  const pending = subs.filter((s) => !isSubmitted(s))
  const groups: Record<Filter, Submission[]> = { all: subs, submitted, pending, late }
  const a = assignment.data
  const enc = encodeURIComponent(id)

  async function remind(s: Submission) {
    try {
      await remindStudent(id, s.student_id)
      setReminded((r) => new Set(r).add(s.student_id))
      show(`Reminder sent to ${s.student_name ?? s.student_id}.`, 'success')
    } catch (err) {
      show(`Couldn't send reminder: ${(err as Error).message}`, 'danger')
    }
  }

  return (
    <WorkflowPage title="Submissions">
      <PageHeader
        back="/teacher/assignments"
        eyebrow="Submissions"
        title={a?.title ?? 'Submissions'}
        subtitle={[a?.class_name, `${enrolled} students`, a?.due_at && `due ${new Date(a.due_at).toLocaleDateString()}`].filter(Boolean).join(' · ')}
        actions={
          <>
            <Link to={`/teacher/assignments/${enc}/analytics`} className="ga-btn ga-btn-secondary">
              Analytics
            </Link>
            <Link to={`/teacher/assignments/${enc}/grading`} className="ga-btn ga-btn-primary">
              Grading run →
            </Link>
          </>
        }
      />
      <AlertStack alerts={alerts} onDismiss={dismiss} />
      {roster.error && <LoadError error={roster.error} onRetry={() => roster.reload()} />}
      {roster.loading && !roster.data && <Spinner label="Loading submissions…" />}
      {roster.data && (
        <>
          <div className="ga-grid-3" style={{ marginBottom: 20 }}>
            <StatTile label="Submitted" value={submitted.length} note={enrolled ? `${Math.round((submitted.length / enrolled) * 100)}%` : undefined} tone="success" />
            <StatTile label="Pending" value={pending.length} note="not yet" tone="warning" />
            <StatTile label="Late" value={late.length} note="after due" tone="danger" />
          </div>
          <Card>
            <div className="ga-tabs">
              {(['all', 'submitted', 'pending', 'late'] as Filter[]).map((f) => (
                <button key={f} type="button" className={`ga-tab${filter === f ? ' is-active' : ''}`} onClick={() => setFilter(f)}>
                  {f[0].toUpperCase() + f.slice(1)} · {groups[f].length}
                </button>
              ))}
            </div>
            {groups[filter].length === 0 ? (
              <EmptyState title="No students here" />
            ) : (
              <div className="ga-table-wrap">
                <table className="ga-table">
                  <thead>
                    <tr>
                      <th>Student</th>
                      <th>Submitted</th>
                      <th>Status</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {groups[filter].map((s) => (
                      <tr key={s.student_id}>
                        <td>
                          <div className="ga-item-title">{s.student_name ?? s.student_id}</div>
                          {s.student_name && <div className="ga-item-meta ga-mono">{s.student_id}</div>}
                        </td>
                        <td>
                          {s.submitted_at ? new Date(s.submitted_at).toLocaleString() : '—'}
                          {s.late && (
                            <>
                              {' '}
                              <Pill tone="danger">Late</Pill>
                            </>
                          )}
                        </td>
                        <td>
                          <StatusPill status={s.status} />
                          {s.recheck && s.recheck.length > 0 && (
                            <div style={{ marginTop: 6 }} title={s.recheck.map((r) => `Q${r.question_number}${r.subpart ?? ''}: ${r.note}`).join('\n')}>
                              <Pill tone="warning">Re-check requested · {s.recheck.map((r) => `Q${r.question_number}${r.subpart ?? ''}`).join(', ')}</Pill>
                            </div>
                          )}
                        </td>
                        <td style={{ textAlign: 'right' }}>
                          {isSubmitted(s) ? (
                            <Link
                              className="ga-btn ga-btn-secondary ga-btn-sm"
                              to={`/teacher/student-details?${new URLSearchParams({ student_id: s.student_id, assignment_id: id, qp_id: a?.qp_id ?? '' })}`}
                            >
                              View
                            </Link>
                          ) : (
                            <button
                              type="button"
                              className="ga-btn ga-btn-ghost ga-btn-sm"
                              disabled={reminded.has(s.student_id)}
                              onClick={() => remind(s)}
                            >
                              {reminded.has(s.student_id) ? 'Reminded' : 'Remind'}
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </WorkflowPage>
  )
}
