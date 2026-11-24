import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { AppHero, AppStat, Panel, Stagger } from '../../components/app/AppKit'
import { EmptyState, GradePill, LoadError, Pill, ProgressBar, Spinner } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { listStudents, type ClassStudent } from '../../lib/workflow'

const initials = (s: ClassStudent) => (s.student_name || s.student_id).split(/\s+/).map((w) => w[0]).slice(0, 2).join('').toUpperCase()

/** §02 Students: the roster across the teacher's classes. */
export default function Students() {
  const data = useAsync((force) => listStudents(force), [])
  const [cls, setCls] = useState('')
  const [q, setQ] = useState('')
  const [riskOnly, setRiskOnly] = useState(false)
  const has = !!data.data
  const all = useMemo(() => data.data?.students ?? [], [data.data])

  const shown = useMemo(
    () =>
      all.filter(
        (s) =>
          (!cls || s.classes.includes(cls)) &&
          (!riskOnly || s.at_risk) &&
          (!q || `${s.student_name ?? ''} ${s.email ?? ''} ${s.student_id}`.toLowerCase().includes(q.toLowerCase())),
      ),
    [all, cls, q, riskOnly],
  )
  const graded = all.filter((s) => s.average != null)
  const classAvg = graded.length ? graded.reduce((n, s) => n + (s.average ?? 0), 0) / graded.length : null

  return (
    <WorkflowPage title="Students">
      <AppHero
        eyebrow="Students"
        title="Your classes at a glance"
        text="Who has submitted, how they're scoring, and who might need a nudge."
        actions={
          <Link to="/teacher/grades" className="ga-btn ga-btn-secondary">
            Look up a grade
          </Link>
        }
      />
      <Stagger className="app-stats">
        <AppStat icon="groups" label="Students" value={has ? all.length : null} tone="sky" />
        <AppStat icon="trending_up" label="Class average" value={classAvg == null ? null : Math.round(classAvg)} suffix="%" tone="accent" />
        <AppStat icon="warning" label="At risk" value={has ? all.filter((s) => s.at_risk).length : null} tone="danger" note="below 50%" />
        <AppStat icon="schedule" label="Late submissions" value={has ? all.reduce((n, s) => n + s.late, 0) : null} tone="warn" />
      </Stagger>

      {data.error && <LoadError error={data.error} onRetry={() => data.reload()} />}
      {data.loading && !has && <Spinner label="Loading students…" />}
      {has && (
        <Panel title={`Roster · ${shown.length}`}>
          <div className="app-toolbar">
            <input className="ga-input" placeholder="Search by name or email" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search students" />
            <select className="ga-select" value={cls} onChange={(e) => setCls(e.target.value)} aria-label="Filter by class">
              <option value="">All classes</option>
              {(data.data?.classes ?? []).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <button type="button" className={`ga-tab${riskOnly ? ' is-active' : ''}`} onClick={() => setRiskOnly((r) => !r)} aria-pressed={riskOnly}>
              At risk only
            </button>
          </div>
          {shown.length === 0 && <EmptyState title="No students match">{all.length ? 'Try clearing the filters.' : 'Students appear once they join a class or submit work.'}</EmptyState>}
          {shown.length > 0 && (
            <div className="ga-table-wrap">
              <table className="ga-table">
                <thead>
                  <tr>
                    <th>Student</th>
                    <th>Class</th>
                    <th>Submitted</th>
                    <th>Average</th>
                    <th>Status</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {shown.map((s) => (
                    <tr key={s.student_id}>
                      <td>
                        <span className="app-person">
                          <span className="app-avatar">{initials(s)}</span>
                          <span>
                            <b>{s.student_name ?? s.student_id}</b>
                            <small>{s.email ?? s.student_id}</small>
                          </span>
                        </span>
                      </td>
                      <td>{s.classes.join(', ') || '—'}</td>
                      <td>
                        <span className="app-meter">
                          <ProgressBar percent={s.assigned ? (s.submitted / s.assigned) * 100 : 0} tone={s.submitted === s.assigned ? 'success' : 'brand'} />
                          <span>
                            {s.submitted}/{s.assigned}
                          </span>
                        </span>
                      </td>
                      <td>
                        {s.average == null ? (
                          <span className="ga-muted">—</span>
                        ) : (
                          <span className="ga-row">
                            <GradePill percent={s.average} /> {Math.round(s.average)}%
                          </span>
                        )}
                      </td>
                      <td>
                        {s.at_risk ? <Pill tone="danger">At risk</Pill> : s.average == null ? <Pill>No grades yet</Pill> : <Pill tone="success">On track</Pill>}
                        {s.late > 0 && (
                          <>
                            {' '}
                            <Pill tone="warning">{s.late} late</Pill>
                          </>
                        )}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {s.latest ? (
                          <Link
                            className="ga-btn ga-btn-secondary ga-btn-sm"
                            to={`/teacher/student-details?${new URLSearchParams({ student_id: s.student_id, assignment_id: s.latest.assignment_id, qp_id: s.latest.qp_id })}`}
                          >
                            Latest paper
                          </Link>
                        ) : (
                          <span className="ga-muted">no papers</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      )}
    </WorkflowPage>
  )
}
