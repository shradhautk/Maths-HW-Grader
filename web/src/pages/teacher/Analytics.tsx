import { useState } from 'react'
import { Link } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { AppHero, AppStat, BarChart, Panel, Stagger } from '../../components/app/AppKit'
import { EmptyState, GradeDistribution, GradePill, LoadError, Pill, Spinner } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { GRADES_STATUS_META } from '../../lib/status'
import { getTeacherAnalytics } from '../../lib/workflow'

/** §11 Class analytics across all assignments (per-assignment drill-down lives under each assignment). */
export default function Analytics() {
  const [cls, setCls] = useState<string | null>(null)
  const data = useAsync((force) => getTeacherAnalytics(cls, force), [cls])
  const a = data.data
  const t = a?.totals
  const weeklyTotal = a?.weekly.reduce((n, w) => n + w.count, 0) ?? 0

  function exportCsv() {
    if (!a) return
    const lines = ['assignment,class,enrolled,submitted,graded,average_percent,at_risk', ...a.assignments.map((r) => [JSON.stringify(r.title), JSON.stringify(r.class_name ?? ''), r.enrolled, r.submitted, r.graded, r.average == null ? '' : r.average.toFixed(1), r.at_risk].join(','))]
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `class-analytics${cls ? `-${cls}` : ''}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <WorkflowPage title="Analytics">
      <AppHero
        eyebrow="Class analytics"
        title={cls ? `${cls} analytics` : 'How your classes are doing'}
        text="Submission rates, score spread and the questions that cost the most marks — click an assignment to drill down."
        actions={
          <>
            <select className="ga-select" style={{ width: 'auto', minWidth: 170 }} value={cls ?? ''} onChange={(e) => setCls(e.target.value || null)} aria-label="Class">
              <option value="">All classes</option>
              {(a?.classes ?? (cls ? [cls] : [])).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <button type="button" className="ga-btn ga-btn-secondary" onClick={exportCsv} disabled={!a?.assignments.length}>
              <span className="material-icons" aria-hidden>
                download
              </span>
              Export CSV
            </button>
          </>
        }
      />

      {data.error && <LoadError error={data.error} onRetry={() => data.reload()} />}
      {data.loading && !a && <Spinner label="Crunching the numbers…" />}

      <Stagger className="app-stats">
        <AppStat icon="percent" label="Submission rate" value={t?.submission_rate == null ? null : Math.round(t.submission_rate)} suffix="%" tone="sky" note={t ? `${t.submitted} of ${t.enrolled} papers` : undefined} />
        <AppStat icon="trending_up" label="Class average" value={t?.average == null ? null : Math.round(t.average)} suffix="%" tone="accent" note={t ? `${t.graded} graded` : undefined} />
        <AppStat icon="assignment" label="Assignments" value={t ? t.assignments : null} tone="brand" />
        <AppStat icon="warning" label="At risk" value={t ? t.at_risk : null} tone="danger" note="students below 50%" />
      </Stagger>

      {a && (
        <>
          <Stagger className="app-grid">
            <Panel title="Weekly grading activity" action={<span className="ga-muted">{weeklyTotal} papers · last 7 days</span>}>
              <BarChart items={a.weekly.map((w) => ({ label: w.label, value: w.count }))} unit=" papers" />
            </Panel>
            <Panel title="Grade distribution">
              {a.totals.graded ? <GradeDistribution counts={a.distribution} /> : <EmptyState title="No graded papers yet" />}
            </Panel>
          </Stagger>

          <Stagger className="app-grid">
            <Panel title="Assignments" action={<span className="ga-muted">click a row to drill down</span>}>
              {a.assignments.length === 0 ? (
                <EmptyState title="No assignments yet">Create one to see analytics here.</EmptyState>
              ) : (
                <div className="ga-table-wrap">
                  <table className="ga-table">
                    <thead>
                      <tr>
                        <th>Assignment</th>
                        <th>Submitted</th>
                        <th>Average</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {a.assignments.map((r) => {
                        const meta = GRADES_STATUS_META[r.grades_status ?? 'not_started']
                        return (
                          <tr key={r.assignment_id}>
                            <td>
                              <Link to={`/teacher/assignments/${encodeURIComponent(r.assignment_id)}/analytics`} className="app-person">
                                <span>
                                  <b>{r.title}</b>
                                  <small>{r.class_name}</small>
                                </span>
                              </Link>
                            </td>
                            <td>
                              {r.submitted}/{r.enrolled}
                            </td>
                            <td>
                              {r.average == null ? (
                                <span className="ga-muted">—</span>
                              ) : (
                                <span className="ga-row">
                                  <GradePill percent={r.average} /> {Math.round(r.average)}%
                                </span>
                              )}
                            </td>
                            <td>
                              <Pill tone={meta.tone}>{meta.label}</Pill>
                              {r.at_risk > 0 && (
                                <>
                                  {' '}
                                  <Pill tone="danger">{r.at_risk} at risk</Pill>
                                </>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </Panel>
            <div>
              <Panel title="Insight">
                {a.insight ? (
                  <div className="app-insight">
                    <span className="material-icons" aria-hidden>
                      auto_awesome
                    </span>
                    <div>
                      <b>AI insight</b>
                      {a.insight}
                    </div>
                  </div>
                ) : (
                  <p className="ga-muted">Insights appear once papers have been graded.</p>
                )}
                {a.hardest.length > 1 && (
                  <div className="app-rows" style={{ marginTop: 14 }}>
                    {a.hardest.slice(0, 4).map((h) => (
                      <div key={`${h.assignment_id}-${h.label}`} className="app-row app-tone-warn">
                        <span className="app-row-main">
                          <span className="app-row-title">{h.label}</span>
                          <span className="app-row-meta">{h.title}</span>
                        </span>
                        <strong>{Math.round(h.average)}%</strong>
                      </div>
                    ))}
                  </div>
                )}
              </Panel>
              <Panel title="Upcoming deadlines">
                {a.upcoming.length === 0 ? (
                  <p className="ga-muted">Nothing due soon.</p>
                ) : (
                  <div className="app-rows">
                    {a.upcoming.map((u) => (
                      <Link key={u.assignment_id} to={`/teacher/assignments/${encodeURIComponent(u.assignment_id)}/submissions`} className="app-row app-tone-sky">
                        <span className="app-row-main">
                          <span className="app-row-title">{u.title}</span>
                          <span className="app-row-meta">{u.class_name}</span>
                        </span>
                        <Pill>{new Date(u.due_at).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}</Pill>
                      </Link>
                    ))}
                  </div>
                )}
              </Panel>
            </div>
          </Stagger>
        </>
      )}
    </WorkflowPage>
  )
}
