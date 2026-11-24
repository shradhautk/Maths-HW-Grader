import { Link } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { AppHero, AppStat, Panel, ScoreRing, Stagger, TrendChart } from '../../components/app/AppKit'
import { EmptyState, GradePill, LoadError, Pill, ProgressBar, Spinner } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { getProgress } from '../../lib/workflow'

const short = (d?: string) => (d ? new Date(d).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : '')

function dueTone(due?: string) {
  if (!due) return { text: 'No due date', tone: 'neutral' as const }
  const days = Math.ceil((new Date(due).getTime() - Date.now()) / 86_400_000)
  if (days < 0) return { text: 'Overdue', tone: 'danger' as const }
  if (days === 0) return { text: 'Due today', tone: 'warning' as const }
  if (days === 1) return { text: 'Due tomorrow', tone: 'warning' as const }
  return { text: `Due in ${days} days`, tone: days <= 5 ? ('warning' as const) : ('neutral' as const) }
}

/** §03 / §12 My progress: trend, by class, due soon and a personal insight (published work only). */
export default function Progress() {
  const data = useAsync((force) => getProgress(force), [])
  const p = data.data
  const best = p?.results.length ? p.results.reduce((a, b) => (b.percent > a.percent ? b : a)) : null
  const last = p?.results.at(-1)
  const prev = p && p.results.length > 1 ? p.results.at(-2) : null
  const delta = last && prev ? Math.round(last.percent - prev.percent) : null

  return (
    <WorkflowPage title="My Progress">
      <AppHero
        eyebrow="My progress"
        title="Your semester at a glance"
        text="Only published grades count here. Tap an assignment to see the feedback behind a score."
        aside={<ScoreRing percent={p?.average ?? null} label={p?.average == null ? 'no grades yet' : 'average score'} />}
        actions={
          <Link to="/student/assignments" className="ga-btn ga-btn-primary">
            My assignments
          </Link>
        }
      />

      {data.error && <LoadError error={data.error} onRetry={() => data.reload()} />}
      {data.loading && !p && <Spinner label="Loading your progress…" />}

      <Stagger className="app-stats">
        <AppStat icon="task_alt" label="Graded" value={p ? p.results.length : null} tone="accent" />
        <AppStat icon="emoji_events" label="Best score" value={best ? Math.round(best.percent) : null} suffix="%" tone="warn" note={best?.title} />
        <AppStat icon="trending_up" label="Latest vs previous" value={delta} suffix=" pts" tone={delta != null && delta < 0 ? 'danger' : 'brand'} note={last?.title} />
        <AppStat icon="local_fire_department" label="On-time streak" value={p ? p.streak : null} tone="sky" />
      </Stagger>

      {p && (
        <>
          <Stagger className="app-grid">
            <Panel title="Grade trend">
              {p.results.length === 0 ? (
                <EmptyState title="No published grades yet">Once your teacher publishes a grade, your trend shows up here.</EmptyState>
              ) : (
                <TrendChart points={p.results.map((r) => ({ label: short(r.date) || r.title, value: r.percent, title: `${r.title}: ${Math.round(r.percent)}%` }))} />
              )}
            </Panel>
            <div>
              <Panel title="By class">
                {p.by_class.length === 0 && <p className="ga-muted">No graded work yet.</p>}
                <div className="app-rows">
                  {p.by_class.map((c) => (
                    <div key={c.class_name} className="app-row app-tone-accent">
                      <span className="app-row-main">
                        <span className="app-row-title">{c.class_name}</span>
                        <span className="app-meter" style={{ marginTop: 8 }}>
                          <ProgressBar percent={c.average} tone="success" />
                          <span>{Math.round(c.average)}%</span>
                        </span>
                      </span>
                      <GradePill percent={c.average} />
                    </div>
                  ))}
                </div>
              </Panel>
              <Panel title="Personal insight">
                {p.insight ? (
                  <div className="app-insight">
                    <span className="material-icons" aria-hidden>
                      auto_awesome
                    </span>
                    <div>
                      <b>Insight</b>
                      {p.insight}
                    </div>
                  </div>
                ) : (
                  <p className="ga-muted">Submit a few more assignments and we'll point out what's working for you.</p>
                )}
              </Panel>
            </div>
          </Stagger>

          <Stagger className="app-grid">
            <Panel title="Graded assignments">
              {p.results.length === 0 && <p className="ga-muted">Nothing graded yet.</p>}
              <div className="app-rows">
                {[...p.results].reverse().map((r) => (
                  <Link key={r.assignment_id} to={`/student/grades?${new URLSearchParams({ assignment_id: r.assignment_id, qp_id: r.qp_id })}`} className="app-row app-tone-accent">
                    <GradePill percent={r.percent} />
                    <span className="app-row-main">
                      <span className="app-row-title">{r.title}</span>
                      <span className="app-row-meta">
                        {[r.class_name, `${r.marks}/${r.max}`, short(r.date)].filter(Boolean).join(' · ')}
                      </span>
                    </span>
                    <span className="app-row-end">
                      {r.early && <Pill tone="success">Early</Pill>}
                      {r.on_time === false && <Pill tone="warning">Late</Pill>}
                      <strong>{Math.round(r.percent)}%</strong>
                    </span>
                  </Link>
                ))}
              </div>
            </Panel>
            <Panel title="Due soon">
              {p.todo.length === 0 && <p className="ga-muted">You're all caught up. 🎉</p>}
              <div className="app-rows">
                {p.todo.map((t) => {
                  const d = dueTone(t.due_at)
                  return (
                    <Link key={t.assignment_id} to={`/student/upload?${new URLSearchParams({ assignment_id: t.assignment_id, qp_id: t.qp_id })}`} className="app-row app-tone-brand">
                      <span className="app-row-main">
                        <span className="app-row-title">{t.title}</span>
                        <span className="app-row-meta">{t.class_name}</span>
                      </span>
                      <Pill tone={d.tone}>{d.text}</Pill>
                    </Link>
                  )
                })}
              </div>
            </Panel>
          </Stagger>
        </>
      )}
    </WorkflowPage>
  )
}
