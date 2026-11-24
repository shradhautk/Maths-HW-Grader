import { Link } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { ActionTile, AppHero, AppStat, Panel, ScoreRing, Stagger } from '../../components/app/AppKit'
import { EmptyState, GradePill, Notice, Pill, StatusPill } from '../../components/ui'
import { useAuth } from '../../auth/AuthContext'
import { useAsync } from '../../hooks/useAsync'
import { useDisplayName } from '../../hooks/useDisplayName'
import { isRouteMissing } from '../../lib/api'
import { gradesRequest, toNumber, type GradeResult } from '../../lib/grading'
import { isPublished, reached } from '../../lib/status'
import { formatWhen } from '../../lib/dates'
import { listAnnouncements, listStudentAssignments, type StudentAssignment } from '../../lib/workflow'

interface Result {
  a: StudentAssignment
  marks: number
  max: number
  percent: number
}

/** First name for the greeting, skipping titles like "Dr.". */
function firstName(full: string) {
  const parts = full.split(/\s+/).filter((w) => !/^(dr|mr|mrs|ms|prof)\.?$/i.test(w))
  return parts[0] ?? full
}

function greeting() {
  const h = new Date().getHours()
  return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
}

function dueText(due?: string) {
  if (!due) return null
  const days = Math.ceil((new Date(due).getTime() - Date.now()) / 86_400_000)
  if (days < 0) return { text: 'Overdue', tone: 'danger' as const }
  if (days === 0) return { text: 'Due today', tone: 'warning' as const }
  return { text: `Due in ${days} day${days === 1 ? '' : 's'}`, tone: days <= 2 ? ('warning' as const) : ('neutral' as const) }
}

/** §03 Student dashboard: what's next, how grading is going, and recent results. */
export default function StudentDashboard() {
  const { session } = useAuth()
  const name = useDisplayName('Student')
  const studentId = session?.userId

  const data = useAsync(
    async (force) => {
      const assignments = await listStudentAssignments(force)
      // Scores for published work come from the grades API (max = the assignment's total marks).
      const published = assignments.filter((a) => isPublished(a.submission_status)).slice(0, 6)
      const results = await Promise.all(
        published.map(async (a): Promise<Result | null> => {
          try {
            const g = await gradesRequest<GradeResult>({ operation: 'getSingleStudentGrade', student_id: studentId, assignment_id: a.assignment_id }, { force })
            const marks = toNumber(g?.summary?.[0]?.total_marks)
            const max = toNumber(a.total_marks) || (g?.details ?? []).reduce((s, q) => s + toNumber(q.max_marks), 0)
            return max > 0 ? { a, marks, max, percent: (marks / max) * 100 } : null
          } catch {
            return null
          }
        }),
      )
      return { assignments, results: results.filter((r): r is Result => !!r) }
    },
    [studentId],
  )

  const notes = useAsync((force) => listAnnouncements(force), [])
  const has = !!data.data
  const all = data.data?.assignments ?? []
  const results = data.data?.results ?? []
  const todo = all.filter((a) => !reached(a.submission_status, 'answer_uploaded')).sort((x, y) => (x.due_at ?? '9').localeCompare(y.due_at ?? '9'))
  const inProgress = all.filter((a) => reached(a.submission_status, 'answer_uploaded') && !isPublished(a.submission_status))
  const graded = all.filter((a) => isPublished(a.submission_status))
  const avg = results.length ? results.reduce((s, r) => s + r.percent, 0) / results.length : null
  const unavailable = !!data.error && isRouteMissing(data.error)
  const next = todo[0]
  const q = (a: StudentAssignment) => new URLSearchParams({ assignment_id: a.assignment_id, qp_id: a.qp_id })

  return (
    <WorkflowPage title="Dashboard">
      <AppHero
        eyebrow="Student dashboard"
        title={
          <>
            {greeting()}, <span className="grad">{firstName(name)}</span>
          </>
        }
        text={
          next
            ? `Next up: ${next.title}${next.due_at ? ` — due ${new Date(next.due_at).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })}` : ''}.`
            : has
              ? 'Nothing due right now — check your latest feedback below.'
              : "Here's an overview of your work and feedback."
        }
        actions={
          <>
            <Link to={next ? `/student/upload?${q(next)}` : '/student/assignments'} className="ga-btn ga-btn-primary">
              <span className="material-icons" aria-hidden>
                upload
              </span>
              {next ? 'Submit next assignment' : 'My assignments'}
            </Link>
            <Link to="/student/grades" className="ga-btn ga-btn-secondary">
              View my grades
            </Link>
          </>
        }
        aside={<ScoreRing percent={avg} label={avg == null ? 'no grades yet' : 'average score'} />}
      />

      {unavailable && (
        <Notice tone="warning">
          Your assignment list isn't reachable — the <code>/workflow</code> API may not be deployed on this server yet. You can still upload
          work and look up grades.
        </Notice>
      )}

      <Stagger className="app-stats">
        <AppStat icon="pending_actions" label="To do" value={has ? todo.length : null} tone="brand" note={next && dueText(next.due_at)?.text} />
        <AppStat icon="hourglass_top" label="Being graded" value={has ? inProgress.length : null} tone="sky" />
        <AppStat icon="task_alt" label="Graded" value={has ? graded.length : null} tone="accent" />
        <AppStat
          icon="trending_up"
          label="Average"
          value={avg == null ? null : Math.round(avg)}
          suffix="%"
          tone="warn"
          note={results.length ? `across ${results.length} result${results.length === 1 ? '' : 's'}` : undefined}
        />
      </Stagger>

      <Stagger className="app-tiles">
        <ActionTile to="/student/assignments" icon="assignment" title="My assignments" text="See what's due and track grading." tone="brand" />
        <ActionTile to="/student/progress" icon="trending_up" title="My progress" text="Your grade trend and what's helping." tone="sky" />
        <ActionTile to="/student/announcements" icon="campaign" title="Announcements" text="News and deadlines from your teachers." tone="accent" />
        <ActionTile to="/student/grades" icon="grade" title="Grades & feedback" text="Per-question marks and comments." tone="ink" />
      </Stagger>

      <Stagger className="app-grid">
        <Panel
          title="Up next"
          action={
            <Link to="/student/assignments" className="ga-btn ga-btn-ghost ga-btn-sm">
              All assignments →
            </Link>
          }
        >
          {!has && !data.error && <p className="ga-muted">Loading…</p>}
          {!has && data.error && <p className="ga-muted">Assignments appear here once the workflow API is connected.</p>}
          {has && todo.length + inProgress.length === 0 && <EmptyState title="You're all caught up">No work waiting on you.</EmptyState>}
          <div className="app-rows">
            {[...todo, ...inProgress].slice(0, 6).map((a) => {
              const due = dueText(a.due_at)
              const isTodo = !reached(a.submission_status, 'answer_uploaded')
              return (
                <Link
                  key={a.assignment_id}
                  to={isTodo ? `/student/upload?${q(a)}` : a.submission_status === 'ocr_done' ? `/student/answer-correction?${q(a)}` : '/student/assignments'}
                  className={`app-row ${isTodo ? 'app-tone-brand' : 'app-tone-sky'}`}
                >
                  <span className="app-row-icon material-icons" aria-hidden>
                    {isTodo ? 'edit_document' : 'hourglass_top'}
                  </span>
                  <span className="app-row-main">
                    <span className="app-row-title">{a.title}</span>
                    <span className="app-row-meta">
                      {[a.class_name, a.question_count !== undefined && `${a.question_count} questions`, a.total_marks !== undefined && `${a.total_marks} marks`]
                        .filter(Boolean)
                        .join(' · ')}
                    </span>
                  </span>
                  <span className="app-row-end">
                    {isTodo && due ? <Pill tone={due.tone}>{due.text}</Pill> : <StatusPill status={a.submission_status} />}
                  </span>
                </Link>
              )
            })}
          </div>
        </Panel>
        <div>
        <Panel
          title="Announcements"
          action={
            <Link to="/student/announcements" className="ga-btn ga-btn-ghost ga-btn-sm">
              All →
            </Link>
          }
        >
          {notes.data && notes.data.length === 0 && <p className="ga-muted">No announcements right now.</p>}
          {!notes.data && <p className="ga-muted">{notes.error ? 'Unavailable until the workflow API is connected.' : 'Loading…'}</p>}
          <div className="app-rows">
            {(notes.data ?? []).slice(0, 2).map((n) => (
              <Link key={n.announcement_id} to="/student/announcements" className="app-row app-tone-sky">
                <span className="app-row-icon material-icons" aria-hidden>
                  campaign
                </span>
                <span className="app-row-main">
                  <span className="app-row-title">{n.title}</span>
                  <span className="app-row-meta">
                    {n.class_name} · {formatWhen(n.created_at)}
                  </span>
                </span>
              </Link>
            ))}
          </div>
        </Panel>
        <Panel title="Recent results">
          {has && results.length === 0 && <p className="ga-muted">Published grades will show up here.</p>}
          {!has && <p className="ga-muted">—</p>}
          <div className="app-rows">
            {results.map((r) => (
              <Link key={r.a.assignment_id} to={`/student/grades?${q(r.a)}`} className="app-row app-tone-accent">
                <GradePill percent={r.percent} />
                <span className="app-row-main">
                  <span className="app-row-title">{r.a.title}</span>
                  <span className="app-row-meta">
                    {r.marks}/{r.max} · {Math.round(r.percent)}%
                  </span>
                </span>
                <span className="material-icons" aria-hidden style={{ color: 'var(--text-subtle)' }}>
                  chevron_right
                </span>
              </Link>
            ))}
          </div>
        </Panel>
        </div>
      </Stagger>
    </WorkflowPage>
  )
}
