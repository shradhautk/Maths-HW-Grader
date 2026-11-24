import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import WorkflowPage from '../../components/WorkflowPage'
import TeacherAnnouncements from '../../components/app/TeacherAnnouncements'
import { ActionTile, AppHero, AppStat, Panel, Stagger } from '../../components/app/AppKit'
import { fadeUp } from '../../components/landing/motion'
import { EmptyState, Notice, Pill, StatusPill } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { useDisplayName } from '../../hooks/useDisplayName'
import { isRouteMissing } from '../../lib/api'
import { GRADES_STATUS_META } from '../../lib/status'
import { listAssignments, type Assignment } from '../../lib/workflow'

type Stage = 'setup' | 'rubric' | 'collecting' | 'reviewing' | 'published'

const STAGES: { key: Stage; label: string }[] = [
  { key: 'setup', label: 'Setup' },
  { key: 'rubric', label: 'Rubric' },
  { key: 'collecting', label: 'Collecting' },
  { key: 'reviewing', label: 'Reviewing' },
  { key: 'published', label: 'Published' },
]

/** Where an assignment sits in the §02 B grading pipeline. */
function stageOf(a: Assignment): Stage {
  if (a.grades_status === 'published') return 'published'
  if (a.grades_status === 'ai_graded' || a.grades_status === 'teacher_reviewing') return 'reviewing'
  if (a.qp_status === 'rubric_reviewed') return 'collecting'
  if (a.qp_status === 'qp_parsed' || a.qp_status?.startsWith('rubric')) return 'rubric'
  return 'setup'
}

function nextLink(a: Assignment): string {
  const id = encodeURIComponent(a.assignment_id)
  const stage = stageOf(a)
  if (stage === 'setup' || stage === 'rubric') return `/teacher/assignments/${id}/setup`
  if (stage === 'collecting') return `/teacher/assignments/${id}/submissions`
  if (stage === 'published') return `/teacher/assignments/${id}/analytics`
  return `/teacher/assignments/${id}/grading`
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

/** §02 Teacher dashboard: overview + the grading pipeline up front. */
export default function TeacherDashboard() {
  const name = useDisplayName('Teacher')
  const list = useAsync((force) => listAssignments(force), [])
  const assignments = list.data ?? []
  const unavailable = !!list.error && isRouteMissing(list.error)
  const has = !!list.data

  const byStage = Object.fromEntries(STAGES.map((s) => [s.key, 0])) as Record<Stage, number>
  assignments.forEach((a) => byStage[stageOf(a)]++)
  const submitted = assignments.reduce((n, a) => n + (a.submitted ?? 0), 0)
  const recent = assignments.slice(0, 5)
  const needsYou = assignments.filter((a) => stageOf(a) === 'reviewing' || stageOf(a) === 'rubric')

  return (
    <WorkflowPage title="Teacher Dashboard">
      <AppHero
        eyebrow="Teacher dashboard"
        title={
          <>
            {greeting()}, <span className="grad">{firstName(name)}</span>
          </>
        }
        text={
          has && needsYou.length
            ? `${needsYou.length} assignment${needsYou.length === 1 ? '' : 's'} need${needsYou.length === 1 ? 's' : ''} your attention today.`
            : 'Create an assignment, let GradeFlow™ do the first pass, then review and publish.'
        }
        actions={
          <>
            <Link to="/teacher/assignments/new" className="ga-btn ga-btn-primary">
              <span className="material-icons" aria-hidden>
                add
              </span>
              New assignment
            </Link>
            <Link to="/teacher/assignments" className="ga-btn ga-btn-secondary">
              View all assignments
            </Link>
          </>
        }
        aside={
          <motion.img
            src="/img/grade-antlogo-removebg-preview.png"
            alt=""
            width={120}
            height={120}
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          />
        }
      />

      {unavailable && (
        <Notice tone="warning">
          Assignment data isn't reachable — the <code>/workflow</code> API may not be deployed on this server yet. Uploads, grades and
          correction still work.
        </Notice>
      )}

      <Stagger className="app-stats">
        <AppStat icon="assignment" label="Assignments" value={has ? assignments.length : null} tone="brand" note={has ? `${byStage.published} published` : undefined} />
        <AppStat icon="upload_file" label="Papers submitted" value={has ? submitted : null} tone="sky" />
        <AppStat icon="rate_review" label="Awaiting review" value={has ? byStage.reviewing : null} tone="warn" note={has && byStage.reviewing ? 'open the grading run' : undefined} />
        <AppStat icon="task_alt" label="Published" value={has ? byStage.published : null} tone="accent" />
      </Stagger>

      <Stagger className="app-tiles">
        <ActionTile to="/teacher/assignments/new" icon="post_add" title="Create assignment" text="Upload a question paper and pick AI or solution-key grading." tone="brand" />
        <ActionTile to="/teacher/queue" icon="inbox" title="Grading queue" text="Batches waiting on a spot-check or publish." tone="accent" />
        <ActionTile to="/teacher/analytics" icon="insights" title="Class analytics" text="Scores, submission rates and tough questions." tone="sky" />
        <ActionTile to="/teacher/assistant" icon="auto_awesome" title="GradeFlow assistant" text="Ask who struggled and what to re-teach." tone="ink" />
      </Stagger>

      <Stagger className="app-grid">
        <div>
          <Panel
            title="Grading pipeline"
            action={<span className="ga-muted">Question › Rubric › Collect › Review › Publish</span>}
          >
            <div className="app-pipeline">
              {STAGES.map((s) => (
                <div key={s.key} className={`app-pipe${has && byStage[s.key] && s.key !== 'published' ? ' is-hot' : ''}`}>
                  <b>{has ? byStage[s.key] : '—'}</b>
                  <span>{s.label}</span>
                </div>
              ))}
            </div>
          </Panel>
          <Panel
            title="Recent assignments"
            action={
              <Link to="/teacher/assignments" className="ga-btn ga-btn-ghost ga-btn-sm">
                See all →
              </Link>
            }
          >
            {has && recent.length === 0 && (
              <EmptyState
                title="No assignments yet"
                action={
                  <Link to="/teacher/assignments/new" className="ga-btn ga-btn-primary">
                    Create your first assignment
                  </Link>
                }
              >
                Upload a question paper to get started.
              </EmptyState>
            )}
            {!has && !list.error && <p className="ga-muted">Loading…</p>}
            {!has && list.error && <p className="ga-muted">Recent assignments appear here once the workflow API is connected.</p>}
            <div className="app-rows">
              {recent.map((a) => {
                const grades = GRADES_STATUS_META[a.grades_status ?? 'not_started']
                return (
                  <Link key={a.assignment_id} to={nextLink(a)} className="app-row app-tone-brand">
                    <span className="app-row-icon material-icons" aria-hidden>
                      description
                    </span>
                    <span className="app-row-main">
                      <span className="app-row-title">{a.title}</span>
                      <span className="app-row-meta">
                        {[a.class_name, a.enrolled !== undefined ? `${a.submitted ?? 0}/${a.enrolled} submitted` : `${a.submitted ?? 0} submitted`, a.due_at && `due ${new Date(a.due_at).toLocaleDateString()}`]
                          .filter(Boolean)
                          .join(' · ')}
                      </span>
                    </span>
                    <span className="app-row-end">
                      {a.qp_status && a.qp_status !== 'rubric_reviewed' && <StatusPill status={a.qp_status} />}
                      <Pill tone={grades.tone}>{grades.label}</Pill>
                    </span>
                  </Link>
                )
              })}
            </div>
          </Panel>
        </div>
        <div>
          <Panel title="Needs your attention">
            {has && needsYou.length === 0 && <p className="ga-muted">You're all caught up. 🎉</p>}
            {!has && <p className="ga-muted">—</p>}
            <div className="app-rows">
              {needsYou.slice(0, 5).map((a) => (
                <Link key={a.assignment_id} to={nextLink(a)} className={`app-row ${stageOf(a) === 'reviewing' ? 'app-tone-warn' : 'app-tone-sky'}`}>
                  <span className="app-row-icon material-icons" aria-hidden>
                    {stageOf(a) === 'reviewing' ? 'flag' : 'checklist'}
                  </span>
                  <span className="app-row-main">
                    <span className="app-row-title">{a.title}</span>
                    <span className="app-row-meta">{stageOf(a) === 'reviewing' ? 'Grades ready to review' : 'Rubric needs approval'}</span>
                  </span>
                </Link>
              ))}
            </div>
          </Panel>
          <motion.div variants={fadeUp}>
            <TeacherAnnouncements classes={[...new Set(assignments.map((a) => a.class_name).filter((c): c is string => !!c))]} />
            <Panel title="How GradeAnt grades">
              <ol className="app-steps">
                <li>
                  <div>
                    <strong>Upload the question paper</strong>
                    <span>Students see the assignment straight away.</span>
                  </div>
                </li>
                <li>
                  <div>
                    <strong>Approve the rubric</strong>
                    <span>Let AI draft it or use your own solution key.</span>
                  </div>
                </li>
                <li>
                  <div>
                    <strong>Spot-check & publish</strong>
                    <span>Only low-confidence answers need you.</span>
                  </div>
                </li>
              </ol>
            </Panel>
          </motion.div>
        </div>
      </Stagger>
    </WorkflowPage>
  )
}
