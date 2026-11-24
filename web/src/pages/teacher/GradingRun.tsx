import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import WorkflowPage from '../../components/WorkflowPage'
import { Card, EmptyState, GradeDistribution, LoadError, Notice, PageHeader, Pill, ProgressBar, Spinner, StatTile, StatusPill, StepProgress } from '../../components/ui'
import { useAlerts } from '../../hooks/useAlerts'
import { useAsync } from '../../hooks/useAsync'
import { getGradeLetter, toNumber } from '../../lib/grading'
import { isPublished, reached } from '../../lib/status'
import { acceptFlagged, getAssignment, getGradingRun, listFlagged, publishGrades, type FlaggedItem } from '../../lib/workflow'

const POLL_MS = 5000
const STEPS = ['Queue', 'Processing', 'Results', 'Spot-check', 'Publish']

/**
 * §15 AI Grading Run: queue → GradeFlow™ processing → results → spot-check,
 * then publish (§19/§20). Polls the run while anything is still grading.
 */
export default function GradingRun() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const { alerts, show, dismiss } = useAlerts(6000)
  const assignment = useAsync((force) => getAssignment(id, force), [id])
  const run = useAsync((force) => getGradingRun(id, force), [id])
  const flagged = useAsync((force) => listFlagged(id, force), [id])
  const [resolved, setResolved] = useState<Set<string>>(new Set())
  const [publishing, setPublishing] = useState(false)

  const subs = run.data?.submissions ?? []
  const total = run.data?.total ?? subs.length
  const graded = subs.filter((s) => reached(s.status, 'ai_graded')).length
  // Waiting on the student (not submitted / OCR not confirmed) never blocks publishing;
  // only papers that are confirmed or mid-grading do (same rule as publishGrades).
  const queued = subs.filter((s) => !reached(s.status, 'corrected')).length
  const grading = subs.filter((s) => s.status === 'grading_triggered' || s.status === 'corrected').length
  const published = subs.length > 0 && subs.every((s) => isPublished(s.status))
  const openFlags = (flagged.data ?? []).filter((f) => !resolved.has(flagKey(f)))
  const step = published ? 4 : grading > 0 ? 1 : graded === 0 ? 0 : openFlags.length > 0 ? 3 : 4

  useEffect(() => {
    if (run.loading || published || grading === 0) return
    const t = window.setTimeout(() => {
      run.reload()
      flagged.reload()
    }, POLL_MS)
    return () => clearTimeout(t)
  }, [run, flagged, grading, published])

  const maxMarks = assignment.data?.total_marks ?? 0
  const counts = { A: 0, B: 0, C: 0, D: 0, F: 0 }
  if (maxMarks > 0) {
    for (const s of subs) {
      if (s.total_marks === undefined || !reached(s.status, 'ai_graded')) continue
      counts[getGradeLetter((toNumber(s.total_marks) / maxMarks) * 100).letter as keyof typeof counts]++
    }
  }

  async function accept(item: FlaggedItem) {
    try {
      await acceptFlagged(item)
      setResolved((r) => new Set(r).add(flagKey(item)))
      run.reload()
    } catch (err) {
      show(`Couldn't accept: ${(err as Error).message}`, 'danger')
    }
  }

  async function publish() {
    setPublishing(true)
    try {
      const res = await publishGrades(id)
      show(`Grades published${res?.notified ? ` — ${res.notified} students notified` : ''}.`, 'success')
      run.reload()
      assignment.reload()
    } catch (err) {
      show(`Couldn't publish: ${(err as Error).message}`, 'danger')
    } finally {
      setPublishing(false)
    }
  }

  const a = assignment.data
  return (
    <WorkflowPage title="Grading Run">
      <PageHeader
        back="/teacher/assignments"
        eyebrow="AI grading run"
        title={a?.title ?? 'Grading run'}
        subtitle={a?.class_name}
        actions={
          <Link to={`/teacher/assignments/${encodeURIComponent(id)}/submissions`} className="ga-btn ga-btn-secondary">
            Submissions
          </Link>
        }
      />
      <StepProgress steps={STEPS} current={step} />
      <AlertStack alerts={alerts} onDismiss={dismiss} />
      {run.error && <LoadError error={run.error} onRetry={() => run.reload()} />}
      {run.loading && !run.data && <Spinner label="Loading grading run…" />}

      {run.data && (
        <>
          <Card title={published ? 'Grades published' : graded === total && total > 0 ? 'Grading complete' : `GradeFlow™ is grading ${total} papers`}>
            <ProgressBar percent={total ? (graded / total) * 100 : 0} tone={graded === total ? 'success' : 'brand'} />
            <p className="ga-muted">
              {graded} of {total} graded · grading runs in the background — you can leave this page.
            </p>
            <div className="ga-grid-4">
              <StatTile label="Graded" value={graded} tone="success" />
              <StatTile label="Grading now" value={grading} tone="brand" />
              <StatTile label="Waiting on students" value={queued} note="not submitted or OCR not confirmed" />
              <StatTile label="Low confidence" value={openFlags.length} tone="warning" />
            </div>
          </Card>

          <div className="ga-grid-2">
            <Card title="Grade distribution">
              {maxMarks > 0 ? (
                <GradeDistribution counts={counts} />
              ) : (
                <p className="ga-muted">Shown once the assignment's total marks are known.</p>
              )}
            </Card>
            <Card title="Queue">
              {subs.length === 0 ? (
                <EmptyState title="No submissions yet">Papers appear here as students upload them.</EmptyState>
              ) : (
                <div className="ga-table-wrap">
                  <table className="ga-table">
                    <thead>
                      <tr>
                        <th>Student</th>
                        <th>Pages</th>
                        <th>State</th>
                      </tr>
                    </thead>
                    <tbody>
                      {subs.map((s) => (
                        <tr key={s.student_id}>
                          <td>{s.student_name ?? s.student_id}</td>
                          <td>{s.pages ?? '–'}</td>
                          <td>
                            <StatusPill status={s.status} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </div>

          <Card
            title={`Spot-check · ${openFlags.length} flagged`}
            actions={openFlags.length > 0 && <Pill tone="warning">Review before publishing</Pill>}
          >
            {flagged.error && <LoadError error={flagged.error} onRetry={() => flagged.reload()} />}
            {openFlags.length === 0 && !flagged.error && <p className="ga-muted">Nothing flagged — every answer scored with high confidence.</p>}
            <div className="ga-list">
              {openFlags.map((f) => (
                <div key={flagKey(f)} className="ga-item is-attention" style={{ alignItems: 'flex-start' }}>
                  <div style={{ flex: 1, minWidth: 240 }}>
                    <div className="ga-row">
                      <span className="ga-item-title">
                        {f.student_name ?? f.student_id} · Q{f.question_number}
                        {f.subpart ?? ''}
                      </span>
                      <Pill tone="warning">Confidence {Math.round(f.confidence * 100)}%</Pill>
                    </div>
                    <div className="ga-grid-2" style={{ marginTop: 8 }}>
                      {f.reference_answer && (
                        <div>
                          <div className="ga-muted">Reference</div>
                          <div className="ga-mono">{f.reference_answer}</div>
                        </div>
                      )}
                      {f.student_answer && (
                        <div>
                          <div className="ga-muted">Student answer (OCR)</div>
                          <div className="ga-mono">{f.student_answer}</div>
                        </div>
                      )}
                    </div>
                    <div className="ga-item-meta">
                      AI proposed: <strong>{f.proposed_marks} / {f.max_marks}</strong>
                    </div>
                  </div>
                  <div className="ga-row">
                    <button type="button" className="ga-btn ga-btn-accent ga-btn-sm" onClick={() => accept(f)}>
                      Accept
                    </button>
                    <button
                      type="button"
                      className="ga-btn ga-btn-secondary ga-btn-sm"
                      onClick={() =>
                        navigate(
                          `/teacher/student-details?${new URLSearchParams({
                            student_id: f.student_id,
                            assignment_id: id,
                            qp_id: a?.qp_id ?? '',
                            focus_q: String(f.question_number),
                            focus_sub: f.subpart ?? '',
                            from: 'correction',
                          })}`,
                        )
                      }
                    >
                      Correct manually →
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {!published && (
            <Notice
              tone={openFlags.length || grading ? 'warning' : 'success'}
              action={
                <button
                  type="button"
                  className="ga-btn ga-btn-primary"
                  disabled={publishing || openFlags.length > 0 || grading > 0 || graded === 0}
                  onClick={publish}
                >
                  {publishing ? 'Publishing…' : 'Publish grades'}
                </button>
              }
            >
              {openFlags.length || grading
                ? 'Publish unlocks once grading has finished and every flagged answer is cleared.'
                : 'All clear — publishing notifies every student at once.'}
            </Notice>
          )}
        </>
      )}
    </WorkflowPage>
  )
}

function flagKey(f: FlaggedItem) {
  return `${f.evaluation_id}:${f.question_number}:${f.subpart ?? ''}`
}
