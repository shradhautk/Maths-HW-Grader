import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import AppShell from '../../components/AppShell'
import PageScope from '../../components/PageScope'
import { useAlerts } from '../../hooks/useAlerts'
import { API } from '../../config'
import { apiRequest } from '../../lib/api'
import { apiCache } from '../../lib/apiCache'
import { getMarksStatus, gradesRequest, type GradeResult, type GradeSummary } from '../../lib/grading'
import '../../styles/themes/aesthetic.css'
import '../../styles/themes/teacher-grade.css'
import '../../styles/themes/teacher-correction.css'

interface WorkItem {
  student_id: string
  question_number: string | number
  subpart: string
  marks: unknown
  max_marks: unknown
  status: 'incorrect' | 'partial'
}
type Filter = 'all' | 'incorrect' | 'partial'

const LAST_QUERY_KEY = 'lastQuery:correction'
const STEPS = ['Worklist', 'Annotate', 'Rubric', 'Override', 'Finalize']

const itemKey = (i: WorkItem) => `${i.student_id}_${i.question_number}_${i.subpart || 'main'}`
const reviewedKey = (qp: string, a: string) => `correction_reviewed_${qp}_${a}`

function loadReviewed(qp: string, a: string): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(reviewedKey(qp, a)) ?? '[]') as string[])
  } catch {
    return new Set()
  }
}

/**
 * Manual Grading & Correction worklist. There's no backend operation that
 * returns flagged answers, so this pulls the roster (getAllStudentGrades) and
 * walks each student's details (getSingleStudentGrade), flagging every
 * partial/incorrect question client-side — N+1 requests, fine for a class.
 *
 * TODO(backend): a "getGradingWorklist" operation returning the flagged
 * items in one call, and a real "finalizeCorrection" operation.
 */
export default function TeacherCorrection() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const { alerts, show, dismiss } = useAlerts()

  const [qpId, setQpId] = useState('')
  const [assignmentId, setAssignmentId] = useState('')
  const [scanned, setScanned] = useState<{ qpId: string; assignmentId: string } | null>(null)
  const [scanning, setScanning] = useState(false)
  const [progress, setProgress] = useState('')
  const [items, setItems] = useState<WorkItem[] | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  // Expanded groups are keyed by student_id (never list position), so
  // opening one student can't reveal another student's questions.
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [reviewed, setReviewed] = useState<Set<string>>(new Set())
  const [finalizing, setFinalizing] = useState(false)

  async function runScan(qp: string, aId: string, force = false) {
    setScanned({ qpId: qp, assignmentId: aId })
    setExpanded(new Set())
    setReviewed(loadReviewed(qp, aId))
    setItems(null)
    setScanning(true)
    setProgress('Loading class roster…')
    try {
      const roster = await gradesRequest<{ grades?: GradeSummary[] }>(
        { operation: 'getAllStudentGrades', qp_id: qp, assignment_id: aId },
        { force },
      )
      apiCache.set(LAST_QUERY_KEY, { qpId: qp, assignmentId: aId })
      const grades = Array.isArray(roster?.grades) ? roster.grades : []

      const found: WorkItem[] = []
      for (let i = 0; i < grades.length; i++) {
        const studentId = String(grades[i].student_id ?? '')
        setProgress(`Scanning ${i + 1} of ${grades.length} — ${studentId || 'student'}…`)
        try {
          const detail = await gradesRequest<GradeResult>(
            { operation: 'getSingleStudentGrade', student_id: studentId, assignment_id: aId },
            { force },
          )
          for (const q of detail?.details ?? []) {
            const status = getMarksStatus(q.marks, q.max_marks)
            if (status === 'incorrect' || status === 'partial') {
              found.push({
                student_id: studentId,
                question_number: q.question_number ?? '',
                subpart: q.subpart || '',
                marks: q.marks,
                max_marks: q.max_marks,
                status,
              })
            }
          }
        } catch {
          /* skip a student whose details fail to load */
        }
      }
      // Needs-correction before partial credit, then by student.
      found.sort((a, b) =>
        a.status !== b.status ? (a.status === 'incorrect' ? -1 : 1) : a.student_id.localeCompare(b.student_id),
      )
      setItems(found)
    } catch (err) {
      show(`Failed to scan for flagged answers: ${(err as Error).message}`, 'danger')
    } finally {
      setScanning(false)
    }
  }

  // Prefill from ?qp_id=&assignment_id= (Students Grades link), else the last scan.
  useEffect(() => {
    const qp = params.get('qp_id')?.trim()
    const aId = params.get('assignment_id')?.trim()
    const last = apiCache.get<{ qpId?: string; assignmentId?: string }>(LAST_QUERY_KEY)
    const initial = qp && aId ? { qpId: qp, assignmentId: aId } : last?.qpId && last.assignmentId ? last : null
    if (qp) setQpId(qp)
    if (aId) setAssignmentId(aId)
    if (initial?.qpId && initial.assignmentId) {
      setQpId(initial.qpId)
      setAssignmentId(initial.assignmentId)
      void runScan(initial.qpId, initial.assignmentId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!qpId.trim() || !assignmentId.trim()) {
      show('Please fill in both Question Paper ID and Assignment ID.', 'danger')
      return
    }
    void runScan(qpId.trim(), assignmentId.trim())
  }

  function markReviewed(item: WorkItem) {
    if (!scanned) return
    const next = new Set(reviewed).add(itemKey(item))
    setReviewed(next)
    try {
      localStorage.setItem(reviewedKey(scanned.qpId, scanned.assignmentId), JSON.stringify([...next]))
    } catch {
      /* storage unavailable: reviewed state lasts for this visit only */
    }
  }

  async function finalize() {
    if (!scanned || !items) return
    setFinalizing(true)
    try {
      // Best-effort: the operation doesn't exist on the backend yet.
      await apiRequest(API.marks, {
        body: {
          operation: 'finalizeCorrection',
          qp_id: scanned.qpId,
          assignment_id: scanned.assignmentId,
          reviewed_count: reviewed.size,
          total_count: items.length,
        },
      })
      apiCache.clearPrefix('grades:')
    } catch {
      /* not implemented server-side yet; continue to grades */
    } finally {
      navigate(`/teacher/grades?${new URLSearchParams({ qp_id: scanned.qpId, assignment_id: scanned.assignmentId })}`)
    }
  }

  const counts = useMemo(() => {
    const c = { all: items?.length ?? 0, incorrect: 0, partial: 0 }
    items?.forEach((i) => c[i.status]++)
    return c
  }, [items])

  const groups = useMemo(() => {
    const map = new Map<string, WorkItem[]>()
    items?.forEach((i) => {
      if (filter !== 'all' && i.status !== filter) return
      map.set(i.student_id, [...(map.get(i.student_id) ?? []), i])
    })
    return [...map.entries()].map(([studentId, list]) => ({ studentId, items: list }))
  }, [items, filter])

  const reviewedCount = items?.filter((i) => reviewed.has(itemKey(i))).length ?? 0

  const detailsLink = (i: WorkItem) =>
    `/teacher/student-details?${new URLSearchParams({
      student_id: i.student_id,
      assignment_id: scanned?.assignmentId ?? '',
      qp_id: scanned?.qpId ?? '',
      focus_q: String(i.question_number),
      focus_sub: i.subpart,
      from: 'correction',
    })}`

  return (
    <PageScope name="teacher-correction" theme={['aesthetic', 'teacher-grade', 'teacher-correction']} title="Manual Grading">
      <AppShell footer={false}>
        <main className="main-content">
          <div className="container">
            <section className="tab-content active" role="tabpanel">
              <div className="card">
                <div className="card-header">
                  <h2>
                    <i className="fas fa-marker" /> Manual Grading &amp; Correction
                  </h2>
                  <p>
                    Worklist &rarr; Annotate &rarr; Rubric score &rarr; Override AI &rarr; Finalize. Fed by AI-flagged
                    answers (and, once the backend supports it, student re-check requests).
                  </p>
                </div>
                <div className="card-body">
                  <div className="correction-steps" aria-label="Correction workflow steps">
                    {STEPS.map((step, i) => (
                      <span key={step} style={{ display: 'contents' }}>
                        {i > 0 && <span className="step-sep">&rsaquo;</span>}
                        <span className={`correction-step${i === 0 ? ' active' : ''}`}>
                          <span className="step-dot">{i + 1}</span>
                          {step}
                        </span>
                      </span>
                    ))}
                  </div>

                  <form
                    className="form"
                    onSubmit={handleSubmit}
                    onReset={() => {
                      setQpId('')
                      setAssignmentId('')
                    }}
                  >
                    <div className="form-row">
                      <div className="form-group">
                        <label htmlFor="qp-id">Question Paper ID</label>
                        <input
                          type="text"
                          id="qp-id"
                          className="form-control"
                          placeholder="Enter question paper ID"
                          required
                          value={qpId}
                          onChange={(e) => setQpId(e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label htmlFor="assignment-id">Assignment ID</label>
                        <input
                          type="text"
                          id="assignment-id"
                          className="form-control"
                          placeholder="Enter assignment ID"
                          required
                          value={assignmentId}
                          onChange={(e) => setAssignmentId(e.target.value)}
                        />
                      </div>
                    </div>
                    <div className="form-actions">
                      <button type="submit" className="btn btn-primary" disabled={scanning}>
                        <i className="fas fa-magnifying-glass" /> Scan for flagged answers
                      </button>
                      <button type="reset" className="btn btn-secondary">
                        <i className="fas fa-redo" /> Clear
                      </button>
                    </div>
                  </form>

                  <AlertStack alerts={alerts} onDismiss={dismiss} />

                  {scanning && (
                    <div className="loader-container" style={{ display: 'block' }}>
                      <div className="loader" />
                      <div className="scan-progress">{progress}</div>
                    </div>
                  )}

                  {items && (
                    <div className="result-container" style={{ display: 'block' }}>
                      <div className="worklist-toolbar">
                        <h4>Needs your attention</h4>
                        <div className="worklist-filters">
                          {(
                            [
                              ['all', 'All', ''],
                              ['incorrect', 'Needs correction', ' filter-danger'],
                              ['partial', 'Partial credit', ' filter-warning'],
                            ] as const
                          ).map(([key, label, cls]) => (
                            <button
                              key={key}
                              type="button"
                              className={`filter-pill${cls}${filter === key ? ' active' : ''}`}
                              onClick={() => setFilter(key)}
                            >
                              {label} &middot; <span>{counts[key]}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="worklist-items">
                        {groups.map(({ studentId, items: list }) => {
                          const open = expanded.has(studentId)
                          const done = list.filter((i) => reviewed.has(itemKey(i))).length
                          return (
                            <div className="worklist-student-group" key={studentId}>
                              <button
                                type="button"
                                className="worklist-student-header"
                                aria-expanded={open}
                                onClick={() =>
                                  setExpanded((s) => {
                                    const next = new Set(s)
                                    if (next.has(studentId)) next.delete(studentId)
                                    else next.add(studentId)
                                    return next
                                  })
                                }
                              >
                                <span className="worklist-item-avatar">{(studentId || '?').slice(0, 2).toUpperCase()}</span>
                                <div className="worklist-item-body">
                                  <div className="worklist-item-title">{studentId}</div>
                                  <div className="worklist-item-meta">
                                    {list.length} flagged answer{list.length === 1 ? '' : 's'} &middot; {done}/{list.length}{' '}
                                    reviewed
                                  </div>
                                </div>
                                <span className="worklist-student-chevron">{open ? '▼' : '▶'}</span>
                              </button>

                              {open && (
                                <div className="worklist-student-body">
                                  {list.map((item) => {
                                    const isReviewed = reviewed.has(itemKey(item))
                                    return (
                                      <div
                                        key={itemKey(item)}
                                        className={`worklist-item status-${item.status}${isReviewed ? ' is-reviewed' : ''}`}
                                      >
                                        <div className="worklist-item-body">
                                          <div className="worklist-item-title">
                                            Q{String(item.question_number)}
                                            {item.subpart}
                                            {isReviewed && <span className="worklist-item-reviewed-tag">✓ reviewed</span>}
                                          </div>
                                          <div className="worklist-item-meta">
                                            {item.status === 'incorrect'
                                              ? 'AI flagged: needs correction'
                                              : 'AI flagged: partial credit'}{' '}
                                            &middot; {String(item.marks)}/{String(item.max_marks)} marks
                                          </div>
                                        </div>
                                        <Link className="btn btn-secondary" to={detailsLink(item)} onClick={() => markReviewed(item)}>
                                          Open &rarr;
                                        </Link>
                                      </div>
                                    )
                                  })}
                                </div>
                              )}
                            </div>
                          )
                        })}
                      </div>

                      {items.length === 0 && (
                        <div className="worklist-empty" style={{ display: 'block' }}>
                          <i className="fas fa-circle-check" /> No flagged answers found for this assignment &mdash; every
                          graded question is at full marks.
                        </div>
                      )}

                      <div className="finalize-panel">
                        <div className="finalize-summary">
                          <div className="finalize-title">Finalize corrections</div>
                          <div
                            className={`finalize-count${items.length > 0 && reviewedCount === items.length ? ' all-done' : ''}`}
                          >
                            {reviewedCount} / {items.length} opened
                          </div>
                        </div>
                        <p className="finalize-note">
                          Opening an item from the worklist marks it reviewed. Finalizing hands this assignment off to
                          grade publishing.
                        </p>
                        <div className="form-actions">
                          <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => scanned && runScan(scanned.qpId, scanned.assignmentId, true)}
                          >
                            <i className="fas fa-rotate" /> Re-scan
                          </button>
                          <button type="button" className="btn btn-primary" disabled={finalizing} onClick={finalize}>
                            {finalizing ? (
                              'Finalizing…'
                            ) : (
                              <>
                                Finalize &amp; view grades <i className="fas fa-arrow-right" />
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>
          </div>
        </main>
      </AppShell>
    </PageScope>
  )
}
