import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import { AssistantDock } from '../../components/assistant/AssistantPanel'
import AppShell from '../../components/AppShell'
import PageScope from '../../components/PageScope'
import QuestionViewer, { FeedbackToggle } from '../../components/grades/QuestionViewer'
import ScoreHero from '../../components/grades/ScoreHero'
import { useAlerts } from '../../hooks/useAlerts'
import { apiRequest } from '../../lib/api'
import { apiCache } from '../../lib/apiCache'
import {
  formatTimestamp,
  gradesRequest,
  maxMarksOf,
  renderContent,
  sortDetails,
  STATUS_BADGE,
  toNumber,
  type GradeResult,
  type MarksStatus,
  type QuestionDetail,
} from '../../lib/grading'
import '../../styles/themes/aesthetic.css'
import '../../styles/themes/teacher-grade.css'

interface RubricRow {
  name: string
  points: number
}
type AnnotationType = 'highlight' | 'strike' | 'comment'
interface Annotation {
  type: AnnotationType
  text: string
  comment?: string
}

function omit<T>(obj: Record<string | number, T>, key: string | number): Record<string | number, T> {
  const next = { ...obj }
  delete next[key]
  return next
}

/** Stable per-question key: q{number}_{subpart|main}. */
const qid = (q: QuestionDetail) => `q${q.question_number}_${q.subpart || 'main'}`

/**
 * Teacher view of one student's assignment: edit per-question marks (inline
 * or via an ad-hoc rubric), annotate answers, see AI-vs-override diffs, and
 * save everything back with the "updateMarks" operation.
 *
 * TODO(backend): ai_marks / rubric / annotations are sent with each detail
 * but the updateMarks operation doesn't persist them yet.
 */
export default function StudentDetails() {
  const [params] = useSearchParams()
  const query = useMemo(
    () => ({
      student_id: params.get('student_id'),
      assignment_id: params.get('assignment_id'),
      qp_id: params.get('qp_id'),
      operation: 'getSingleStudentGrade',
    }),
    [params],
  )
  const fromCorrection = params.get('from') === 'correction'
  const focusQ = params.get('focus_q')
  const focusSub = params.get('focus_sub') ?? ''

  const { alerts, show, dismiss } = useAlerts()
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<GradeResult | null>(null)
  const [updating, setUpdating] = useState(false)
  const [preview, setPreview] = useState(false)

  // Inline edit state: summary row index / question id -> draft value.
  const [summaryDraft, setSummaryDraft] = useState<Record<number, string>>({})
  const [questionDraft, setQuestionDraft] = useState<Record<string, string>>({})

  // Correction-workspace extras, keyed by qid().
  const aiMarks = useRef(new Map<string, number>())
  const [rubrics, setRubrics] = useState<Record<string, RubricRow[]>>({})
  const [rubricOpen, setRubricOpen] = useState<Record<string, boolean>>({})
  const annotations = useRef(new Map<string, Annotation[]>())
  const originalAnswerHtml = useRef(new Map<string, string>())
  const answerRefs = useRef(new Map<string, HTMLPreElement>())
  const [pendingComment, setPendingComment] = useState<{ id: string; range: Range; text: string } | null>(null)

  const load = useCallback(
    async (force = false) => {
      if (!query.student_id || !query.assignment_id) {
        show('Missing required parameters: student_id or assignment_id', 'danger')
        return
      }
      setLoading(true)
      try {
        const result = await gradesRequest(query, { force })
        if (!Array.isArray(result?.summary)) {
          show('No summary data available', 'warning')
          return
        }
        // Remember each question's AI mark the first time we see it, so the
        // override diff survives a refresh after saving.
        result.details?.forEach((q) => {
          if (!aiMarks.current.has(qid(q))) aiMarks.current.set(qid(q), toNumber(q.marks))
        })
        setData(structuredClone(result))
        setSummaryDraft({})
        setQuestionDraft({})
      } catch (err) {
        show(`Failed to load data: ${(err as Error).message}`, 'danger')
      } finally {
        setLoading(false)
      }
    },
    [query, show],
  )

  useEffect(() => {
    void load()
  }, [load])

  const sorted = useMemo(() => sortDetails(data?.details ?? []), [data?.details])
  const focusIndex = useMemo(() => {
    if (!focusQ) return undefined
    const i = sorted.findIndex((q) => String(q.question_number) === focusQ && (q.subpart || '') === focusSub)
    return i >= 0 ? i : undefined
    // Only resolve the deep link once per question set, not on every mark edit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusQ, focusSub, sorted.length])

  // ---- Marks ----

  /** Sets one question's marks and re-derives the summary total from all questions. */
  function setQuestionMarks(id: string, value: number) {
    if (!data?.details) return
    const details = data.details.map((q) => (qid(q) === id ? { ...q, marks: value } : q))
    const total = details.reduce((s, q) => s + toNumber(q.marks), 0)
    const summary = data.summary?.map((s, i) => (i === 0 ? { ...s, total_marks: total } : s))
    setData({ ...data, details, summary })
    show(`Total marks automatically updated to ${total}`, 'success')
  }

  function toggleQuestionEdit(q: QuestionDetail) {
    const id = qid(q)
    if (!(id in questionDraft)) {
      setQuestionDraft((d) => ({ ...d, [id]: String(q.marks ?? 0) }))
      return
    }
    const value = parseFloat(questionDraft[id]) || 0
    const max = toNumber(q.max_marks) || 100
    if (value > max) {
      show(`Marks cannot exceed maximum of ${max}`, 'warning')
      setQuestionDraft((d) => ({ ...d, [id]: String(max) }))
      return
    }
    if (value < 0) {
      show('Marks cannot be negative', 'warning')
      setQuestionDraft((d) => ({ ...d, [id]: '0' }))
      return
    }
    setQuestionDraft((d) => omit(d, id))
    setQuestionMarks(id, value)
  }

  function toggleSummaryEdit(index: number) {
    if (!(index in summaryDraft)) {
      setSummaryDraft((d) => ({ ...d, [index]: String(data?.summary?.[index]?.total_marks ?? 0) }))
      return
    }
    const value = parseFloat(summaryDraft[index]) || 0
    setSummaryDraft((d) => omit(d, index))
    setData((d) => d && { ...d, summary: d.summary?.map((s, i) => (i === index ? { ...s, total_marks: value } : s)) })
  }

  // ---- Rubric ----

  const rubricSum = (id: string) => (rubrics[id] ?? []).reduce((s, r) => s + (Number(r.points) || 0), 0)

  function toggleRubric(id: string) {
    setRubricOpen((o) => ({ ...o, [id]: !o[id] }))
    if (!rubricOpen[id] && !(rubrics[id]?.length)) setRubrics((r) => ({ ...r, [id]: [{ name: '', points: 0 }] }))
  }

  function updateRubric(id: string, fn: (rows: RubricRow[]) => RubricRow[]) {
    setRubrics((r) => ({ ...r, [id]: fn(r[id] ?? []) }))
  }

  function applyRubric(q: QuestionDetail) {
    const id = qid(q)
    const max = toNumber(q.max_marks) || 100
    const sum = rubricSum(id)
    const clamped = Math.max(0, Math.min(sum, max))
    if (sum > max) show(`Rubric total (${sum}) exceeds this question's max marks (${max}) — applied as ${clamped}.`, 'warning')
    setQuestionDraft((d) => omit(d, id))
    setQuestionMarks(id, clamped)
    show('Rubric score applied to this question.', 'success')
  }

  // ---- Annotation (operates on the selection inside an answer <pre>) ----

  function selectionIn(id: string): Range | null {
    const pre = answerRefs.current.get(id)
    const sel = window.getSelection()
    if (!pre || !sel || sel.rangeCount === 0 || sel.isCollapsed) return null
    const range = sel.getRangeAt(0)
    return pre.contains(range.commonAncestorContainer) ? range : null
  }

  function wrapRange(id: string, range: Range, type: AnnotationType, comment?: string) {
    const pre = answerRefs.current.get(id)
    if (!pre) return
    // Capture the pristine answer once so "clear" can restore it exactly.
    if (!originalAnswerHtml.current.has(id)) originalAnswerHtml.current.set(id, pre.innerHTML)
    const wrapper = document.createElement(type === 'strike' ? 'del' : type === 'comment' ? 'span' : 'mark')
    wrapper.className = `annotation-${type}`
    if (comment) wrapper.title = comment
    const text = range.toString()
    try {
      range.surroundContents(wrapper)
    } catch {
      show('Could not annotate that selection — try selecting text within a single line.', 'warning')
      return
    }
    window.getSelection()?.removeAllRanges()
    annotations.current.set(id, [...(annotations.current.get(id) ?? []), { type, text, comment }])
  }

  function annotate(id: string, type: AnnotationType) {
    const range = selectionIn(id)
    if (!range) {
      show('Select some text inside the answer first, then choose an annotation.', 'warning')
      return
    }
    if (type === 'comment') {
      // Ask inline (the old page used window.prompt).
      setPendingComment({ id, range: range.cloneRange(), text: '' })
      return
    }
    wrapRange(id, range, type)
  }

  function clearAnnotations(id: string) {
    const pre = answerRefs.current.get(id)
    const original = originalAnswerHtml.current.get(id)
    if (pre && original !== undefined) pre.innerHTML = original
    annotations.current.set(id, [])
  }

  // ---- Save ----

  async function handleUpdateMarks() {
    if (!data?.summary) {
      show('No data available to update', 'warning')
      return
    }
    // Round-trip the full original records; only marks/total_marks changed.
    const details = sortDetails(data.details ?? []).map((q) => ({
      ...q,
      subpart: q.subpart || null,
      ai_marks: aiMarks.current.get(qid(q)) ?? q.marks,
      rubric: rubrics[qid(q)] ?? [],
      annotations: annotations.current.get(qid(q)) ?? [],
    }))

    const seen = new Set<string>()
    for (const d of details) {
      const key = `${d.question_number}-${d.subpart || 'main'}`
      const invalid =
        toNumber(d.marks) > toNumber(d.max_marks) ||
        typeof d.question_number !== 'number' ||
        typeof d.marks !== 'number' ||
        typeof d.max_marks !== 'number' ||
        !d.evaluation_id ||
        typeof d.timestamp !== 'number' ||
        seen.has(key)
      if (invalid) {
        show('Failed to update marks: Data validation failed - marks and max_marks mismatch detected', 'danger')
        return
      }
      seen.add(key)
    }

    setUpdating(true)
    try {
      const response = await apiRequest<{ success?: boolean; statusCode?: number; message?: string; error?: string }>(
        '/updatedb',
        { body: { operation: 'updateMarks', summary: data.summary.map((g) => ({ ...g })), details }, raw: true },
      )
      apiCache.clearPrefix('grades:')
      if (!(response?.success || response?.statusCode === 200)) {
        throw new Error(response?.message || response?.error || 'Update failed')
      }
      show('All marks updated successfully!', 'success')
      await load(true)
    } catch (err) {
      show(`Failed to update marks: ${(err as Error).message}`, 'danger')
    } finally {
      setUpdating(false)
    }
  }

  // ---- Render ----

  function renderCard(q: QuestionDetail, _i: number, status: MarksStatus) {
    const id = qid(q)
    const ts = formatTimestamp(q.timestamp)
    const editing = id in questionDraft
    const ai = aiMarks.current.get(id)
    const marks = toNumber(q.marks)
    const max = q.max_marks || 0
    const rows = rubrics[id] ?? []

    return (
      <>
        <div className="question-header">
          <h3>
            Question {String(q.question_number ?? '')}
            {q.subpart ? ` (${q.subpart})` : ''}
          </h3>
          <span className={`status-badge status-badge-${status}`}>{STATUS_BADGE[status]}</span>
        </div>

        <div className="question-meta">
          <span className="marks">
            Marks:{' '}
            <span className="obtained-marks-container">
              {editing ? (
                <input
                  type="number"
                  className="marks-input"
                  style={{ display: 'inline-block' }}
                  min={0}
                  max={Number(max) || 100}
                  autoFocus
                  value={questionDraft[id]}
                  onChange={(e) => setQuestionDraft((d) => ({ ...d, [id]: e.target.value }))}
                  onKeyDown={(e) => e.key === 'Enter' && toggleQuestionEdit(q)}
                />
              ) : (
                <span className="marks-display">{String(q.marks || 0)}</span>
              )}
            </span>{' '}
            / <span className="max">{String(max)}</span>
            <button
              className={`edit-question-marks-btn${editing ? ' save-mode' : ''}`}
              style={{ marginLeft: 10, padding: '2px 8px', fontSize: 12 }}
              onClick={() => toggleQuestionEdit(q)}
            >
              {editing ? 'Save' : 'Edit'}
            </button>
          </span>
          {ai !== undefined && ai !== marks && (
            <span className="override-diff">
              AI: {ai} → Override: {marks}
            </span>
          )}
          {ts && <span className="timestamp">Answered on: {ts}</span>}
        </div>

        <p className="question-text" dangerouslySetInnerHTML={{ __html: renderContent(q.question || 'Question not available') }} />

        <div className="answer-section">
          <div className="answer-section-header">
            <h4>Your Answer:</h4>
            <div className="annotate-toolbar">
              <span className="annotate-toolbar-label">Annotate selection:</span>
              <button type="button" className="annotate-btn" title="Highlight selected text" onClick={() => annotate(id, 'highlight')}>
                <i className="fas fa-highlighter" />
              </button>
              <button type="button" className="annotate-btn" title="Strike through selected text" onClick={() => annotate(id, 'strike')}>
                <i className="fas fa-strikethrough" />
              </button>
              <button type="button" className="annotate-btn" title="Add a comment on selected text" onClick={() => annotate(id, 'comment')}>
                <i className="fas fa-comment-dots" />
              </button>
              <button
                type="button"
                className="annotate-btn annotate-btn-clear"
                title="Clear all annotations"
                onClick={() => clearAnnotations(id)}
              >
                <i className="fas fa-eraser" />
              </button>
            </div>
          </div>
          {pendingComment?.id === id && (
            <form
              className="rubric-row"
              onSubmit={(e) => {
                e.preventDefault()
                if (pendingComment.text.trim()) wrapRange(id, pendingComment.range, 'comment', pendingComment.text.trim())
                setPendingComment(null)
              }}
            >
              <input
                type="text"
                className="rubric-row-name"
                placeholder="Comment on the selected text"
                autoFocus
                value={pendingComment.text}
                onChange={(e) => setPendingComment({ ...pendingComment, text: e.target.value })}
              />
              <button type="submit" className="btn btn-secondary">
                Add
              </button>
              <button type="button" className="rubric-row-remove" aria-label="Cancel" onClick={() => setPendingComment(null)}>
                &times;
              </button>
            </form>
          )}
          <pre
            ref={(el) => {
              if (el) answerRefs.current.set(id, el)
            }}
            dangerouslySetInnerHTML={{ __html: renderContent(q.student_answer || 'No answer provided') }}
          />
        </div>

        <div className="reasoning-section rubric">
          <FeedbackToggle
            feedback={q.feedback}
            status={status}
            defaultOpen={focusQ === String(q.question_number) && focusSub === (q.subpart || '')}
          />
        </div>

        <div className="rubric-panel">
          <button type="button" className="rubric-toggle-btn" aria-expanded={!!rubricOpen[id]} onClick={() => toggleRubric(id)}>
            <span>Rubric score</span>
            <span className="feedback-toggle-icon">▾</span>
          </button>
          <div className="rubric-body" hidden={!rubricOpen[id]}>
            <div className="rubric-rows">
              {rows.length === 0 && <p className="rubric-empty">No criteria yet — add one below.</p>}
              {rows.map((row, ri) => (
                <div className="rubric-row" key={ri}>
                  <input
                    type="text"
                    className="rubric-row-name"
                    placeholder="Criterion (e.g. Correct method)"
                    value={row.name}
                    onChange={(e) => updateRubric(id, (rs) => rs.map((r, j) => (j === ri ? { ...r, name: e.target.value } : r)))}
                  />
                  <input
                    type="number"
                    className="rubric-row-points"
                    min={0}
                    step={0.5}
                    placeholder="pts"
                    value={row.points}
                    onChange={(e) =>
                      updateRubric(id, (rs) => rs.map((r, j) => (j === ri ? { ...r, points: parseFloat(e.target.value) || 0 } : r)))
                    }
                  />
                  <button
                    type="button"
                    className="rubric-row-remove"
                    aria-label="Remove criterion"
                    onClick={() => updateRubric(id, (rs) => rs.filter((_, j) => j !== ri))}
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
            <button type="button" className="rubric-add-btn" onClick={() => updateRubric(id, (rs) => [...rs, { name: '', points: 0 }])}>
              <i className="fas fa-plus" /> Add criterion
            </button>
            <div className="rubric-sum-row">
              <span>
                Rubric total: <strong>{rubricSum(id)}</strong> / {String(max)}
              </span>
              <button type="button" className="btn btn-secondary rubric-apply-btn" onClick={() => applyRubric(q)}>
                Apply to score
              </button>
            </div>
          </div>
        </div>
      </>
    )
  }

  const worklistLink =
    query.qp_id && query.assignment_id
      ? `/teacher/correction?${new URLSearchParams({ qp_id: query.qp_id, assignment_id: query.assignment_id })}`
      : null

  return (
    <PageScope
      name="student-details"
      theme={['aesthetic', 'teacher-grade']}
      className={preview ? 'student-preview-mode' : undefined}
      title="Student Details"
    >
      <AppShell footer={false}>
        <main className="main-content">
          <div className="container">
            <section className="tab-content active" role="tabpanel">
              <div className="card">
                <div className="card-header">
                  <h2>
                    <i className="fas fa-chart-bar" /> View Grades
                  </h2>
                </div>
                <div className="card-body">
                  <nav className="secondary-nav" aria-label="User Type Navigation">
                    <ul className="nav-tabs" role="tablist">
                      <li role="presentation">
                        <button className="tab-btn active" aria-selected="true">
                          <i className="fas fa-user-graduate" /> Single Student's Assignment Info View
                        </button>
                      </li>
                    </ul>
                  </nav>

                  <h1>Student Assignment Details</h1>

                  <div className="sub-tab-content active">
                    {worklistLink && (
                      <Link to={worklistLink} className="back-to-worklist-link">
                        {fromCorrection ? (
                          <>
                            <i className="fas fa-arrow-left" /> Back to correction worklist
                          </>
                        ) : (
                          <>
                            <i className="fas fa-list-check" /> Open correction worklist for this assignment
                          </>
                        )}
                      </Link>
                    )}

                    {loading && (
                      <div className="loader-container">
                        <div className="loader" id="student-loader" />
                      </div>
                    )}

                    <AlertStack alerts={alerts} onDismiss={dismiss} />

                    {data?.summary && (
                      <div className="result-container is-revealed" style={{ display: 'block' }}>
                        {data.summary[0] && (
                          <ScoreHero totalMarks={data.summary[0].total_marks} maxMarks={maxMarksOf(data.details)} details={data.details} />
                        )}
                        <h4>Your Assignment Summary</h4>
                        <table className="summary-table">
                          <thead>
                            <tr>
                              <th>Student ID</th>
                              <th>Assignment ID</th>
                              <th>Total Marks</th>
                              <th>Evaluation ID</th>
                              <th>QP ID</th>
                              <th>Date</th>
                              <th className="actions-col">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {data.summary.map((g, i) => (
                              <tr key={i}>
                                <td>{g.student_id || 'N/A'}</td>
                                <td>{g.assignment_id || 'N/A'}</td>
                                <td>
                                  {i in summaryDraft ? (
                                    <input
                                      type="number"
                                      className="marks-input"
                                      style={{ display: 'inline-block' }}
                                      min={0}
                                      max={100}
                                      autoFocus
                                      value={summaryDraft[i]}
                                      onChange={(e) => setSummaryDraft((d) => ({ ...d, [i]: e.target.value }))}
                                    />
                                  ) : (
                                    <span className="marks-display">{g.total_marks ?? 'N/A'}</span>
                                  )}
                                </td>
                                <td>{g.evaluation_id || 'N/A'}</td>
                                <td>{g.qp_id || 'N/A'}</td>
                                <td>{formatTimestamp(g.timestamp) || 'N/A'}</td>
                                <td className="actions-col">
                                  <button
                                    className={`edit-marks-btn${i in summaryDraft ? ' save-mode' : ''}`}
                                    onClick={() => toggleSummaryEdit(i)}
                                  >
                                    {i in summaryDraft ? 'Save' : 'Edit'}
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <button
                          className="update-marks-button"
                          disabled={updating}
                          style={{ cursor: updating ? 'not-allowed' : 'pointer' }}
                          onClick={handleUpdateMarks}
                        >
                          {updating ? 'Updating...' : 'Update Marks'}
                        </button>
                      </div>
                    )}

                    <div className="questions-toolbar">
                      <h4>Question Breakdown</h4>
                      <button type="button" className="preview-toggle-btn" onClick={() => setPreview((p) => !p)}>
                        {preview ? (
                          <>
                            <i className="fas fa-pen" /> Back to Editing
                          </>
                        ) : (
                          <>
                            <i className="fas fa-eye" /> Preview as Student
                          </>
                        )}
                      </button>
                    </div>
                    <div className="preview-banner" style={{ display: preview ? 'flex' : 'none' }}>
                      <i className="fas fa-eye" /> Student preview — this is exactly what the student sees for this
                      assignment (no edit controls).
                    </div>

                    {data?.details && <QuestionViewer details={data.details} renderCard={renderCard} initialIndex={focusIndex} />}

                    <button className="print-button" onClick={() => window.print()}>
                      Print
                    </button>
                  </div>
                </div>
              </div>
            </section>
          </div>
        </main>
      </AppShell>
      {query.assignment_id && (
        <AssistantDock
          context={{ assignment_id: query.assignment_id, student_id: query.student_id ?? undefined }}
          subtitle={query.student_id ? `Paper · ${query.student_id}` : undefined}
          greeting="I can explain any score on this paper, or draft feedback. What do you want to check?"
          suggestions={['Why did this question lose marks?', 'Draft feedback for this paper', 'Show the rubric for the lowest-scoring question']}
          placeholder="Ask about this paper…"
        />
      )}
    </PageScope>
  )
}
