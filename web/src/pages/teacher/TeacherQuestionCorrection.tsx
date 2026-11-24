import { useEffect, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import AppShell from '../../components/AppShell'
import { EditorCard, useCorrectionDrafts, type Draft } from '../../components/correction/CorrectionEditor'
import PageScope from '../../components/PageScope'
import { useAlerts } from '../../hooks/useAlerts'
import { apiCache } from '../../lib/apiCache'
import { correctionRequest } from '../../lib/correction'
import '../../styles/themes/aesthetic.css'
import '../../styles/themes/teacher-grade.css'
import '../../styles/themes/teacher-correction.css'
import '../../styles/pages/teacher-question-correction.css'

interface Question {
  question_number: string | number
  subpart?: string
  question?: string
  is_latex?: boolean
  max_marks?: number | string
  question_reviewed?: boolean
}

const LAST_QUERY_KEY = 'lastQuery:questionCorrection'
const getText = (q: Question) => q.question ?? ''
const getLatex = (q: Question) => !!q.is_latex
const markReviewed = (q: Question, d: Draft): Question => ({ ...q, question: d.text, is_latex: d.latex, question_reviewed: true })

/**
 * Pre-grading pass over the OCR'd question paper. A correction fans out to
 * every student's copy of that question. Reached directly or from the upload
 * page after a teacher upload (?qp_id=&assignment_id=). Distinct from the
 * post-grading Manual Grading worklist.
 */
export default function TeacherQuestionCorrection() {
  const [params] = useSearchParams()
  const { alerts, show, dismiss } = useAlerts(6000)

  const [qpId, setQpId] = useState('')
  const [assignmentId, setAssignmentId] = useState('')
  const [loaded, setLoaded] = useState<{ qpId: string; assignmentId: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [savingAll, setSavingAll] = useState(false)

  const { items: questions, drafts, setItems, updateDraft, saveOne } = useCorrectionDrafts<Question>({
    getText,
    getLatex,
    emptyMessage: "Question text can't be empty.",
    savingMessage: 'Saving to whole class…',
    save: async (q, d) => {
      const data = await correctionRequest<{ updated_questions?: { records_updated?: number }[] }>({
        operation: 'submitQuestionCorrections',
        qp_id: loaded?.qpId,
        assignment_id: loaded?.assignmentId,
        corrections: [{ question_number: q.question_number, subpart: q.subpart, corrected_question: d.text, is_latex: d.latex }],
      })
      apiCache.clearPrefix('questionCorrection:')
      return `Saved — updated ${data?.updated_questions?.[0]?.records_updated ?? 0} student record(s).`
    },
  })

  async function runLoad(qp: string, aId: string, force = false) {
    setLoaded({ qpId: qp, assignmentId: aId })
    setProcessing(false)
    setItems([])
    setLoading(true)
    try {
      const data = await apiCache.withCache(`questionCorrection:${qp}:${aId}`, force, () =>
        correctionRequest<{ questions?: Question[] }>({
          operation: 'getQuestionsForCorrection',
          qp_id: qp,
          assignment_id: aId,
        }),
      )
      apiCache.set(LAST_QUERY_KEY, { qpId: qp, assignmentId: aId })
      const list = Array.isArray(data?.questions) ? data.questions : []
      if (list.length === 0) setProcessing(true)
      else setItems(list)
    } catch (err) {
      show(`Failed to load questions: ${(err as Error).message}`, 'danger')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const qp = params.get('qp_id')?.trim()
    const aId = params.get('assignment_id')?.trim()
    const last = apiCache.get<{ qpId?: string; assignmentId?: string }>(LAST_QUERY_KEY)
    if (qp) setQpId(qp)
    if (aId) setAssignmentId(aId)
    const initial = qp && aId ? { qpId: qp, assignmentId: aId } : last?.qpId && last.assignmentId ? last : null
    if (initial?.qpId && initial.assignmentId) {
      setQpId(initial.qpId)
      setAssignmentId(initial.assignmentId)
      void runLoad(initial.qpId, initial.assignmentId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!qpId.trim() || !assignmentId.trim()) {
      show('Please fill in both Question Paper ID and Assignment ID.', 'danger')
      return
    }
    void runLoad(qpId.trim(), assignmentId.trim())
  }

  async function saveAll() {
    setSavingAll(true)
    try {
      let ok = 0
      for (let i = 0; i < questions.length; i++) if (await saveOne(i, markReviewed)) ok++
      show(`Saved ${ok} of ${questions.length} question(s).`, ok === questions.length ? 'success' : 'warning')
    } finally {
      setSavingAll(false)
    }
  }

  const reload = () => loaded && runLoad(loaded.qpId, loaded.assignmentId, true)
  const reviewedCount = questions.filter((q) => q.question_reviewed).length

  return (
    <PageScope
      name="teacher-question-correction"
      theme={['aesthetic', 'teacher-grade', 'teacher-correction']}
      title="Review Question Paper"
    >
      <AppShell footer={false}>
        <main className="main-content">
          <div className="container">
            <section className="tab-content active" role="tabpanel">
              <div className="card">
                <div className="card-header">
                  <h2>
                    <i className="fas fa-file-circle-check" /> Review &amp; Correct Question Text
                  </h2>
                  <p>
                    GradeAnt's OCR read the question paper you uploaded. Fix anything it misread before grading starts
                    &mdash; use LaTeX (e.g. <code>x^2 + 1</code>, <code>{'\\frac{1}{2}'}</code>) for math, or plain text
                    otherwise. A fix here updates that question for every student in the class.
                  </p>
                </div>
                <div className="card-body">
                  <form className="form" onSubmit={handleSubmit}>
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
                      <button type="submit" className="btn btn-primary" disabled={loading}>
                        <i className="fas fa-magnifying-glass" /> Load questions
                      </button>
                    </div>
                  </form>

                  {loading && (
                    <div className="loader-container" style={{ display: 'block' }}>
                      <div className="loader" />
                      <p className="scan-progress">Loading questions&hellip;</p>
                    </div>
                  )}

                  <AlertStack alerts={alerts} onDismiss={dismiss} />

                  {processing && (
                    <div className="processing-state" style={{ display: 'block' }}>
                      <i className="fas fa-hourglass-half" />
                      <h4>Still processing this question paper</h4>
                      <p>
                        GradeAnt hasn't finished reading this question paper (and at least one student submission) yet.
                      </p>
                      <button type="button" className="btn btn-secondary" onClick={reload}>
                        <i className="fas fa-rotate" /> Check again
                      </button>
                    </div>
                  )}

                  {questions.length > 0 && (
                    <div className="result-container" style={{ display: 'block' }}>
                      <div className="correction-toolbar">
                        <h4>Questions in this paper</h4>
                        <span className="correction-count">
                          {questions.length} question{questions.length === 1 ? '' : 's'}
                        </span>
                      </div>

                      <div className="answers-list">
                        {questions.map((q, i) => (
                          <EditorCard
                            key={`${q.question_number}-${q.subpart ?? ''}-${i}`}
                            label={`Q${q.question_number}${q.subpart ?? ''}`}
                            done={!!q.question_reviewed}
                            doneLabel="Reviewed"
                            maxMarks={q.max_marks}
                            editorLabel="Question text"
                            saveLabel="Save (applies to whole class)"
                            draft={drafts[i]}
                            onChange={(patch) => updateDraft(i, patch)}
                            onSave={() => void saveOne(i, markReviewed)}
                          />
                        ))}
                      </div>

                      <div className="finalize-panel">
                        <div className="finalize-summary">
                          <div className="finalize-title">Submit corrections</div>
                          <div className={`finalize-count${reviewedCount === questions.length ? ' all-done' : ''}`}>
                            {reviewedCount} / {questions.length} reviewed
                          </div>
                        </div>
                        <p className="finalize-note">Corrections apply to every student's copy of that question.</p>
                        <div className="form-actions">
                          <button type="button" className="btn btn-secondary" onClick={reload}>
                            <i className="fas fa-rotate" /> Reload
                          </button>
                          <button type="button" className="btn btn-primary" disabled={savingAll} onClick={saveAll}>
                            {savingAll ? (
                              'Saving all…'
                            ) : (
                              <>
                                <i className="fas fa-check-double" /> Save all corrections
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
