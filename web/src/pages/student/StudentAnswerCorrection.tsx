import { useEffect, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext'
import AlertStack from '../../components/AlertStack'
import AppShell from '../../components/AppShell'
import { EditorCard, useCorrectionDrafts, type Draft } from '../../components/correction/CorrectionEditor'
import PageScope from '../../components/PageScope'
import { useAlerts } from '../../hooks/useAlerts'
import { apiCache } from '../../lib/apiCache'
import { correctionRequest } from '../../lib/correction'
import '../../styles/themes/aesthetic.css'
import '../../styles/pages/student-answer-correction.css'

interface Answer {
  evaluation_id: string
  question_number: string | number
  subpart?: string
  question?: string
  student_answer?: string
  is_latex?: boolean
  max_marks?: number | string
  status?: string
}

const LAST_QUERY_KEY = 'lastQuery:answerCorrection'
const getText = (a: Answer) => a.student_answer ?? ''
const getLatex = (a: Answer) => !!a.is_latex

async function saveAnswer(a: Answer, d: Draft): Promise<string> {
  await correctionRequest({
    operation: 'submitAnswerCorrections',
    corrections: [
      { evaluation_id: a.evaluation_id, question_number: a.question_number, corrected_answer: d.text, is_latex: d.latex },
    ],
  })
  apiCache.clearPrefix('answerCorrection:')
  return 'Saved.'
}

/**
 * Review & correct OCR'd answers before grading runs. Reached directly, or
 * from the upload page after a student upload (?assignment_id=&qp_id=). No
 * OCR rows yet means "still processing", which is expected right after upload.
 */
export default function StudentAnswerCorrection() {
  const { session } = useAuth()
  const [params] = useSearchParams()
  const { alerts, show, dismiss } = useAlerts(6000)

  const [assignmentId, setAssignmentId] = useState('')
  const [qpId, setQpId] = useState('')
  const [loaded, setLoaded] = useState<{ assignmentId: string; qpId: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [savingAll, setSavingAll] = useState(false)
  const { items: answers, drafts, setItems, updateDraft, saveOne } = useCorrectionDrafts<Answer>({
    getText,
    getLatex,
    save: saveAnswer,
    emptyMessage: "Answer can't be empty.",
    savingMessage: 'Saving…',
  })
  const markCorrected = (a: Answer, d: Draft): Answer => ({ ...a, student_answer: d.text, is_latex: d.latex, status: 'corrected' })

  async function runLoad(aId: string, qp: string, force = false) {
    const studentId = session?.userId
    setLoaded({ assignmentId: aId, qpId: qp })
    setProcessing(false)
    setItems([])
    if (!studentId) {
      show('Could not find your student ID — please log in again.', 'danger')
      return
    }

    setLoading(true)
    try {
      const data = await apiCache.withCache(`answerCorrection:${studentId}:${aId}:${qp}`, force, () =>
        correctionRequest<{ answers?: Answer[] }>({
          operation: 'getAnswersForCorrection',
          student_id: studentId,
          assignment_id: aId,
          qp_id: qp || undefined,
        }),
      )
      apiCache.set(LAST_QUERY_KEY, { assignmentId: aId, qpId: qp })
      const list = Array.isArray(data?.answers) ? data.answers : []
      if (list.length === 0) setProcessing(true)
      else setItems(list)
    } catch (err) {
      show(`Failed to load your answers: ${(err as Error).message}`, 'danger')
    } finally {
      setLoading(false)
    }
  }

  // Prefill from the upload redirect, else restore the last lookup.
  useEffect(() => {
    const aId = params.get('assignment_id')?.trim()
    const qp = params.get('qp_id')?.trim() ?? ''
    const last = apiCache.get<{ assignmentId?: string; qpId?: string }>(LAST_QUERY_KEY)
    const initial = aId
      ? { assignmentId: aId, qpId: qp }
      : last?.assignmentId
        ? { assignmentId: last.assignmentId, qpId: last.qpId ?? '' }
        : null
    if (initial) {
      setAssignmentId(initial.assignmentId)
      setQpId(initial.qpId)
      void runLoad(initial.assignmentId, initial.qpId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!assignmentId.trim()) {
      show('Please enter an Assignment ID.', 'danger')
      return
    }
    void runLoad(assignmentId.trim(), qpId.trim())
  }

  async function saveAll() {
    setSavingAll(true)
    try {
      let ok = 0
      for (let i = 0; i < answers.length; i++) if (await saveOne(i, markCorrected)) ok++
      show(`Saved ${ok} of ${answers.length} answer(s).`, ok === answers.length ? 'success' : 'warning')
    } finally {
      setSavingAll(false)
    }
  }

  const reload = () => loaded && runLoad(loaded.assignmentId, loaded.qpId, true)
  const savedCount = answers.filter((a) => a.status === 'corrected').length

  return (
    <PageScope name="student-answer-correction" theme="aesthetic" title="Review Your Answers">
      <AppShell footer={false}>
        <main className="main-content">
          <div className="container">
            <section className="tab-content active" role="tabpanel">
              <div className="card">
                <div className="card-header">
                  <span className="header-icon-badge">
                    <i className="fas fa-pen-ruler" />
                  </span>
                  <h2>Review &amp; Correct Your Answers</h2>
                  <p>
                    Your upload is being read by GradeAnt's OCR. Check each answer below and fix anything it misread
                    before grading runs &mdash; use LaTeX (e.g. <code>x^2 + 1</code>, <code>{'\\frac{1}{2}'}</code>) for
                    math, or plain text otherwise.
                  </p>
                </div>
                <div className="card-body">
                  <form className="form" onSubmit={handleSubmit}>
                    <div className="form-row">
                      <div className="form-group">
                        <label htmlFor="assignment-id">Assignment ID</label>
                        <input
                          type="text"
                          id="assignment-id"
                          className="form-control"
                          placeholder="e.g. assignment456"
                          required
                          value={assignmentId}
                          onChange={(e) => setAssignmentId(e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label htmlFor="qp-id">Question Paper ID</label>
                        <input
                          type="text"
                          id="qp-id"
                          className="form-control"
                          placeholder="e.g. physics_101"
                          value={qpId}
                          onChange={(e) => setQpId(e.target.value)}
                        />
                      </div>
                    </div>
                    <div className="form-actions">
                      <button type="submit" className="btn btn-primary" disabled={loading}>
                        <i className="fas fa-magnifying-glass" /> Load my answers
                      </button>
                    </div>
                  </form>

                  {loading && (
                    <div className="loader-container" style={{ display: 'block' }}>
                      <div className="loader" />
                      <p className="scan-progress">Loading your answers&hellip;</p>
                    </div>
                  )}

                  <AlertStack alerts={alerts} onDismiss={dismiss} />

                  {processing && (
                    <div className="processing-state" style={{ display: 'block' }}>
                      <i className="fas fa-hourglass-half" />
                      <h4>Still processing your upload</h4>
                      <p>
                        GradeAnt hasn't finished reading this submission yet. This usually takes a minute or two after
                        upload.
                      </p>
                      <button type="button" className="btn btn-secondary" onClick={reload}>
                        <i className="fas fa-rotate" /> Check again
                      </button>
                    </div>
                  )}

                  {answers.length > 0 && (
                    <div className="result-container" style={{ display: 'block' }}>
                      <div className="correction-toolbar">
                        <h4>Your answers</h4>
                        <span className="correction-count">
                          {answers.length} question{answers.length === 1 ? '' : 's'}
                        </span>
                      </div>

                      <div className="answers-list">
                        {answers.map((a, i) => (
                          <EditorCard
                            key={`${a.evaluation_id}-${i}`}
                            label={`Q${a.question_number}${a.subpart ?? ''}`}
                            done={a.status === 'corrected'}
                            doneLabel="Corrected"
                            maxMarks={a.max_marks}
                            question={a.question || 'Question text not available yet'}
                            editorLabel="Your answer"
                            saveLabel="Save this answer"
                            draft={drafts[i]}
                            onChange={(patch) => updateDraft(i, patch)}
                            onSave={() => void saveOne(i, markCorrected)}
                          />
                        ))}
                      </div>

                      <div className="finalize-panel">
                        <div className="finalize-summary">
                          <div className="finalize-title">Submit corrections</div>
                          <div className={`finalize-count${savedCount === answers.length ? ' all-done' : ''}`}>
                            {savedCount} / {answers.length} saved
                          </div>
                        </div>
                        <p className="finalize-note">
                          Save each answer you edit, or use "Save all" once you're done reviewing every question.
                        </p>
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
