import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import WorkflowPage from '../../components/WorkflowPage'
import { Card, LoadError, Notice, PageHeader, Spinner, StepProgress } from '../../components/ui'
import { useAuth } from '../../auth/AuthContext'
import { useAlerts } from '../../hooks/useAlerts'
import { useAsync } from '../../hooks/useAsync'
import { gradesRequest, sortDetails, toNumber, type GradeResult } from '../../lib/grading'
import { requestRecheck } from '../../lib/workflow'

/**
 * §14 step 5: pick questions and say why. The request reopens the paper in
 * the teacher's correction queue (§16); their edit comes back as published_revised.
 */
export default function Recheck() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const { session } = useAuth()
  const { alerts, show, dismiss } = useAlerts(6000)
  const result = useAsync(
    (force) => gradesRequest<GradeResult>({ operation: 'getSingleStudentGrade', student_id: session?.userId, assignment_id: id }, { force }),
    [id, session?.userId],
  )
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [sending, setSending] = useState(false)

  const details = sortDetails(result.data?.details ?? [])
  const evaluationId = result.data?.summary?.[0]?.evaluation_id ?? details[0]?.evaluation_id
  const keyOf = (q: { question_number?: unknown; subpart?: string }) => `${q.question_number}${q.subpart ?? ''}`

  function toggle(k: string) {
    setPicked((p) => {
      const next = new Set(p)
      if (next.has(k)) next.delete(k)
      else next.add(k)
      return next
    })
  }

  async function send() {
    const items = details
      .filter((q) => picked.has(keyOf(q)))
      .map((q) => ({ question_number: q.question_number ?? '', subpart: q.subpart, note: (notes[keyOf(q)] ?? '').trim() }))
    if (items.length === 0) return show('Pick at least one question.', 'danger')
    if (items.some((i) => !i.note)) return show('Tell your teacher why for each question you picked.', 'danger')
    if (!evaluationId) return show('This result has no evaluation id yet — try again later.', 'danger')
    setSending(true)
    try {
      await requestRecheck(id, String(evaluationId), items)
      show('Re-check request sent to your teacher.', 'success')
      window.setTimeout(() => navigate('/student/assignments'), 1200)
    } catch (err) {
      show(`Couldn't send the request: ${(err as Error).message}`, 'danger')
    } finally {
      setSending(false)
    }
  }

  return (
    <WorkflowPage title="Request a Re-check">
      <PageHeader back="/student/assignments" eyebrow="Request a re-check" title="Which answers should be re-checked?" subtitle="Select the questions and tell your teacher why. They'll see this in their correction queue." />
      <StepProgress steps={['Receive', 'Submit', 'Track', 'Feedback', 'Re-check']} current={4} />
      <AlertStack alerts={alerts} onDismiss={dismiss} />
      {result.error && <LoadError error={result.error} onRetry={() => result.reload()} />}
      {result.loading && !result.data && <Spinner label="Loading your result…" />}
      {result.data && (
        <Card>
          <div className="ga-list">
            {details.map((q) => {
              const k = keyOf(q)
              const on = picked.has(k)
              return (
                <div key={k} className={`ga-item${on ? ' is-attention' : ''}`} style={{ alignItems: 'flex-start' }}>
                  <label className="ga-check" style={{ flex: 1, minWidth: 240 }}>
                    <input type="checkbox" checked={on} onChange={() => toggle(k)} />
                    <span style={{ flex: 1 }}>
                      <span className="ga-item-title">Q{k}</span>
                      {q.question && <span className="ga-item-meta"> — {String(q.question).slice(0, 120)}</span>}
                      {on && (
                        <textarea
                          className="ga-textarea"
                          style={{ marginTop: 8 }}
                          placeholder="e.g. I believe my sign was correct because…"
                          value={notes[k] ?? ''}
                          onChange={(e) => setNotes((n) => ({ ...n, [k]: e.target.value }))}
                        />
                      )}
                    </span>
                  </label>
                  <strong className="ga-mono">
                    {toNumber(q.marks)}/{toNumber(q.max_marks)}
                  </strong>
                </div>
              )
            })}
          </div>
          <Notice tone="info">⤺ This request reopens the item in your teacher's correction workspace.</Notice>
          <div className="ga-row-between">
            <button type="button" className="ga-btn ga-btn-secondary" onClick={() => navigate('/student/assignments')}>
              Cancel
            </button>
            <button type="button" className="ga-btn ga-btn-primary" disabled={sending || picked.size === 0} onClick={send}>
              {sending ? 'Sending…' : 'Send re-check request'}
            </button>
          </div>
        </Card>
      )}
    </WorkflowPage>
  )
}
