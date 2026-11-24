import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import AlertStack from '../../components/AlertStack'
import WorkflowPage from '../../components/WorkflowPage'
import FileDrop from '../../components/ui/FileDrop'
import { Card, LoadError, Notice, PageHeader, Pill, ProgressBar, Spinner, StatusPill, StepProgress } from '../../components/ui'
import { useAlerts } from '../../hooks/useAlerts'
import { useAsync } from '../../hooks/useAsync'
import { apiCache } from '../../lib/apiCache'
import { correctionRequest } from '../../lib/correction'
import { toNumber } from '../../lib/grading'
import { renameSolutionFile, uploadFiles } from '../../lib/upload'
import {
  approveRubric,
  confirmQuestions,
  createAssignment,
  getAssignment,
  getRubric,
  setGradingMethod,
  startGrading,
  type Assignment,
  type GradingMethod,
  type ParsedQuestion,
  type Persona,
  type RubricItem,
} from '../../lib/workflow'

/**
 * §13 Create Assignment: upload question → review parsed questions → choose
 * method → (standard: upload solution → review answers | AI: tune rubric) →
 * final verification → start grading (§15).
 *
 * The assignment id lives in the URL (/teacher/assignments/:id/setup?step=)
 * so a teacher can leave and resume; step 0 is /teacher/assignments/new.
 */

type StepKey = 'question' | 'review' | 'method' | 'solution' | 'rubric' | 'verify'

const STEPS: Record<GradingMethod, { key: StepKey; label: string }[]> = {
  standard: [
    { key: 'question', label: 'Question' },
    { key: 'review', label: 'Review Qs' },
    { key: 'method', label: 'Method' },
    { key: 'solution', label: 'Solution' },
    { key: 'rubric', label: 'Review' },
    { key: 'verify', label: 'Verify' },
  ],
  ai: [
    { key: 'question', label: 'Question' },
    { key: 'review', label: 'Review Qs' },
    { key: 'method', label: 'Method' },
    { key: 'rubric', label: 'Rubric' },
    { key: 'verify', label: 'Verify' },
  ],
}

const LOW_CONFIDENCE = 0.9
const qLabel = (q: { question_number: string | number; subpart?: string }) => `Q${q.question_number}${q.subpart ?? ''}`

export default function CreateAssignment() {
  const { id } = useParams()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const { alerts, show, dismiss } = useAlerts(6000)

  const assignment = useAsync<Assignment | null>((force) => (id ? getAssignment(id, force) : Promise.resolve(null)), [id])
  const a = assignment.data ?? null
  const [chosenMethod, setMethod] = useState<GradingMethod | null>(null)
  const method: GradingMethod = chosenMethod ?? a?.method ?? 'standard'

  const steps = STEPS[method]
  const requested = params.get('step') as StepKey | null
  const stepKey: StepKey = id ? (requested && steps.some((s) => s.key === requested) ? requested : 'review') : 'question'
  const stepIndex = steps.findIndex((s) => s.key === stepKey)
  const go = (key: StepKey) => setParams({ step: key })

  const header = (
    <>
      <PageHeader
        back="/teacher/assignments"
        eyebrow="New assignment"
        title={a?.title ?? 'Create assignment'}
        subtitle={a ? [a.class_name, a.qp_id && `QP ${a.qp_id}`].filter(Boolean).join(' · ') : 'Name it, pick the class and upload the question paper.'}
        actions={a?.qp_status && <StatusPill status={a.qp_status} />}
      />
      <StepProgress steps={steps.map((s) => s.label)} current={stepIndex} />
      <AlertStack alerts={alerts} onDismiss={dismiss} />
    </>
  )

  if (id && assignment.loading && !a) {
    return (
      <WorkflowPage title="Create Assignment">
        {header}
        <Spinner label="Loading assignment…" />
      </WorkflowPage>
    )
  }
  if (id && assignment.error) {
    return (
      <WorkflowPage title="Create Assignment">
        {header}
        <LoadError error={assignment.error} onRetry={() => assignment.reload()} />
      </WorkflowPage>
    )
  }

  return (
    <WorkflowPage title="Create Assignment">
      {header}
      {stepKey === 'question' && (
        <QuestionStep
          onError={(m) => show(m, 'danger')}
          onCreated={(newId) => navigate(`/teacher/assignments/${encodeURIComponent(newId)}/setup?step=review`)}
        />
      )}
      {a && stepKey === 'review' && <ReviewStep assignment={a} onError={(m) => show(m, 'danger')} onDone={() => go('method')} />}
      {a && stepKey === 'method' && (
        <MethodStep
          assignment={a}
          method={method}
          onBack={() => go('review')}
          onError={(m) => show(m, 'danger')}
          onDone={(m) => {
            setMethod(m)
            assignment.reload()
            setParams({ step: m === 'standard' ? 'solution' : 'rubric' })
          }}
        />
      )}
      {a && stepKey === 'solution' && (
        <SolutionStep assignment={a} onBack={() => go('method')} onError={(m) => show(m, 'danger')} onDone={() => go('rubric')} />
      )}
      {a && stepKey === 'rubric' && (
        <RubricStep
          assignment={a}
          method={method}
          onBack={() => go(method === 'standard' ? 'solution' : 'method')}
          onError={(m) => show(m, 'danger')}
          onDone={() => {
            assignment.reload()
            go('verify')
          }}
        />
      )}
      {a && stepKey === 'verify' && (
        <VerifyStep
          assignment={a}
          method={method}
          go={go}
          onError={(m) => show(m, 'danger')}
          onStarted={() => navigate(`/teacher/assignments/${encodeURIComponent(a.assignment_id)}/grading`)}
        />
      )}
    </WorkflowPage>
  )
}

// ---- Step 1: set up + upload the question paper ----

function QuestionStep({ onCreated, onError }: { onCreated: (id: string) => void; onError: (m: string) => void }) {
  const [title, setTitle] = useState('')
  const [className, setClassName] = useState('')
  const [due, setDue] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)

  async function submit() {
    if (!title.trim() || !className.trim()) return onError('Give the assignment a title and a class.')
    if (files.length === 0) return onError('Add the question paper PDF.')
    setBusy(true)
    try {
      const created = await createAssignment({ title: title.trim(), class_name: className.trim(), due_at: due || undefined })
      setProgress(0)
      // Same tagging the grading Lambda already expects from the old upload page.
      await uploadFiles({ uploadType: 'teacher', assignmentId: created.assignment_id, courseId: created.qp_id }, files, setProgress)
      apiCache.clearPrefix('workflow:')
      onCreated(created.assignment_id)
    } catch (err) {
      onError(`Couldn't create the assignment: ${(err as Error).message}`)
      setProgress(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card title="Set up the assignment">
      <div className="ga-grid-3">
        <div className="ga-field">
          <label htmlFor="ca-title">Title</label>
          <input id="ca-title" className="ga-input" placeholder="e.g. Thermodynamics HW" value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
        <div className="ga-field">
          <label htmlFor="ca-class">Class</label>
          <input id="ca-class" className="ga-input" placeholder="e.g. PHY-204" value={className} onChange={(e) => setClassName(e.target.value)} />
        </div>
        <div className="ga-field">
          <label htmlFor="ca-due">Due date</label>
          <input id="ca-due" type="date" className="ga-input" value={due} onChange={(e) => setDue(e.target.value)} />
        </div>
      </div>
      <FileDrop label="Drag & drop the question paper" files={files} onChange={setFiles} onReject={onError} />
      {progress !== null && (
        <div style={{ marginTop: 16 }}>
          <ProgressBar percent={progress} />
          <p className="ga-muted">Uploading… {progress}%</p>
        </div>
      )}
      <Notice tone="info">Only the question paper is needed here — you'll review the parsed questions next.</Notice>
      <div className="ga-row" style={{ justifyContent: 'flex-end' }}>
        <button type="button" className="ga-btn ga-btn-primary" disabled={busy} onClick={submit}>
          {busy ? 'Uploading…' : 'Review parsed questions →'}
        </button>
      </div>
    </Card>
  )
}

// ---- Step 2: review what OCR read (text + marks) ----

function ReviewStep({ assignment, onDone, onError }: { assignment: Assignment; onDone: () => void; onError: (m: string) => void }) {
  const query = { qp_id: assignment.qp_id, assignment_id: assignment.assignment_id }
  const questions = useAsync(
    (force) =>
      apiCache
        .withCache(`questionCorrection:${query.qp_id}:${query.assignment_id}`, force, () =>
          correctionRequest<{ questions?: ParsedQuestion[] }>({ operation: 'getQuestionsForCorrection', ...query }),
        )
        .then((d) => (Array.isArray(d?.questions) ? d.questions : [])),
    [assignment.assignment_id],
  )
  const [edits, setEdits] = useState<Record<number, string>>({})
  const [saving, setSaving] = useState<number | null>(null)
  const [confirming, setConfirming] = useState(false)
  const [showAll, setShowAll] = useState(false)

  const list = questions.data ?? []
  // Saved this session — the correction API may not echo question_reviewed back.
  const [confirmed, setConfirmed] = useState<Set<number>>(new Set())
  const lowConfidence = (q: ParsedQuestion) =>
    q.ocr_confidence !== undefined && q.ocr_confidence < LOW_CONFIDENCE && !q.question_reviewed && !confirmed.has(list.indexOf(q))
  const marksTotal = list.reduce((s, q) => s + toNumber(q.max_marks), 0)
  const needsAttention = list.filter(lowConfidence).length
  const visible = showAll ? list : list.filter((q, i) => i < 4 || lowConfidence(q))

  async function save(i: number) {
    const q = list[i]
    const text = (edits[i] ?? q.question ?? '').trim()
    if (!text) return onError("Question text can't be empty.")
    setSaving(i)
    try {
      await correctionRequest({
        operation: 'submitQuestionCorrections',
        ...query,
        corrections: [{ question_number: q.question_number, subpart: q.subpart, corrected_question: text, is_latex: false }],
      })
      apiCache.clearPrefix('questionCorrection:')
      setConfirmed((c) => new Set(c).add(i))
      setEdits((ed) => {
        const next = { ...ed }
        delete next[i]
        return next
      })
      questions.reload()
    } catch (err) {
      onError(`Couldn't save ${qLabel(q)}: ${(err as Error).message}`)
    } finally {
      setSaving(null)
    }
  }

  async function confirm() {
    setConfirming(true)
    try {
      await confirmQuestions(assignment.assignment_id)
      onDone()
    } catch (err) {
      onError(`Couldn't confirm questions: ${(err as Error).message}`)
    } finally {
      setConfirming(false)
    }
  }

  if (questions.loading && !questions.data) return <Spinner label="Reading the question paper…" />
  if (questions.error) return <LoadError error={questions.error} onRetry={() => questions.reload()} />
  if (list.length === 0) {
    return (
      <Card>
        <Notice tone="brand">GradeAnt is still reading this question paper. This usually takes under a minute.</Notice>
        <button type="button" className="ga-btn ga-btn-secondary" onClick={() => questions.reload()}>
          Check again
        </button>
      </Card>
    )
  }

  return (
    <Card
      title="Check what the OCR read"
      actions={needsAttention > 0 ? <Pill tone="warning">{needsAttention} needs attention</Pill> : <Pill tone="success">All clear</Pill>}
    >
      <p className="ga-muted">The rubric and solution matching are built from these questions — fix any text or marks now.</p>
      <div className="ga-list">
        {visible.map((q) => {
          const i = list.indexOf(q)
          const attention = lowConfidence(q)
          return (
            <div key={`${qLabel(q)}-${i}`} className={`ga-item${attention ? ' is-attention' : ''}`}>
              <div style={{ flex: 1, minWidth: 220 }}>
                <div className="ga-row">
                  <span className="ga-item-title">{qLabel(q)}</span>
                  {q.max_marks !== undefined && <Pill>{toNumber(q.max_marks)} pts</Pill>}
                  {q.ocr_confidence !== undefined && (
                    <Pill tone={attention ? 'warning' : 'neutral'}>OCR {Math.round(q.ocr_confidence * 100)}%</Pill>
                  )}
                  {q.question_reviewed && <Pill tone="success">Reviewed</Pill>}
                </div>
                <textarea
                  className="ga-textarea"
                  style={{ marginTop: 8 }}
                  value={edits[i] ?? q.question ?? ''}
                  onChange={(e) => setEdits((ed) => ({ ...ed, [i]: e.target.value }))}
                  aria-label={`${qLabel(q)} text`}
                />
                {attention && <div className="ga-item-meta">Low OCR confidence — confirm or re-type before continuing.</div>}
              </div>
              <button
                type="button"
                className="ga-btn ga-btn-secondary ga-btn-sm"
                disabled={saving === i || (!attention && edits[i] === undefined)}
                onClick={() => save(i)}
              >
                {saving === i ? 'Saving…' : attention ? (edits[i] === undefined ? 'Confirm text' : 'Save fix') : 'Save'}
              </button>
            </div>
          )
        })}
      </div>
      {!showAll && visible.length < list.length && (
        <button type="button" className="ga-btn ga-btn-ghost" onClick={() => setShowAll(true)}>
          + {list.length - visible.length} more questions · expand
        </button>
      )}
      <Notice tone="neutral">
        Σ Marks total <strong>{marksTotal}</strong> across {list.length} questions. Confirming locks the question set — rubric and
        solution matching use these exact questions.
      </Notice>
      <div className="ga-row" style={{ justifyContent: 'flex-end' }}>
        <button type="button" className="ga-btn ga-btn-primary" disabled={confirming || needsAttention > 0} onClick={confirm}>
          {confirming ? 'Confirming…' : 'Confirm questions · choose method →'}
        </button>
      </div>
    </Card>
  )
}

// ---- Step 3: AI grading or standard solution ----

function MethodStep({
  assignment,
  method,
  onBack,
  onDone,
  onError,
}: {
  assignment: Assignment
  method: GradingMethod
  onBack: () => void
  onDone: (m: GradingMethod) => void
  onError: (m: string) => void
}) {
  const [choice, setChoice] = useState<GradingMethod>(method)
  const [busy, setBusy] = useState(false)

  async function next() {
    setBusy(true)
    try {
      await setGradingMethod(assignment.assignment_id, choice)
      onDone(choice)
    } catch (err) {
      onError(`Couldn't save the grading method: ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card title="How should GradeAnt grade this?">
      <p className="ga-muted">Pick one. You can switch methods later before publishing.</p>
      <div className="ga-grid-2">
        <button type="button" className={`ga-option${choice === 'ai' ? ' is-selected' : ''}`} onClick={() => setChoice('ai')}>
          <Pill tone="brand">✦ Fastest</Pill>
          <h3>AI grading</h3>
          <p className="ga-muted">GradeFlow™ builds the rubric from the question paper and grades every answer automatically.</p>
          <ul>
            <li>No answer key needed</li>
            <li>Grades in seconds, not hours</li>
            <li>You still review &amp; edit scores</li>
          </ul>
        </button>
        <button type="button" className={`ga-option${choice === 'standard' ? ' is-selected' : ''}`} onClick={() => setChoice('standard')}>
          <Pill tone="success">📄 Most control</Pill>
          <h3>Standard solution</h3>
          <p className="ga-muted">Upload your own model answer key. You review &amp; edit each answer before it grades.</p>
          <ul>
            <li>Graded against your exact key</li>
            <li>Edit marks and criteria per answer</li>
            <li>Manual verification per answer</li>
          </ul>
        </button>
      </div>
      <div className="ga-row-between" style={{ marginTop: 20 }}>
        <button type="button" className="ga-btn ga-btn-secondary" onClick={onBack}>
          Back
        </button>
        <button type="button" className="ga-btn ga-btn-primary" disabled={busy} onClick={next}>
          {choice === 'ai' ? 'Use AI grading →' : 'Upload standard solution →'}
        </button>
      </div>
    </Card>
  )
}

// ---- Step 4 (standard): upload the model answer key ----

function SolutionStep({
  assignment,
  onBack,
  onDone,
  onError,
}: {
  assignment: Assignment
  onBack: () => void
  onDone: () => void
  onError: (m: string) => void
}) {
  const [files, setFiles] = useState<File[]>([])
  const [progress, setProgress] = useState<number | null>(null)

  async function upload() {
    if (files.length === 0) return onError('Add the solution key PDF.')
    setProgress(0)
    try {
      const renamed = files.map((f) => renameSolutionFile(f, assignment.assignment_id, assignment.qp_id))
      await uploadFiles(
        { uploadType: 'answer', assignmentId: assignment.assignment_id, courseId: assignment.qp_id, qpId: assignment.qp_id },
        renamed,
        setProgress,
      )
      apiCache.clearPrefix('workflow:')
      onDone()
    } catch (err) {
      onError(`Upload failed: ${(err as Error).message}`)
      setProgress(null)
    }
  }

  return (
    <Card title="Upload the standard solution">
      <p className="ga-muted">This becomes the grading key. After upload you'll review &amp; edit each model answer.</p>
      <FileDrop label="Drag & drop the solution key" hint="PDF · one model answer per question" files={files} onChange={setFiles} onReject={onError} />
      {progress !== null && (
        <div style={{ marginTop: 16 }}>
          <ProgressBar percent={progress} />
          <p className="ga-muted">Uploading… {progress}%</p>
        </div>
      )}
      <div className="ga-row-between" style={{ marginTop: 20 }}>
        <button type="button" className="ga-btn ga-btn-secondary" onClick={onBack}>
          Back
        </button>
        <button type="button" className="ga-btn ga-btn-primary" disabled={progress !== null} onClick={upload}>
          Review &amp; edit answers →
        </button>
      </div>
    </Card>
  )
}

// ---- Step 4/5: review the rubric (AI draft, or answers matched from the key) ----

function RubricStep({
  assignment,
  method,
  onBack,
  onDone,
  onError,
}: {
  assignment: Assignment
  method: GradingMethod
  onBack: () => void
  onDone: () => void
  onError: (m: string) => void
}) {
  const rubric = useAsync((force) => getRubric(assignment.assignment_id, force), [assignment.assignment_id])
  const [items, setItems] = useState<RubricItem[] | null>(null)
  // Defaults come from the teacher's Settings (applied by the API when the method was chosen).
  const [persona, setPersona] = useState<Persona>((assignment.persona as Persona | undefined) ?? 'balanced')
  const [partial, setPartial] = useState(assignment.partial_credit ?? true)
  const [busy, setBusy] = useState(false)
  const rows = items ?? rubric.data?.rubric ?? []
  const generating = rubric.data?.qp_status === 'rubric_generating' || (!rubric.data?.rubric?.length && !rubric.error)
  const total = rows.reduce((s, r) => s + toNumber(r.max_marks), 0)

  // Poll while GradeFlow is still drafting the rubric / matching the key.
  useEffect(() => {
    if (!generating || rubric.loading) return
    const t = window.setTimeout(() => rubric.reload(), 4000)
    return () => clearTimeout(t)
  }, [generating, rubric])

  const update = (i: number, patch: Partial<RubricItem>) => setItems(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  async function approve() {
    setBusy(true)
    try {
      if (method === 'ai') await setGradingMethod(assignment.assignment_id, 'ai', { persona, partial_credit: partial })
      await approveRubric(assignment.assignment_id, rows)
      onDone()
    } catch (err) {
      onError(`Couldn't approve the rubric: ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  if (rubric.error) return <LoadError error={rubric.error} onRetry={() => rubric.reload()} />

  return (
    <Card
      title={method === 'ai' ? 'Tune the AI grader' : 'Review & edit model answers'}
      actions={<Pill>{rows.length} questions · {total} marks</Pill>}
    >
      <p className="ga-muted">
        {method === 'ai'
          ? 'GradeFlow™ drafted a rubric from your reviewed questions. Adjust it or keep the defaults.'
          : 'Each question is matched to its model answer from your key. Change wording, marks or criteria before grading.'}
      </p>
      {generating ? (
        <Spinner label={method === 'ai' ? 'GradeFlow™ is drafting the rubric…' : 'Reading model answers · matching to questions…'} />
      ) : (
        <div className="ga-list">
          {rows.map((r, i) => (
            <div key={`${qLabel(r)}-${i}`} className="ga-item" style={{ alignItems: 'flex-start' }}>
              <div style={{ flex: 1, minWidth: 220 }}>
                <div className="ga-item-title">
                  {qLabel(r)}
                  {r.title ? ` · ${r.title}` : ''}
                </div>
                <textarea
                  className="ga-textarea"
                  style={{ marginTop: 8 }}
                  value={r.criteria ?? ''}
                  placeholder={method === 'ai' ? 'What earns the marks' : 'Model answer'}
                  onChange={(e) => update(i, { criteria: e.target.value })}
                  aria-label={`${qLabel(r)} criteria`}
                />
              </div>
              <div className="ga-field" style={{ width: 90, marginBottom: 0 }}>
                <label htmlFor={`marks-${i}`}>Marks</label>
                <input
                  id={`marks-${i}`}
                  type="number"
                  min={0}
                  className="ga-input"
                  value={r.max_marks}
                  onChange={(e) => update(i, { max_marks: toNumber(e.target.value) })}
                />
              </div>
            </div>
          ))}
        </div>
      )}
      {method === 'ai' && (
        <div className="ga-grid-2" style={{ marginTop: 20 }}>
          <div className="ga-field">
            <label>Grading persona</label>
            <div className="ga-tabs">
              {(['balanced', 'strict', 'lenient'] as Persona[]).map((p) => (
                <button key={p} type="button" className={`ga-tab${persona === p ? ' is-active' : ''}`} onClick={() => setPersona(p)}>
                  {p[0].toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <label className="ga-check">
            <input type="checkbox" checked={partial} onChange={(e) => setPartial(e.target.checked)} />
            <span>
              <strong>Partial credit</strong> — award method marks
            </span>
          </label>
        </div>
      )}
      <div className="ga-row-between" style={{ marginTop: 20 }}>
        <button type="button" className="ga-btn ga-btn-secondary" onClick={onBack}>
          Back
        </button>
        <button type="button" className="ga-btn ga-btn-primary" disabled={busy || generating || rows.length === 0} onClick={approve}>
          {busy ? 'Saving…' : 'Approve · final verification →'}
        </button>
      </div>
    </Card>
  )
}

// ---- Step 6: final verification (both methods converge) ----

function VerifyStep({
  assignment,
  method,
  go,
  onStarted,
  onError,
}: {
  assignment: Assignment
  method: GradingMethod
  go: (k: StepKey) => void
  onStarted: () => void
  onError: (m: string) => void
}) {
  const rubric = useAsync((force) => getRubric(assignment.assignment_id, force), [assignment.assignment_id])
  const [busy, setBusy] = useState(false)
  const rubricTotal = (rubric.data?.rubric ?? []).reduce((s, r) => s + toNumber(r.max_marks), 0)
  const status = assignment.qp_status

  const items = useMemo(() => {
    const parsed = status !== undefined && status !== 'upload_requested' && status !== 'qp_uploaded'
    const list: { ok: boolean; title: string; detail: string; step: StepKey }[] = [
      { ok: parsed, title: 'Question paper attached', detail: `${assignment.question_count ?? '–'} questions detected`, step: 'question' },
      { ok: parsed, title: 'Parsed questions reviewed', detail: 'Question text and marks confirmed', step: 'review' },
      { ok: !!assignment.method, title: 'Grading method chosen', detail: method === 'ai' ? 'AI grading · GradeFlow™ rubric' : 'Standard solution · graded against your key', step: 'method' },
      {
        ok: status === 'rubric_reviewed',
        title: method === 'ai' ? 'Rubric approved' : 'All model answers reviewed',
        detail: status === 'rubric_reviewed' ? 'Approved' : 'Not approved yet',
        step: 'rubric',
      },
    ]
    if (assignment.total_marks !== undefined) {
      list.push({
        ok: rubricTotal === assignment.total_marks,
        title: `Marks total = ${assignment.total_marks}`,
        detail: rubricTotal === assignment.total_marks ? 'Matches the paper' : `Currently ${rubricTotal} — adjust the rubric marks`,
        step: 'rubric',
      })
    }
    return list
  }, [assignment, method, status, rubricTotal])

  const done = items.filter((i) => i.ok).length
  const ready = done === items.length

  async function start() {
    setBusy(true)
    try {
      await startGrading(assignment.assignment_id)
      onStarted()
    } catch (err) {
      onError(`Couldn't start grading: ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="ga-grid-2" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)' }}>
      <Card title="Verify each item">
        <p className="ga-muted">Everything below must check out before grading can start.</p>
        <ul className="ga-checklist">
          {items.map((i) => (
            <li key={i.title} className={i.ok ? 'is-ok' : 'is-warn'}>
              <span className="ga-check-dot">{i.ok ? '✓' : '!'}</span>
              <div style={{ flex: 1 }}>
                <div className="ga-item-title">{i.title}</div>
                <div className="ga-item-meta">{i.detail}</div>
              </div>
              <button type="button" className="ga-btn ga-btn-ghost ga-btn-sm" onClick={() => go(i.step)}>
                {i.ok ? 'View' : 'Fix'}
              </button>
            </li>
          ))}
        </ul>
      </Card>
      <Card title="Ready to grade?">
        <p className="ga-muted">{ready ? 'All items confirmed.' : `Resolve ${items.length - done} open item(s) to enable.`}</p>
        <ProgressBar percent={(done / items.length) * 100} tone={ready ? 'success' : 'brand'} />
        <p className="ga-muted">
          {done} of {items.length} items confirmed
        </p>
        <button type="button" className="ga-btn ga-btn-primary" style={{ width: '100%' }} disabled={!ready || busy} onClick={start}>
          {busy ? 'Starting…' : 'Confirm & start grading'}
        </button>
        <p className="ga-muted" style={{ marginTop: 12 }}>
          Students' answers grade as soon as they confirm their OCR — nobody waits on anyone.
        </p>
      </Card>
    </div>
  )
}
