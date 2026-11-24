import { useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { Card, EmptyState, GradeDistribution, GradePill, LoadError, Notice, PageHeader, Spinner, StatTile } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { apiCache } from '../../lib/apiCache'
import { correctionRequest } from '../../lib/correction'
import { getGradeLetter, gradesRequest, toNumber, type GradeResult, type GradeSummary } from '../../lib/grading'
import { getAssignment } from '../../lib/workflow'

type Letter = 'A' | 'B' | 'C' | 'D' | 'F'
const AT_RISK = 50

/**
 * §19 step 3: cohort analytics — submission rate, score spread and
 * per-question difficulty, each tile drilling down. Built on the existing
 * grades API (getAllStudentGrades + getSingleStudentGrade) so it works for
 * any graded assignment; ?qp_id= skips the workflow lookup.
 */
export default function CohortAnalytics() {
  const { id = '' } = useParams()
  const [params] = useSearchParams()
  const qpParam = params.get('qp_id')
  const assignment = useAsync((force) => (qpParam ? Promise.resolve(null) : getAssignment(id, force)), [id, qpParam])
  const qpId = qpParam ?? assignment.data?.qp_id ?? ''
  const totalMarks = assignment.data?.total_marks ?? 0

  const data = useAsync(
    async (force) => {
      if (!qpId) return null
      const [grades, questions] = await Promise.all([
        gradesRequest<{ grades?: GradeSummary[] }>({ operation: 'getAllStudentGrades', qp_id: qpId, assignment_id: id }, { force }),
        apiCache.withCache(`questionCorrection:${qpId}:${id}`, force, () =>
          correctionRequest<{ questions?: { max_marks?: number | string }[] }>({ operation: 'getQuestionsForCorrection', qp_id: qpId, assignment_id: id }),
        ),
      ])
      const rows = (grades?.grades ?? []).filter((g) => g.student_id)
      // Per-question results come from each student's detail record (cached per student).
      const details = await Promise.all(
        rows.map((g) =>
          gradesRequest<GradeResult>({ operation: 'getSingleStudentGrade', student_id: g.student_id, assignment_id: id }, { force }).catch(() => null),
        ),
      )
      // Prefer the assignment's own total, then the largest graded paper's, then the question paper's.
      const fromDetails = Math.max(0, ...details.map((d) => (d?.details ?? []).reduce((s, q) => s + toNumber(q.max_marks), 0)))
      const fromQuestions = (questions?.questions ?? []).reduce((s, q) => s + toNumber(q.max_marks), 0)
      const maxMarks = totalMarks || fromDetails || fromQuestions
      return { rows, details, maxMarks }
    },
    [id, qpId, totalMarks],
  )

  const [selected, setSelected] = useState<string | null>(null)

  const stats = useMemo(() => {
    if (!data.data) return null
    const { rows, details, maxMarks } = data.data
    const students = rows.map((g) => {
      const marks = toNumber(g.total_marks)
      const percent = maxMarks > 0 ? (marks / maxMarks) * 100 : 0
      return { id: String(g.student_id), marks, percent, letter: getGradeLetter(percent).letter as Letter }
    })
    const percents = students.map((s) => s.percent).sort((a, b) => a - b)
    const avg = percents.length ? percents.reduce((a, b) => a + b, 0) / percents.length : 0
    const median = percents.length ? percents[Math.floor(percents.length / 2)] : 0
    const counts: Record<Letter, number> = { A: 0, B: 0, C: 0, D: 0, F: 0 }
    students.forEach((s) => counts[s.letter]++)

    const perQ = new Map<string, { label: string; text: string; got: number; max: number }>()
    for (const d of details) {
      for (const q of d?.details ?? []) {
        const key = `Q${q.question_number}${q.subpart ?? ''}`
        const e = perQ.get(key) ?? { label: key, text: String(q.question ?? ''), got: 0, max: 0 }
        e.got += toNumber(q.marks)
        e.max += toNumber(q.max_marks)
        perQ.set(key, e)
      }
    }
    const hardest = [...perQ.values()]
      .filter((q) => q.max > 0)
      .map((q) => ({ ...q, avg: (q.got / q.max) * 100 }))
      .sort((a, b) => a.avg - b.avg)
      .slice(0, 5)
    return { students, avg, median, counts, hardest, atRisk: students.filter((s) => s.percent < AT_RISK) }
  }, [data.data])

  function exportCsv() {
    if (!stats) return
    const lines = ['student_id,marks,percent,grade', ...stats.students.map((s) => `${s.id},${s.marks},${s.percent.toFixed(1)},${s.letter}`)]
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${id}-analytics.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  const a = assignment.data
  const drill = stats?.students.filter((s) => (selected === 'risk' ? s.percent < AT_RISK : s.letter === selected)) ?? []
  const error = assignment.error ?? data.error

  return (
    <WorkflowPage title="Cohort Analytics">
      <PageHeader
        back={qpParam ? '/teacher/grades' : `/teacher/assignments/${encodeURIComponent(id)}/submissions`}
        eyebrow="Cohort analytics"
        title={a?.title ?? id}
        subtitle={a?.class_name}
        actions={
          <button type="button" className="ga-btn ga-btn-secondary" disabled={!stats?.students.length} onClick={exportCsv}>
            Export CSV
          </button>
        }
      />
      {error && <LoadError error={error} onRetry={() => (assignment.error ? assignment.reload() : data.reload())} />}
      {(assignment.loading || data.loading) && !stats && !error && <Spinner label="Crunching the class results…" />}
      {stats && stats.students.length === 0 && (
        <Card>
          <EmptyState title="No graded papers yet">Analytics fill in as soon as grading finishes.</EmptyState>
        </Card>
      )}
      {stats && stats.students.length > 0 && (
        <>
          <div className="ga-grid-4" style={{ marginBottom: 20 }}>
            <StatTile
              label="Graded"
              value={stats.students.length}
              note={a?.enrolled ? `${Math.round((stats.students.length / a.enrolled) * 100)}% of ${a.enrolled}` : undefined}
            />
            <StatTile label="Class average" value={`${Math.round(stats.avg)}%`} tone="success" />
            <StatTile label="Median" value={`${Math.round(stats.median)}%`} note={`grade ${getGradeLetter(stats.median).letter}`} />
            <button type="button" className="ga-option" style={{ padding: 0, border: 'none' }} onClick={() => setSelected(selected === 'risk' ? null : 'risk')}>
              <StatTile label="At risk" value={stats.atRisk.length} note={`below ${AT_RISK}% · click to list`} tone="danger" />
            </button>
          </div>
          <div className="ga-grid-2">
            <Card title="Score distribution" actions={<span className="ga-muted">click a bar to list those students</span>}>
              <GradeDistribution counts={stats.counts} selected={selected} onSelect={setSelected} />
              {selected && (
                <div className="ga-list" style={{ marginTop: 16 }}>
                  {drill.length === 0 && <p className="ga-muted">No students.</p>}
                  {drill.map((s) => (
                    <div key={s.id} className="ga-item">
                      <span className="ga-mono">{s.id}</span>
                      <div className="ga-row">
                        <span className="ga-muted">{Math.round(s.percent)}%</span>
                        <GradePill percent={s.percent} />
                        <Link
                          className="ga-btn ga-btn-ghost ga-btn-sm"
                          to={`/teacher/student-details?${new URLSearchParams({ student_id: s.id, assignment_id: id, qp_id: qpId })}`}
                        >
                          Open
                        </Link>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>
            <Card title="Hardest questions">
              {stats.hardest.length === 0 ? (
                <p className="ga-muted">Per-question results aren't available yet.</p>
              ) : (
                <>
                  <div className="ga-list">
                    {stats.hardest.map((q) => (
                      <div key={q.label} className="ga-item">
                        <div style={{ minWidth: 0, flex: 1 }}>
                          <div className="ga-item-title">{q.label}</div>
                          {q.text && <div className="ga-item-meta">{q.text.slice(0, 90)}</div>}
                        </div>
                        <strong>{Math.round(q.avg)}% avg</strong>
                      </div>
                    ))}
                  </div>
                  <Notice tone="brand">
                    {stats.hardest[0].label} is the class blind spot ({Math.round(stats.hardest[0].avg)}% average) — consider re-teaching it.
                  </Notice>
                </>
              )}
            </Card>
          </div>
        </>
      )}
    </WorkflowPage>
  )
}
