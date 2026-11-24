import { useState } from 'react'
import { Link } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { Card, EmptyState, LoadError, PageHeader, Pill, Spinner, StatusPill } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { GRADES_STATUS_META } from '../../lib/status'
import { listAssignments, type Assignment } from '../../lib/workflow'

type Filter = 'all' | 'draft' | 'grading' | 'published'

const FILTERS: { key: Filter; label: string; match: (a: Assignment) => boolean }[] = [
  { key: 'all', label: 'All', match: () => true },
  { key: 'draft', label: 'Drafts', match: (a) => a.content_status !== 'published' },
  { key: 'grading', label: 'Grading', match: (a) => a.grades_status === 'ai_graded' || a.grades_status === 'teacher_reviewing' },
  { key: 'published', label: 'Published', match: (a) => a.grades_status === 'published' },
]

/** Where an assignment card leads, based on how far it got (§13 → §15 → §19). */
function nextLink(a: Assignment): { to: string; label: string } {
  const id = encodeURIComponent(a.assignment_id)
  if (a.qp_status !== 'rubric_reviewed') return { to: `/teacher/assignments/${id}/setup`, label: 'Continue setup' }
  if (!a.grades_status || a.grades_status === 'not_started') return { to: `/teacher/assignments/${id}/submissions`, label: 'Submissions' }
  if (a.grades_status === 'published') return { to: `/teacher/assignments/${id}/analytics`, label: 'Analytics' }
  return { to: `/teacher/assignments/${id}/grading`, label: 'Grading run' }
}

/** §13 entry point: Assignments → + New Assignment. */
export default function AssignmentList() {
  const list = useAsync((force) => listAssignments(force), [])
  const [filter, setFilter] = useState<Filter>('all')
  const all = list.data ?? []
  const shown = all.filter(FILTERS.find((f) => f.key === filter)!.match)

  return (
    <WorkflowPage title="Assignments">
      <PageHeader
        eyebrow="Teacher"
        title="Assignments"
        subtitle="Create an assignment, follow it through grading, and publish results to your class."
        actions={
          <Link to="/teacher/assignments/new" className="ga-btn ga-btn-primary">
            + New assignment
          </Link>
        }
      />
      <div className="ga-tabs">
        {FILTERS.map((f) => (
          <button key={f.key} type="button" className={`ga-tab${filter === f.key ? ' is-active' : ''}`} onClick={() => setFilter(f.key)}>
            {f.label} · {all.filter(f.match).length}
          </button>
        ))}
      </div>
      {list.error && <LoadError error={list.error} onRetry={() => list.reload()} />}
      {list.loading && !list.data && <Spinner label="Loading assignments…" />}
      {list.data && shown.length === 0 && (
        <Card>
          <EmptyState
            title={all.length === 0 ? 'No assignments yet' : 'Nothing here'}
            action={
              all.length === 0 && (
                <Link to="/teacher/assignments/new" className="ga-btn ga-btn-primary">
                  Create your first assignment
                </Link>
              )
            }
          >
            {all.length === 0 ? 'Upload a question paper to get started.' : 'No assignments match this filter.'}
          </EmptyState>
        </Card>
      )}
      <div className="ga-list">
        {shown.map((a) => {
          const next = nextLink(a)
          const grades = GRADES_STATUS_META[a.grades_status ?? 'not_started']
          return (
            <div key={a.assignment_id} className="ga-item">
              <div>
                <div className="ga-item-title">{a.title}</div>
                <div className="ga-item-meta">
                  {[
                    a.class_name,
                    a.question_count !== undefined && `${a.question_count} questions`,
                    a.total_marks !== undefined && `${a.total_marks} marks`,
                    a.due_at && `due ${new Date(a.due_at).toLocaleDateString()}`,
                    a.enrolled !== undefined && `${a.submitted ?? 0}/${a.enrolled} submitted`,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
                </div>
              </div>
              <div className="ga-row">
                {a.qp_status && a.qp_status !== 'rubric_reviewed' && <StatusPill status={a.qp_status} />}
                <Pill tone={grades.tone}>{grades.label}</Pill>
                <Link to={next.to} className="ga-btn ga-btn-secondary ga-btn-sm">
                  {next.label} →
                </Link>
              </div>
            </div>
          )
        })}
      </div>
    </WorkflowPage>
  )
}
