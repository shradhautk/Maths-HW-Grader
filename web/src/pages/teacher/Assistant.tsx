import { useSearchParams } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import AssistantPanel from '../../components/assistant/AssistantPanel'
import { Card, PageHeader } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { listAssignments } from '../../lib/workflow'

/** §04 B: full assistant — batch-level questions and actions for one assignment. */
export default function Assistant() {
  const [params, setParams] = useSearchParams()
  const assignmentId = params.get('assignment_id') ?? ''
  const list = useAsync((force) => listAssignments(force), [])
  const current = list.data?.find((a) => a.assignment_id === assignmentId)

  return (
    <WorkflowPage title="GradeFlow Assistant">
      <PageHeader
        eyebrow="AI tools"
        title="GradeFlow Assistant"
        subtitle="Ask about a whole batch: who struggled, which questions tripped the class up, what to re-teach."
      />
      <Card>
        <div className="ga-field" style={{ maxWidth: 420, marginBottom: 0 }}>
          <label htmlFor="as-assignment">Assignment</label>
          <select
            id="as-assignment"
            className="ga-select"
            value={assignmentId}
            onChange={(e) => setParams(e.target.value ? { assignment_id: e.target.value } : {})}
          >
            <option value="">All my assignments</option>
            {(list.data ?? []).map((a) => (
              <option key={a.assignment_id} value={a.assignment_id}>
                {a.title}
                {a.class_name ? ` · ${a.class_name}` : ''}
              </option>
            ))}
          </select>
        </div>
      </Card>
      <AssistantPanel
        key={assignmentId}
        context={assignmentId ? { assignment_id: assignmentId } : {}}
        subtitle={current ? current.title : 'All assignments'}
        greeting={current ? `I've read every graded paper in ${current.title}. What would you like to know?` : 'Pick an assignment above, or ask about all of your classes.'}
        suggestions={['Which students struggled most?', 'Which question did the class find hardest?', 'Draft a re-teach note for the hardest question']}
      />
    </WorkflowPage>
  )
}
