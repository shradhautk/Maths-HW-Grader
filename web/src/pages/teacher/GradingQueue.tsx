import { useState } from 'react'
import { Link } from 'react-router-dom'
import WorkflowPage from '../../components/WorkflowPage'
import { AppHero, AppStat, Panel, Stagger } from '../../components/app/AppKit'
import { EmptyState, LoadError, Pill, ProgressBar, Spinner } from '../../components/ui'
import type { Tone } from '../../lib/status'
import { useAsync } from '../../hooks/useAsync'
import { getGradingQueue, type QueueItem, type QueueState } from '../../lib/workflow'

const STATE: Record<QueueState, { label: string; tone: Tone; icon: string; cta: string }> = {
  review: { label: 'Needs review', tone: 'warning', icon: 'flag', cta: 'Spot-check' },
  ready: { label: 'Ready to publish', tone: 'success', icon: 'task_alt', cta: 'Review & publish' },
  grading: { label: 'Grading…', tone: 'brand', icon: 'hourglass_top', cta: 'View progress' },
  collecting: { label: 'Collecting papers', tone: 'info', icon: 'inbox', cta: 'Submissions' },
  setup: { label: 'Setup unfinished', tone: 'neutral', icon: 'construction', cta: 'Continue setup' },
  published: { label: 'Published', tone: 'success', icon: 'verified', cta: 'Analytics' },
}

function destination(q: QueueItem): string {
  const id = encodeURIComponent(q.assignment_id)
  if (q.state === 'setup') return `/teacher/assignments/${id}/setup`
  if (q.state === 'collecting') return `/teacher/assignments/${id}/submissions`
  if (q.state === 'published') return `/teacher/assignments/${id}/analytics`
  return `/teacher/assignments/${id}/grading`
}

/** §02 Grading Queue: every batch and what it's waiting on. */
export default function GradingQueue() {
  const queue = useAsync((force) => getGradingQueue(force), [])
  const [tab, setTab] = useState<'action' | 'all'>('action')
  const all = queue.data?.queue ?? []
  const count = (s: QueueState) => all.filter((q) => q.state === s).length
  const needsAction = all.filter((q) => ['review', 'ready', 'grading'].includes(q.state))
  const shown = tab === 'action' ? needsAction : all
  const waiting = queue.data?.waiting ?? 0
  const has = !!queue.data

  return (
    <WorkflowPage title="Grading Queue">
      <AppHero
        eyebrow="Grading queue"
        title={has ? (waiting ? `${waiting} batch${waiting === 1 ? '' : 'es'} waiting to review` : "You're all caught up") : 'Grading queue'}
        text="Every assignment, in the order it needs you: flagged answers first, then batches ready to publish."
        actions={
          <>
            <Link to="/teacher/correction" className="ga-btn ga-btn-primary">
              <span className="material-icons" aria-hidden>
                rate_review
              </span>
              Manual correction
            </Link>
            <Link to="/teacher/assignments/new" className="ga-btn ga-btn-secondary">
              New assignment
            </Link>
          </>
        }
      />
      <Stagger className="app-stats">
        <AppStat icon="flag" label="Needs review" value={has ? count('review') : null} tone="warn" />
        <AppStat icon="task_alt" label="Ready to publish" value={has ? count('ready') : null} tone="accent" />
        <AppStat icon="hourglass_top" label="Grading now" value={has ? count('grading') : null} tone="brand" />
        <AppStat icon="inbox" label="Collecting papers" value={has ? count('collecting') : null} tone="sky" />
      </Stagger>

      {queue.error && <LoadError error={queue.error} onRetry={() => queue.reload()} />}
      {queue.loading && !has && <Spinner label="Loading the queue…" />}
      {has && (
        <Panel
          title="Batches"
          action={
            <div className="ga-tabs" style={{ margin: 0 }}>
              <button type="button" className={`ga-tab${tab === 'action' ? ' is-active' : ''}`} onClick={() => setTab('action')}>
                Needs action · {needsAction.length}
              </button>
              <button type="button" className={`ga-tab${tab === 'all' ? ' is-active' : ''}`} onClick={() => setTab('all')}>
                All · {all.length}
              </button>
            </div>
          }
        >
          {shown.length === 0 && (
            <EmptyState title={tab === 'action' ? 'Nothing waiting on you' : 'No assignments yet'}>
              {tab === 'action' ? 'New batches appear here as soon as students submit and grading finishes.' : 'Create an assignment to get started.'}
            </EmptyState>
          )}
          <div className="app-rows">
            {shown.map((q) => {
              const st = STATE[q.state]
              return (
                <div key={q.assignment_id} className={`app-row app-tone-${q.state === 'review' ? 'warn' : q.state === 'ready' || q.state === 'published' ? 'accent' : 'brand'}`}>
                  <span className="app-row-icon material-icons" aria-hidden>
                    {st.icon}
                  </span>
                  <span className="app-row-main">
                    <span className="app-row-title">{q.title}</span>
                    <span className="app-row-meta">
                      {[q.class_name, `${q.papers} paper${q.papers === 1 ? '' : 's'}`, q.due_at && `due ${new Date(q.due_at).toLocaleDateString()}`].filter(Boolean).join(' · ')}
                    </span>
                    {q.papers > 0 && (
                      <span className="app-meter" style={{ marginTop: 8 }}>
                        <ProgressBar percent={(q.graded / q.papers) * 100} tone={q.graded === q.papers ? 'success' : 'brand'} />
                        <span>
                          {q.graded}/{q.papers} graded
                        </span>
                      </span>
                    )}
                  </span>
                  <span className="app-row-end">
                    {q.flagged > 0 && <Pill tone="warning">{q.flagged} flagged</Pill>}
                    <Pill tone={st.tone}>{st.label}</Pill>
                    <Link to={destination(q)} className="ga-btn ga-btn-secondary ga-btn-sm">
                      {st.cta} →
                    </Link>
                  </span>
                </div>
              )
            })}
          </div>
        </Panel>
      )}
    </WorkflowPage>
  )
}
