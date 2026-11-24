import WorkflowPage from '../../components/WorkflowPage'
import { AppHero, Panel, Stagger } from '../../components/app/AppKit'
import { EmptyState, LoadError, Pill, Spinner } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { formatWhen } from '../../lib/dates'
import { listAnnouncements } from '../../lib/workflow'

/** §12 Announcements from your teachers, newest first. */
export default function Announcements() {
  const list = useAsync((force) => listAnnouncements(force), [])
  const items = list.data ?? []

  return (
    <WorkflowPage title="Announcements">
      <AppHero eyebrow="Announcements" title="News from your teachers" text="Deadlines, quiz dates and anything your class needs to know." />
      {list.error && <LoadError error={list.error} onRetry={() => list.reload()} />}
      {list.loading && !list.data && <Spinner label="Loading announcements…" />}
      {list.data && (
        <Stagger>
          <Panel title={`${items.length} announcement${items.length === 1 ? '' : 's'}`}>
            {items.length === 0 && <EmptyState title="Nothing new">When a teacher posts to your class, it appears here.</EmptyState>}
            <div className="app-rows">
              {items.map((n) => (
                <article key={n.announcement_id} className="app-note">
                  <div className="app-note-head">
                    <h3>{n.title}</h3>
                    <Pill tone="brand">{n.class_name}</Pill>
                  </div>
                  <p>{n.message}</p>
                  <small>
                    {n.author_name ?? 'Your teacher'} · {formatWhen(n.created_at)}
                  </small>
                </article>
              ))}
            </div>
          </Panel>
        </Stagger>
      )}
    </WorkflowPage>
  )
}
