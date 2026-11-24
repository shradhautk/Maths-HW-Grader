import type { ReactNode } from 'react'
import AppShell from './AppShell'
import PageScope from './PageScope'
import '../styles/ui.css'

/** Student/teacher shell for the screens built on components/ui (the wireframe kit). */
export default function WorkflowPage({ title, children }: { title: string; children: ReactNode }) {
  return (
    <PageScope name="workflow" title={title}>
      <AppShell footer={false}>
        <div className="ga-page">{children}</div>
      </AppShell>
    </PageScope>
  )
}
