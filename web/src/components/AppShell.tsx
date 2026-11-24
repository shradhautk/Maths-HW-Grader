import type { ReactNode } from 'react'
import BottomNav from './BottomNav'
import Sidebar from './Sidebar'
import '../styles/app-bridge.css'

const YEAR = new Date().getFullYear()

interface Props {
  children: ReactNode
  footer?: boolean
  /** Extra classes for <main class="app-main">. */
  mainClassName?: string
}

/** Sidebar + main column used by every student/teacher page (.app-shell in the page CSS). */
export default function AppShell({ children, footer = true, mainClassName }: Props) {
  return (
    <>
      <div className="app-shell">
        <Sidebar />
        <main className={mainClassName ? `app-main ${mainClassName}` : 'app-main'}>{children}</main>
      </div>
      <BottomNav />
      {footer && (
        <footer className="site-footer">
          <div className="container">
            <p>&copy; {YEAR} GradeAnt. All rights reserved.</p>
          </div>
        </footer>
      )}
    </>
  )
}
