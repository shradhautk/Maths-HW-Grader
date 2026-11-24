import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext'
import PageScope from '../../components/PageScope'
import SidebarToggle, { useSidebarClosed } from '../../components/SidebarToggle'
import '../../styles/themes/aesthetic.css'
import '../../styles/themes/admin.css'

type AlertType = 'success' | 'warning' | 'danger'
type ShowAlert = (message: string, type?: AlertType) => void

const AdminAlertContext = createContext<ShowAlert>(() => {})

/** The page-level status pill (was showAdminAlert in admin-common.js). */
// eslint-disable-next-line react-refresh/only-export-components
export const useAdminAlert = () => useContext(AdminAlertContext)

const NAV = [
  { to: '/admin/dashboard', icon: 'dashboard', label: 'Overview' },
  { to: '/admin/users', icon: 'group', label: 'Users' },
  { to: '/admin/reset-requests', icon: 'lock_reset', label: 'Reset Requests' },
  { to: '/admin/content', icon: 'article', label: 'Content & Blog' },
  { to: '/admin/analytics', icon: 'bar_chart', label: 'Analytics' },
  { to: '/admin/tickets', icon: 'confirmation_number', label: 'Support Tickets' },
  { to: '/admin/audit-logs', icon: 'history', label: 'Audit Logs' },
  { to: '/admin/settings', icon: 'settings', label: 'System Settings' },
]

interface Props {
  title: ReactNode
  /** Omit for pages whose header is just the title. */
  subtitle?: string
  /** Buttons on the right of the page header (in .admin-page-actions). */
  actions?: ReactNode
  /** Unwrapped header control (Analytics' range select). */
  headerAside?: ReactNode
  /** Rendered after .admin-shell (e.g. a modal overlay). */
  overlay?: ReactNode
  docTitle: string
  children: ReactNode
}

/** Admin console shell: sidebar, page header and status pill (was admin-common.js). */
export default function AdminLayout({ title, subtitle, actions, headerAside, overlay, docTitle, children }: Props) {
  const { session } = useAuth()
  const { pathname } = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)
  // Small screens: the nav collapses behind a menu button in the top bar.
  const [navOpen, setNavOpen] = useState(false)
  const [closed, setClosed] = useSidebarClosed()
  const [alert, setAlert] = useState<{ message: string; type: AlertType } | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const showAlert = useCallback<ShowAlert>((message, type = 'success') => {
    setAlert({ message, type })
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setAlert(null), 5000)
  }, [])
  useEffect(() => () => window.clearTimeout(timer.current), [])

  const [lastPath, setLastPath] = useState(pathname)
  if (pathname !== lastPath) {
    setLastPath(pathname)
    setMenuOpen(false)
    setNavOpen(false)
  }
  useEffect(() => {
    if (!menuOpen) return
    const close = () => setMenuOpen(false)
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [menuOpen])

  return (
    <AdminAlertContext.Provider value={showAlert}>
      <PageScope name="admin" theme={['aesthetic', 'admin']} title={`Admin · ${docTitle}`}>
        <div className="admin-shell">
          <aside className={`admin-sidebar${closed ? ' is-closed' : ''}`}>
            <SidebarToggle closed={closed} onChange={setClosed} />
            <Link to="/admin/dashboard" className="admin-sidebar-logo">
              <img className="logo-mark" src="/img/gradeant-logo.webp" alt="" width={28} height={28} />
              <span className="logo-word">
                Grade<b>Ant</b>
              </span>
              <span className="admin-badge">ADMIN</span>
            </Link>
            <button
              type="button"
              className="admin-nav-toggle"
              aria-controls="admin-nav"
              aria-expanded={navOpen}
              aria-label={navOpen ? 'Close menu' : 'Open menu'}
              onClick={() => setNavOpen((o) => !o)}
            >
              <span className="material-icons">{navOpen ? 'close' : 'menu'}</span>
            </button>
            <div id="admin-nav" className={`admin-nav${navOpen ? ' open' : ''}`}>
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} className={({ isActive }) => `admin-nav-link${isActive ? ' active' : ''}`}>
                <span className="material-icons">{item.icon}</span>
                {item.label}
              </NavLink>
            ))}
            <div className="admin-sidebar-footer">
              <div
                className="admin-user-card"
                onClick={(e) => {
                  e.stopPropagation()
                  setMenuOpen((o) => !o)
                }}
              >
                <span className="admin-user-avatar" />
                <div className="admin-user-meta">
                  <div className="admin-user-name">{session?.userName || 'Admin'}</div>
                  <div className="admin-user-role">Administrator</div>
                </div>
                <div className={`admin-user-menu${menuOpen ? ' show' : ''}`}>
                  <Link to="/admin/profile">
                    <span className="material-icons" style={{ fontSize: 16 }}>person</span>My Profile
                  </Link>
                  <Link to="/logout" className="danger">
                    <span className="material-icons" style={{ fontSize: 16 }}>logout</span>Logout
                  </Link>
                </div>
              </div>
            </div>
            </div>
          </aside>

          <main className="admin-main">
            <div className="admin-page-header">
              {subtitle === undefined ? (
                <div className="admin-page-title">{title}</div>
              ) : (
                <div>
                  <div className="admin-page-title">{title}</div>
                  <div className="admin-page-subtitle">{subtitle}</div>
                </div>
              )}
              {actions && <div className="admin-page-actions">{actions}</div>}
              {headerAside}
            </div>

            {alert && (
              <span
                className={`status-pill ${alert.type}`}
                style={{ display: 'inline-flex', marginBottom: 16 }}
                role="status"
              >
                {alert.message}
              </span>
            )}

            {children}
          </main>
        </div>
        {overlay}
      </PageScope>
    </AdminAlertContext.Provider>
  )
}

export function capitalize(s?: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''
}
