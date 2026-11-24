import { Fragment, useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import { useAuth } from '../auth/AuthContext'
import { useDisplayName } from '../hooks/useDisplayName'
import SidebarToggle, { useSidebarClosed } from './SidebarToggle'
import '../styles/sidebar.css'

interface NavItem {
  to: string
  label: string
  icon: string
  /** Other routes that should keep this item highlighted. */
  also?: string[]
  /** Heading shown above this item (starts a new group). */
  section?: string
}

/**
 * Sidebar items per the website wireframes: teacher §02/§11 (Dashboard ·
 * Assignments · Grading Queue · Students · Analytics · Settings) and student
 * §12 (Dashboard · Assignments · Grades · Progress · Announcements). The
 * teacher's "Tools" group keeps the grade lookup, manual correction workspace
 * and assistant that already existed.
 */
const NAV: Record<'student' | 'teacher', { items: NavItem[]; avatar: string; role: string; profile: string; cta: { to: string; label: string } }> = {
  student: {
    avatar: '/img/student.jpg',
    role: 'Student',
    profile: '/student/profile',
    cta: { to: '/student/assignments', label: 'Submit work' },
    items: [
      { to: '/student/dashboard', label: 'Dashboard', icon: 'space_dashboard', section: 'Menu' },
      { to: '/student/assignments', label: 'Assignments', icon: 'assignment', also: ['/student/upload', '/student/answer-correction'] },
      { to: '/student/grades', label: 'Grades', icon: 'grade' },
      { to: '/student/progress', label: 'Progress', icon: 'trending_up' },
      { to: '/student/announcements', label: 'Announcements', icon: 'campaign' },
    ],
  },
  teacher: {
    avatar: '/img/teacher-avatar.jpg',
    role: 'Teacher',
    profile: '/teacher/profile',
    cta: { to: '/teacher/assignments/new', label: 'New assignment' },
    items: [
      { to: '/teacher/dashboard', label: 'Dashboard', icon: 'space_dashboard', section: 'Menu' },
      { to: '/teacher/assignments', label: 'Assignments', icon: 'assignment', also: ['/teacher/upload', '/teacher/question-correction'] },
      { to: '/teacher/queue', label: 'Grading Queue', icon: 'inbox' },
      { to: '/teacher/students', label: 'Students', icon: 'groups', also: ['/teacher/student-details'] },
      { to: '/teacher/analytics', label: 'Analytics', icon: 'insights' },
      { to: '/teacher/settings', label: 'Settings', icon: 'settings' },
      { to: '/teacher/grades', label: 'Grade lookup', icon: 'leaderboard', section: 'Tools' },
      { to: '/teacher/correction', label: 'Manual correction', icon: 'rate_review' },
      { to: '/teacher/assistant', label: 'GradeFlow Assistant', icon: 'auto_awesome' },
    ],
  },
}

/** Student/teacher sidebar — replaces nav.html + nav.js + component-loader.js. */
export default function Sidebar() {
  const { session } = useAuth()
  const { pathname } = useLocation()
  const name = useDisplayName('User')
  const [menuOpen, setMenuOpen] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [closed, setClosed] = useSidebarClosed()
  const nav = NAV[session?.userType === 'teacher' ? 'teacher' : 'student']

  // Close menus on navigation and on any outside click.
  const [lastPath, setLastPath] = useState(pathname)
  if (pathname !== lastPath) {
    setLastPath(pathname)
    setMenuOpen(false)
    setDropdownOpen(false)
  }
  useEffect(() => {
    if (!dropdownOpen) return
    const close = () => setDropdownOpen(false)
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [dropdownOpen])

  const isActive = (item: NavItem) =>
    pathname === item.to || pathname.startsWith(item.to + '/') || !!item.also?.some((p) => pathname.startsWith(p))

  return (
    <aside className={`app-sidebar${closed ? ' is-closed' : ''}`}>
      <SidebarToggle closed={closed} onChange={setClosed} />
      <div className="sidebar-inner">
        <Link to="/" className="sidebar-logo" aria-label="GradeAnt home">
          <img src="/img/grade-antlogo-removebg-preview.png" alt="" className="logo-img" />
          <span className="logo-word">
            Grade<b>Ant</b>
          </span>
        </Link>
        <button className="mobile-menu-btn" onClick={() => setMenuOpen((o) => !o)} aria-label="Toggle menu" aria-expanded={menuOpen}>
          <span className="material-icons">{menuOpen ? 'close' : 'menu'}</span>
        </button>

        <nav className={`nav-links${menuOpen ? ' active' : ''}`} aria-label="Main">
          {nav.items.map((item) => {
            const active = isActive(item)
            return (
              <Fragment key={item.to}>
                {item.section && <span className="nav-section">{item.section}</span>}
                <NavLink to={item.to} className={active ? 'active' : ''} aria-current={active ? 'page' : undefined}>
                  {active && <motion.span layoutId="side-active" className="nav-active-bg" transition={{ type: 'spring', stiffness: 420, damping: 34 }} />}
                  <span className="material-icons" aria-hidden>
                    {item.icon}
                  </span>
                  <span className="nav-label">{item.label}</span>
                </NavLink>
              </Fragment>
            )
          })}
          <Link to={nav.cta.to} className="sidebar-cta">
            <span className="material-icons" aria-hidden>
              add_circle
            </span>
            {nav.cta.label}
          </Link>
        </nav>

        <div className="user-dropdown">
          <button
            type="button"
            className="user-dropdown-toggle"
            aria-expanded={dropdownOpen}
            onClick={(e) => {
              e.stopPropagation()
              setDropdownOpen((o) => !o)
            }}
          >
            <img src={nav.avatar} alt="" className="avatar" />
            <span className="user-info">
              <span className="user-name">{name}</span>
              <span className="user-role">{nav.role}</span>
            </span>
            <span className="material-icons">unfold_more</span>
          </button>
          <AnimatePresence>
            {dropdownOpen && (
              <motion.div
                className="dropdown-menu show"
                initial={{ opacity: 0, y: 8, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.97 }}
                transition={{ duration: 0.16 }}
              >
                <Link to={nav.profile}>
                  <span className="material-icons">person</span>
                  My Profile
                </Link>
                <Link to={nav.profile.replace('/profile', '/change-password')}>
                  <span className="material-icons">lock</span>
                  Change Password
                </Link>
                <Link to={`${nav.profile}#request-reset`}>
                  <span className="material-icons">help_outline</span>
                  Request Password Reset
                </Link>
                <div className="dropdown-divider" />
                <Link to="/logout">
                  <span className="material-icons">logout</span>
                  Logout
                </Link>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </aside>
  )
}
