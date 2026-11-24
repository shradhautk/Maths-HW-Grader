import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import '../styles/ui.css'

const ITEMS = {
  student: [
    { to: '/student/dashboard', icon: 'home', label: 'Home' },
    { to: '/student/assignments', icon: 'assignment', label: 'Assignments' },
    { to: '/student/grades', icon: 'grade', label: 'Grades' },
    { to: '/student/profile', icon: 'person', label: 'Profile' },
  ],
  teacher: [
    { to: '/teacher/dashboard', icon: 'home', label: 'Home' },
    { to: '/teacher/assignments', icon: 'assignment', label: 'Assignments' },
    { to: '/teacher/queue', icon: 'inbox', label: 'Queue' },
    { to: '/teacher/profile', icon: 'person', label: 'Profile' },
  ],
}

/** §10 mobile bottom navigation; hidden above 768px (styles/ui.css). */
export default function BottomNav() {
  const { session } = useAuth()
  const items = ITEMS[session?.userType === 'teacher' ? 'teacher' : 'student']
  return (
    <nav className="ga-bottom-nav" aria-label="Primary">
      {items.map((i) => (
        <NavLink key={i.to} to={i.to}>
          <span className="material-icons" aria-hidden>
            {i.icon}
          </span>
          {i.label}
        </NavLink>
      ))}
    </nav>
  )
}
