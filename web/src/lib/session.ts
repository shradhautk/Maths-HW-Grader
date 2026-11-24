import { apiCache } from './apiCache'

export type UserRole = 'student' | 'teacher' | 'admin'

export interface Session {
  token: string
  userId: string
  userType: UserRole
  userName: string
  loginTime: string
  userData: Record<string, unknown>
}

// Same keys the static site used, so a user logged in on the old pages
// stays logged in on the React app and vice versa during the migration.
const KEYS = ['authToken', 'userId', 'userType', 'userName', 'userData', 'loginTime'] as const

function read(key: string): string | null {
  return sessionStorage.getItem(key) ?? localStorage.getItem(key)
}

export function loadSession(): Session | null {
  const token = read('authToken')
  if (!token) return null
  let userData: Record<string, unknown> = {}
  try {
    userData = JSON.parse(read('userData') ?? '{}')
  } catch {
    /* keep empty */
  }
  return {
    token,
    userId: read('userId') ?? '',
    userType: (read('userType') as UserRole) || 'student',
    userName: read('userName') ?? 'User',
    loginTime: read('loginTime') ?? '',
    userData,
  }
}

/** Persists a fresh login. `remember` uses localStorage, otherwise this tab only. */
export function saveSession(s: Omit<Session, 'loginTime'>, remember: boolean): Session {
  clearSession()
  const storage = remember ? localStorage : sessionStorage
  const session: Session = { ...s, loginTime: new Date().toISOString() }
  storage.setItem('authToken', session.token)
  storage.setItem('userId', session.userId)
  storage.setItem('userType', session.userType)
  storage.setItem('userName', session.userName)
  storage.setItem('loginTime', session.loginTime)
  storage.setItem('userData', JSON.stringify(session.userData))
  return session
}

/** Merges fields into the stored login data, in whichever storage holds the session. */
export function patchUserData(patch: Record<string, unknown>): Session | null {
  const current = loadSession()
  if (!current) return null
  const storage = sessionStorage.getItem('authToken') !== null ? sessionStorage : localStorage
  const userData = { ...current.userData, ...patch }
  storage.setItem('userData', JSON.stringify(userData))
  return { ...current, userData }
}

export function clearSession(): void {
  KEYS.forEach((k) => {
    localStorage.removeItem(k)
    sessionStorage.removeItem(k)
  })
  // Never reuse responses cached for a previous user on a shared machine.
  apiCache.clearAll()
}

export function getToken(): string | null {
  return read('authToken')
}

export function dashboardPath(role: UserRole | null | undefined): string {
  if (role === 'admin') return '/admin/dashboard'
  if (role === 'teacher') return '/teacher/dashboard'
  if (role === 'student') return '/student/dashboard'
  return '/'
}

/** Where a temp password gets replaced: admins change it on their profile page. */
export function changePasswordPath(role: UserRole): string {
  return role === 'admin' ? '/admin/profile' : `/${role}/change-password`
}

const ROLE_AREAS = ['/student', '/teacher', '/admin']

/**
 * A ?returnTo= target that is safe to follow after login: a same-origin path
 * (not //host or /\host) that isn't another role's area. Anything else falls
 * back to the user's dashboard.
 */
export function safeReturnTo(returnTo: string | null | undefined, role: UserRole): string {
  if (!returnTo || !returnTo.startsWith('/') || returnTo.startsWith('//') || returnTo.startsWith('/\\')) return dashboardPath(role)
  const area = ROLE_AREAS.find((a) => returnTo === a || returnTo.startsWith(`${a}/`))
  if (area && area !== `/${role}`) return dashboardPath(role)
  return returnTo
}
