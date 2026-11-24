import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { changePasswordPath, type UserRole } from '../lib/session'
import { useAuth } from './AuthContext'

/**
 * Route guard replacing checkAuthentication() in nav.html and
 * requireAdminAuth() in admin-common.js: no session -> login,
 * wrong role -> unauthorized, temp password still active -> forced change.
 */
export default function RequireRole({ roles }: { roles: UserRole[] }) {
  const { session } = useAuth()
  const location = useLocation()

  if (!session) {
    // Someone who was never signed in goes to the login page; "session
    // expired" is only for a session that actually ended (see lib/api.ts).
    const returnTo = encodeURIComponent(location.pathname + location.search)
    return <Navigate to={`/login?returnTo=${returnTo}`} replace />
  }
  if (!roles.includes(session.userType)) {
    return <Navigate to="/unauthorized" replace />
  }

  // An admin-issued temp password must be replaced before anything else works.
  const changePath = changePasswordPath(session.userType)
  if (session.userData.must_change_password === true && location.pathname !== changePath) {
    return <Navigate to={changePath} replace />
  }
  return <Outlet />
}
