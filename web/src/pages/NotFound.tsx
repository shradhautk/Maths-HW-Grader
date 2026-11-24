import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { dashboardPath } from '../lib/session'
import StateShell from './session/StateShell'

export default function NotFound() {
  const { pathname } = useLocation()
  const { session } = useAuth()
  return (
    <StateShell
      title="Page not found"
      icon="travel_explore"
      tone="info"
      heading="We couldn't find that page"
      actions={
        <>
          {session && (
            <Link to={dashboardPath(session.userType)} className="lp-btn lp-btn-primary lp-btn-lg">
              Go to my dashboard
            </Link>
          )}
          <Link to="/" className={`lp-btn lp-btn-lg ${session ? 'lp-btn-secondary' : 'lp-btn-primary'}`}>
            Go to the home page
          </Link>
        </>
      }
    >
      <p>
        There's no page at <code>{pathname}</code>. It may have moved, or the link may have a typo.
      </p>
    </StateShell>
  )
}
