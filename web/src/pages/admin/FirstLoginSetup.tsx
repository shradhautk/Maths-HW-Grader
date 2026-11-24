import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext'
import { API, roleLabel, type RoleName } from '../../config'
import { apiRequest } from '../../lib/api'
import { dashboardPath, type UserRole } from '../../lib/session'
import { AuthShell, Notice, PasswordInput, PasswordRules, StepsPreview } from '../auth/AuthLayout'
import { unmetPasswordRule } from '../auth/passwordRules'

interface Invite {
  role?: string
  fullName?: string
  className?: string
}

/**
 * Step 3 of onboarding: set a password from an *unauthenticated* invite link
 * (?token=...). Not behind RequireRole — the visitor has no session yet.
 */
export default function FirstLoginSetup() {
  const [params] = useSearchParams()
  const token = params.get('token') ?? ''
  const navigate = useNavigate()
  const { login } = useAuth()

  const [invite, setInvite] = useState<Invite | null>(null)
  const [linkError, setLinkError] = useState<string | null>(token ? null : 'This invite link is missing part of its address.')
  const [error, setError] = useState<string | null>(null)
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [agreed, setAgreed] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (!token) return
    apiRequest<Invite>(API.admin, { auth: false, body: { operation: 'getInviteDetails', token } })
      .then(setInvite)
      .catch((err) => setLinkError((err as Error).message || 'This invite link doesn’t work.'))
  }, [token])

  const role = invite?.role ? roleLabel(invite.role as RoleName) : ''
  const edited = <T,>(set: (v: T) => void) => (v: T) => {
    set(v)
    setError(null) // an error stays until the person changes something
  }

  async function finish(e?: FormEvent) {
    e?.preventDefault()
    if (!invite || submitting) return
    const unmet = unmetPasswordRule(password)
    if (unmet) return setError(`Your password needs ${unmet.label.toLowerCase()}.`)
    if (password !== confirm) return setError('The two passwords don’t match.')
    if (!agreed) return setError('Accept the acceptable-use and privacy policy to finish setting up.')

    setSubmitting(true)
    try {
      const result = await apiRequest<{ token: string; userId: string; role: UserRole; fullName: string }>(API.admin, {
        auth: false,
        body: { operation: 'completeFirstLogin', token, password },
      })
      login({ token: result.token, userId: result.userId, userType: result.role, userName: result.fullName, userData: result }, false)
      navigate(dashboardPath(result.role), { replace: true })
    } catch (err) {
      setError((err as Error).message || 'Setup didn’t finish. Try again.')
      setSubmitting(false)
    }
  }

  const aside = (
    <StepsPreview
      k={invite ? 'invite-ready' : 'invite-waiting'}
      title={invite ? `Welcome${invite.fullName ? `, ${invite.fullName.split(' ')[0]}` : ''}. One step to go` : 'Setting up an invited account'}
      text={invite ? 'Your administrator has already created your account.' : 'Invites come from your school’s GradeAnt administrator.'}
      steps={[
        [
          'You were invited',
          invite ? `As ${/^[aeiou]/i.test(role) ? 'an' : 'a'} ${role.toLowerCase() || 'member'}${invite.className ? ` for ${invite.className}` : ''}.` : 'Your administrator sends a link to your email.',
        ],
        ['Choose your password', 'You’ll use it with your email to log in from now on.'],
        [role ? `Open your ${role.toLowerCase()} dashboard` : 'Open your dashboard', 'You go straight there when you finish.'],
      ]}
      done={invite ? 1 : 0}
    />
  )

  if (linkError) {
    return (
      <AuthShell title="Set up your account" aside={aside}>
        <h1>This invite link doesn’t work</h1>
        <Notice tone="error">{linkError}</Notice>
        <p className="au-lead" style={{ marginTop: 18 }}>
          Invite links can only be used once and expire after a while. Ask your administrator to send you a new invite.
        </p>
        <div className="lp-form au-form">
          <Link to="/login" className="lp-btn lp-btn-primary lp-btn-lg au-submit">
            I already have an account
          </Link>
          <Link to="/" className="lp-btn lp-btn-secondary lp-btn-lg au-submit">
            Go to the home page
          </Link>
        </div>
      </AuthShell>
    )
  }

  return (
    <AuthShell title="Set up your account" aside={aside}>
      <h1>Set up your account</h1>
      <p className="au-lead">
        {invite ? (
          <>
            You were invited as {/^[aeiou]/i.test(role) ? 'an' : 'a'} <b>{role.toLowerCase()}</b>
            {invite.className ? (
              <>
                {' '}for <b>{invite.className}</b>
              </>
            ) : null}
            . Choose a password to finish.
          </>
        ) : (
          'Checking your invite…'
        )}
      </p>

      {error && <Notice tone="error">{error}</Notice>}

      <form id="firstlogin-form" className="lp-form au-form" onSubmit={finish} noValidate>
        <div className="lp-field">
          <label htmlFor="createPassword">Password</label>
          <PasswordInput
            id="createPassword"
            autoComplete="new-password"
            describedBy="firstlogin-rules"
            value={password}
            onChange={(e) => edited(setPassword)(e.target.value)}
          />
          <PasswordRules id="firstlogin-rules" value={password} />
        </div>

        <div className="lp-field">
          <label htmlFor="confirmPassword">Type it again</label>
          <PasswordInput id="confirmPassword" autoComplete="new-password" value={confirm} onChange={(e) => edited(setConfirm)(e.target.value)} />
        </div>

        <label className="au-check au-terms">
          <input type="checkbox" checked={agreed} onChange={(e) => edited(setAgreed)(e.target.checked)} />
          <span>
            I agree to the{' '}
            <Link to="/terms" className="au-link">
              acceptable-use terms
            </Link>{' '}
            and{' '}
            <Link to="/privacy" className="au-link">
              privacy policy
            </Link>
          </span>
        </label>

        <button type="submit" className="lp-btn lp-btn-primary lp-btn-lg au-submit" disabled={!invite || submitting}>
          {submitting ? 'Setting up your account…' : role ? `Finish and open my ${role.toLowerCase()} dashboard` : 'Finish setting up'}
        </button>
      </form>
    </AuthShell>
  )
}
