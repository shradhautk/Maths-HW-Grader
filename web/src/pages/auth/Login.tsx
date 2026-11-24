import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext'
import { API, API_BASE_URL, ROLES, type RoleName } from '../../config'
import { apiRequest, isRouteMissing } from '../../lib/api'
import { changePasswordPath, safeReturnTo, type UserRole } from '../../lib/session'
import { AuthShell, LoginPreview, Notice, PasswordInput, RoleSwitch } from './AuthLayout'

interface LoginResponse {
  token: string
  user_id: string
  userRole?: UserRole
  name?: string
  [key: string]: unknown
}

type Message = { text: string; tone: 'error' | 'success' } | null

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [params] = useSearchParams()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(false)
  const [userType, setUserType] = useState<RoleName>('student')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<Message>(null)
  const redirectTimer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(redirectTimer.current), [])

  // An error stays on screen until the person changes something.
  const edited = <T,>(set: (v: T) => void) => (v: T) => {
    set(v)
    if (message?.tone === 'error') setMessage(null)
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    // Autofill and phone keyboards often add a trailing space.
    const typedEmail = email.trim()
    if (!typedEmail || !password) {
      setMessage({ text: 'Enter your email and password.', tone: 'error' })
      return
    }

    setLoading(true)
    setMessage(null)
    try {
      const data = await apiRequest<LoginResponse>(API.auth.login, {
        auth: false,
        body: { email: typedEmail, password, userType },
      })
      if (!data?.token) throw new Error('The server sent an unexpected response. Try again.')

      const role = data.userRole || userType
      login({ token: data.token, userId: data.user_id, userType: role, userName: data.name ?? 'User', userData: data }, remember)
      setMessage({ text: `Welcome back${data.name ? `, ${data.name}` : ''}. Opening your dashboard…`, tone: 'success' })

      // A temporary password (issued after an admin approves a reset) must be replaced first.
      const target = data.must_change_password === true ? changePasswordPath(role) : safeReturnTo(params.get('returnTo'), role)
      redirectTimer.current = window.setTimeout(() => navigate(target, { replace: true }), 700)
    } catch (err) {
      const text = err instanceof Error ? err.message : ''
      const label = (ROLES.find((r) => r.name === userType)?.label ?? userType).toLowerCase()
      const article = /^[aeiou]/.test(label) ? 'an' : 'a'
      setMessage({
        text:
          text === 'Invalid credentials'
            ? `That email and password don't match ${article} ${label} account. Check both, and that the right role is selected.` +
              // Easy to miss when a password is pasted; not trimmed, since a real password may contain spaces.
              (password !== password.trim() ? ' Your password starts or ends with a space — remove it unless it is part of your password.' : '')
            : isRouteMissing(err)
              ? `Can't reach the GradeAnt server (${API_BASE_URL}). Check your connection; if you're running locally, start the API with "cd backend && npm run dev".`
              : text || 'Login failed. Try again.',
        tone: 'error',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthShell title="Log in" aside={<LoginPreview role={userType} />}>
      <h1>Welcome back</h1>
      <p className="au-lead">Log in to pick up where you left off.</p>

      {message && <Notice tone={message.tone}>{message.text}</Notice>}

      <form className="lp-form au-form" onSubmit={handleSubmit} noValidate>
        {/* Roles and their labels come from config/app.yaml. */}
        <RoleSwitch name="userType" legend="I am a" roles={ROLES} value={userType} onChange={edited(setUserType)} />

        <div className="lp-field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            name="email"
            className="lp-input"
            type="email"
            autoComplete="username"
            required
            value={email}
            onChange={(e) => edited(setEmail)(e.target.value)}
          />
        </div>

        <div className="lp-field">
          <label htmlFor="password">Password</label>
          <PasswordInput id="password" autoComplete="current-password" value={password} onChange={(e) => edited(setPassword)(e.target.value)} />
        </div>

        <div className="au-row">
          <label className="au-check">
            <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
            Keep me signed in on this device
          </label>
          <Link to="/forgot-password" className="au-link">
            Forgot password?
          </Link>
        </div>

        <button type="submit" className="lp-btn lp-btn-primary lp-btn-lg au-submit" disabled={loading}>
          {loading ? 'Logging in…' : 'Log in'}
        </button>
      </form>

      <p className="au-switch">
        New to GradeAnt?{' '}
        <Link to="/signup" className="au-link">
          Create an account
        </Link>
      </p>
    </AuthShell>
  )
}
