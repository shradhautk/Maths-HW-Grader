import { useState, type ChangeEvent, type FormEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext'
import { API, ROLES, SELF_SIGNUP_ROLES, type RoleName } from '../../config'
import { apiRequest } from '../../lib/api'
import { dashboardPath } from '../../lib/session'
import { AuthShell, Notice, PasswordInput, PasswordRules, RoleSwitch, SignupPreview } from './AuthLayout'
import { unmetPasswordRule } from './passwordRules'

// Which roles can sign up here comes from config/app.yaml (roles.<role>.signup
// includes "self"). Invite-only roles are created from an admin's emailed invite.
const INVITE_ONLY = ROLES.filter((r) => !r.signup.includes('self') && r.signup.includes('invite') && r.name !== 'admin')

const EMPTY = {
  firstName: '',
  lastName: '',
  email: '',
  studentId: '',
  gradeLevel: '',
  institution: '',
  department: '',
  password: '',
}

export default function Signup() {
  const { login } = useAuth()
  const navigate = useNavigate()
  // ?role=teacher preselects a role (e.g. from the demo page), if that role can self-sign-up.
  const [params] = useSearchParams()
  const wanted = SELF_SIGNUP_ROLES.find((r) => r.name === params.get('role'))
  const [role, setRole] = useState<RoleName>(wanted?.name ?? SELF_SIGNUP_ROLES[0]?.name ?? 'student')
  const [form, setForm] = useState(EMPTY)
  const [terms, setTerms] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const field = (name: keyof typeof EMPTY) => ({
    id: name,
    name,
    className: 'lp-input',
    value: form[name],
    onChange: (e: ChangeEvent<HTMLInputElement>) => {
      setForm((f) => ({ ...f, [name]: e.target.value }))
      setError(null) // an error stays until the person changes something
    },
  })

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    const firstName = form.firstName.trim()
    const lastName = form.lastName.trim()
    const email = form.email.trim()

    if (!firstName || !lastName || !email || !form.password) {
      setError('Fill in your name, email and a password.')
      return
    }
    const unmet = unmetPasswordRule(form.password)
    if (unmet) {
      setError(`Your password needs ${unmet.label.toLowerCase()}.`)
      return
    }
    if (!terms) {
      setError('Accept the Terms of Service and Privacy Policy to create your account.')
      return
    }

    const roleSpecific =
      role === 'teacher'
        ? { institution: form.institution.trim(), department: form.department.trim() }
        : { studentId: form.studentId.trim(), gradeLevel: form.gradeLevel.trim() }
    const name = `${firstName} ${lastName}`

    setSubmitting(true)
    try {
      const data = await apiRequest<{ token: string; user_id: string }>(API.auth.signup, {
        auth: false,
        body: { email, password: form.password, name, role, ...roleSpecific },
      })
      login({ token: data.token, userId: data.user_id, userType: role, userName: name, userData: data }, true)
      navigate(dashboardPath(role), { replace: true })
    } catch (err) {
      const text = err instanceof Error ? err.message : ''
      setError(
        text === 'User already exists'
          ? 'An account with this email already exists. Log in instead, or use a different email.'
          : text || 'Sign up failed. Try again.',
      )
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthShell title="Sign up" aside={<SignupPreview role={role} />}>
      <h1>Create your account</h1>
      <p className="au-lead">Free to start. Takes about a minute.</p>

      {INVITE_ONLY.length > 0 && (
        <Notice tone="info">
          {INVITE_ONLY.map((r) => r.label).join(' and ')} accounts are created by your school administrator. Check your
          email for an invite.
        </Notice>
      )}
      {error && <Notice tone="error">{error}</Notice>}

      <form className="lp-form au-form" onSubmit={handleSubmit} noValidate>
        {SELF_SIGNUP_ROLES.length > 1 && (
          <RoleSwitch name="role" legend="I am a" roles={SELF_SIGNUP_ROLES} value={role} onChange={setRole} />
        )}

        <div className="lp-form-row">
          <div className="lp-field">
            <label htmlFor="firstName">First name</label>
            <input type="text" autoComplete="given-name" required {...field('firstName')} />
          </div>
          <div className="lp-field">
            <label htmlFor="lastName">Last name</label>
            <input type="text" autoComplete="family-name" required {...field('lastName')} />
          </div>
        </div>

        <div className="lp-field">
          <label htmlFor="email">Email</label>
          <input type="email" autoComplete="email" required {...field('email')} />
        </div>

        {role === 'teacher' ? (
          <div className="lp-form-row">
            <div className="lp-field">
              <label htmlFor="institution">
                School <span className="au-optional">(optional)</span>
              </label>
              <input type="text" autoComplete="organization" {...field('institution')} />
            </div>
            <div className="lp-field">
              <label htmlFor="department">
                Department <span className="au-optional">(optional)</span>
              </label>
              <input type="text" {...field('department')} />
            </div>
          </div>
        ) : (
          <div className="lp-form-row">
            <div className="lp-field">
              <label htmlFor="studentId">
                Student ID <span className="au-optional">(optional)</span>
              </label>
              <input type="text" {...field('studentId')} />
            </div>
            <div className="lp-field">
              <label htmlFor="gradeLevel">
                Grade or year <span className="au-optional">(optional)</span>
              </label>
              <input type="text" {...field('gradeLevel')} />
            </div>
          </div>
        )}

        <div className="lp-field">
          <label htmlFor="password">Password</label>
          <PasswordInput
            id="password"
            autoComplete="new-password"
            describedBy="password-rules"
            value={form.password}
            onChange={field('password').onChange}
          />
          <PasswordRules id="password-rules" value={form.password} />
        </div>

        <label className="au-check au-terms">
          <input
            type="checkbox"
            checked={terms}
            onChange={(e) => {
              setTerms(e.target.checked)
              setError(null)
            }}
          />
          <span>
            I agree to the{' '}
            <Link to="/terms" className="au-link">
              Terms of Service
            </Link>{' '}
            and{' '}
            <Link to="/privacy" className="au-link">
              Privacy Policy
            </Link>
          </span>
        </label>

        <button type="submit" className="lp-btn lp-btn-primary lp-btn-lg au-submit" disabled={submitting || !SELF_SIGNUP_ROLES.length}>
          {submitting ? 'Creating your account…' : 'Create account'}
        </button>
      </form>

      <p className="au-switch">
        Already have an account?{' '}
        <Link to="/login" className="au-link">
          Log in
        </Link>
      </p>
    </AuthShell>
  )
}
