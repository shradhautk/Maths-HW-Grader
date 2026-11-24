import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext'
import AppShell from '../../components/AppShell'
import PageScope from '../../components/PageScope'
import { changePassword } from '../../lib/account'
import { meetsRequiredRules, PASSWORD_RULES, passwordStrength } from '../../lib/password'
import { dashboardPath } from '../../lib/session'
import '../../styles/pages/profile.css'
import '../../styles/themes/account.css'

/**
 * Wireframe §07 "Change Password — after login". Doubles as the forced-change
 * screen after an admin-issued temporary password (must_change_password).
 */
export default function ChangePassword({ role }: { role: 'student' | 'teacher' }) {
  const { session, updateUserData } = useAuth()
  const navigate = useNavigate()
  const forced = session?.userData.must_change_password === true

  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)

  const strength = passwordStrength(next)
  const mismatch = confirm !== '' && confirm !== next

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    if (!current) return setError(forced ? 'Enter the temporary password you were given.' : 'Enter your current password.')
    if (!meetsRequiredRules(next)) return setError('Your new password must be 8+ characters with an uppercase letter and a number.')
    if (next !== confirm) return setError("New password and confirmation don't match.")
    if (next === current) return setError('Choose a password different from your current one.')

    setSaving(true)
    try {
      await changePassword(current, next)
      updateUserData({ must_change_password: false })
      setCurrent('')
      setNext('')
      setConfirm('')
      if (forced) {
        navigate(dashboardPath(role), { replace: true })
      } else {
        setDone(true)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not update your password. Please try again.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <PageScope name="profile" theme="account" className={role === 'teacher' ? 'profile--teacher' : undefined} title="Change Password">
      <AppShell footer={false} mainClassName="profile-container">
        <div className="container">
          <div className="profile-card account-card">
            <div className="profile-card-header">
              <h1>Change password</h1>
              <span className="role-pill">{role === 'teacher' ? 'Teacher' : 'Student'}</span>
            </div>

            {forced && (
              <div className="account-notice account-notice--warn" role="status">
                An administrator issued you a temporary password. Choose a new one to continue.
              </div>
            )}
            {done && (
              <div className="account-notice account-notice--ok" role="status">
                Password updated.
              </div>
            )}
            {error && (
              <div className="account-notice account-notice--error" role="alert">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="account-form" noValidate>
              <div className="account-field">
                <label htmlFor="current-password">{forced ? 'Temporary password' : 'Current password'}</label>
                <input
                  id="current-password"
                  type="password"
                  autoComplete="current-password"
                  value={current}
                  onChange={(e) => setCurrent(e.target.value)}
                />
              </div>

              <div className="account-field">
                <label htmlFor="new-password">New password</label>
                <input
                  id="new-password"
                  type="password"
                  autoComplete="new-password"
                  value={next}
                  onChange={(e) => setNext(e.target.value)}
                />
                <div className="pw-meter" aria-live="polite">
                  <div className={`pw-meter-bar pw-${strength.level}`}>
                    {[1, 2, 3, 4].map((n) => (
                      <span key={n} className={n <= strength.score ? 'on' : ''} />
                    ))}
                  </div>
                  <span className="pw-meter-label">Strength{strength.label && `: ${strength.label}`}</span>
                </div>
                <ul className="pw-rules">
                  {PASSWORD_RULES.map((rule) => (
                    <li key={rule.id} className={rule.test(next) ? 'met' : ''}>
                      <span aria-hidden="true">{rule.test(next) ? '✓' : '○'}</span>
                      {rule.label}
                      {!rule.required && <em> (optional)</em>}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="account-field">
                <label htmlFor="confirm-password">Confirm new password</label>
                <input
                  id="confirm-password"
                  type="password"
                  autoComplete="new-password"
                  aria-invalid={mismatch}
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                />
                {mismatch && <span className="account-hint account-hint--error">Passwords don't match.</span>}
              </div>

              <div className="profile-actions">
                {!forced && (
                  <Link to={`/${role}/profile`} className="btn btn-secondary">
                    Back to profile
                  </Link>
                )}
                <button type="submit" className="btn btn-primary" disabled={saving}>
                  <span className="material-icons">lock</span>
                  {saving ? 'Updating…' : 'Update Password'}
                </button>
              </div>
            </form>

            {!forced && (
              <p className="account-hint">
                Don't remember your current password? Teachers and students can't reset it directly — request a reset
                from your <Link to={`/${role}/profile`}>profile</Link> and an administrator will issue a temporary one.
              </p>
            )}
          </div>
        </div>
      </AppShell>
    </PageScope>
  )
}
