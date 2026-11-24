import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../../auth/AuthContext'
import AppShell from '../../components/AppShell'
import PageScope from '../../components/PageScope'
import { fetchMe, requestPasswordReset, uploadProfilePhoto, type MeResponse } from '../../lib/account'
import '../../styles/pages/profile.css'
import '../../styles/themes/account.css'

const ROLE_FIELDS = {
  student: [
    { icon: 'school', label: 'Grade Level', key: 'gradeLevel', empty: 'Not specified' },
    { icon: 'assignment', label: 'Assignments Submitted', key: 'assignmentsSubmitted', empty: '0' },
    { icon: 'grade', label: 'Average Grade', key: 'averageGrade', empty: 'N/A' },
  ],
  teacher: [
    { icon: 'subject', label: 'Subject', key: 'subject', empty: 'Not specified' },
    { icon: 'school', label: 'Department', key: 'department', empty: 'Not specified' },
    { icon: 'assignment', label: 'Assignments Created', key: 'assignmentsCreated', empty: '0' },
    { icon: 'people', label: 'Students', key: 'studentCount', empty: '0' },
  ],
}

// Fields shown elsewhere on the card, or private.
const HIDDEN_KEYS = new Set([
  'name', 'userRole', 'userType', 'user_id', 'userId', 'email', 'loginTime', 'authToken', 'token', 'password',
  'photoUrl', 'must_change_password', 'message',
])

/** "gradeLevel" / "grade_level" -> "Grade Level". */
function formatLabel(key: string): string {
  return key
    .replace(/([A-Z])/g, ' $1')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (l) => l.toUpperCase())
    .trim()
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  return typeof v === 'object' ? JSON.stringify(v) : String(v)
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  if (!iso || Number.isNaN(d.getTime())) return 'Not available'
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

type ResetStep = 'idle' | 'confirm' | 'sending' | 'sent'

/** Student/teacher profile, built from the stored login response (was user_profile.js). */
export default function Profile({ role }: { role: 'student' | 'teacher' }) {
  const { session, updateUserData } = useAuth()
  const { hash } = useLocation()
  const [notice, setNotice] = useState<string | null>(null)
  const [me, setMe] = useState<MeResponse | null>(null)
  const [photoBusy, setPhotoBusy] = useState(false)
  const [photoError, setPhotoError] = useState<string | null>(null)
  const [resetStep, setResetStep] = useState<ResetStep>('idle')
  const [resetError, setResetError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const resetRef = useRef<HTMLDivElement>(null)

  const data = session?.userData ?? {}
  const roleLabel = role === 'teacher' ? 'Teacher' : 'Student'
  const fields = ROLE_FIELDS[role]
  const fieldKeys = new Set(fields.map((f) => f.key))
  const extras = Object.entries(data).filter(([k]) => !HIDDEN_KEYS.has(k) && !fieldKeys.has(k))
  const photoUrl = typeof data.photoUrl === 'string' && data.photoUrl ? data.photoUrl : null

  // Role and status are admin-managed, so show the server's copy rather than the
  // login-time snapshot. The same call refreshes the photo's presigned URL, which
  // expires after an hour.
  useEffect(() => {
    let cancelled = false
    fetchMe()
      .then((m) => {
        if (cancelled) return
        setMe(m)
        updateUserData({ photoUrl: m.photoUrl })
      })
      .catch(() => {
        /* the profile still renders from the stored login data */
      })
    return () => {
      cancelled = true
    }
  }, [updateUserData])

  useEffect(() => {
    if (hash === '#request-reset') resetRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [hash])

  async function handlePhoto(file: File | undefined) {
    if (!file) return
    setPhotoBusy(true)
    setPhotoError(null)
    try {
      updateUserData({ photoUrl: await uploadProfilePhoto(file) })
    } catch (err) {
      setPhotoError(err instanceof Error ? err.message : 'Could not update your photo.')
    } finally {
      setPhotoBusy(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function sendResetRequest() {
    const email = typeof data.email === 'string' ? data.email : ''
    if (!email) {
      setResetError('We could not find your email address — please log in again.')
      return
    }
    setResetStep('sending')
    setResetError(null)
    try {
      await requestPasswordReset(email)
      setResetStep('sent')
    } catch (err) {
      setResetError(err instanceof Error ? err.message : 'Could not submit your request.')
      setResetStep('confirm')
    }
  }

  const lockedFields = [
    { label: `${roleLabel} ID`, value: session?.userId || 'Not available', plain: true },
    { label: 'Role', value: me?.userRole || role },
    { label: 'Status', value: me?.status ?? '—' },
  ]

  return (
    <PageScope name="profile" theme="account" className={role === 'teacher' ? 'profile--teacher' : undefined} title={`${roleLabel} Profile`}>
      <AppShell footer={false} mainClassName="profile-container">
        <div className="container">
          <div className="profile-card">
            {notice && <div className="error-message">{notice}</div>}
            <div className="profile-card-header">
              <h1>{roleLabel} Profile</h1>
              <span className="role-pill">{roleLabel}</span>
            </div>

            <div className="profile-header">
              <div className="profile-avatar-wrap">
                <div className="profile-avatar">
                  <img
                    src={photoUrl ?? (role === 'teacher' ? '/img/teacher-avatar.jpg' : '/img/student.jpg')}
                    alt={`${roleLabel} Avatar`}
                  />
                </div>
                <div className="photo-actions">
                  <input
                    ref={fileInput}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    hidden
                    onChange={(e) => void handlePhoto(e.target.files?.[0])}
                  />
                  <button type="button" className="btn btn-secondary" disabled={photoBusy} onClick={() => fileInput.current?.click()}>
                    <span className="material-icons">photo_camera</span>
                    {photoBusy ? 'Uploading…' : 'Change photo'}
                  </button>
                  {photoError && (
                    <span className="photo-error" role="alert">
                      {photoError}
                    </span>
                  )}
                </div>
              </div>
              <div className="profile-title">
                <h2>{session?.userName || 'Unknown User'}</h2>
              </div>
            </div>

            <div className="profile-details">
              <div className="detail-item">
                <span className="material-icons">email</span>
                <div>
                  <h3>Email Address</h3>
                  <p>{formatValue(data.email ?? 'Not available')}</p>
                </div>
              </div>
              {fields.slice(0, role === 'teacher' ? 2 : 1).map((f) => (
                <div className="detail-item" key={f.key}>
                  <span className="material-icons">{f.icon}</span>
                  <div>
                    <h3>{f.label}</h3>
                    <p>{data[f.key] ? formatValue(data[f.key]) : f.empty}</p>
                  </div>
                </div>
              ))}
              <div className="detail-item">
                <span className="material-icons">calendar_today</span>
                <div>
                  <h3>Last Login</h3>
                  <p>{formatDate(session?.loginTime ?? '')}</p>
                </div>
              </div>
              {fields.slice(role === 'teacher' ? 2 : 1).map((f) => (
                <div className="detail-item" key={f.key}>
                  <span className="material-icons">{f.icon}</span>
                  <div>
                    <h3>{f.label}</h3>
                    <p>{data[f.key] ? formatValue(data[f.key]) : f.empty}</p>
                  </div>
                </div>
              ))}
              {extras.map(([key, value]) => (
                <div className="detail-item" key={key}>
                  <span className="material-icons">info</span>
                  <div>
                    <h3>{formatLabel(key)}</h3>
                    <p>{formatValue(value)}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="locked-fields">
              {lockedFields.map((f) => (
                <div key={f.label}>
                  <div className="locked-field-label">
                    <span>{f.label}</span>
                    <span className="locked-pill">LOCKED</span>
                  </div>
                  <div className="locked-field-value" style={f.plain ? undefined : { textTransform: 'capitalize' }}>
                    {f.value}
                  </div>
                </div>
              ))}
              <p className="account-hint">Your ID, role and status are managed by an administrator.</p>
            </div>

            <div className="profile-actions">
              {/* Editing details has no backend yet (the old page showed an alert). */}
              <button className="btn btn-primary" onClick={() => setNotice('Editing your profile is coming soon.')}>
                <span className="material-icons">edit</span>
                Edit Profile
              </button>
              <Link className="btn btn-secondary" to={`/${role}/change-password`}>
                <span className="material-icons">lock</span>
                Change Password
              </Link>
            </div>

            <div className="reset-request" id="request-reset" ref={resetRef}>
              <strong>Forgot your password?</strong>
              {resetStep === 'sent' ? (
                <p role="status">Your password reset request has been sent to the administrator.</p>
              ) : (
                <>
                  <p>
                    You can't reset it yourself — an administrator approves the request and issues a temporary password.
                  </p>
                  {resetError && (
                    <p className="photo-error" role="alert">
                      {resetError}
                    </p>
                  )}
                  <div className="reset-request-actions">
                    {resetStep === 'idle' ? (
                      <button className="btn btn-secondary" onClick={() => setResetStep('confirm')}>
                        Request Reset
                      </button>
                    ) : (
                      <>
                        <button className="btn btn-primary" disabled={resetStep === 'sending'} onClick={() => void sendResetRequest()}>
                          {resetStep === 'sending' ? 'Sending…' : 'Send request to administrator'}
                        </button>
                        <button className="btn btn-secondary" disabled={resetStep === 'sending'} onClick={() => setResetStep('idle')}>
                          Cancel
                        </button>
                      </>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </AppShell>
    </PageScope>
  )
}
