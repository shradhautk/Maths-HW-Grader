import { useEffect, useState, type FormEvent } from 'react'
import { adminApiRequest } from '../../lib/api'
import AdminLayout, { useAdminAlert } from './AdminLayout'
import { AdminModal, useAdminData } from './adminShared'

interface Profile {
  fullName?: string
  username?: string
  email?: string
  phone?: string
  adminLevel?: string
}

const EMPTY_PASSWORDS = { current: '', next: '', confirm: '' }

/** Admin self-edit (including username) and a direct password change — no reset request needed. */
function ProfileBody() {
  const showAlert = useAdminAlert()
  const { data, error, reload } = useAdminData<Profile>('getProfile')
  const [form, setForm] = useState({ fullName: '', username: '', email: '', phone: '' })
  const [saving, setSaving] = useState(false)
  const [pwOpen, setPwOpen] = useState(false)
  const [pw, setPw] = useState(EMPTY_PASSWORDS)
  const [pwSaving, setPwSaving] = useState(false)

  useEffect(() => {
    if (error) showAlert(`Couldn't load profile: ${error}`, 'danger')
  }, [error, showAlert])
  useEffect(() => {
    if (data) setForm({ fullName: data.fullName || '', username: data.username || '', email: data.email || '', phone: data.phone || '' })
  }, [data])

  async function save() {
    setSaving(true)
    try {
      await adminApiRequest('updateProfile', {
        fullName: form.fullName.trim(),
        username: form.username.trim(),
        email: form.email.trim(),
        phone: form.phone.trim(),
      })
      showAlert('Profile updated.', 'success')
      void reload(true)
    } catch (err) {
      showAlert(`Couldn't save profile: ${(err as Error).message}`, 'danger')
    } finally {
      setSaving(false)
    }
  }

  function closePw() {
    setPwOpen(false)
    setPw(EMPTY_PASSWORDS)
  }

  async function changePassword(e?: FormEvent) {
    e?.preventDefault()
    if (pwSaving) return
    if (!pw.current || !pw.next) {
      showAlert('Enter your current and new password.', 'warning')
      return
    }
    if (pw.next !== pw.confirm) {
      showAlert("New password and confirmation don't match.", 'warning')
      return
    }
    setPwSaving(true)
    try {
      await adminApiRequest('changeOwnPassword', { currentPassword: pw.current, newPassword: pw.next })
      showAlert('Password updated.', 'success')
      closePw()
    } catch (err) {
      showAlert(`Couldn't change password: ${(err as Error).message}`, 'danger')
    } finally {
      setPwSaving(false)
    }
  }

  const field = (name: keyof typeof form, label: string) => (
    <div className="form-group">
      <label htmlFor={name}>{label}</label>
      <input type="text" className="form-control" id={name} value={form[name]} onChange={(e) => setForm({ ...form, [name]: e.target.value })} />
    </div>
  )
  const pwField = (name: keyof typeof pw, id: string, label: string) => (
    <div className="form-group">
      <label htmlFor={id}>{label}</label>
      <input
        type="password"
        className="form-control"
        id={id}
        autoComplete={name === 'current' ? 'current-password' : 'new-password'}
        value={pw[name]}
        onChange={(e) => setPw({ ...pw, [name]: e.target.value })}
      />
    </div>
  )

  return (
    <>
      <div className="admin-panel" style={{ maxWidth: 640, padding: 0 }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '18px 22px',
            borderBottom: '1px solid var(--border-subtle)',
          }}
        >
          <span className="admin-panel-title" style={{ margin: 0 }}>
            My profile
          </span>
          <span className="status-pill warning">Administrator</span>
        </div>
        <div style={{ padding: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
            <span style={{ width: 64, height: 64, borderRadius: '50%', background: 'var(--gradient-brand)', display: 'block' }} />
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 16, color: 'var(--ink-900)' }}>
                {data?.fullName || 'Admin'}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-subtle)', fontFamily: 'var(--font-mono)' }}>
                {data ? `${data.username || ''} · ${data.adminLevel || 'admin'}` : ''}
              </div>
              <span
                style={{ display: 'inline-block', marginTop: 6, fontSize: 12, fontWeight: 600, color: 'var(--text-link)', cursor: 'pointer' }}
              >
                Change photo
              </span>
            </div>
          </div>

          <div className="form">
            {field('fullName', 'Full Name')}
            {field('username', 'Username')}
            <div className="form-row">
              {field('email', 'Email')}
              {field('phone', 'Phone')}
            </div>
          </div>

          <div className="form-actions" style={{ marginTop: 20 }}>
            <button type="button" className="btn btn-primary" disabled={saving} onClick={save}>
              Save Changes
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => setPwOpen(true)}>
              Change Password
            </button>
          </div>
          <div style={{ marginTop: 14, fontSize: 12, color: 'var(--text-subtle)', lineHeight: 1.4 }}>
            Admin may edit any field including username, and change their own password directly — unlike
            teachers/students, who must go through the reset-request workflow on the Reset Requests page.
          </div>
        </div>
      </div>

      <AdminModal
        open={pwOpen}
        title="Change password"
        style={{ maxWidth: 420 }}
        onClose={closePw}
        footer={
          <>
            <button type="button" className="btn btn-secondary" onClick={closePw}>
              Cancel
            </button>
            <button type="submit" form="admin-password-form" className="btn btn-primary" disabled={pwSaving}>
              Update Password
            </button>
          </>
        }
      >
        {/* A real form, so Enter submits and password managers recognise the fields. */}
        <form id="admin-password-form" className="form" onSubmit={changePassword} noValidate>
          {pwField('current', 'currentPassword', 'Current password')}
          {pwField('next', 'newPassword', 'New password')}
          {pwField('confirm', 'confirmNewPassword', 'Confirm new password')}
        </form>
      </AdminModal>
    </>
  )
}

export default function AdminProfile() {
  return (
    <AdminLayout docTitle="My Profile" title="My profile">
      <ProfileBody />
    </AdminLayout>
  )
}
