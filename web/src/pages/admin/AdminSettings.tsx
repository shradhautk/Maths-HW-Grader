import { useEffect, useState } from 'react'
import { adminApiRequest } from '../../lib/api'
import AdminLayout, { useAdminAlert } from './AdminLayout'
import { useAdminData } from './adminShared'

interface Settings {
  platformName?: string
  supportEmail?: string
  sessionTimeoutMinutes?: number
  maintenanceMode?: boolean
  allowSelfSignup?: boolean
}

const CHECK_LABEL = { display: 'flex', alignItems: 'center', gap: 9, fontSize: 13, color: 'var(--ink-700)' } as const

function SettingsForm() {
  const showAlert = useAdminAlert()
  const { data, error } = useAdminData<Settings>('getSettings')
  const [form, setForm] = useState({
    platformName: '',
    supportEmail: '',
    sessionTimeout: '30',
    maintenanceMode: false,
    allowSelfSignup: false,
  })
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (error) showAlert(`Couldn't load settings: ${error}`, 'danger')
  }, [error, showAlert])

  useEffect(() => {
    if (!data) return
    setForm({
      platformName: data.platformName || '',
      supportEmail: data.supportEmail || '',
      sessionTimeout: String(data.sessionTimeoutMinutes || 30),
      maintenanceMode: !!data.maintenanceMode,
      allowSelfSignup: !!data.allowSelfSignup,
    })
  }, [data])

  async function save() {
    setSaving(true)
    try {
      await adminApiRequest('updateSettings', {
        platformName: form.platformName.trim(),
        supportEmail: form.supportEmail.trim(),
        sessionTimeoutMinutes: Number(form.sessionTimeout) || 30,
        maintenanceMode: form.maintenanceMode,
        allowSelfSignup: form.allowSelfSignup,
      })
      showAlert('Settings saved.', 'success')
    } catch (err) {
      showAlert(`Couldn't save settings: ${(err as Error).message}`, 'danger')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="admin-panel" style={{ maxWidth: 640 }}>
      <div className="admin-panel-title">Platform</div>
      <div className="form">
        <div className="form-group">
          <label htmlFor="platformName">Platform name</label>
          <input
            type="text"
            className="form-control"
            id="platformName"
            placeholder="GradeAnt"
            value={form.platformName}
            onChange={(e) => setForm({ ...form, platformName: e.target.value })}
          />
        </div>
        <div className="form-group">
          <label htmlFor="supportEmail">Support contact email</label>
          <input
            type="email"
            className="form-control"
            id="supportEmail"
            placeholder="support@gradeant.io"
            value={form.supportEmail}
            onChange={(e) => setForm({ ...form, supportEmail: e.target.value })}
          />
        </div>
        <div className="form-group">
          <label htmlFor="sessionTimeout">Session timeout (minutes)</label>
          <input
            type="number"
            className="form-control"
            id="sessionTimeout"
            min={5}
            max={480}
            placeholder="30"
            value={form.sessionTimeout}
            onChange={(e) => setForm({ ...form, sessionTimeout: e.target.value })}
          />
        </div>
        <label style={CHECK_LABEL}>
          <input
            type="checkbox"
            checked={form.maintenanceMode}
            onChange={(e) => setForm({ ...form, maintenanceMode: e.target.checked })}
          />
          Maintenance mode — signs out non-admins and blocks new logins
        </label>
        <label style={CHECK_LABEL}>
          <input
            type="checkbox"
            checked={form.allowSelfSignup}
            onChange={(e) => setForm({ ...form, allowSelfSignup: e.target.checked })}
          />
          Allow public self-signup (off = invite-only, per the Onboarding flow)
        </label>
      </div>
      <div className="form-actions">
        <button type="button" className="btn btn-primary" disabled={saving} onClick={save}>
          Save Changes
        </button>
      </div>
    </div>
  )
}

export default function AdminSettings() {
  return (
    <AdminLayout docTitle="System Settings" title="System settings">
      <SettingsForm />
    </AdminLayout>
  )
}
