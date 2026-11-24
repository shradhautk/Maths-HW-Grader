import { Fragment, useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { adminApiRequest } from '../../lib/api'
import AdminLayout, { capitalize, useAdminAlert } from './AdminLayout'
import { AdminModal, ConfirmButton, FilterPills, timeAgo, useAdminData } from './adminShared'

type Filter = 'pending' | 'approved' | 'rejected'

interface ResetRequest {
  id: string
  userName: string
  userEmail: string
  role: string
  requestedAt: string | number
  status: Filter
}

type Permission = 'yes' | 'self' | 'no'
interface Feature {
  label: string
  admin: Permission
  teacher: Permission
  student: Permission
}

const WORKFLOW = [
  { title: 'Forgot password', sub: 'User submits request', tone: '', icon: 'help' },
  { title: 'Admin notified', sub: 'Verifies user', tone: ' sky', icon: 'notifications' },
  { title: 'Admin resets', sub: 'Temp password generated', tone: '', icon: 'key' },
  { title: 'User logs in', sub: 'With temp password', tone: ' sky', icon: 'login' },
  { title: 'Forced change', sub: 'System requires new pw', tone: '', icon: 'lock_reset' },
  { title: 'Dashboard', sub: 'Access granted', tone: ' brand', icon: 'check' },
]
const SMALL_BTN = { padding: '6px 11px', fontSize: 12 }

function Cell({ value }: { value: Permission }) {
  return <div className={`cell ${value === 'yes' ? 'yes' : value === 'self' ? 'self' : 'no'}`}>{value === 'yes' ? '✓' : value === 'self' ? 'Self' : '✗'}</div>
}

function RequestsBody() {
  const showAlert = useAdminAlert()
  const [filter, setFilter] = useState<Filter>('pending')
  const [tempPassword, setTempPassword] = useState<string | null>(null)
  const requests = useAdminData<{ requests?: ResetRequest[]; pendingCount?: number }>('listResetRequests', { status: filter })
  const matrix = useAdminData<{ features?: Feature[] }>('getPermissionMatrix')

  // Dashboard's "Permission matrix" link lands on the matrix once it has loaded.
  const { hash } = useLocation()
  useEffect(() => {
    if (hash === '#permission-matrix' && matrix.data) document.getElementById('permission-matrix')?.scrollIntoView({ behavior: 'smooth' })
  }, [hash, matrix.data])

  async function act(operation: string, requestId: string, success?: string) {
    try {
      const result = await adminApiRequest<{ tempPassword?: string }>(operation, { requestId })
      if (operation === 'generateTempPassword') setTempPassword(result.tempPassword ?? '')
      if (success) showAlert(success, 'success')
      void requests.reload(true)
    } catch (err) {
      const what = { approveResetRequest: 'approve request', rejectResetRequest: 'reject request', generateTempPassword: 'generate a temp password' }[operation]
      showAlert(`Couldn't ${what}: ${(err as Error).message}`, 'danger')
    }
  }

  const list = requests.data?.requests ?? []
  const pendingCount = requests.data?.pendingCount

  return (
    <>
      <div className="admin-rule-banner">
        Rule: teachers &amp; students <b>cannot self-reset</b> passwords. They submit a request → admin approves → temp
        password issued → forced change on next login.
      </div>

      <div className="admin-workflow">
        {WORKFLOW.map((step, i) => (
          <Fragment key={step.title}>
            {i > 0 && <span className="admin-workflow-arrow">→</span>}
            <div className={`admin-workflow-step${step.tone}`}>
              <div className="admin-workflow-icon">
                <span className="material-icons" aria-hidden>
                  {step.icon}
                </span>
              </div>
              <div className="admin-workflow-step-title">{step.title}</div>
              <div className="admin-workflow-step-sub">{step.sub}</div>
            </div>
          </Fragment>
        ))}
      </div>

      <div className="admin-analytics-row admin-reset-row">
        <div>
          <FilterPills<Filter>
            value={filter}
            onChange={setFilter}
            options={[
              { value: 'pending', label: typeof pendingCount === 'number' ? `Pending ${pendingCount}` : 'Pending' },
              { value: 'approved', label: 'Approved' },
              { value: 'rejected', label: 'Rejected' },
            ]}
          />
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Role</th>
                  <th>Requested</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {requests.error ? (
                  <tr>
                    <td colSpan={4}>Couldn't load requests: {requests.error}</td>
                  </tr>
                ) : !requests.data ? (
                  <tr>
                    <td colSpan={4}>Loading requests…</td>
                  </tr>
                ) : list.length === 0 ? (
                  <tr>
                    <td colSpan={4}>No {filter} requests.</td>
                  </tr>
                ) : (
                  list.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <div className="admin-cell-name">{r.userName}</div>
                        <div className="admin-cell-sub" style={{ fontFamily: 'var(--font-mono)' }}>
                          {r.userEmail}
                        </div>
                      </td>
                      <td>
                        <span className={`status-pill ${r.role === 'teacher' || r.role === 'student' ? 'info' : 'warning'}`}>
                          {capitalize(r.role)}
                        </span>
                      </td>
                      <td>{timeAgo(r.requestedAt)}</td>
                      <td className="admin-inline-actions">
                        {r.status === 'pending' ? (
                          <>
                            <button
                              type="button"
                              className="btn btn-primary"
                              style={{ ...SMALL_BTN, background: 'var(--accent)' }}
                              onClick={() => act('approveResetRequest', r.id, 'Request approved. Generate a temp password when ready.')}
                            >
                              Approve
                            </button>
                            <ConfirmButton
                              label="Reject"
                              confirmLabel="Confirm reject"
                              style={SMALL_BTN}
                              onConfirm={() => act('rejectResetRequest', r.id, 'Request rejected.')}
                            />
                          </>
                        ) : r.status === 'approved' ? (
                          <button type="button" className="btn btn-primary" style={SMALL_BTN} onClick={() => act('generateTempPassword', r.id)}>
                            Generate Temp Password
                          </button>
                        ) : (
                          <span className="form-text">{capitalize(r.status)}</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="admin-panel" id="permission-matrix">
          <div className="admin-panel-title">Permissions by role</div>
          <div className="admin-matrix">
            <div className="head">Feature</div>
            <div className="head role-admin">Admin</div>
            <div className="head role-teacher">Teacher</div>
            <div className="head role-student">Student</div>
            {matrix.error ? (
              <div className="feature" style={{ gridColumn: '1 / -1' }}>
                Couldn't load permission matrix: {matrix.error}
              </div>
            ) : (
              matrix.data?.features?.map((f) => (
                <Fragment key={f.label}>
                  <div className="feature">{f.label}</div>
                  <Cell value={f.admin} />
                  <Cell value={f.teacher} />
                  <Cell value={f.student} />
                </Fragment>
              ))
            )}
          </div>
        </div>
      </div>

      <AdminModal
        open={tempPassword !== null}
        title="Temporary password issued"
        style={{ maxWidth: 420 }}
        onClose={() => setTempPassword(null)}
        footerStyle={{ justifyContent: 'flex-end' }}
        footer={
          <button type="button" className="btn btn-primary" onClick={() => setTempPassword(null)}>
            Done
          </button>
        }
      >
        <p className="form-text" style={{ marginBottom: 10 }}>
          Share this with the user through a verified channel (not email/chat where it could be intercepted). It must be
          changed on their next login.
        </p>
        <div className="admin-callout-warning" style={{ fontFamily: 'var(--font-mono)', fontSize: 15, textAlign: 'center' }}>
          {tempPassword || '••••••••'}
        </div>
      </AdminModal>
    </>
  )
}

/** Password reset request queue + the read-only permission matrix. */
export default function AdminResetRequests() {
  return (
    <AdminLayout docTitle="Reset Requests" title="Password reset requests" subtitle="Approve & issue temp password">
      <RequestsBody />
    </AdminLayout>
  )
}
