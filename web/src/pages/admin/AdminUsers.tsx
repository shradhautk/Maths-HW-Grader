import { useEffect, useRef, useState, type RefObject } from 'react'
import { adminApiRequest } from '../../lib/api'
import AdminLayout, { capitalize, useAdminAlert } from './AdminLayout'
import { AdminModal, ConfirmButton, useAdminData, useDebounced } from './adminShared'

interface User {
  id: string
  name: string
  email: string
  role: string
  school?: string
  status: 'active' | 'invited' | 'suspended'
}

interface Invitee {
  email: string
  name: string
  role?: string
  className?: string
}

const STATUS_PILL: Record<User['status'], string> = { active: 'success', invited: 'warning', suspended: 'info' }

/** "email,name" per line — the roster bulk-add case, not a general CSV parser. */
function parseCsv(text: string): Invitee[] {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((line) => {
      const [email = '', name = ''] = line.split(',').map((s) => s.trim())
      return { email, name: name || email }
    })
    .filter((p) => p.email.includes('@'))
}

/**
 * "Add people" wizard: invite by email or import CSV → assign roles → send.
 * Inviting is how a user gets added here.
 */
function AddPeopleWizard({
  open,
  startWithImport,
  fileRef,
  onClose,
  onSent,
}: {
  open: boolean
  startWithImport: boolean
  /** Owned by the page so "Bulk Import" can open the picker within its own click (browsers require a user gesture). */
  fileRef: RefObject<HTMLInputElement | null>
  onClose: () => void
  onSent: () => void
}) {
  const showAlert = useAdminAlert()
  const [step, setStep] = useState<1 | 2>(1)
  const [mode, setMode] = useState<'invite' | 'import'>('invite')
  const [queue, setQueue] = useState<Invitee[]>([])
  const [email, setEmail] = useState('')
  const [name, setName] = useState('')
  const [sending, setSending] = useState(false)

  // Reset each time the wizard opens.
  useEffect(() => {
    if (!open) return
    setStep(1)
    setQueue([])
    setEmail('')
    setName('')
    setMode(startWithImport ? 'import' : 'invite')
  }, [open, startWithImport])

  function addToList() {
    if (!email.trim().includes('@')) {
      showAlert('Enter a valid email address.', 'warning')
      return
    }
    setQueue((q) => [...q, { email: email.trim(), name: name.trim() || email.trim() }])
    setEmail('')
    setName('')
  }

  function importCsv(file: File) {
    const reader = new FileReader()
    reader.onload = () => {
      const parsed = parseCsv(String(reader.result))
      if (!parsed.length) {
        showAlert('No valid rows found in that CSV (expected email,name per line).', 'warning')
        return
      }
      setQueue((q) => [...q, ...parsed])
      showAlert(`Added ${parsed.length} people from CSV.`, 'success')
    }
    reader.readAsText(file)
  }

  async function send() {
    setSending(true)
    try {
      // Unassigned rows default to student, matching the role select's default.
      await adminApiRequest('inviteUsers', { invitees: queue.map((p) => ({ ...p, role: p.role || 'student' })) })
      showAlert(`Sent ${queue.length} invite(s).`, 'success')
      onClose()
      onSent()
    } catch (err) {
      showAlert(`Couldn't send invites: ${(err as Error).message}`, 'danger')
    } finally {
      setSending(false)
    }
  }

  const update = (i: number, patch: Partial<Invitee>) => setQueue((q) => q.map((p, j) => (j === i ? { ...p, ...patch } : p)))
  const stepClass = (s: number) => `step${s < step ? ' done' : s === step ? ' current' : ''}`

  return (
    <AdminModal
      open={open}
      title="Add people"
      onClose={onClose}
      header={
        <div className="admin-stepper">
          <span className={stepClass(1)}>
            <span className="dot">1</span>Invite
          </span>
          <span className="sep">›</span>
          <span className={stepClass(2)}>
            <span className="dot">2</span>Roles
          </span>
          <span className="sep">›</span>
          <span className="step">First login</span>
        </div>
      }
    >
      <div className={`admin-wizard-step${step === 1 ? ' active' : ''}`}>
        <div className="admin-onboard-choice">
          <div className={`admin-onboard-choice-card${mode === 'invite' ? ' active' : ''}`} onClick={() => setMode('invite')}>
            <div className="admin-onboard-choice-icon">
              <span className="material-icons">mail</span>
            </div>
            <div className="admin-onboard-choice-title">Invite by email</div>
            <div className="admin-onboard-choice-sub">Add a few people one at a time.</div>
          </div>
          <div
            className={`admin-onboard-choice-card${mode === 'import' ? ' active' : ''}`}
            onClick={() => {
              setMode('import')
              fileRef.current?.click()
            }}
          >
            <div className="admin-onboard-choice-icon">
              <span className="material-icons">upload</span>
            </div>
            <div className="admin-onboard-choice-title">Import CSV</div>
            <div className="admin-onboard-choice-sub">Bulk-add a whole roster at once.</div>
          </div>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) importCsv(file)
            e.target.value = ''
          }}
        />

        {mode === 'invite' && (
          <div className="admin-panel">
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="inviteEmail">Email address</label>
                <input
                  type="email"
                  className="form-control"
                  id="inviteEmail"
                  placeholder="name@school.edu"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label htmlFor="inviteName">Full name</label>
                <input
                  type="text"
                  className="form-control"
                  id="inviteName"
                  placeholder="Jane Doe"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="form-text">An invite email with a setup link will be sent.</span>
              <button type="button" className="btn btn-secondary" onClick={addToList}>
                + Add to list
              </button>
            </div>
          </div>
        )}

        <div className="admin-invite-list">
          {queue.map((p, i) => (
            <div className="admin-invite-list-item" key={`${p.email}-${i}`}>
              <span>
                {p.name} · {p.email}
              </span>
              <button type="button" title="Remove" onClick={() => setQueue((q) => q.filter((_, j) => j !== i))}>
                &times;
              </button>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 'var(--space-4)' }}>
          <button type="button" className="btn btn-primary" disabled={queue.length === 0} onClick={() => setStep(2)}>
            Continue to roles →
          </button>
        </div>
      </div>

      <div className={`admin-wizard-step${step === 2 ? ' active' : ''}`}>
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Person</th>
                <th>Role</th>
                <th>Class</th>
              </tr>
            </thead>
            <tbody>
              {queue.map((p, i) => (
                <tr key={`${p.email}-${i}`}>
                  <td>
                    <div className="admin-cell-user">
                      <span className="admin-cell-avatar" />
                      <span>{p.name}</span>
                    </div>
                  </td>
                  <td>
                    <select className="admin-role-select" value={p.role || 'student'} onChange={(e) => update(i, { role: e.target.value })}>
                      <option value="teacher">Teacher</option>
                      <option value="student">Student</option>
                      <option value="admin">Admin</option>
                    </select>
                  </td>
                  <td>
                    <input
                      type="text"
                      className="form-control"
                      placeholder="e.g. PHY-204"
                      value={p.className || ''}
                      style={{ maxWidth: 140, padding: '6px 10px' }}
                      onChange={(e) => update(i, { className: e.target.value })}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="admin-info-callout">
          <span className="material-icons" style={{ fontSize: 16 }}>info</span>Role sets each person's permissions &amp;
          navigation — see the permission matrix on the Reset Requests page.
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <button type="button" className="btn btn-secondary" onClick={() => setStep(1)}>
            Back
          </button>
          <button type="button" className="btn btn-primary" disabled={sending} onClick={send}>
            {sending ? 'Sending…' : 'Send invites →'}
          </button>
        </div>
      </div>
    </AdminModal>
  )
}

function UsersBody({
  onCount,
  wizard,
  closeWizard,
  fileRef,
}: {
  onCount: (n: number) => void
  wizard: 'closed' | 'invite' | 'import'
  closeWizard: () => void
  fileRef: RefObject<HTMLInputElement | null>
}) {
  const showAlert = useAdminAlert()
  const [search, setSearch] = useState('')
  const [role, setRole] = useState('')
  const [status, setStatus] = useState('')
  const debouncedSearch = useDebounced(search.trim())
  const { data, error, reload } = useAdminData<{ users?: User[]; total?: number }>('listUsers', { search: debouncedSearch, role, status })
  const users = data?.users ?? []
  const total = data ? (data.total ?? users.length) : null
  useEffect(() => {
    if (total !== null) onCount(total)
  }, [total, onCount])

  async function toggleStatus(u: User) {
    const next = u.status === 'suspended' ? 'active' : 'suspended'
    try {
      await adminApiRequest('updateUserStatus', { userId: u.id, status: next })
      showAlert(`User ${next === 'suspended' ? 'suspended' : 'activated'}.`, 'success')
      void reload(true)
    } catch (err) {
      showAlert(`Couldn't update status: ${(err as Error).message}`, 'danger')
    }
  }

  async function remove(u: User) {
    try {
      await adminApiRequest('deleteUser', { userId: u.id })
      showAlert('User deleted.', 'success')
      void reload(true)
    } catch (err) {
      showAlert(`Couldn't delete user: ${(err as Error).message}`, 'danger')
    }
  }

  return (
    <>
      <div className="admin-toolbar">
        <input
          type="text"
          className="form-control admin-toolbar-search"
          placeholder="Search by name or email…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select className="form-control" value={role} onChange={(e) => setRole(e.target.value)}>
          <option value="">Role</option>
          <option value="teacher">Teacher</option>
          <option value="student">Student</option>
          <option value="admin">Admin</option>
        </select>
        <select className="form-control" value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">Status</option>
          <option value="active">Active</option>
          <option value="invited">Invited</option>
          <option value="suspended">Suspended</option>
        </select>
      </div>

      <div className="table-responsive">
        <table className="data-table">
          <thead>
            <tr>
              <th>User</th>
              <th>Role</th>
              <th>School</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {error ? (
              <tr>
                <td colSpan={5}>Couldn't load users: {error}</td>
              </tr>
            ) : !data ? (
              <tr>
                <td colSpan={5}>Loading users…</td>
              </tr>
            ) : users.length === 0 ? (
              <tr>
                <td colSpan={5}>No users match this search.</td>
              </tr>
            ) : (
              users.map((u) => (
                <tr key={u.id}>
                  <td>
                    <div className="admin-cell-user">
                      <span className="admin-cell-avatar" />
                      <div>
                        <div className="admin-cell-name">{u.name}</div>
                        <div className="admin-cell-sub">{u.email}</div>
                      </div>
                    </div>
                  </td>
                  <td>{capitalize(u.role)}</td>
                  <td>{u.school || '—'}</td>
                  <td>
                    <span className={`status-pill ${STATUS_PILL[u.status] ?? 'info'}`}>{capitalize(u.status)}</span>
                  </td>
                  <td className="admin-inline-actions">
                    <button type="button" className="btn btn-secondary" onClick={() => toggleStatus(u)}>
                      {u.status === 'suspended' ? 'Activate' : 'Suspend'}
                    </button>
                    <ConfirmButton
                      label="Delete"
                      confirmLabel="Delete permanently"
                      style={{ color: 'var(--color-danger)' }}
                      onConfirm={() => remove(u)}
                    />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <AddPeopleWizard
        open={wizard !== 'closed'}
        startWithImport={wizard === 'import'}
        fileRef={fileRef}
        onClose={closeWizard}
        onSent={() => void reload(true)}
      />
    </>
  )
}

/** User list/search/filter + the "Add people" wizard. */
export default function AdminUsers() {
  const [count, setCount] = useState<number | null>(null)
  const [wizard, setWizard] = useState<'closed' | 'invite' | 'import'>('closed')
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <AdminLayout
      docTitle="Users"
      title={
        <>
          Users <span className="count">{count !== null ? `· ${count}` : ''}</span>
        </>
      }
      subtitle="Teachers & students, roles, bulk import"
      actions={
        <>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => {
              setWizard('import')
              fileRef.current?.click()
            }}
          >
            <span className="material-icons" style={{ fontSize: 18 }}>upload_file</span>Bulk Import (CSV)
          </button>
          <button type="button" className="btn btn-primary" onClick={() => setWizard('invite')}>
            <span className="material-icons" style={{ fontSize: 18 }}>person_add</span>+ Add User
          </button>
        </>
      }
    >
      <UsersBody onCount={setCount} wizard={wizard} closeWizard={() => setWizard('closed')} fileRef={fileRef} />
    </AdminLayout>
  )
}
