import { useState } from 'react'
import AdminLayout from './AdminLayout'
import { useAdminData, useDebounced } from './adminShared'

interface AuditEntry {
  timestamp: string | number
  actor: string
  action: string
  target: string
}

function formatTimestamp(v: string | number) {
  return (typeof v === 'number' ? new Date(v * 1000) : new Date(v)).toLocaleString()
}

/** Read-only trail of admin actions. */
export default function AdminAuditLogs() {
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('')
  const debouncedSearch = useDebounced(search.trim())
  const { data, error } = useAdminData<{ entries?: AuditEntry[] }>('listAuditLogs', { search: debouncedSearch, category })
  const entries = data?.entries ?? []

  return (
    <AdminLayout
      docTitle="Audit Logs"
      title="Audit logs"
      subtitle="Every admin action — approvals, role changes, deletions, publishes. Admin-only per the permission matrix."
    >
      <div className="admin-toolbar">
        <input
          type="text"
          className="form-control admin-toolbar-search"
          placeholder="Search by actor, action, or target…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select className="form-control" value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="">All actions</option>
          <option value="user">User management</option>
          <option value="reset">Password resets</option>
          <option value="content">Content</option>
          <option value="ticket">Tickets</option>
          <option value="settings">Settings</option>
        </select>
      </div>

      <div className="table-responsive">
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Actor</th>
              <th>Action</th>
              <th>Target</th>
            </tr>
          </thead>
          <tbody>
            {error ? (
              <tr>
                <td colSpan={4}>Couldn't load audit logs: {error}</td>
              </tr>
            ) : !data ? (
              <tr>
                <td colSpan={4}>Loading audit logs…</td>
              </tr>
            ) : entries.length === 0 ? (
              <tr>
                <td colSpan={4}>No matching audit log entries.</td>
              </tr>
            ) : (
              entries.map((e, i) => (
                <tr key={i}>
                  <td className="admin-cell-sub" style={{ fontFamily: 'var(--font-mono)' }}>
                    {formatTimestamp(e.timestamp)}
                  </td>
                  <td>{e.actor}</td>
                  <td>{e.action}</td>
                  <td>{e.target}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </AdminLayout>
  )
}
