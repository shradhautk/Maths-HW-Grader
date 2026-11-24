import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import AdminLayout, { useAdminAlert } from './AdminLayout'
import { StatGrid, useAdminData, type Analytics } from './adminShared'

const ATTENTION = [
  { icon: 'lock_reset', color: 'var(--color-warning)', text: 'Pending password reset requests', to: '/admin/reset-requests' },
  { icon: 'article', color: 'var(--color-warning)', text: 'Blog posts pending review', to: '/admin/content' },
  { icon: 'confirmation_number', color: 'var(--color-danger)', text: 'Open support tickets', to: '/admin/tickets' },
]

const QUICK_LINKS = [
  { icon: 'group_add', text: 'Manage users', to: '/admin/users' },
  { icon: 'shield', text: 'Permission matrix', to: '/admin/reset-requests#permission-matrix' },
  { icon: 'insights', text: 'Full analytics', to: '/admin/analytics' },
  { icon: 'tune', text: 'System settings', to: '/admin/settings' },
]

function OverviewBody() {
  const showAlert = useAdminAlert()
  // Same summary numbers the Analytics page shows in full.
  const { data, error } = useAdminData<Analytics>('getAnalytics', { range: '30d' })
  useEffect(() => {
    if (error) showAlert(`Couldn't load platform stats: ${error}`, 'danger')
  }, [error, showAlert])

  return (
    <>
      <StatGrid analytics={data} flatUptimeArrow />
      <div className="admin-analytics-row">
        <div className="admin-panel">
          <div className="admin-panel-title">Needs attention</div>
          <div className="admin-invite-list">
            {ATTENTION.map((a) => (
              <div className="admin-invite-list-item" key={a.to}>
                <span>
                  <span className="material-icons" style={{ fontSize: 16, verticalAlign: 'middle', color: a.color }}>
                    {a.icon}
                  </span>{' '}
                  {a.text}
                </span>
                <Link to={a.to} className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: 12 }}>
                  Review
                </Link>
              </div>
            ))}
          </div>
        </div>
        <div className="admin-panel">
          <div className="admin-panel-title">Quick links</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {QUICK_LINKS.map((q) => (
              <Link key={q.text} to={q.to} className="btn btn-secondary" style={{ justifyContent: 'flex-start' }}>
                <span className="material-icons" style={{ fontSize: 18 }}>
                  {q.icon}
                </span>
                {q.text}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}

export default function AdminDashboard() {
  return (
    <AdminLayout
      docTitle="Overview"
      title="Overview"
      subtitle="Platform health at a glance"
      actions={
        <Link to="/admin/users" className="btn btn-primary">
          <span className="material-icons" style={{ fontSize: 18 }}>person_add</span>Add People
        </Link>
      }
    >
      <OverviewBody />
    </AdminLayout>
  )
}
