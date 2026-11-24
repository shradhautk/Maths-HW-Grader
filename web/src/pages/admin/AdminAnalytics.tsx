import { useEffect, useState } from 'react'
import AdminLayout, { useAdminAlert } from './AdminLayout'
import { StatGrid, useAdminData, type Analytics } from './adminShared'

function AnalyticsBody({ range }: { range: string }) {
  const showAlert = useAdminAlert()
  const { data, error } = useAdminData<Analytics>('getAnalytics', { range })
  useEffect(() => {
    if (error) showAlert(`Couldn't load analytics: ${error}`, 'danger')
  }, [error, showAlert])

  const weekly = data?.weeklyPapers ?? []
  const schools = data?.topSchools ?? []
  const weeklyMax = Math.max(...weekly.map((w) => w.value), 1)
  const schoolsMax = Math.max(...schools.map((s) => s.value), 1)

  return (
    <>
      <StatGrid analytics={data} />
      <div className="admin-analytics-row">
        <div className="admin-panel">
          <div className="admin-panel-title">Papers graded / week</div>
          <div className="admin-bar-chart">
            {weekly.map((w) => (
              <div
                key={w.label}
                className="bar"
                style={{ height: `${Math.round((w.value / weeklyMax) * 100)}%` }}
                title={`${w.label}: ${w.value}`}
              />
            ))}
          </div>
        </div>
        <div className="admin-panel">
          <div className="admin-panel-title">Top schools</div>
          <div className="admin-ranked-list">
            {schools.map((s) => (
              <div key={s.name}>
                <div className="admin-ranked-row-labels">
                  <span>{s.name}</span>
                  <span className="value">{s.label}</span>
                </div>
                <div className="admin-progress-track">
                  <div className="admin-progress-fill" style={{ width: `${Math.round((s.value / schoolsMax) * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}

export default function AdminAnalytics() {
  const [range, setRange] = useState('30d')
  return (
    <AdminLayout
      docTitle="Analytics"
      title="Platform analytics"
      headerAside={
        <select className="form-control" style={{ width: 'auto' }} value={range} onChange={(e) => setRange(e.target.value)}>
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="90d">Last 90 days</option>
        </select>
      }
    >
      <AnalyticsBody range={range} />
    </AdminLayout>
  )
}
