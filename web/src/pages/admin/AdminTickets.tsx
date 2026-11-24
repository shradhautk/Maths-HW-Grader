import { useEffect, useState } from 'react'
import { adminApiRequest } from '../../lib/api'
import AdminLayout, { capitalize, useAdminAlert } from './AdminLayout'
import { FilterPills, useAdminData } from './adminShared'

type Filter = 'open' | 'mine' | 'resolved'

interface Ticket {
  id: string
  priority: 'high' | 'med' | 'low' | string
  subject: string
  ageLabel: string
  requester: string
  assignee?: string
}

const PRIORITY_CLASS: Record<string, string> = { high: 'priority-high', low: 'priority-low' }
const SMALL_BTN = { padding: '5px 10px', fontSize: 12 }

function TicketsTable({ filter, onCount }: { filter: Filter; onCount: (n: number | null) => void }) {
  const showAlert = useAdminAlert()
  const { data, error, reload } = useAdminData<{ tickets?: Ticket[]; openCount?: number }>('listTickets', { filter })
  const openCount = typeof data?.openCount === 'number' ? data.openCount : null
  useEffect(() => onCount(openCount), [openCount, onCount])

  async function act(operation: 'assignTicket' | 'resolveTicket', ticketId: string) {
    try {
      await adminApiRequest(operation, { ticketId })
      if (operation === 'resolveTicket') showAlert('Ticket resolved.', 'success')
      void reload(true)
    } catch (err) {
      showAlert(`Couldn't ${operation === 'assignTicket' ? 'assign' : 'resolve'} ticket: ${(err as Error).message}`, 'danger')
    }
  }

  const tickets = data?.tickets ?? []
  return (
    <div className="table-responsive">
      <table className="data-table">
        <thead>
          <tr>
            <th>Priority</th>
            <th>Subject</th>
            <th>Requester</th>
            <th>Assignee</th>
          </tr>
        </thead>
        <tbody>
          {error ? (
            <tr>
              <td colSpan={4}>Couldn't load tickets: {error}</td>
            </tr>
          ) : !data ? (
            <tr>
              <td colSpan={4}>Loading tickets…</td>
            </tr>
          ) : tickets.length === 0 ? (
            <tr>
              <td colSpan={4}>No tickets in this view.</td>
            </tr>
          ) : (
            tickets.map((t) => (
              <tr key={t.id}>
                <td>
                  <span className={`status-pill ${PRIORITY_CLASS[t.priority] ?? 'priority-med'}`}>{capitalize(t.priority)}</span>
                </td>
                <td>
                  <div className="admin-cell-name">{t.subject}</div>
                  <div className="admin-cell-sub" style={{ fontFamily: 'var(--font-mono)' }}>
                    #{t.id} · {t.ageLabel}
                  </div>
                </td>
                <td>{t.requester}</td>
                <td>
                  {t.assignee ? (
                    <span>{t.assignee}</span>
                  ) : (
                    <button type="button" className="btn btn-secondary" style={SMALL_BTN} onClick={() => act('assignTicket', t.id)}>
                      Assign to me
                    </button>
                  )}
                  {filter !== 'resolved' && (
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ ...SMALL_BTN, marginLeft: 6 }}
                      onClick={() => act('resolveTicket', t.id)}
                    >
                      Resolve
                    </button>
                  )}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

export default function AdminTickets() {
  const [filter, setFilter] = useState<Filter>('open')
  const [openCount, setOpenCount] = useState<number | null>(null)
  return (
    <AdminLayout
      docTitle="Support Tickets"
      title={
        <>
          Support tickets <span className="count">{openCount !== null ? `· ${openCount} open` : ''}</span>
        </>
      }
    >
      <FilterPills<Filter>
        value={filter}
        onChange={setFilter}
        options={[
          { value: 'open', label: 'Open' },
          { value: 'mine', label: 'Mine' },
          { value: 'resolved', label: 'Resolved' },
        ]}
      />
      <TicketsTable filter={filter} onCount={setOpenCount} />
    </AdminLayout>
  )
}
