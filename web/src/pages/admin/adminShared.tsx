import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { adminApiRequest } from '../../lib/api'

/**
 * Loads an admin Lambda read operation and reloads whenever `payload`
 * changes. `reload(true)` bypasses the cache (after a mutation).
 */
export function useAdminData<T>(operation: string, payload: Record<string, unknown> = {}) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const key = JSON.stringify(payload)
  const latest = useRef(0)

  const reload = useCallback(
    async (forceRefresh = false) => {
      const id = ++latest.current
      try {
        const result = await adminApiRequest<T>(operation, JSON.parse(key), { forceRefresh })
        // Ignore responses to superseded requests (e.g. fast typing in search).
        if (id !== latest.current) return
        setData(result)
        setError(null)
      } catch (err) {
        if (id === latest.current) setError((err as Error).message)
      }
    },
    [operation, key],
  )

  useEffect(() => {
    void reload()
  }, [reload])

  return { data, error, reload }
}

/** `value` trailing `delayMs` behind its input (search boxes). */
export function useDebounced<T>(value: T, delayMs = 300): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(t)
  }, [value, delayMs])
  return debounced
}

export interface Stat {
  value: string | number
  trend?: string
  direction?: 'up' | 'down' | 'flat'
}

export interface Analytics {
  activeUsers?: Stat
  papersGraded?: Stat
  schools?: Stat
  uptime?: Stat
  weeklyPapers?: { label: string; value: number }[]
  topSchools?: { name: string; label: string; value: number }[]
}

const STAT_TILES = [
  { key: 'activeUsers', label: 'Active Users', variant: '' },
  { key: 'papersGraded', label: 'Papers Graded', variant: ' accent' },
  { key: 'schools', label: 'Schools', variant: ' info' },
  { key: 'uptime', label: 'Avg Uptime', variant: ' warning', flat: true },
] as const

/** The four platform stat tiles (Overview and Analytics). */
export function StatGrid({ analytics, flatUptimeArrow = false }: { analytics: Analytics | null; flatUptimeArrow?: boolean }) {
  return (
    <div className="admin-stat-grid">
      {STAT_TILES.map((tile) => {
        const stat = analytics?.[tile.key]
        const flat = (tile.key === 'uptime' && flatUptimeArrow) || stat?.direction === 'flat'
        const arrow = flat ? '→' : stat?.direction === 'down' ? '↓' : '↑'
        return (
          <div className={`admin-stat-card${tile.variant}`} key={tile.key}>
            <span className="admin-stat-label">{tile.label}</span>
            <div className="admin-stat-value">{stat?.value ?? '—'}</div>
            <div className={`admin-stat-trend${'flat' in tile ? ' flat' : ''}`}>
              {stat?.trend && (
                <>
                  <span>{arrow}</span>
                  <span>{stat.trend}</span>
                </>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function FilterPills<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: ReactNode }[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="admin-filter-pills">
      {options.map((o) => (
        <span
          key={o.value}
          className={`admin-filter-pill${o.value === value ? ' active' : ''}`}
          role="button"
          tabIndex={0}
          onClick={() => onChange(o.value)}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onChange(o.value)}
        >
          {o.label}
        </span>
      ))}
    </div>
  )
}

/** Admin modal (overlay click and × close it). */
export function AdminModal({
  open,
  title,
  header,
  footer,
  footerStyle,
  onClose,
  style,
  children,
}: {
  open: boolean
  title: string
  /** Extra header content between title and close button (wizard stepper). */
  header?: ReactNode
  footer?: ReactNode
  footerStyle?: CSSProperties
  onClose: () => void
  style?: CSSProperties
  children: ReactNode
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])

  return (
    <div className={`admin-modal-overlay${open ? ' show' : ''}`} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="admin-modal" style={style} role="dialog" aria-modal="true" aria-label={title}>
        <div className="admin-modal-header">
          <span className="admin-modal-title">{title}</span>
          {header}
          <button type="button" className="admin-modal-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>
        <div className="admin-modal-body">{children}</div>
        {footer && (
          <div className="admin-modal-footer" style={footerStyle}>
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}

/** Inline confirmation (replaces window.confirm, which the old pages used). */
export function ConfirmButton({
  label,
  confirmLabel,
  onConfirm,
  className = 'btn btn-secondary',
  style,
}: {
  label: string
  confirmLabel: string
  onConfirm: () => void
  className?: string
  style?: CSSProperties
}) {
  const [asking, setAsking] = useState(false)
  if (!asking) {
    return (
      <button type="button" className={className} style={style} onClick={() => setAsking(true)}>
        {label}
      </button>
    )
  }
  return (
    <>
      <button
        type="button"
        className={className}
        style={{ ...style, color: 'var(--color-danger)' }}
        onClick={() => {
          setAsking(false)
          onConfirm()
        }}
      >
        {confirmLabel}
      </button>
      <button type="button" className="btn btn-secondary" style={style} onClick={() => setAsking(false)}>
        Cancel
      </button>
    </>
  )
}

export function timeAgo(isoOrEpoch: string | number): string {
  const then = typeof isoOrEpoch === 'number' ? isoOrEpoch * 1000 : new Date(isoOrEpoch).getTime()
  const minutes = Math.max(1, Math.round((Date.now() - then) / 60000))
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}
