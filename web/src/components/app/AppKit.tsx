import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { fadeUp, stagger } from '../landing/motion'
import CountUp from '../motion/CountUp'
import '../../styles/app.css'

/**
 * Building blocks for the signed-in pages, matching the marketing site:
 * gradient hero, quick-action tiles, animated stat cards, staggered reveal.
 */

export function Stagger({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div className={className} variants={stagger} initial="hidden" animate="show">
      {children}
    </motion.div>
  )
}

/** Greeting banner at the top of a dashboard or tool page. */
export function AppHero({
  eyebrow,
  title,
  text,
  actions,
  aside,
}: {
  eyebrow: string
  title: ReactNode
  text?: ReactNode
  actions?: ReactNode
  aside?: ReactNode
}) {
  return (
    <motion.section className="app-hero" variants={fadeUp} initial="hidden" animate="show">
      <motion.span className="app-hero-orb a" animate={{ x: [0, 24, 0], y: [0, -16, 0] }} transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }} aria-hidden />
      <motion.span className="app-hero-orb b" animate={{ x: [0, -20, 0], y: [0, 18, 0] }} transition={{ duration: 15, repeat: Infinity, ease: 'easeInOut' }} aria-hidden />
      <div className="app-hero-copy">
        <span className="app-eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        {text && <p>{text}</p>}
        {actions && <div className="app-hero-actions">{actions}</div>}
      </div>
      {aside && <div className="app-hero-aside">{aside}</div>}
    </motion.section>
  )
}

export function ActionTile({ to, icon, title, text, tone = 'brand' }: { to: string; icon: string; title: string; text: string; tone?: 'brand' | 'accent' | 'sky' | 'ink' }) {
  return (
    <motion.div variants={fadeUp} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 320, damping: 22 }}>
      <Link to={to} className={`app-tile app-tone-${tone}`}>
        <span className="app-tile-icon material-icons" aria-hidden>
          {icon}
        </span>
        <span className="app-tile-title">{title}</span>
        <span className="app-tile-text">{text}</span>
        <span className="app-tile-go material-icons" aria-hidden>
          arrow_forward
        </span>
      </Link>
    </motion.div>
  )
}

/** Stat card; `value` null shows a dash (data not available). */
export function AppStat({
  icon,
  label,
  value,
  suffix,
  decimals,
  note,
  tone = 'brand',
}: {
  icon: string
  label: string
  value: number | null | undefined
  suffix?: string
  decimals?: number
  note?: ReactNode
  tone?: 'brand' | 'accent' | 'sky' | 'warn' | 'danger'
}) {
  return (
    <motion.div className={`app-stat app-tone-${tone}`} variants={fadeUp}>
      <span className="app-stat-icon material-icons" aria-hidden>
        {icon}
      </span>
      <div>
        <div className="app-stat-label">{label}</div>
        <div className="app-stat-value">{value == null ? '—' : <CountUp to={value} suffix={suffix} decimals={decimals} />}</div>
        {note && <div className="app-stat-note">{note}</div>}
      </div>
    </motion.div>
  )
}

/** Circular score meter (0–100). */
export function ScoreRing({ percent, label, size = 120 }: { percent: number | null; label?: string; size?: number }) {
  const r = 46
  const c = 2 * Math.PI * r
  const p = percent == null ? 0 : Math.max(0, Math.min(100, percent))
  return (
    <div className="app-ring" style={{ width: size, height: size }}>
      <svg viewBox="0 0 108 108" aria-hidden>
        <circle cx="54" cy="54" r={r} className="app-ring-track" />
        <motion.circle
          cx="54"
          cy="54"
          r={r}
          className="app-ring-fill"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - p / 100) }}
          transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
        />
      </svg>
      <div className="app-ring-label">
        <b>{percent == null ? '—' : <CountUp to={Math.round(p)} suffix="%" />}</b>
        {label && <small>{label}</small>}
      </div>
    </div>
  )
}

export function Panel({ title, action, children, className }: { title: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <motion.section className={`app-panel${className ? ` ${className}` : ''}`} variants={fadeUp}>
      <header className="app-panel-head">
        <h2>{title}</h2>
        {action}
      </header>
      {children}
    </motion.section>
  )
}

/** Line chart of 0–100 scores over time (progress, trends). */
export function TrendChart({ points, height = 220 }: { points: { label: string; value: number; title?: string }[]; height?: number }) {
  const W = 640
  const H = height
  const pad = { l: 36, r: 16, t: 14, b: 30 }
  const x = (i: number) => (points.length === 1 ? (W + pad.l - pad.r) / 2 : pad.l + (i / (points.length - 1)) * (W - pad.l - pad.r))
  const y = (v: number) => pad.t + (1 - v / 100) * (H - pad.t - pad.b)
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')
  const area = points.length > 1 ? `${path} L${x(points.length - 1).toFixed(1)},${y(0)} L${x(0).toFixed(1)},${y(0)} Z` : ''
  return (
    <svg className="app-trend" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Score trend: ${points.map((p) => `${p.label} ${Math.round(p.value)}%`).join(', ')}`}>
      <defs>
        <linearGradient id="app-trend-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--orange-400)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="var(--orange-400)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 25, 50, 75, 100].map((g) => (
        <g key={g}>
          <line x1={pad.l} x2={W - pad.r} y1={y(g)} y2={y(g)} className="app-trend-grid" />
          <text x={pad.l - 8} y={y(g) + 4} textAnchor="end" className="app-trend-axis">
            {g}
          </text>
        </g>
      ))}
      {area && <motion.path d={area} fill="url(#app-trend-fill)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8, duration: 0.6 }} />}
      {points.length > 1 && (
        <motion.path d={path} className="app-trend-line" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }} />
      )}
      {points.map((p, i) => (
        <g key={`${p.label}-${i}`}>
          <motion.circle cx={x(i)} cy={y(p.value)} r={5} className="app-trend-dot" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.3 + i * 0.12, type: 'spring', stiffness: 300 }}>
            <title>{p.title ?? `${p.label}: ${Math.round(p.value)}%`}</title>
          </motion.circle>
          {(points.length <= 8 || i % Math.ceil(points.length / 8) === 0) && (
            <text x={x(i)} y={H - 8} textAnchor="middle" className="app-trend-axis">
              {p.label.length > 12 ? `${p.label.slice(0, 11)}…` : p.label}
            </text>
          )}
        </g>
      ))}
    </svg>
  )
}

/** Simple vertical bars (e.g. papers submitted per day). */
export function BarChart({ items, unit = '' }: { items: { label: string; value: number }[]; unit?: string }) {
  const max = Math.max(1, ...items.map((i) => i.value))
  return (
    <div className="app-bars" role="img" aria-label={items.map((i) => `${i.label} ${i.value}${unit}`).join(', ')}>
      {items.map((i, n) => (
        <div key={`${i.label}-${n}`} className="app-bar">
          <span className="app-bar-val">{i.value || ''}</span>
          <motion.span
            className="app-bar-fill"
            initial={{ height: 0 }}
            animate={{ height: `${(i.value / max) * 100}%` }}
            transition={{ duration: 0.7, delay: n * 0.05, ease: [0.22, 1, 0.36, 1] }}
            style={{ minHeight: i.value ? 6 : 3 }}
          />
          <span className="app-bar-label">{i.label}</span>
        </div>
      ))}
    </div>
  )
}
