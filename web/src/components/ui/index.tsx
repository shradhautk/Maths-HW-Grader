import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { getGradeLetter } from '../../lib/grading'
import { isRouteMissing } from '../../lib/api'
import { statusMeta, type Tone } from '../../lib/status'
import '../../styles/ui.css'

/**
 * Shared building blocks from the wireframes, styled from tokens.css via the
 * global `ga-` classes in styles/ui.css (not page-scoped, so any page can use them).
 */

export function Pill({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`ga-pill ga-tone-${tone}`}>{children}</span>
}

/** A QUESTION_PAPERS / SUBMISSIONS status rendered with its §20 label and tone. */
export function StatusPill({ status }: { status: string | undefined }) {
  const { label, tone } = statusMeta(status)
  return (
    <Pill tone={tone}>
      {tone === 'brand' && <span className="ga-pulse" aria-hidden />}
      {label}
    </Pill>
  )
}

/** Letter grade chip on the --grade-a…f scale. */
export function GradePill({ percent }: { percent: number }) {
  const { letter, cssVar } = getGradeLetter(percent)
  return (
    <span className="ga-grade" style={{ background: `var(${cssVar})` }} title={`${Math.round(percent)}%`}>
      {letter}
    </span>
  )
}

/** The "Question › Review Qs › Method › …" bar every create/submit step carries. */
export function StepProgress({ steps, current }: { steps: string[]; current: number }) {
  return (
    <ol className="ga-steps" aria-label="Progress">
      {steps.map((label, i) => {
        const state = i < current ? 'done' : i === current ? 'current' : 'todo'
        return (
          <li key={label} className={`ga-step is-${state}`} aria-current={state === 'current' ? 'step' : undefined}>
            <span className="ga-step-dot">{state === 'done' ? '✓' : i + 1}</span>
            <span className="ga-step-label">{label}</span>
          </li>
        )
      })}
    </ol>
  )
}

export function StatTile({
  label,
  value,
  note,
  tone = 'neutral',
}: {
  label: string
  value: ReactNode
  note?: ReactNode
  tone?: Tone
}) {
  return (
    <div className={`ga-stat ga-tone-${tone}`}>
      <div className="ga-stat-label">{label}</div>
      <div className="ga-stat-value">{value}</div>
      {note && <div className="ga-stat-note">{note}</div>}
    </div>
  )
}

export function Card({ title, actions, children, className }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={className ? `ga-card ${className}` : 'ga-card'}>
      {(title || actions) && (
        <header className="ga-card-head">
          {title && <h2 className="ga-card-title">{title}</h2>}
          {actions && <div className="ga-card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function PageHeader({ eyebrow, title, subtitle, back, actions }: { eyebrow?: string; title: ReactNode; subtitle?: ReactNode; back?: string; actions?: ReactNode }) {
  return (
    <header className="ga-page-head">
      <div>
        {back && (
          <Link to={back} className="ga-back">
            ← Back
          </Link>
        )}
        {eyebrow && <div className="ga-eyebrow">{eyebrow}</div>}
        <h1 className="ga-h1">{title}</h1>
        {subtitle && <p className="ga-subtitle">{subtitle}</p>}
      </div>
      {actions && <div className="ga-page-actions">{actions}</div>}
    </header>
  )
}

export function Notice({ tone = 'info', children, action }: { tone?: Tone; children: ReactNode; action?: ReactNode }) {
  return (
    <div className={`ga-notice ga-tone-${tone}`}>
      <div className="ga-notice-body">{children}</div>
      {action}
    </div>
  )
}

/** Load error with retry; spells out when the workflow API isn't deployed yet. */
export function LoadError({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  // API Gateway answers an undeployed route with 403 "Missing Authentication
  // Token" (no token sent) or "Invalid key=value pair … in Authorization header".
  const missing = isRouteMissing(error)
  return (
    <Notice
      tone="danger"
      action={
        onRetry && (
          <button type="button" className="ga-btn ga-btn-ghost ga-btn-sm" onClick={onRetry}>
            Try again
          </button>
        )
      }
    >
      <strong>{missing ? 'The assignment workflow service isn’t reachable.' : 'Couldn’t load this.'}</strong>{' '}
      {missing ? 'The /workflow API may not be deployed yet — ask your administrator.' : error.message}
    </Notice>
  )
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="ga-empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}

export function ProgressBar({ percent, tone = 'brand' }: { percent: number; tone?: Tone }) {
  const p = Math.max(0, Math.min(100, Math.round(percent)))
  return (
    <div className={`ga-progress ga-tone-${tone}`} role="progressbar" aria-valuenow={p} aria-valuemin={0} aria-valuemax={100}>
      <span style={{ width: `${p}%` }} />
    </div>
  )
}

/** A–F bars; clicking a bar reports its letter (§15 results, §19 drill-down). */
export function GradeDistribution({
  counts,
  selected,
  onSelect,
}: {
  counts: Record<'A' | 'B' | 'C' | 'D' | 'F', number>
  selected?: string | null
  onSelect?: (letter: string | null) => void
}) {
  const max = Math.max(1, ...Object.values(counts))
  const vars: Record<string, string> = { A: '--grade-a', B: '--grade-b', C: '--grade-c', D: '--grade-d', F: '--grade-f' }
  return (
    <div className="ga-dist">
      {(Object.keys(counts) as (keyof typeof counts)[]).map((letter) => (
        <button
          key={letter}
          type="button"
          className={`ga-dist-col${selected === letter ? ' is-selected' : ''}`}
          onClick={() => onSelect?.(selected === letter ? null : letter)}
          disabled={!onSelect}
        >
          <span className="ga-dist-bar" style={{ height: `${(counts[letter] / max) * 100}%`, background: `var(${vars[letter]})` }} />
          <span className="ga-dist-label">
            {letter} · {counts[letter]}
          </span>
        </button>
      ))}
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="ga-loading">
      <span className="ga-spinner" aria-hidden />
      {label}
    </div>
  )
}
