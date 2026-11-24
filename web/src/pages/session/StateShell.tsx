// Focused page for session states (log out, expired, no access, new sign-in) and
// the 404 page: Home's logo and background around one card. No marketing nav —
// these are moments between screens, so the only choices are the card's buttons.
import { useEffect, type ReactNode } from 'react'
import { Logo } from '../../components/landing/LandingKit'
import '../../styles/state-landing.css'

type Tone = 'brand' | 'warn' | 'danger' | 'info'

export default function StateShell({
  title,
  icon,
  tone,
  heading,
  children,
  actions,
}: {
  /** Browser tab title. */
  title: string
  /** Material icon name. */
  icon: string
  tone: Tone
  heading: string
  children: ReactNode
  actions: ReactNode
}) {
  useEffect(() => {
    document.title = `GradeAnt | ${title}`
  }, [title])

  return (
    <div className="lp st">
      <header className="lp-container st-top">
        <Logo />
      </header>
      <main className="st-main" id="main">
        <div className="lp-hero-bg" aria-hidden>
          <div className="lp-grid-bg" />
          <div className="lp-blob lp-blob-a" />
          <div className="lp-blob lp-blob-b" />
        </div>
        <section className="st-card" aria-labelledby="st-heading">
          <span className={`st-icon st-icon-${tone}`} aria-hidden>
            <span className="material-icons">{icon}</span>
          </span>
          <h1 id="st-heading">{heading}</h1>
          <div className="st-text">{children}</div>
          <div className="st-actions">{actions}</div>
        </section>
      </main>
    </div>
  )
}

/** "Riya Sharma, Student" card for the signed-in person. */
export function WhoChip({ name, role }: { name: string; role: string }) {
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase())
    .join('')
  return (
    <div className="st-who">
      <span className="st-avatar" aria-hidden>
        {initials || '?'}
      </span>
      <span>
        <span className="st-who-label">Signed in as</span>
        <b>{name}</b>
        {role && <small>, {role}</small>}
      </span>
    </div>
  )
}
