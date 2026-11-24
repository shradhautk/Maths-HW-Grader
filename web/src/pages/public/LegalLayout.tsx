// Shared layout for the legal pages (Terms, Privacy, Cookies), in the marketing
// site's design: site header/footer, a switcher between the three documents, an
// "On this page" list built from the section headings, and a readable text column.
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { motion } from 'motion/react'
import { LandingPage } from '../../components/landing/LandingKit'
import { EASE } from '../../components/landing/motion'
import '../../styles/legal-landing.css'

const DOCS = [
  { to: '/terms', label: 'Terms of Service' },
  { to: '/privacy', label: 'Privacy Policy' },
  { to: '/cookies', label: 'Cookie Policy' },
]

type Heading = { id: string; title: string }

export default function LegalLayout({
  title,
  effective,
  intro,
  children,
}: {
  title: string
  /** Shown under the title, e.g. "Effective December 5, 2025". */
  effective: string
  intro: ReactNode
  /** Sections: elements with an id and an <h2> (class "lg-section"). */
  children: ReactNode
}) {
  const article = useRef<HTMLDivElement>(null)
  const [headings, setHeadings] = useState<Heading[]>([])
  const [current, setCurrent] = useState<string | null>(null)

  // Build "On this page" from the sections' headings, so it can't drift from the text.
  useEffect(() => {
    const root = article.current
    if (!root) return
    const sections = [...root.querySelectorAll<HTMLElement>('.lg-section[id]')]
    setHeadings(sections.map((s) => ({ id: s.id, title: s.querySelector('h2')?.textContent?.trim() ?? s.id })))

    // Highlight the section being read.
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible[0]) setCurrent(visible[0].target.id)
      },
      { rootMargin: '-96px 0px -60% 0px' },
    )
    sections.forEach((s) => io.observe(s))
    return () => io.disconnect()
  }, [])

  const toc = (
    <ol className="lg-toc-list">
      {headings.map((h) => (
        <li key={h.id}>
          <a href={`#${h.id}`} className={current === h.id ? 'is-current' : ''} aria-current={current === h.id ? 'true' : undefined}>
            {h.title}
          </a>
        </li>
      ))}
    </ol>
  )

  return (
    <LandingPage title={title}>
      <section className="lg-hero">
        <div className="lp-hero-bg" aria-hidden>
          <div className="lp-grid-bg" />
          <div className="lp-blob lp-blob-a" />
          <div className="lp-blob lp-blob-b" />
        </div>
        <div className="lp-container lg-hero-inner">
          <nav className="lg-docs" aria-label="Legal documents">
            {DOCS.map((d) => (
              <NavLink key={d.to} to={d.to} className={({ isActive }) => `lg-doc-link${isActive ? ' is-on' : ''}`} end>
                {({ isActive }) => (
                  <>
                    {isActive && <motion.span layoutId="lg-doc-pill" className="lg-doc-pill" transition={{ duration: 0.35, ease: EASE }} />}
                    <span>{d.label}</span>
                  </>
                )}
              </NavLink>
            ))}
          </nav>
          <h1>{title}</h1>
          <p className="lg-effective">{effective}</p>
        </div>
      </section>

      <div className="lp-container lg-body">
        <aside className="lg-toc" aria-label="On this page">
          <p className="lg-toc-title">On this page</p>
          {toc}
        </aside>

        <div className="lg-main">
          <details className="lg-toc-mobile">
            <summary>On this page</summary>
            {toc}
          </details>
          <div className="lg-intro">{intro}</div>
          <div className="lg-article" ref={article}>
            {children}
          </div>
        </div>
      </div>
    </LandingPage>
  )
}
