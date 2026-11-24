import { useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { AnimatePresence, MotionConfig, motion, useMotionValueEvent, useScroll, useSpring } from 'motion/react'
import { EASE, fadeUp, stagger } from './motion'
import { CONTACT, RESEARCH, SITE_URL, SOCIAL } from './site'
import { Magnetic, RouteFade, TextReveal, trackSpotlight } from '../motion/kit'
import '../../styles/landing.css'
import '../../styles/landing-brand.css'
import '../../styles/landing-motion.css'

/** Cards across the public pages that get the pointer spotlight. */
const SPOT = '.fx-spot, .lp-feature, .lp-role, .lp-tech-card, .lp-panel, .lp-info-item, .lp-paper-card, .lp-person, .lp-cards > *'
const onSpot = trackSpotlight(SPOT)

/**
 * Shared shell and building blocks for the marketing pages (Home, Product,
 * Research, About, Careers, Contact, Security, Request a pilot): header, footer,
 * per-page meta tags, reveal-on-scroll, section heads, inner-page hero, FAQ,
 * product shots and CTA band.
 */

const YEAR = new Date().getFullYear()

const NAV = [
  { to: '/features', label: 'Product' },
  { to: '/research', label: 'Research' },
  { to: '/about', label: 'About' },
  { to: '/careers', label: 'Careers' },
  { to: '/contact', label: 'Contact' },
]

const DEFAULT_DESCRIPTION =
  'GradeAnt is AI that grades handwritten math and physics step by step against your rubric, and drafts feedback for the teacher to review.'

/** Sets a <meta> tag's content, creating the tag if the page doesn't have it yet. */
function setMeta(attr: 'name' | 'property', key: string, content: string) {
  let el = document.head.querySelector<HTMLMetaElement>(`meta[${attr}="${key}"]`)
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute(attr, key)
    document.head.appendChild(el)
  }
  el.content = content
}

function setCanonical(href: string) {
  let el = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]')
  if (!el) {
    el = document.createElement('link')
    el.rel = 'canonical'
    document.head.appendChild(el)
  }
  el.href = href
}

/** Page wrapper: reduced-motion aware, skip link, progress bar, header, footer, #hash scrolling. */
export function LandingPage({ title, description = DEFAULT_DESCRIPTION, children }: { title: string; description?: string; children: ReactNode }) {
  const { pathname, hash } = useLocation()

  // Title, description and sharing tags per page (search engines and link previews read these).
  useEffect(() => {
    const fullTitle = `GradeAnt | ${title}`
    document.title = fullTitle
    setMeta('name', 'description', description)
    setMeta('property', 'og:title', fullTitle)
    setMeta('property', 'og:description', description)
    setMeta('property', 'og:url', SITE_URL + pathname)
    setMeta('name', 'twitter:title', fullTitle)
    setMeta('name', 'twitter:description', description)
    setCanonical(SITE_URL + pathname)
  }, [title, description, pathname])

  // Router navigation doesn't scroll to #anchors (e.g. /#how, /features#technology).
  useEffect(() => {
    if (!hash) {
      window.scrollTo(0, 0)
      return
    }
    const t = window.setTimeout(() => document.querySelector(hash)?.scrollIntoView({ behavior: 'smooth' }), 60)
    return () => clearTimeout(t)
  }, [pathname, hash])

  return (
    <MotionConfig reducedMotion="user">
      <div className="lp" onPointerMove={onSpot}>
        <a href="#main" className="lp-skip">
          Skip to content
        </a>
        <ScrollProgress />
        <Header />
        <main id="main">
          <RouteFade routeKey={pathname}>{children}</RouteFade>
        </main>
        <Footer />
      </div>
    </MotionConfig>
  )
}

function ScrollProgress() {
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30 })
  return <motion.div className="lp-progress" style={{ scaleX }} aria-hidden />
}

export function Logo() {
  return (
    <Link to="/" className="lp-logo" aria-label="GradeAnt home">
      <img src="/img/gradeant-logo.webp" alt="" width={36} height={36} />
      <span>
        Grade<b>Ant</b>
      </span>
    </Link>
  )
}

function Header() {
  const { scrollY } = useScroll()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  // Which nav link the pointer is over: a pill slides between them.
  const [hovered, setHovered] = useState<string | null>(null)
  useMotionValueEvent(scrollY, 'change', (y) => setScrolled(y > 12))

  return (
    <header className={`lp-header${scrolled || open ? ' is-scrolled' : ''}`}>
      <div className="lp-container lp-header-inner">
        <Logo />
        <nav className={`lp-nav${open ? ' is-open' : ''}`} aria-label="Main" onPointerLeave={() => setHovered(null)}>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              onClick={() => setOpen(false)}
              onPointerEnter={() => setHovered(n.to)}
              onFocus={() => setHovered(n.to)}
              onBlur={() => setHovered(null)}
              className={({ isActive }) => (isActive ? 'is-active' : '')}
            >
              {hovered === n.to && (
                <motion.span layoutId="lp-nav-pill" className="lp-nav-pill" transition={{ type: 'spring', stiffness: 420, damping: 34 }} aria-hidden />
              )}
              <span className="lp-nav-label">{n.label}</span>
            </NavLink>
          ))}
          <div className="lp-nav-cta">
            {/* Log in is for pilot users; no public sign-up link until sign-up opens. */}
            <Link to="/login" className="lp-btn lp-btn-ghost" onClick={() => setOpen(false)}>
              Log in
            </Link>
            <Magnetic strength={0.2}>
              <Link to="/request-a-pilot" className="lp-btn lp-btn-primary" onClick={() => setOpen(false)}>
                Request a pilot
              </Link>
            </Magnetic>
          </div>
        </nav>
        <button
          type="button"
          className="lp-menu"
          aria-label={open ? 'Close menu' : 'Open menu'}
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
        >
          <span className="material-icons">{open ? 'close' : 'menu'}</span>
        </button>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="lp-footer">
      <div className="lp-container lp-footer-grid">
        <div className="lp-footer-brand">
          <Logo />
          <p>AI that grades handwritten math and physics step by step, with the teacher reviewing every grade.</p>
          <ul className="lp-social" aria-label="GradeAnt on social media">
            {SOCIAL.map((s) => (
              <li key={s.label}>
                <a href={s.href} target="_blank" rel="noreferrer" aria-label={`GradeAnt on ${s.label}`}>
                  <img src={s.icon} alt="" width={18} height={18} />
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Product</h3>
          <ul>
            <li>
              <Link to="/features">Product</Link>
            </li>
            <li>
              <Link to="/#how">How it works</Link>
            </li>
            <li>
              <Link to="/demo">Demo</Link>
            </li>
            <li>
              <Link to="/security">Security and privacy</Link>
            </li>
            <li>
              <Link to="/request-a-pilot">Request a pilot</Link>
            </li>
          </ul>
        </div>
        <div>
          <h3>Company</h3>
          <ul>
            <li>
              <Link to="/research">Research</Link>
            </li>
            <li>
              <Link to="/about">About</Link>
            </li>
            <li>
              <Link to="/careers">Careers</Link>
            </li>
            <li>
              <Link to="/contact">Contact</Link>
            </li>
          </ul>
        </div>
        <div>
          <h3>Legal</h3>
          <ul>
            <li>
              <Link to="/terms">Terms of Service</Link>
            </li>
            <li>
              <Link to="/privacy">Privacy Policy</Link>
            </li>
            <li>
              <Link to="/cookies">Cookies</Link>
            </li>
          </ul>
        </div>
        <div>
          <h3>Contact</h3>
          <ul>
            <li>
              <a href={`mailto:${CONTACT.email}`}>{CONTACT.email}</a>
            </li>
            <li>
              <a href={`tel:${CONTACT.phone}`}>{CONTACT.phone}</a>
            </li>
            <li>{CONTACT.hours}</li>
            <li>{CONTACT.address}</li>
          </ul>
        </div>
      </div>
      <div className="lp-container lp-footer-bottom">
        <p>© {YEAR} GradeAnt. All rights reserved.</p>
        <a href="#main" className="lp-top" aria-label="Back to top">
          <span className="material-icons">arrow_upward</span>
        </a>
      </div>
    </footer>
  )
}


// ---- Building blocks ----

export function Reveal({ children, className, as = 'div' }: { children: ReactNode; className?: string; as?: 'div' | 'section' | 'ul' }) {
  const Tag = motion[as]
  return (
    <Tag className={className} variants={stagger} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.15 }}>
      {children}
    </Tag>
  )
}

export function SectionHead({ eyebrow, title, text, light }: { eyebrow: string; title: ReactNode; text?: string; light?: boolean }) {
  return (
    <Reveal className={`lp-section-head${light ? ' is-light' : ''}`}>
      <motion.span className="lp-eyebrow" variants={fadeUp}>
        {eyebrow}
      </motion.span>
      {typeof title === 'string' ? <TextReveal text={title} inView /> : <motion.h2 variants={fadeUp}>{title}</motion.h2>}
      {text && (
        <motion.p className="lp-lead" variants={fadeUp}>
          {text}
        </motion.p>
      )}
    </Reveal>
  )
}

/** Hero for inner pages: eyebrow, h1, lead, optional actions, soft animated backdrop. */
export function PageHero({ eyebrow, title, lead, children }: { eyebrow: string; title: ReactNode; lead: string; children?: ReactNode }) {
  return (
    <section className="lp-page-hero">
      <div className="lp-hero-bg" aria-hidden>
        <motion.span className="lp-blob lp-blob-a" animate={{ x: [0, 30, 0], y: [0, -20, 0] }} transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.span className="lp-blob lp-blob-b" animate={{ x: [0, -20, 0], y: [0, 30, 0] }} transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }} />
        <span className="lp-grid-bg" />
      </div>
      <motion.div className="lp-container lp-page-hero-inner" variants={stagger} initial="hidden" animate="show">
        <motion.span className="lp-eyebrow" variants={fadeUp}>
          <span className="lp-dot" /> {eyebrow}
        </motion.span>
        {typeof title === 'string' ? <TextReveal as="h1" text={title} delay={0.1} /> : <motion.h1 variants={fadeUp}>{title}</motion.h1>}
        <motion.p className="lp-lead" variants={fadeUp}>
          {lead}
        </motion.p>
        {children && (
          <motion.div className="lp-hero-ctas" variants={fadeUp}>
            {children}
          </motion.div>
        )}
      </motion.div>
    </section>
  )
}

/** Accordion FAQ with FAQPage JSON-LD. */
export function FaqList({ items }: { items: { q: string; a: string; link?: { to: string; label: string } }[] }) {
  const [open, setOpen] = useState<number | null>(0)
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: items.map((f) => ({ '@type': 'Question', name: f.q, acceptedAnswer: { '@type': 'Answer', text: f.a } })),
  }
  return (
    <>
      <Reveal className="lp-faq">
        {items.map((f, n) => {
          const isOpen = open === n
          return (
            <motion.div key={f.q} className={`lp-faq-item${isOpen ? ' is-open' : ''}`} variants={fadeUp}>
              <button type="button" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : n)}>
                <span>{f.q}</span>
                <motion.span className="material-icons" animate={{ rotate: isOpen ? 45 : 0 }} aria-hidden>
                  add
                </motion.span>
              </button>
              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    className="lp-faq-a"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: EASE }}
                  >
                    <p>
                      {f.a}
                      {f.link && (
                        <>
                          {' '}
                          <Link to={f.link.to} className="lp-inline-link">
                            {f.link.label}
                          </Link>
                        </>
                      )}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )
        })}
      </Reveal>
      <script type="application/ld+json">{JSON.stringify(jsonLd)}</script>
    </>
  )
}

/** Gradient call-to-action band used at the bottom of every marketing page. */
export function CtaBand({
  title = 'Try GradeAnt with your class',
  text = 'Pilots are open for math and physics courses. Tell us about your class and we will set you up.',
}: {
  title?: string
  text?: string
}) {
  return (
    <section className="lp-cta-section">
      <motion.div
        className="lp-container lp-cta"
        initial={{ opacity: 0, scale: 0.96 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true, amount: 0.4 }}
        transition={{ duration: 0.7, ease: EASE }}
      >
        <motion.span className="lp-cta-glow" animate={{ rotate: 360 }} transition={{ duration: 30, repeat: Infinity, ease: 'linear' }} aria-hidden />
        <h2>{title}</h2>
        <p>{text}</p>
        <div className="lp-hero-ctas">
          <Magnetic>
            <Link to="/request-a-pilot" className="lp-btn lp-btn-light lp-btn-lg">
              Request a pilot →
            </Link>
          </Magnetic>
          <Link to="/#how" className="lp-btn lp-btn-outline-light lp-btn-lg">
            See how it works
          </Link>
        </div>
      </motion.div>
    </section>
  )
}

/** "Built on research" strip: links the product to the published benchmark work. */
export function ResearchStrip() {
  const { linalgBench } = RESEARCH
  return (
    <section className="lp-research-strip" aria-label="Built on research">
      <Reveal className="lp-container lp-research-strip-inner">
        <motion.span className="lp-eyebrow" variants={fadeUp}>
          Built on research
        </motion.span>
        <motion.p variants={fadeUp}>
          <b>{linalgBench.name}</b>, our benchmark of 10 frontier AI models on step-by-step linear algebra, maps where their reasoning
          breaks down as problems grow. That research shapes how GradeAnt checks each step.
        </motion.p>
        <motion.div variants={fadeUp}>
          <Link to="/research" className="lp-btn lp-btn-secondary">
            Read the research →
          </Link>
        </motion.div>
      </Reveal>
    </section>
  )
}

/**
 * A product screen in a light browser frame. Shows the real screenshot when `src`
 * is set (see MEDIA in site.ts); until then, the drawn illustration in `children`.
 */
export function ProductShot({ src, alt, bar, children }: { src: string | null; alt: string; bar: string; children: ReactNode }) {
  return (
    <figure className="lp-shot">
      <div className="lp-mock-window">
        <div className="lp-mock-bar" aria-hidden>
          <span />
          <span />
          <span />
          <em>{bar}</em>
        </div>
        {src ? (
          <img src={src} alt={alt} loading="lazy" width={1600} height={1000} />
        ) : (
          <div className="lp-shot-draw" role="img" aria-label={`Illustration: ${alt}`}>
            {children}
          </div>
        )}
      </div>
    </figure>
  )
}
