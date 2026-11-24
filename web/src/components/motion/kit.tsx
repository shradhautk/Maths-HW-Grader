import { Fragment, useRef, type CSSProperties, type PointerEvent, type ReactNode } from 'react'
import { motion, useMotionValue, useReducedMotion, useScroll, useSpring, useTransform, type Variants } from 'motion/react'
import { EASE } from '../landing/motion'
import '../../styles/motion.css'

/**
 * Motion kit shared by the public site and the app. Every piece is decorative:
 * content is readable with JavaScript animations off, and each one either
 * respects prefers-reduced-motion itself or runs inside <MotionConfig
 * reducedMotion="user"> (LandingPage, AppShell), which drops transforms.
 */

const finePointer = () => typeof window !== 'undefined' && window.matchMedia('(hover: hover) and (pointer: fine)').matches

// ---- Text reveal ----

const word: Variants = {
  hidden: { y: '110%' },
  show: { y: '0%', transition: { duration: 0.8, ease: EASE } },
}

/**
 * Heading whose words slide up out of a mask one after another. `highlight`
 * is a trailing phrase of `text` drawn in the brand gradient.
 */
export function TextReveal({
  text,
  highlight,
  as = 'h2',
  className,
  delay = 0,
  inView = false,
}: {
  text: string
  highlight?: string
  as?: 'h1' | 'h2' | 'h3' | 'p'
  className?: string
  delay?: number
  /** Start when scrolled into view instead of on mount. */
  inView?: boolean
}) {
  const Tag = motion[as]
  const cut = highlight && text.endsWith(highlight) ? text.length - highlight.length : text.length
  const plain = text.slice(0, cut).split(/\s+/).filter(Boolean)
  const accent = text.slice(cut).split(/\s+/).filter(Boolean)
  const trigger = inView
    ? { initial: 'hidden', whileInView: 'show', viewport: { once: true, amount: 0.6 } }
    : { initial: 'hidden', animate: 'show' }

  return (
    <Tag
      className={className ? `fx-text ${className}` : 'fx-text'}
      aria-label={text}
      variants={{ hidden: {}, show: { transition: { staggerChildren: 0.055, delayChildren: delay } } }}
      {...trigger}
    >
      {[...plain.map((w) => ({ w, hi: false })), ...accent.map((w) => ({ w, hi: true }))].map(({ w, hi }, i, all) => (
        // The space sits between the masked words: inside an inline-block it would collapse.
        <Fragment key={i}>
          <span className="fx-word" aria-hidden>
            <motion.span className={hi ? 'fx-word-in fx-gradient-text' : 'fx-word-in'} variants={word}>
              {w}
            </motion.span>
          </span>
          {i < all.length - 1 && ' '}
        </Fragment>
      ))}
    </Tag>
  )
}

// ---- Magnetic ----

/** Wrapper that drifts toward the pointer while hovered (buttons, icons). Mouse only. */
export function Magnetic({ children, strength = 0.28, className }: { children: ReactNode; strength?: number; className?: string }) {
  const reduce = useReducedMotion()
  const x = useSpring(useMotionValue(0), { stiffness: 260, damping: 18, mass: 0.4 })
  const y = useSpring(useMotionValue(0), { stiffness: 260, damping: 18, mass: 0.4 })

  function move(e: PointerEvent<HTMLSpanElement>) {
    if (reduce || e.pointerType !== 'mouse') return
    const r = e.currentTarget.getBoundingClientRect()
    x.set((e.clientX - (r.left + r.width / 2)) * strength)
    y.set((e.clientY - (r.top + r.height / 2)) * strength)
  }
  function reset() {
    x.set(0)
    y.set(0)
  }

  return (
    <motion.span className={className ? `fx-magnetic ${className}` : 'fx-magnetic'} style={{ x, y }} onPointerMove={move} onPointerLeave={reset} whileTap={{ scale: 0.97 }}>
      {children}
    </motion.span>
  )
}

// ---- Spotlight ----

/**
 * Pointer handler for a container: any descendant with `.fx-spot` (or matching
 * `selector`) gets --mx/--my set so its CSS spotlight follows the cursor.
 * One listener per page, no React state, so it costs nothing per card.
 */
export function trackSpotlight(selector = '.fx-spot') {
  return (e: PointerEvent<HTMLElement>) => {
    if (e.pointerType !== 'mouse') return
    const el = (e.target as Element).closest<HTMLElement>(selector)
    if (!el) return
    const r = el.getBoundingClientRect()
    el.style.setProperty('--mx', `${e.clientX - r.left}px`)
    el.style.setProperty('--my', `${e.clientY - r.top}px`)
  }
}

// ---- Tilt ----

/** 3D tilt toward the pointer (product shots). Mouse only; flat with reduced motion. */
export function Tilt({ children, max = 6, className }: { children: ReactNode; max?: number; className?: string }) {
  const reduce = useReducedMotion()
  const px = useMotionValue(0.5)
  const py = useMotionValue(0.5)
  const rotateY = useSpring(useTransform(px, [0, 1], [-max, max]), { stiffness: 150, damping: 20 })
  const rotateX = useSpring(useTransform(py, [0, 1], [max, -max]), { stiffness: 150, damping: 20 })

  function move(e: PointerEvent<HTMLDivElement>) {
    if (reduce || !finePointer()) return
    const r = e.currentTarget.getBoundingClientRect()
    px.set((e.clientX - r.left) / r.width)
    py.set((e.clientY - r.top) / r.height)
  }
  function reset() {
    px.set(0.5)
    py.set(0.5)
  }

  return (
    <div className={className ? `fx-tilt ${className}` : 'fx-tilt'} onPointerMove={move} onPointerLeave={reset}>
      <motion.div className="fx-tilt-inner" style={reduce ? undefined : { rotateX, rotateY }}>
        {children}
      </motion.div>
    </div>
  )
}

// ---- Parallax ----

/** Moves its children by up to ±`offset`px as the element crosses the viewport. */
export function Parallax({ children, offset = 40, className }: { children: ReactNode; offset?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const reduce = useReducedMotion()
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] })
  const y = useTransform(scrollYProgress, [0, 1], [offset, -offset])
  return (
    <motion.div ref={ref} className={className} style={reduce ? undefined : { y }}>
      {children}
    </motion.div>
  )
}

// ---- Marquee ----

/** Infinite horizontal ticker; pauses on hover/focus, static (wrapped) with reduced motion. */
export function Marquee({ items, label, speed = 40 }: { items: ReactNode[]; label: string; speed?: number }) {
  return (
    <div className="fx-marquee" role="region" aria-label={label} style={{ '--fx-marquee-dur': `${speed}s` } as CSSProperties}>
      <ul className="fx-marquee-track">
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ul>
      <ul className="fx-marquee-track" aria-hidden>
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ul>
    </div>
  )
}

// ---- Route transition ----

/** Fades and lifts a page in when the route changes (enter only, so navigation never waits). */
export function RouteFade({ routeKey, children, className }: { routeKey: string; children: ReactNode; className?: string }) {
  return (
    <motion.div
      key={routeKey}
      className={className}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: EASE }}
    >
      {children}
    </motion.div>
  )
}

// ---- Skeleton ----

/** Shimmering placeholder block shown while data loads (keeps layout from shifting). */
export function Skeleton({ width = '100%', height = 14, radius = 8, className }: { width?: number | string; height?: number | string; radius?: number; className?: string }) {
  return <span className={className ? `fx-skeleton ${className}` : 'fx-skeleton'} style={{ width, height, borderRadius: radius }} aria-hidden />
}

/** A few skeleton lines with a screen-reader label. */
export function SkeletonText({ lines = 3, label = 'Loading' }: { lines?: number; label?: string }) {
  return (
    <div className="fx-skeleton-stack" role="status" aria-label={label}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} width={i === lines - 1 ? '62%' : '100%'} />
      ))}
    </div>
  )
}
