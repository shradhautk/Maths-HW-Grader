import { motion, useReducedMotion } from 'motion/react'
import CountUp from '../motion/CountUp'
import { EASE } from './motion'

// Drawn stand-ins for product screenshots (hero, how-it-works steps, before/after).
// Each one draws a real GradeAnt screen with a made-up sample answer: no student
// names, no invented usage numbers. Replace with real screenshots via MEDIA in site.ts.

/** One worked step of a handwritten answer as GradeAnt marks it. */
const PHYSICS_STEPS = [
  { hand: 'v² = u² − 2gh', mark: '2/2', ok: true, note: 'Right equation for maximum height' },
  { hand: '0 = 12² − 2(9.8)h', mark: '1/1', ok: true, note: 'Values substituted correctly' },
  { hand: 'h = 144 / 9.8 = 14.7 m', mark: '0/2', ok: false, note: 'The factor of 2 in 2g was dropped' },
]

/**
 * Hero: scanned handwritten answer (left), GradeAnt's step-by-step marks (right).
 * `animated` plays the grading as a short sequence: the handwriting is written
 * line by line, each step's mark lands beside it, then the draft feedback.
 */
export function HeroSplit({ animated: wanted = false, delay = 0.5 }: { animated?: boolean; delay?: number }) {
  const reduce = useReducedMotion()
  const animated = wanted && !reduce
  const at = (i: number) => delay + i * 0.45
  const write = (i: number) =>
    animated
      ? { initial: { clipPath: 'inset(0 100% 0 0)' }, animate: { clipPath: 'inset(0 0% 0 0)' }, transition: { duration: 0.7, ease: EASE, delay: at(i) } }
      : {}
  const land = (i: number) =>
    animated ? { initial: { opacity: 0, x: 14 }, animate: { opacity: 1, x: 0 }, transition: { duration: 0.5, ease: EASE, delay: at(i) + 0.35 } } : {}
  const done = at(PHYSICS_STEPS.length) + 0.2

  return (
    <div className="lp-split-view">
      <div className="lp-scan-paper">
        <small>Q2 · A ball is thrown straight up at 12 m/s. How high does it go?</small>
        {PHYSICS_STEPS.map((s, i) => (
          <motion.p key={s.hand} className={`lp-hand${!s.ok ? ' is-struck' : ''}`} {...write(i)}>
            {s.hand}
          </motion.p>
        ))}
      </div>
      <div className="lp-marks">
        <div className="lp-marks-head">
          <b>Step-by-step marks</b>
          <span>{animated ? <CountUp to={3} suffix=" / 5" /> : '3 / 5'}</span>
        </div>
        <ol>
          {PHYSICS_STEPS.map((s, i) => (
            <motion.li key={s.hand} className={s.ok ? 'is-ok' : 'is-off'} {...land(i)}>
              <span className="lp-step-tag">Step {i + 1}</span>
              <span>{s.note}</span>
              <strong>{s.mark}</strong>
            </motion.li>
          ))}
        </ol>
        <motion.div
          className="lp-feedback"
          {...(animated ? { initial: { opacity: 0, y: 10 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.6, ease: EASE, delay: done } } : {})}
        >
          <b>Draft feedback · for teacher review</b>
          <p>Good setup. In the last step divide by 2g, not g: h = 144 / 19.6 ≈ 7.3 m.</p>
        </motion.div>
      </div>
    </div>
  )
}

export function UploadScreen() {
  return (
    <div className="lp-draw">
      <div className="lp-drop">
        <span className="material-icons" aria-hidden>
          upload_file
        </span>
        <b>Drop scanned answer sheets here</b>
        <small>PDF, JPG or PNG · one file per student or one combined scan</small>
      </div>
      <ul className="lp-files">
        <li>
          <span className="material-icons" aria-hidden>
            description
          </span>
          section-a-scans.pdf <em>Uploaded</em>
        </li>
        <li>
          <span className="material-icons" aria-hidden>
            description
          </span>
          section-b-scans.pdf <em>Uploading…</em>
        </li>
      </ul>
    </div>
  )
}

export function RubricScreen() {
  const rows = [
    ['Correct equation chosen', '2'],
    ['Values substituted with units', '1'],
    ['Algebra and final answer', '2'],
  ]
  return (
    <div className="lp-draw">
      <div className="lp-draw-title">
        Rubric · Q2 <em>5 marks</em>
      </div>
      <ul className="lp-rubric">
        {rows.map(([t, m]) => (
          <li key={t}>
            <span>{t}</span>
            <b>{m}</b>
          </li>
        ))}
      </ul>
      <div className="lp-toggle-row">
        <span>Partial credit for method</span>
        <i className="lp-toggle is-on" />
      </div>
    </div>
  )
}

export function GradingScreen() {
  return (
    <div className="lp-draw">
      <div className="lp-draw-title">
        Grading in progress <em>Q1–Q4</em>
      </div>
      <div className="lp-bar">
        <i style={{ width: '64%' }} />
      </div>
      <ul className="lp-rubric">
        <li>
          <span>Reading handwriting</span>
          <b className="is-done">Done</b>
        </li>
        <li>
          <span>Matching answers to questions</span>
          <b className="is-done">Done</b>
        </li>
        <li>
          <span>Checking each step against the rubric</span>
          <b>Running</b>
        </li>
        <li>
          <span>Drafting feedback</span>
          <b className="is-wait">Waiting</b>
        </li>
      </ul>
    </div>
  )
}

export function ReviewScreen() {
  return (
    <div className="lp-draw">
      <div className="lp-draw-title">
        Teacher review <em>Flagged for you</em>
      </div>
      <ul className="lp-rubric">
        <li>
          <span>Q2 · Step 3 · low confidence reading “9.8”</span>
          <b className="is-wait">Check</b>
        </li>
        <li>
          <span>Q4 · unusual method, marks need a look</span>
          <b className="is-wait">Check</b>
        </li>
      </ul>
      <div className="lp-draw-actions">
        <span className="lp-btn lp-btn-secondary">Edit marks</span>
        <span className="lp-btn lp-btn-primary">Approve and publish</span>
      </div>
    </div>
  )
}

/** Before and after: a messy derivation next to the feedback it produced. */
export function BeforeAfterScreen() {
  return (
    <div className="lp-before-after">
      <div className="lp-scan-paper">
        <small>Q3 · Find ∫ x eˣ dx</small>
        <p className="lp-hand">u = x, dv = eˣ dx</p>
        <p className="lp-hand">= x eˣ − ∫ eˣ dx</p>
        <p className="lp-hand is-struck">= x eˣ + eˣ</p>
      </div>
      <div className="lp-marks">
        <div className="lp-marks-head">
          <b>GradeAnt feedback</b>
          <span>2 / 4</span>
        </div>
        <ol>
          <li className="is-ok">
            <span className="lp-step-tag">Step 1</span>
            <span>Good choice of u and dv</span>
            <strong>1/1</strong>
          </li>
          <li className="is-ok">
            <span className="lp-step-tag">Step 2</span>
            <span>Integration by parts applied correctly</span>
            <strong>1/1</strong>
          </li>
          <li className="is-off">
            <span className="lp-step-tag">Step 3</span>
            <span>Sign error, and the constant C is missing</span>
            <strong>0/2</strong>
          </li>
        </ol>
        <div className="lp-feedback">
          <b>Draft feedback · for teacher review</b>
          <p>∫ eˣ dx = eˣ, so subtracting gives x eˣ − eˣ + C.</p>
        </div>
      </div>
    </div>
  )
}
