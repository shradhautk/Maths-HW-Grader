// Shared pieces for the login and signup pages, built on the marketing site's
// design (components/landing/LandingKit + styles/landing.css) so they read as
// part of the same site as the Home page.
import { useState, type ChangeEvent, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { LandingPage } from '../../components/landing/LandingKit'
import { EASE } from '../../components/landing/motion'
import type { RoleName } from '../../config'
import { PASSWORD_RULES } from './passwordRules'
import '../../styles/auth-landing.css'

/** Page shell: site header/footer, Home's hero background, form card + preview. */
export function AuthShell({ title, aside, children }: { title: string; aside: ReactNode; children: ReactNode }) {
  return (
    <LandingPage title={title}>
      <section className="au">
        <div className="lp-hero-bg" aria-hidden>
          <div className="lp-grid-bg" />
          <div className="lp-blob lp-blob-a" />
          <div className="lp-blob lp-blob-b" />
        </div>
        <div className="lp-container au-grid">
          <div className="au-card">{children}</div>
          <div className="au-aside">{aside}</div>
        </div>
      </section>
    </LandingPage>
  )
}

/** Segmented role switch: real radio inputs, with a pill that slides to the chosen role. */
export function RoleSwitch({
  name,
  legend,
  roles,
  value,
  onChange,
}: {
  name: string
  legend: string
  roles: { name: RoleName; label: string }[]
  value: RoleName
  onChange: (role: RoleName) => void
}) {
  return (
    <fieldset className="au-roles">
      <legend>{legend}</legend>
      <div className="au-roles-track" style={{ gridTemplateColumns: `repeat(${roles.length}, minmax(0, 1fr))` }}>
        {roles.map((r) => (
          <label key={r.name} className={`au-role${value === r.name ? ' is-on' : ''}`}>
            {value === r.name && (
              <motion.span layoutId={`${name}-pill`} className="au-role-pill" transition={{ duration: 0.35, ease: EASE }} />
            )}
            <input type="radio" name={name} value={r.name} checked={value === r.name} onChange={() => onChange(r.name)} />
            <span>{r.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}

/** Password field with a show/hide button. */
export function PasswordInput({
  id,
  value,
  onChange,
  autoComplete,
  describedBy,
}: {
  id: string
  value: string
  onChange: (e: ChangeEvent<HTMLInputElement>) => void
  autoComplete: 'current-password' | 'new-password'
  describedBy?: string
}) {
  const [shown, setShown] = useState(false)
  return (
    <div className="au-pw">
      <input
        id={id}
        name={id}
        className="lp-input"
        type={shown ? 'text' : 'password'}
        autoComplete={autoComplete}
        required
        value={value}
        onChange={onChange}
        aria-describedby={describedBy}
      />
      <button
        type="button"
        className="au-pw-toggle"
        aria-label={shown ? 'Hide password' : 'Show password'}
        aria-pressed={shown}
        onClick={() => setShown((s) => !s)}
      >
        <span className="material-icons" aria-hidden>
          {shown ? 'visibility_off' : 'visibility'}
        </span>
      </button>
    </div>
  )
}

/** Live checklist under a new-password field; give the field aria-describedby={id}. */
export function PasswordRules({ id, value }: { id: string; value: string }) {
  return (
    <ul className="au-rules" id={id} aria-label="Password requirements">
      {PASSWORD_RULES.map((r) => {
        const met = r.test(value)
        return (
          <li key={r.id} className={met ? 'is-met' : ''}>
            <span className="material-icons" aria-hidden>
              {met ? 'check_circle' : 'radio_button_unchecked'}
            </span>
            {r.label}
            <span className="au-sr-only">{met ? ' (done)' : ''}</span>
          </li>
        )
      })}
    </ul>
  )
}

/** Form message. Errors stay until the person changes something, so they can't be missed. */
export function Notice({ tone, children }: { tone: 'error' | 'success' | 'info'; children: ReactNode }) {
  const icon = tone === 'error' ? 'error' : tone === 'success' ? 'check_circle' : 'info'
  return (
    <div className={`au-note au-note-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <span className="material-icons" aria-hidden>
        {icon}
      </span>
      <span>{children}</span>
    </div>
  )
}

// ---- Previews (right-hand column; decorative, with a readable caption) ----

function Ring({ value, of, grade }: { value: number; of: number; grade: string }) {
  const R = 34
  const C = 2 * Math.PI * R
  return (
    <div className="lp-ring">
      <svg viewBox="0 0 80 80" width="80" height="80" aria-hidden>
        <circle cx="40" cy="40" r={R} className="lp-ring-track" />
        <motion.circle
          cx="40"
          cy="40"
          r={R}
          className="lp-ring-fill"
          strokeDasharray={C}
          initial={{ strokeDashoffset: C }}
          animate={{ strokeDashoffset: C * (1 - value / of) }}
          transition={{ duration: 1.1, delay: 0.15, ease: EASE }}
        />
      </svg>
      <div className="lp-ring-label">
        <b>{value}</b>
        <small>
          / {of} · {grade}
        </small>
      </div>
    </div>
  )
}

function Chip({ tone, tag, text, score }: { tone: 'ok' | 'warn'; tag: string; text: string; score?: string }) {
  return (
    <div className={`lp-chip lp-chip-${tone}`}>
      <b>{tag}</b>
      <span>{text}</span>
      <strong>{score ?? ''}</strong>
    </div>
  )
}

const LOGIN_PREVIEW: Record<RoleName, { title: string; text: string; window: string; body: ReactNode }> = {
  student: {
    title: 'Your marks and feedback, question by question',
    text: 'See how each answer was graded, read the feedback, and ask for a re-check if something looks wrong.',
    window: 'My grades · Thermodynamics HW',
    body: (
      <>
        <Ring value={88} of={100} grade="A" />
        <Chip tone="ok" tag="Q3" text="First law stated correctly" score="10/10" />
        <Chip tone="warn" tag="Q5" text="Entropy sign reversed" score="6/10" />
        <Chip tone="ok" tag="Q7" text="Clear diagram and working" score="8/8" />
      </>
    ),
  },
  teacher: {
    title: 'Your grading run, where you left it',
    text: 'Papers are scored against your rubric. Review only the answers GradeFlow flagged, then publish.',
    window: 'PHY-204 · Grading run',
    body: (
      <>
        <div className="au-stat">
          <b>31 of 31</b>
          <small>papers graded</small>
          <div className="au-bar">
            <motion.i initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 1, delay: 0.15, ease: EASE }} />
          </div>
        </div>
        <Chip tone="warn" tag="Q5" text="3 answers flagged for your review" />
        <Chip tone="ok" tag="Rubric" text="Approved, 4 questions" />
        <Chip tone="ok" tag="Ready" text="28 papers ready to publish" />
      </>
    ),
  },
  admin: {
    title: "Your school's accounts and settings",
    text: 'Invite teachers and students, handle password resets, and check the audit log.',
    window: 'Admin console',
    body: (
      <>
        <div className="au-stat">
          <b>2</b>
          <small>password reset requests waiting</small>
        </div>
        <Chip tone="warn" tag="Reset" text="Sam Okafor asked for a new password" />
        <Chip tone="ok" tag="Invite" text="Dr. Priya Nair finished setting up" />
        <Chip tone="ok" tag="Log" text="Every change is in the audit log" />
      </>
    ),
  },
}

/** Login preview: what's waiting for the chosen role. */
export function LoginPreview({ role }: { role: RoleName }) {
  const p = LOGIN_PREVIEW[role] ?? LOGIN_PREVIEW.student
  return <Preview k={`login-${role}`} title={p.title} text={p.text} window={p.window} body={p.body} />
}

const SIGNUP_STEPS: Partial<Record<RoleName, { title: string; text: string; steps: [string, string][] }>> = {
  teacher: {
    title: 'Your first assignment, in four steps',
    text: 'You approve every grade before students see it.',
    steps: [
      ['Create an assignment', 'Pick the class and due date, then upload the question paper.'],
      ['Students upload their answers', 'Photos or PDFs of handwritten work, straight from a phone.'],
      ['GradeFlow scores every answer', 'Against your rubric, with partial credit and feedback.'],
      ['You review and publish', 'Check the flagged answers, adjust any mark, then release grades.'],
    ],
  },
  student: {
    title: 'From answer sheet to feedback',
    text: 'Your teacher reviews every grade before you see it.',
    steps: [
      ['Find your assignment', "It appears when your teacher publishes it, with the due date."],
      ['Upload your answers', 'Photograph or scan your pages, or upload a PDF.'],
      ['Check how they were read', 'Fix anything that was misread before grading starts.'],
      ['Get your marks and feedback', 'Question by question, with a re-check if you need one.'],
    ],
  },
}

/** Signup preview: the first steps for the chosen role. */
export function SignupPreview({ role }: { role: RoleName }) {
  const p = SIGNUP_STEPS[role] ?? SIGNUP_STEPS.student!
  return <StepsPreview k={`signup-${role}`} title={p.title} text={p.text} steps={p.steps} />
}

/**
 * A numbered process (only for content that really is a sequence). Steps before
 * `done` show a tick, so the list can show how far someone has got.
 */
export function StepsPreview({
  k,
  title,
  text,
  steps,
  done = 0,
}: {
  k: string
  title: string
  text: string
  steps: [string, string][]
  done?: number
}) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={k}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -12 }}
        transition={{ duration: 0.3, ease: EASE }}
      >
        <div className="au-caption">
          <h2>{title}</h2>
          <p>{text}</p>
        </div>
        <ol className="au-steps">
          {steps.map(([stepTitle, stepText], i) => (
            <li key={stepTitle} className={`au-step${i < done ? ' is-done' : ''}${i === done && done > 0 ? ' is-current' : ''}`}>
              <span className="au-step-num" aria-hidden>
                {i < done ? <span className="material-icons">check</span> : i + 1}
              </span>
              <div>
                <h3>
                  {stepTitle}
                  {i < done && <span className="au-sr-only"> (done)</span>}
                </h3>
                <p>{stepText}</p>
              </div>
            </li>
          ))}
        </ol>
      </motion.div>
    </AnimatePresence>
  )
}

function Preview({ k, title, text, window: windowTitle, body }: { k: string; title: string; text: string; window: string; body: ReactNode }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={k}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -12 }}
        transition={{ duration: 0.3, ease: EASE }}
      >
        <div className="au-caption">
          <h2>{title}</h2>
          <p>{text}</p>
        </div>
        <div className="lp-mock-window au-preview" aria-hidden>
          <div className="lp-mock-bar">
            <span />
            <span />
            <span />
            <em>{windowTitle}</em>
          </div>
          <div className="au-preview-body">{body}</div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
