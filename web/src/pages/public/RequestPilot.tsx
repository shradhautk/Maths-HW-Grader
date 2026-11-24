import { useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { LandingPage, PageHero, Reveal } from '../../components/landing/LandingKit'
import { fadeUp } from '../../components/landing/motion'
import { CONTACT } from '../../components/landing/site'
import { sendContactMessage } from '../../lib/contact'

const SUBJECTS = ['Math', 'Physics', 'Math and physics', 'Other STEM subject']

const NEXT = [
  ['We reply within two working days', 'To agree on the course, dates and what you want to try.'],
  ['We set up your course', 'Teacher accounts, and student accounts if you want students to see feedback.'],
  ['You grade one assignment', 'Upload scans, review what GradeAnt marked, and tell us what to improve.'],
]

export default function RequestPilot() {
  const [form, setForm] = useState({ name: '', email: '', school: '', subject: SUBJECTS[0], students: '', message: '', website: '' })
  const [state, setState] = useState<{ status: 'idle' | 'sending' | 'sent' | 'error'; text?: string }>({ status: 'idle' })
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) => {
    setForm((f) => ({ ...f, [k]: e.target.value }))
    if (state.status === 'error') setState({ status: 'idle' })
  }

  async function submit(e: FormEvent) {
    e.preventDefault()
    setState({ status: 'sending' })
    try {
      setState({ status: 'sent', text: await sendContactMessage({ kind: 'pilot', ...form }) })
      // The form is replaced by a short confirmation; bring it into view.
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (err) {
      setState({
        status: 'error',
        text: err instanceof Error && err.message ? err.message : `Your request didn't send. Please email us at ${CONTACT.email}.`,
      })
    }
  }

  return (
    <LandingPage
      title="Request a pilot"
      description="Try GradeAnt with your math or physics course. Tell us your school, subject and class size and we will set up a pilot."
    >
      <PageHero
        eyebrow="Request a pilot"
        title="Try GradeAnt with your class"
        lead="Pilots are open for math and physics courses. Tell us a little about your class and we will get in touch."
      />

      <section className="lp-section" style={{ paddingTop: 8 }}>
        <Reveal className="lp-container lp-contact">
          <motion.div className="lp-panel" variants={fadeUp}>
            {state.status === 'sent' ? (
              <div className="lp-success" role="status">
                <span className="material-icons" aria-hidden>
                  mark_email_read
                </span>
                {state.text}
              </div>
            ) : (
              <form className="lp-form" onSubmit={submit}>
                <div className="lp-form-row">
                  <div className="lp-field">
                    <label htmlFor="p-name">Your name</label>
                    <input id="p-name" className="lp-input" required maxLength={120} autoComplete="name" value={form.name} onChange={set('name')} />
                  </div>
                  <div className="lp-field">
                    <label htmlFor="p-email">Work email</label>
                    <input id="p-email" className="lp-input" type="email" required maxLength={200} autoComplete="email" value={form.email} onChange={set('email')} />
                  </div>
                </div>
                <div className="lp-field">
                  <label htmlFor="p-school">School or university</label>
                  <input id="p-school" className="lp-input" required maxLength={200} autoComplete="organization" value={form.school} onChange={set('school')} />
                </div>
                <div className="lp-form-row">
                  <div className="lp-field">
                    <label htmlFor="p-subject">Subject</label>
                    <select id="p-subject" className="lp-input" value={form.subject} onChange={set('subject')}>
                      {SUBJECTS.map((s) => (
                        <option key={s}>{s}</option>
                      ))}
                    </select>
                  </div>
                  <div className="lp-field">
                    <label htmlFor="p-students">Number of students</label>
                    <input
                      id="p-students"
                      className="lp-input"
                      type="number"
                      inputMode="numeric"
                      min={1}
                      max={999999}
                      step={1}
                      required
                      value={form.students}
                      onChange={set('students')}
                    />
                  </div>
                </div>
                <div className="lp-field">
                  <label htmlFor="p-message">
                    Anything else? <span className="lp-optional">(optional)</span>
                  </label>
                  <textarea id="p-message" className="lp-input" maxLength={5000} value={form.message} onChange={set('message')} />
                </div>
                <div className="lp-hp" aria-hidden>
                  <label htmlFor="p-website">Website</label>
                  <input id="p-website" tabIndex={-1} autoComplete="off" value={form.website} onChange={set('website')} />
                </div>
                {state.status === 'error' && (
                  <div className="lp-error" role="alert">
                    <span className="material-icons" aria-hidden>
                      error
                    </span>
                    {state.text}
                  </div>
                )}
                <button type="submit" className="lp-btn lp-btn-primary lp-btn-lg" disabled={state.status === 'sending'}>
                  {state.status === 'sending' ? 'Sending…' : 'Request a pilot'}
                </button>
                <p className="lp-form-note">
                  We use these details only to reply to you. See our <Link to="/privacy">privacy policy</Link>.
                </p>
              </form>
            )}
          </motion.div>

          <motion.div className="lp-info" variants={fadeUp}>
            <h2 className="lp-info-title">What happens next</h2>
            <ol className="lp-next">
              {NEXT.map(([t, d]) => (
                <li key={t}>
                  <b>{t}</b>
                  <span>{d}</span>
                </li>
              ))}
            </ol>
            <p className="lp-form-note">
              Prefer email? Write to <a href={`mailto:${CONTACT.email}`}>{CONTACT.email}</a>.
            </p>
          </motion.div>
        </Reveal>
      </section>
    </LandingPage>
  )
}
