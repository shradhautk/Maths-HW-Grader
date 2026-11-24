import { useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import { FaqList, LandingPage, PageHero, Reveal, SectionHead } from '../../components/landing/LandingKit'
import { EASE, fadeUp } from '../../components/landing/motion'
import { CONTACT, SOCIAL } from '../../components/landing/site'
import { sendContactMessage } from '../../lib/contact'

const FAQ = [
  {
    q: 'How do I start using GradeAnt?',
    a: 'GradeAnt is running pilots with math and physics courses. Request a pilot and we will set up your course and accounts.',
    link: { to: '/request-a-pilot', label: 'Request a pilot.' },
  },
  {
    q: 'Does GradeAnt integrate with my existing LMS?',
    a: 'LMS integration with Canvas, Blackboard, Moodle and Google Classroom is in progress. For now, you upload scans and download grades from GradeAnt.',
  },
  {
    q: "Is my students' data secure?",
    a: 'We are FERPA compliant. Student work is visible only to the teachers of that course, and we never sell or share it.',
    link: { to: '/security', label: 'Read our security and privacy page.' },
  },
  {
    q: 'Can I customise the grading criteria?',
    a: 'Yes. Upload your own solution key or edit the drafted rubric, and decide how much partial credit each step earns.',
  },
]

const SUBJECTS = ['General question', 'Support', 'Partnership', 'Research', 'Press', 'Feedback']

const MAP_URL = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(CONTACT.address)}`
const LINKEDIN = SOCIAL.find((s) => s.label === 'LinkedIn')

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', subject: SUBJECTS[0], message: '', website: '' })
  const [state, setState] = useState<{ status: 'idle' | 'sending' | 'sent' | 'error'; text?: string }>({ status: 'idle' })
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) => {
    setForm((f) => ({ ...f, [k]: e.target.value }))
    if (state.status === 'error') setState({ status: 'idle' })
  }

  async function submit(e: FormEvent) {
    e.preventDefault()
    setState({ status: 'sending' })
    try {
      const text = await sendContactMessage({ kind: 'message', ...form })
      setState({ status: 'sent', text })
      setForm((f) => ({ ...f, message: '' }))
    } catch (err) {
      setState({
        status: 'error',
        text: err instanceof Error && err.message ? err.message : `Your message didn't send. Please email us at ${CONTACT.email}.`,
      })
    }
  }

  return (
    <LandingPage title="Contact" description={`Contact the GradeAnt team in Rolla, Missouri: ${CONTACT.email}, ${CONTACT.phone}, ${CONTACT.hours}.`}>
      <PageHero
        eyebrow="Contact"
        title="Talk to the GradeAnt team"
        lead="Questions, support, research or partnership ideas: send us a message and we will reply by email."
      />

      <section className="lp-section" style={{ paddingTop: 8 }}>
        <Reveal className="lp-container lp-contact">
          <motion.div className="lp-panel" variants={fadeUp}>
            <h2>Send us a message</h2>
            <p>
              Teaching a math or physics course? <Link to="/request-a-pilot">Request a pilot</Link> instead.
            </p>
            <form className="lp-form" onSubmit={submit}>
              <div className="lp-form-row">
                <div className="lp-field">
                  <label htmlFor="c-name">Full name</label>
                  <input id="c-name" className="lp-input" required maxLength={120} autoComplete="name" value={form.name} onChange={set('name')} />
                </div>
                <div className="lp-field">
                  <label htmlFor="c-email">Email address</label>
                  <input id="c-email" className="lp-input" type="email" required maxLength={200} autoComplete="email" value={form.email} onChange={set('email')} />
                </div>
              </div>
              <div className="lp-field">
                <label htmlFor="c-subject">Subject</label>
                <select id="c-subject" className="lp-input" value={form.subject} onChange={set('subject')}>
                  {SUBJECTS.map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div className="lp-field">
                <label htmlFor="c-message">Message</label>
                <textarea id="c-message" className="lp-input" required maxLength={5000} value={form.message} onChange={set('message')} />
              </div>
              {/* Honeypot for spam bots: hidden from people and screen readers. */}
              <div className="lp-hp" aria-hidden>
                <label htmlFor="c-website">Website</label>
                <input id="c-website" tabIndex={-1} autoComplete="off" value={form.website} onChange={set('website')} />
              </div>
              <AnimatePresence>
                {(state.status === 'sent' || state.status === 'error') && (
                  <motion.div
                    className={state.status === 'sent' ? 'lp-success' : 'lp-error'}
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ ease: EASE }}
                    role={state.status === 'error' ? 'alert' : 'status'}
                  >
                    <span className="material-icons" aria-hidden>
                      {state.status === 'sent' ? 'mark_email_read' : 'error'}
                    </span>
                    {state.text}
                  </motion.div>
                )}
              </AnimatePresence>
              <button type="submit" className="lp-btn lp-btn-primary lp-btn-lg" disabled={state.status === 'sending'}>
                {state.status === 'sending' ? 'Sending…' : 'Send message'}
              </button>
            </form>
          </motion.div>

          <motion.div className="lp-info" variants={fadeUp}>
            <a className="lp-info-item" href={`mailto:${CONTACT.email}`}>
              <span className="lp-info-icon material-icons" aria-hidden>
                mail
              </span>
              <div>
                <b>Email</b>
                <span>{CONTACT.email}</span>
              </div>
            </a>
            <a className="lp-info-item" href={`tel:${CONTACT.phone}`}>
              <span className="lp-info-icon material-icons" aria-hidden>
                call
              </span>
              <div>
                <b>Phone</b>
                <span>
                  {CONTACT.phone}
                  <br />
                  {CONTACT.hours}
                </span>
              </div>
            </a>
            <a className="lp-info-item" href={MAP_URL} target="_blank" rel="noreferrer">
              <span className="lp-info-icon material-icons" aria-hidden>
                location_on
              </span>
              <div>
                <b>Location</b>
                <span>{CONTACT.address}</span>
              </div>
            </a>
            {LINKEDIN && (
              <a className="lp-info-item" href={LINKEDIN.href} target="_blank" rel="noreferrer">
                <span className="lp-info-icon" aria-hidden>
                  <img src={LINKEDIN.icon} alt="" width={20} height={20} />
                </span>
                <div>
                  <b>LinkedIn</b>
                  <span>Follow GradeAnt for news and new roles</span>
                </div>
              </a>
            )}
          </motion.div>
        </Reveal>
      </section>

      <section className="lp-section lp-section-soft">
        <div className="lp-container lp-faq-wrap">
          <SectionHead eyebrow="FAQ" title="Frequently asked questions" />
          <FaqList items={FAQ} />
        </div>
      </section>
    </LandingPage>
  )
}
