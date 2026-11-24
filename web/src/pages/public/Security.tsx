import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { LandingPage, PageHero, Reveal } from '../../components/landing/LandingKit'
import { fadeUp } from '../../components/landing/motion'
import { CONTACT } from '../../components/landing/site'

/**
 * Security and privacy. Content owners: engineers + Missouri S&T compliance.
 * Every statement must match how the system works today and the privacy policy
 * (retention and research use are copied from it). The data region is still being
 * confirmed with S&T compliance, and the page says so rather than guess.
 */
const SECTIONS = [
  {
    icon: 'verified',
    title: 'FERPA',
    body: ['GradeAnt is FERPA compliant. We use student education records only to provide grading and feedback to the course that submitted them, and never sell or share them.'],
  },
  {
    icon: 'cloud',
    title: 'Where data is stored',
    body: [
      'Accounts, grades and feedback are stored in Amazon DynamoDB, and uploaded scans in Amazon S3, both on Amazon Web Services (AWS). Data is encrypted in transit (HTTPS) and at rest.',
      'We are confirming with Missouri S&T compliance which AWS region US student data must be stored in, and will move it there if needed.',
    ],
  },
  {
    icon: 'group',
    title: 'Who can see it',
    body: [
      'Students see only their own work, marks and feedback.',
      'Teachers see the work of students in their own classes.',
      "Your school's administrator manages accounts and can see account details; admin actions are recorded in an audit log.",
      'Passwords are stored only as salted hashes (PBKDF2), never in plain text.',
    ],
  },
  {
    icon: 'schedule',
    title: 'How long it is kept',
    body: [
      'Submitted work and grading results are kept for about one year, so students and their school can look back at them. After that they are deleted or fully anonymised.',
      'Account details are kept while the account is active, and deleted about six months after it was last used.',
    ],
  },
  {
    icon: 'delete',
    title: 'Deletion on request',
    body: [
      `You can ask us to delete an account and the work linked to it at any time by emailing ${CONTACT.email}. We may need to confirm your identity first.`,
    ],
  },
  {
    icon: 'science',
    title: 'Research use',
    body: [
      'Anonymised work may be used for education research under IRB guidelines; names are removed from answer sheets first. You can opt out when you create your account.',
      'We do not use your homework submissions to train our main grading models without your consent.',
    ],
  },
]

export default function Security() {
  return (
    <LandingPage
      title="Security and privacy"
      description="How GradeAnt stores and protects student data: FERPA, where data is stored, who can see it, how long it is kept, and deletion on request."
    >
      <PageHero
        eyebrow="Security and privacy"
        title="How we handle student data"
        lead="Where it is stored, who can see it, how long we keep it, and how to have it deleted."
      />

      <section className="lp-section" style={{ paddingTop: 8 }}>
        <Reveal className="lp-container lp-doc">
          {SECTIONS.map((s) => (
            <motion.section key={s.title} className="lp-doc-section" variants={fadeUp}>
              <h2>
                <span className="material-icons" aria-hidden>
                  {s.icon}
                </span>
                {s.title}
              </h2>
              {s.body.map((p) => (
                <p key={p}>{p}</p>
              ))}
            </motion.section>
          ))}
          <motion.p className="lp-form-note" variants={fadeUp}>
            Questions about security or privacy? Email <a href={`mailto:${CONTACT.email}`}>{CONTACT.email}</a>. The full legal terms are in
            our <Link to="/privacy">privacy policy</Link>.
          </motion.p>
        </Reveal>
      </section>
    </LandingPage>
  )
}
