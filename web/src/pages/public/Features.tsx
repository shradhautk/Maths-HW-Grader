import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { CtaBand, LandingPage, PageHero, ProductShot, Reveal, SectionHead } from '../../components/landing/LandingKit'
import { fadeUp } from '../../components/landing/motion'
import { BeforeAfterScreen, HeroSplit, ReviewScreen } from '../../components/landing/Screens'
import { MEDIA } from '../../components/landing/site'

// Product page. Only claims that are true today: no accuracy figures until there
// is a published measurement to link to, and only the subjects tested (math, physics).

const KEY_FEATURES = [
  { icon: 'checklist', title: 'Step-by-step grading', text: 'Every step of a worked answer is marked against your rubric, with partial credit for correct method.', tone: 'brand' },
  { icon: 'chat', title: 'Drafted feedback', text: 'A short note on each answer: what was right, what went wrong and how to fix it. You edit before it is sent.', tone: 'accent' },
  { icon: 'document_scanner', title: 'Handwriting and scans', text: 'Reads handwritten working, equations and units from scanned pages, photos and PDFs.', tone: 'brand' },
  { icon: 'flag', title: 'Flags what it is unsure of', text: 'Low-confidence readings and unusual methods go to a review queue instead of being guessed.', tone: 'accent' },
  { icon: 'insights', title: 'Class view', text: 'Score spread and the hardest questions for each assignment, once grading finishes.', tone: 'brand' },
  { icon: 'lock', title: 'Security and privacy', text: 'Role-based access: student work is visible only to the teachers of that course.', tone: 'accent', link: { to: '/security', label: 'How we handle data →' } },
] as const

const ROWS: { eyebrow: string; title: string; text: string; points: string[]; bar: string; src: string | null; alt: string; draw: ReactNode }[] = [
  {
    eyebrow: 'Grading',
    title: 'Marks for every step, not just the final answer',
    text: 'GradeAnt finds each question on the page, follows the working line by line and marks it against your rubric, so a slip in the last line does not cost the whole question.',
    points: ['Questions identified automatically', 'Partial credit for correct method', 'Feedback on every answer'],
    bar: 'Q2 · step-by-step grading',
    src: MEDIA.hero,
    alt: 'A handwritten physics answer with marks for each step',
    draw: <HeroSplit />,
  },
  {
    eyebrow: 'Feedback',
    title: 'Feedback the student can act on',
    text: 'For each answer GradeAnt drafts a note that points to the step that went wrong and shows the fix. You can edit any note before it is published.',
    points: ['Points to the exact step', 'Shows the corrected working', 'Edited and approved by the teacher'],
    bar: 'Q3 · before and after',
    src: MEDIA.beforeAfter,
    alt: 'A handwritten integration answer next to the feedback it produced',
    draw: <BeforeAfterScreen />,
  },
  {
    eyebrow: 'Review',
    title: 'You approve every grade',
    text: 'Answers GradeAnt is unsure about are queued for you. Change any mark, then publish when you are ready. Students see nothing before that.',
    points: ['Review queue for flagged answers', 'Change any mark', 'Publish only when you are ready'],
    bar: 'Teacher review',
    src: MEDIA.review,
    alt: 'Teacher review screen with flagged answers',
    draw: <ReviewScreen />,
  },
]

export default function Features() {
  return (
    <LandingPage
      title="Product"
      description="GradeAnt reads handwritten math and physics, marks every step against your rubric, drafts feedback and flags what it is unsure of for teacher review."
    >
      <PageHero
        eyebrow="Product"
        title="Step-by-step grading for handwritten math and physics"
        lead="GradeAnt reads scanned student work, marks each step against your rubric and drafts feedback. You review and publish."
      >
        <Link to="/request-a-pilot" className="lp-btn lp-btn-primary lp-btn-lg">
          Request a pilot
        </Link>
        <Link to="/demo" className="lp-btn lp-btn-secondary lp-btn-lg">
          Take the product tour
        </Link>
      </PageHero>

      <section className="lp-section" style={{ paddingTop: 24 }}>
        <div className="lp-container">
          <SectionHead eyebrow="Key features" title="What GradeAnt does today" />
          <Reveal className="lp-cards">
            {KEY_FEATURES.map((f) => (
              <motion.article key={f.title} className={`lp-feature lp-tone-${f.tone}`} variants={fadeUp}>
                <span className="lp-feature-icon material-icons" aria-hidden>
                  {f.icon}
                </span>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
                {'link' in f && (
                  <Link to={f.link.to} className="lp-inline-link">
                    {f.link.label}
                  </Link>
                )}
              </motion.article>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="lp-section lp-section-soft">
        <div className="lp-container lp-feature-rows">
          {ROWS.map((r, i) => (
            <Reveal key={r.title} className={`lp-split lp-feature-row${i % 2 ? ' is-flipped' : ''}`}>
              <motion.div variants={fadeUp}>
                <span className="lp-eyebrow">{r.eyebrow}</span>
                <h2>{r.title}</h2>
                <p>{r.text}</p>
                <ul className="lp-checks">
                  {r.points.map((p) => (
                    <li key={p}>
                      <span className="material-icons" aria-hidden>
                        check_circle
                      </span>
                      {p}
                    </li>
                  ))}
                </ul>
              </motion.div>
              <motion.div variants={fadeUp}>
                <ProductShot src={r.src} alt={r.alt} bar={r.bar}>
                  {r.draw}
                </ProductShot>
              </motion.div>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="lp-section lp-dark" id="technology">
        <div className="lp-container">
          <SectionHead
            light
            eyebrow="The technology"
            title={<>GradeFlow™, our grading pipeline</>}
            text="A vision model reads the page, then language-model agents match answers to questions, check each step against the rubric and draft feedback."
          />
          <Reveal className="lp-tech-cards">
            {[
              { icon: 'visibility', title: 'Reading', steps: ['Handwriting to text', 'Equations to math', 'Low-confidence text flagged'] },
              { icon: 'hub', title: 'Grading', steps: ['Answers matched to questions', 'Each step checked', 'Marks against your rubric'] },
              { icon: 'rate_review', title: 'Feedback', steps: ['What was right', 'Where it went wrong', 'How to fix it'] },
            ].map((m) => (
              <motion.div key={m.title} className="lp-tech-card" variants={fadeUp}>
                <span className="material-icons" aria-hidden>
                  {m.icon}
                </span>
                <h3>{m.title}</h3>
                <ol className="lp-model-steps">
                  {m.steps.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ol>
              </motion.div>
            ))}
          </Reveal>
          <p className="lp-dark-note">
            Want to see how we test this? Our benchmark work is on the <Link to="/research">Research page</Link>.
          </p>
        </div>
      </section>

      <div style={{ height: 104 }} />
      <CtaBand />
    </LandingPage>
  )
}
