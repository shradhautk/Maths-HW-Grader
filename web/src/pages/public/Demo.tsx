import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { CtaBand, LandingPage, PageHero, ProductShot, Reveal } from '../../components/landing/LandingKit'
import { fadeUp } from '../../components/landing/motion'
import { BeforeAfterScreen, GradingScreen, ReviewScreen, RubricScreen, UploadScreen } from '../../components/landing/Screens'
import { MEDIA } from '../../components/landing/site'

/**
 * Public product tour (the "See how it works" / "Take the product tour" links).
 * Each step shows the real screen once MEDIA has a screenshot, otherwise a drawn
 * illustration of it. No pitch-deck slides.
 */
const TOUR: { eyebrow: string; title: string; text: string; points: string[]; bar: string; src: string | null; alt: string; draw: ReactNode }[] = [
  {
    eyebrow: 'Step 1 · Upload',
    title: 'Upload the question paper and scanned answers',
    text: 'Create an assignment, upload the question paper, then the scanned student answers: one file per student or one combined scan.',
    points: ['Questions identified automatically', 'Scans, photos or PDFs', 'You confirm the parsed questions'],
    bar: 'Upload answer sheets',
    src: MEDIA.upload,
    alt: 'Upload screen with scanned answer sheets',
    draw: <UploadScreen />,
  },
  {
    eyebrow: 'Step 2 · Rubric',
    title: 'Use your rubric, or approve a drafted one',
    text: 'Upload your solution key, or start from a rubric GradeAnt drafts from the question paper. Set the marks for each step and approve it before grading starts.',
    points: ['Marks per step', 'Partial credit for method', 'Nothing is graded until you approve'],
    bar: 'Rubric settings',
    src: MEDIA.rubric,
    alt: 'Rubric settings with marks for each step',
    draw: <RubricScreen />,
  },
  {
    eyebrow: 'Step 3 · Grading',
    title: 'Every step checked against your rubric',
    text: 'GradeAnt reads the handwriting, matches answers to questions, marks each step and drafts feedback. Anything it is unsure of is flagged for you instead of being guessed.',
    points: ['Step-by-step marks', 'Draft feedback on every answer', 'Low-confidence answers flagged'],
    bar: 'Grading in progress',
    src: MEDIA.grading,
    alt: 'Grading progress for a batch of answers',
    draw: <GradingScreen />,
  },
  {
    eyebrow: 'Step 4 · Teacher review',
    title: 'Review the flagged answers, then publish',
    text: 'Clear the review queue, change any mark or note, and publish when you are ready. Students see nothing before that.',
    points: ['You approve every grade', 'Re-check requests in one place', 'Score spread and hardest questions'],
    bar: 'Teacher review',
    src: MEDIA.review,
    alt: 'Teacher review screen with flagged answers',
    draw: <ReviewScreen />,
  },
  {
    eyebrow: 'Result',
    title: 'What the student gets back',
    text: 'Marks for each step and a short note that points to what went wrong and how to fix it.',
    points: ['Marks for each step', 'Points to the exact mistake', 'Shows the corrected working'],
    bar: 'Q3 · before and after',
    src: MEDIA.beforeAfter,
    alt: 'A handwritten integration answer next to the feedback it produced',
    draw: <BeforeAfterScreen />,
  },
]

export default function Demo() {
  return (
    <LandingPage
      title="Product tour"
      description="Follow one assignment through GradeAnt: upload scans, set the rubric, step-by-step grading, teacher review and the feedback students get."
    >
      <PageHero
        eyebrow="Product tour"
        title="See how GradeAnt works"
        lead="Follow one assignment from scanned answers to reviewed grades, and see the feedback the student gets back."
      >
        <a href="#tour" className="lp-btn lp-btn-primary lp-btn-lg">
          Start the tour ↓
        </a>
        <Link to="/request-a-pilot" className="lp-btn lp-btn-secondary lp-btn-lg">
          Request a pilot
        </Link>
      </PageHero>

      {MEDIA.demoVideo && (
        <section className="lp-section" style={{ paddingTop: 0 }}>
          <div className="lp-container lp-video">
            <video controls muted playsInline preload="metadata" poster={MEDIA.demoPoster ?? undefined} aria-label="GradeAnt grading one handwritten answer">
              <source src={MEDIA.demoVideo} type="video/mp4" />
              {MEDIA.demoCaptions && <track kind="captions" src={MEDIA.demoCaptions} srcLang="en" label="English" default />}
            </video>
          </div>
        </section>
      )}

      <section className="lp-section lp-section-soft" id="tour">
        <div className="lp-container lp-feature-rows">
          {TOUR.map((r, i) => (
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

      <div style={{ height: 104 }} />
      <CtaBand />
    </LandingPage>
  )
}
