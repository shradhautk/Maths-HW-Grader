import { useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { motion, useMotionValueEvent, useScroll, useSpring } from 'motion/react'
import { Magnetic, Marquee, TextReveal, Tilt } from '../../components/motion/kit'
import { CtaBand, FaqList, LandingPage, ProductShot, ResearchStrip, Reveal, SectionHead } from '../../components/landing/LandingKit'
import { EASE, fadeUp, stagger } from '../../components/landing/motion'
import { BeforeAfterScreen, GradingScreen, HeroSplit, ReviewScreen, RubricScreen, UploadScreen } from '../../components/landing/Screens'
import { MEDIA } from '../../components/landing/site'

/**
 * Marketing homepage. Every claim here must be true today and checkable
 * (GradeAnt Website Plan, priority 0): no usage numbers, accuracy figures or
 * subjects we haven't tested. Send changed copy to Shradha before it goes live.
 */

const FEATURES = [
  { icon: 'draw', title: 'Reads handwritten work', text: 'Scanned working, equations and units are read and matched to the right question.', tone: 'brand', span: 2 },
  { icon: 'checklist', title: 'Checks every step', text: 'Each step is marked against your rubric, with partial credit for correct method.', tone: 'accent', span: 1 },
  { icon: 'tune', title: 'Your rubric', text: 'Upload a solution key, or start from an AI-drafted rubric that you edit and approve.', tone: 'accent', span: 1 },
  { icon: 'functions', title: 'Math and physics', text: 'Built and tested on math and physics today, including units, signs and method marks.', tone: 'brand', span: 2 },
  { icon: 'insights', title: 'Class view', text: 'See the score spread and the hardest questions once a batch is graded.', tone: 'brand', span: 2 },
  { icon: 'verified_user', title: 'You approve every grade', text: 'Uncertain answers are flagged for you. Nothing reaches students until you publish.', tone: 'accent', span: 1 },
] as const

const STEPS: { title: string; text: string; bar: string; src: string | null; alt: string; draw: ReactNode }[] = [
  {
    title: 'Upload',
    text: 'Upload the question paper and the scanned student answers.',
    bar: 'Upload answer sheets',
    src: MEDIA.upload,
    alt: 'Upload screen with scanned answer sheets',
    draw: <UploadScreen />,
  },
  {
    title: 'Set the rubric',
    text: 'Use your solution key, or edit the drafted rubric and approve it.',
    bar: 'Rubric settings',
    src: MEDIA.rubric,
    alt: 'Rubric settings with marks for each step',
    draw: <RubricScreen />,
  },
  {
    title: 'Grading',
    text: 'GradeAnt checks every step against the rubric and drafts feedback.',
    bar: 'Grading in progress',
    src: MEDIA.grading,
    alt: 'Grading progress for a batch of answers',
    draw: <GradingScreen />,
  },
  {
    title: 'Teacher review',
    text: 'Check what was flagged, adjust any mark, then publish.',
    bar: 'Teacher review',
    src: MEDIA.review,
    alt: 'Teacher review screen with flagged answers',
    draw: <ReviewScreen />,
  },
]

const PIPELINE = [
  { label: 'Read', detail: 'Handwriting to text and math' },
  { label: 'Match', detail: 'Answers mapped to questions' },
  { label: 'Check', detail: 'Each step against your rubric' },
  { label: 'Explain', detail: 'Draft feedback per answer' },
  { label: 'Review', detail: 'You approve before students see it' },
]

const FAQ = [
  {
    q: 'Can GradeAnt really read handwritten answers?',
    a: 'Yes. Teachers upload scanned pages and GradeAnt reads the handwriting, equations and working. Anything it reads with low confidence is flagged for a person to check.',
  },
  {
    q: 'Do I need to provide an answer key?',
    a: 'No. GradeAnt can draft a rubric from your question paper for you to edit and approve, or you can upload your own solution key.',
  },
  {
    q: 'Who has the final say on grades?',
    a: 'You do. Uncertain answers are queued for your review, you can change any mark, and nothing is visible to students until you publish.',
  },
  {
    q: 'Which subjects does it support?',
    a: 'Math and physics. Those are the subjects GradeAnt has been built and tested on so far.',
  },
  {
    q: 'How is student data handled?',
    a: 'Student work is only visible to the teachers of that course, and you can ask us to delete it.',
    link: { to: '/security', label: 'Read how we store and protect data.' },
  },
]

export default function Home() {
  return (
    <LandingPage
      title="AI that grades handwritten math and physics"
      description="Teachers upload scanned work. GradeAnt checks every step of handwritten math and physics against your rubric and drafts feedback for you to review."
    >
      <Hero />
      <ResearchStrip />
      <Features />
      <HowItWorks />
      <InAction />
      <Technology />
      <Roles />
      <section className="lp-section" id="faq">
        <div className="lp-container lp-faq-wrap">
          <SectionHead eyebrow="FAQ" title="Questions teachers ask us" />
          <FaqList items={FAQ} />
        </div>
      </section>
      <CtaBand />
    </LandingPage>
  )
}

// ---- Hero ----

const HERO_POINTS = ['Your rubric', 'Partial credit for method', 'You approve every grade']

/** What GradeAnt marks, shown as a slow ticker under the hero (all shown in the examples on this page). */
const CHECKS = [
  ['functions', 'Handwritten equations'],
  ['straighten', 'Units'],
  ['swap_vert', 'Sign conventions'],
  ['account_tree', 'Method marks'],
  ['percent', 'Partial credit'],
  ['input', 'Substituted values'],
  ['timeline', 'Multi-step derivations'],
  ['task_alt', 'Final answers'],
  ['flag', 'Low-confidence reads, flagged'],
]

function Hero() {
  return (
    <section className="lp-hero lp-hero-v2">
      <div className="lp-container lp-hero-inner">
        <motion.div className="lp-hero-copy" variants={stagger} initial="hidden" animate="show">
          <motion.span className="lp-eyebrow" variants={fadeUp}>
            <span className="lp-dot" /> For math and physics teachers
          </motion.span>
          <TextReveal as="h1" text="AI that grades handwritten math and physics, step by step." highlight="step by step." delay={0.15} />
          <motion.p className="lp-lead" variants={fadeUp}>
            Teachers upload scanned work. GradeAnt checks every step against your rubric and drafts feedback for you to review.
          </motion.p>
          <motion.div className="lp-hero-ctas" variants={fadeUp}>
            <Magnetic>
              <Link to="/request-a-pilot" className="lp-btn lp-btn-primary lp-btn-lg">
                Request a pilot
                <span className="material-icons lp-btn-arrow" aria-hidden>
                  arrow_forward
                </span>
              </Link>
            </Magnetic>
            <Link to="/#how" className="lp-btn lp-btn-secondary lp-btn-lg">
              See how it works
            </Link>
          </motion.div>
          <motion.ul className="lp-hero-points" variants={fadeUp}>
            {HERO_POINTS.map((p) => (
              <li key={p}>
                <span className="material-icons" aria-hidden>
                  check_circle
                </span>
                {p}
              </li>
            ))}
          </motion.ul>
        </motion.div>

        <motion.div className="lp-hero-visual" initial={{ opacity: 0, y: 32, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.9, ease: EASE, delay: 0.25 }}>
          <span className="lp-hero-glow" aria-hidden />
          <Tilt max={5}>
            <div className="lp-hero-frame fx-border">
              <ProductShot src={MEDIA.hero} alt="A scanned handwritten physics answer next to GradeAnt's step-by-step marks and feedback" bar="Q2 · step-by-step grading">
                <HeroSplit animated={!MEDIA.hero} delay={0.9} />
              </ProductShot>
            </div>
          </Tilt>
          <motion.div
            className="lp-float lp-float-b"
            initial={{ opacity: 0, y: 12, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.6, ease: EASE, delay: 2.6 }}
            aria-hidden
          >
            <span className="material-icons">flag</span>
            <span>
              <b>Step 3 flagged</b>
              <small>Waiting for your review</small>
            </span>
          </motion.div>
          <motion.div
            className="lp-float lp-float-a"
            initial={{ opacity: 0, y: 12, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.6, ease: EASE, delay: 2.1 }}
            aria-hidden
          >
            <span className="material-icons">verified</span>
            <span>
              <b>Method mark kept</b>
              <small>Right equation, wrong arithmetic</small>
            </span>
          </motion.div>
        </motion.div>
      </div>

      <div className="lp-container lp-checks-strip">
        <span className="lp-checks-label">What GradeAnt checks</span>
        <Marquee
          label="What GradeAnt checks"
          items={CHECKS.map(([icon, text]) => (
            <span key={text} className="lp-check-chip">
              <span className="material-icons" aria-hidden>
                {icon}
              </span>
              {text}
            </span>
          ))}
        />
      </div>
    </section>
  )
}

// ---- Features ----

function Features() {
  return (
    <section className="lp-section" id="features">
      <div className="lp-container">
        <SectionHead
          eyebrow="What it does"
          title="Grading support that shows its working"
          text="GradeAnt handles the reading and first-pass marking. The judgement stays with you."
        />
        <Reveal className="lp-bento">
          {FEATURES.map((f) => (
            <motion.article key={f.title} className={`lp-feature lp-tone-${f.tone}${f.span === 2 ? ' is-wide' : ''}`} variants={fadeUp}>
              <span className="lp-feature-icon material-icons" aria-hidden>
                {f.icon}
              </span>
              <h3>{f.title}</h3>
              <p>{f.text}</p>
            </motion.article>
          ))}
        </Reveal>
      </div>
    </section>
  )
}

// ---- How it works ----

/**
 * Sticky scroll story (desktop): the step list stays pinned on the left while
 * each step's screen scrolls past on the right; the step whose screen is in
 * view lights up and the progress rail fills. On narrow screens it is a plain
 * stacked list (CSS hides the rail and un-sticks the list).
 */
function HowItWorks() {
  const [active, setActive] = useState(0)
  const story = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({ target: story, offset: ['start 60%', 'end 60%'] })
  const fill = useSpring(scrollYProgress, { stiffness: 120, damping: 28 })

  function jump(i: number) {
    document.getElementById(`how-step-${i}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  return (
    <section className="lp-section lp-section-soft" id="how">
      <div className="lp-container">
        <SectionHead eyebrow="How it works" title="From scanned answers to reviewed grades in four steps" />
        <div className="lp-story" ref={story}>
          <div className="lp-story-rail">
            <span className="lp-story-track" aria-hidden>
              <motion.span style={{ scaleY: fill }} />
            </span>
            <ol className="lp-story-list">
              {STEPS.map((s, i) => (
                <li key={s.title} className={i === active ? 'is-active' : i < active ? 'is-done' : undefined}>
                  <button type="button" onClick={() => jump(i)} aria-current={i === active ? 'step' : undefined}>
                    <span className="lp-story-num">{i + 1}</span>
                    <span>
                      <b>{s.title}</b>
                      <small>{s.text}</small>
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </div>
          <div className="lp-story-shots">
            {STEPS.map((s, i) => (
              <motion.figure
                key={s.title}
                id={`how-step-${i}`}
                className="lp-story-shot"
                initial={{ opacity: 0.35, scale: 0.96 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ amount: 0.6 }}
                onViewportEnter={() => setActive(i)}
                transition={{ duration: 0.5, ease: EASE }}
              >
                <figcaption className="lp-story-caption">
                  <small>Step {i + 1}</small>
                  <h3>{s.title}</h3>
                  <p>{s.text}</p>
                </figcaption>
                <ProductShot src={s.src} alt={s.alt} bar={s.bar}>
                  {s.draw}
                </ProductShot>
              </motion.figure>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

// ---- See it in action: demo video (when recorded) + before and after ----

function InAction() {
  return (
    <section className="lp-section" id="demo">
      <div className="lp-container">
        <SectionHead
          eyebrow="See GradeAnt in action"
          title="One answer, before and after"
          text="A handwritten derivation as the student wrote it, next to the step marks and feedback GradeAnt drafted for the teacher."
        />
        {MEDIA.demoVideo && (
          <div className="lp-video">
            <video controls muted playsInline preload="metadata" poster={MEDIA.demoPoster ?? undefined} aria-label="GradeAnt grading one handwritten answer">
              <source src={MEDIA.demoVideo} type="video/mp4" />
              {MEDIA.demoCaptions && <track kind="captions" src={MEDIA.demoCaptions} srcLang="en" label="English" default />}
            </video>
          </div>
        )}
        <Reveal>
          <motion.div variants={fadeUp}>
            <ProductShot src={MEDIA.beforeAfter} alt="A handwritten integration-by-parts answer next to GradeAnt's feedback" bar="Q3 · before and after">
              <BeforeAfterScreen />
            </ProductShot>
          </motion.div>
        </Reveal>
        <p className="lp-center-link">
          <Link to="/demo">Take the full product tour →</Link>
        </p>
      </div>
    </section>
  )
}

// ---- Technology (dark) ----

/** The pipeline draws itself as the section scrolls through: the line fills and each stage lights up when reached. */
function Technology() {
  const ref = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start 85%', 'end 45%'] })
  const line = useSpring(scrollYProgress, { stiffness: 110, damping: 26 })
  const [lit, setLit] = useState(0)
  useMotionValueEvent(scrollYProgress, 'change', (v) => setLit(Math.min(PIPELINE.length, Math.floor(v * PIPELINE.length + 0.35))))

  return (
    <section className="lp-section lp-dark lp-tech" id="technology">
      <div className="lp-container">
        <SectionHead
          light
          eyebrow="The technology"
          title={<>How GradeFlow™ works</>}
          text="GradeFlow™ is our grading pipeline. Each stage leaves a result you can check."
        />
        <div className="lp-pipeline-wrap" ref={ref}>
          <span className="lp-pipe-line" aria-hidden>
            <motion.span style={{ scaleX: line }} />
          </span>
          <ol className="lp-pipeline lp-pipeline-v2">
            {PIPELINE.map((p, i) => (
              <li key={p.label} className={i < lit ? 'is-lit' : undefined}>
                <span className="lp-pipe-node">{i + 1}</span>
                <b>{p.label}</b>
                <small>{p.detail}</small>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  )
}

// ---- Roles ----

function Roles() {
  const roles = [
    {
      who: 'For teachers',
      title: 'Less time on first-pass marking',
      icon: 'school',
      points: ['Upload scans for a whole class', 'Use your rubric or approve a drafted one', 'Review only what is flagged', 'Publish when you are ready'],
      to: '/request-a-pilot',
      cta: 'Request a pilot',
      tone: 'brand',
    },
    {
      who: 'For students in a pilot course',
      title: 'Feedback on every step',
      icon: 'backpack',
      points: ['See marks for each step', 'Read what to fix and why', 'Ask your teacher for a re-check'],
      to: '/login',
      cta: 'Log in',
      tone: 'accent',
    },
  ] as const
  return (
    <section className="lp-section lp-section-soft">
      <div className="lp-container">
        <SectionHead eyebrow="Who it is for" title="Teachers grade, students learn from the feedback" />
        <Reveal className="lp-roles">
          {roles.map((r) => (
            <motion.article key={r.who} className={`lp-role lp-tone-${r.tone}`} variants={fadeUp}>
              <span className="lp-role-icon material-icons" aria-hidden>
                {r.icon}
              </span>
              <small>{r.who}</small>
              <h3>{r.title}</h3>
              <ul>
                {r.points.map((p) => (
                  <li key={p}>
                    <span className="material-icons" aria-hidden>
                      check
                    </span>
                    {p}
                  </li>
                ))}
              </ul>
              <Link to={r.to} className="lp-btn lp-btn-secondary">
                {r.cta} →
              </Link>
            </motion.article>
          ))}
        </Reveal>
      </div>
    </section>
  )
}
