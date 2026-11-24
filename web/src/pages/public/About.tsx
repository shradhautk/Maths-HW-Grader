import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { CtaBand, LandingPage, PageHero, Reveal, SectionHead } from '../../components/landing/LandingKit'
import { fadeUp } from '../../components/landing/motion'
import { FUNDERS } from '../../components/landing/site'

const TEAM = [
  {
    name: 'Dr. Shradha Agarwal',
    role: 'Head of AI',
    subject: 'Physics',
    photo: '/img/shradha.webp',
    bio: 'Head of ML/AI at a stealth startup, leading LLM work in RAG, prompt engineering, subject classification and OCR-based image analysis. Specialises in NLP, computer vision and generative AI.',
  },
  {
    name: 'Dr. Nilanjan Chakraborty',
    role: 'Assistant Professor, Mathematics & Statistics',
    subject: 'Mathematics & Statistics',
    photo: '/img/nirlanjan_chakarborty.webp',
    bio: 'Assistant Professor of Mathematics and Statistics at Missouri S&T, focusing on advanced statistical inference and AI.',
  },
  {
    name: 'Dr. Donald C. Wunsch',
    role: 'Professor, Electrical & Computer Engineering',
    subject: 'Electrical & Computer Engineering',
    photo: '/img/Don_Wunsh.webp',
    bio: 'Distinguished professor and director of the Kummer Centre at Missouri S&T, specialising in neural networks, reinforcement learning and XAI.',
  },
  {
    name: 'Dr. Beth Kania-Gosche',
    role: 'Professor',
    subject: 'Education & Cognitive Science',
    photo: '/img/Kania_beth.webp',
    bio: 'Chair of Education at Missouri S&T, with expertise in STEM education and teacher preparation programmes.',
  },
  {
    name: 'Tariq Jamil',
    role: 'Head of Pedagogy',
    photo: '/img/tariq.webp',
    bio: 'Former principal and curriculum specialist focused on assessment strategies.',
  },
  {
    name: 'Deepak Rajbhar',
    role: 'AI Engineer',
    photo: '/img/deepak.webp',
    bio: 'AI/ML engineer for EdTech and AI.',
  },
  {
    name: 'Narayan Adhikari',
    role: 'AI Engineer',
    photo: '/img/narayan.webp',
    bio: 'AI researcher specialising in natural language processing and education.',
  },
]

const VALUES = [
  { icon: 'school', title: 'Teacher in control', text: 'GradeAnt drafts; the teacher decides. No grade reaches a student without a teacher publishing it.', tone: 'brand' },
  { icon: 'science', title: 'Research first', text: 'We test where AI grading goes wrong before we rely on it, and publish what we find.', tone: 'accent' },
  { icon: 'fact_check', title: 'Only true claims', text: 'We say what GradeAnt does today, for the subjects we have tested, and nothing more.', tone: 'brand' },
  { icon: 'shield', title: 'Student privacy', text: 'Student work is visible only to the teachers of that course, and deleted on request.', tone: 'accent' },
] as const

function Person({ p }: { p: (typeof TEAM)[number] }) {
  return (
    <motion.article className="lp-person" variants={fadeUp} whileHover={{ y: -6 }}>
      <img src={p.photo} alt={`Portrait of ${p.name}`} loading="lazy" width={240} height={240} />
      <h3>{p.name}</h3>
      <div className="lp-role-tag">{p.role}</div>
      {p.subject && <span className="lp-subject">{p.subject}</span>}
      <p>{p.bio}</p>
    </motion.article>
  )
}

export default function About() {
  return (
    <LandingPage
      title="About"
      description="GradeAnt was founded in 2024 at Missouri S&T by a team of seven researchers, educators and engineers building AI that grades handwritten math and physics."
    >
      <PageHero
        eyebrow="About GradeAnt"
        title="Giving teachers their time back"
        lead="We build AI that does the first pass of grading handwritten math and physics, so teachers can spend their time on feedback and teaching."
      />

      <section className="lp-section" style={{ paddingTop: 24 }}>
        <Reveal className="lp-container lp-split">
          <motion.div className="lp-photo" variants={fadeUp}>
            <img src="/img/group.webp" alt="Portraits of the seven GradeAnt team members, each with their name" loading="lazy" width={1280} height={960} />
            <span className="lp-photo-badge">Founded in 2024 at Missouri S&amp;T</span>
          </motion.div>
          <motion.div variants={fadeUp}>
            <span className="lp-eyebrow">Our story</span>
            <h2>Born from too many late nights of grading</h2>
            <p>
              GradeAnt began in 2024 at Missouri S&amp;T in Rolla, Missouri, when a group of educators and AI researchers set out to
              tackle the grading bottleneck in STEM courses.
            </p>
            <p>
              Grading handwritten working takes hours, and students often wait a week for feedback. We are building an AI assistant
              that reads that working, marks it step by step against the teacher's rubric, and leaves the final call to the teacher.
            </p>
            <p>Today we are piloting GradeAnt with math and physics courses.</p>
          </motion.div>
        </Reveal>
      </section>

      <section className="lp-section lp-section-soft">
        <div className="lp-container">
          <SectionHead
            eyebrow="The team"
            title="Seven researchers, educators and engineers"
            text="One team from Missouri S&T and beyond, building AI that understands how math and physics are taught."
          />
          <Reveal className="lp-cards is-4">
            {TEAM.map((p) => (
              <Person key={p.name} p={p} />
            ))}
          </Reveal>
        </div>
      </section>

      <section className="lp-section">
        <div className="lp-container">
          <SectionHead eyebrow="Our values" title="What we stand for" />
          <Reveal className="lp-cards is-4">
            {VALUES.map((v) => (
              <motion.article key={v.title} className={`lp-icon-card lp-tone-${v.tone}`} variants={fadeUp} whileHover={{ y: -6 }}>
                <span className="lp-feature-icon material-icons" aria-hidden>
                  {v.icon}
                </span>
                <h3>{v.title}</h3>
                <p>{v.text}</p>
              </motion.article>
            ))}
          </Reveal>
        </div>
      </section>

      {FUNDERS.length > 0 && (
        <section className="lp-section lp-section-soft" aria-labelledby="backed-by">
          <div className="lp-container">
            <h2 id="backed-by" className="lp-backed-title">
              Backed by
            </h2>
            <ul className="lp-backed">
              {FUNDERS.map((f) => (
                <li key={f.name}>
                  <img src={f.logo} alt={f.name} height={40} loading="lazy" />
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      <CtaBand />
      <section className="lp-section lp-careers-band">
        <div className="lp-container">
          <p>
            Want to work with us? See <Link to="/careers">open roles</Link>.
          </p>
        </div>
      </section>
    </LandingPage>
  )
}
