import { motion } from 'motion/react'
import { LandingPage, PageHero, Reveal } from '../../components/landing/LandingKit'
import { fadeUp } from '../../components/landing/motion'
import { OPEN_ROLES, SOCIAL } from '../../components/landing/site'

const LINKEDIN = SOCIAL.find((s) => s.label === 'LinkedIn')!

export default function Careers() {
  return (
    <LandingPage title="Careers" description="Open roles at GradeAnt, an AI grading startup founded at Missouri S&T. Follow GradeAnt on LinkedIn for new roles.">
      <PageHero
        eyebrow="Careers"
        title="Work on AI that helps teachers"
        lead="GradeAnt is a small team from Missouri S&T building AI that grades handwritten math and physics. We post every opening on LinkedIn."
      />

      <section className="lp-section" style={{ paddingTop: 8 }}>
        <div className="lp-container">
          {OPEN_ROLES.length > 0 ? (
            <Reveal className="lp-roles-list" as="ul">
              {OPEN_ROLES.map((r) => (
                <motion.li key={r.href} variants={fadeUp}>
                  <div>
                    <h2>{r.title}</h2>
                    <span>
                      {r.type} · {r.location}
                    </span>
                  </div>
                  <a href={r.href} target="_blank" rel="noreferrer" className="lp-btn lp-btn-secondary">
                    View on LinkedIn →
                  </a>
                </motion.li>
              ))}
            </Reveal>
          ) : (
            <Reveal className="lp-empty">
              <motion.p variants={fadeUp}>There are no open roles right now.</motion.p>
            </Reveal>
          )}

          <div className="lp-follow">
            <img src={LINKEDIN.icon} alt="" width={28} height={28} />
            <div>
              <b>Follow us on LinkedIn for new roles</b>
              <span>New openings are posted there first.</span>
            </div>
            <a href={LINKEDIN.href} target="_blank" rel="noreferrer" className="lp-btn lp-btn-primary">
              Follow GradeAnt
            </a>
          </div>
        </div>
      </section>
    </LandingPage>
  )
}
