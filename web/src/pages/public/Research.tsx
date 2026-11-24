import { motion } from 'motion/react'
import FailureChart from '../../components/landing/FailureChart'
import { CtaBand, LandingPage, PageHero, Reveal, SectionHead } from '../../components/landing/LandingKit'
import { ACCURACY, EIGEN_DET_5X5, FAILURE_TYPES, SIZES } from '../../components/landing/linalgBench'
import { fadeUp } from '../../components/landing/motion'
import { RESEARCH } from '../../components/landing/site'

/**
 * Research page. Content from the LinAlg-Bench dataset card and its published
 * results (see components/landing/linalgBench.ts). Owner: Shradha. The dataset card
 * asks that results not be used for marketing or model rankings, so this page
 * reports all-model totals only and makes no claim about GradeAnt's own accuracy.
 */

const count = (type: string, size: (typeof SIZES)[number]) => FAILURE_TYPES.find((f) => f.type === type)!.counts[size]

const FINDINGS = [
  `Accuracy falls as matrices grow: across all 10 models, ${ACCURACY['3×3'].percent}% of answers were correct on 3×3 problems, ${ACCURACY['4×4'].percent}% on 4×4 and ${ACCURACY['5×5'].percent}% on 5×5.`,
  `Failures cluster by operation. At 5×5, eigenvalue and determinant problems account for ${EIGEN_DET_5X5} of the ${ACCURACY['5×5'].fail} failures.`,
  `The main kind of mistake changes with size: sign errors lead at 4×4 (${count('Sign error', '4×4')} of ${ACCURACY['4×4'].fail}), while hallucinated values or steps lead at 5×5 (${count('Hallucination', '5×5')} of ${ACCURACY['5×5'].fail}).`,
  'The paper concludes that these failures are structured by algorithm family and matrix size rather than random.',
]

const DETAILS = [
  ['660 problems', '220 each at 3×3, 4×4 and 5×5, with ground truth checked in SymPy'],
  ['9 operation types', 'Including determinant, eigenvalues, rank, nullity, matrix power and multiplication'],
  ['10 frontier models', '6,600 responses in total, at temperature 0 and without tools'],
  ['Forensic error labels', 'Each failure classified by a three-stage automated pipeline with a meta-auditor'],
]

const LIMITS = [
  'Linear algebra only; other areas of math are an open question.',
  'A snapshot: all model runs were made in May 2026.',
  'Matrices up to 5×5; larger sizes were not tested.',
  'Problems use small whole numbers from −10 to 10, which may not match real coursework.',
]

export default function Research() {
  const b = RESEARCH.linalgBench
  const links = [
    { href: b.arxiv, label: 'Read the paper on arXiv', icon: 'article' },
    { href: b.huggingFace, label: 'Dataset on Hugging Face', icon: 'dataset' },
    { href: b.github, label: 'Code on GitHub', icon: 'code' },
  ]

  return (
    <LandingPage
      title="Research"
      description="LinAlg-Bench, our benchmark of 660 linear algebra problems across 10 frontier AI models, shows where step-by-step math reasoning breaks down as matrices grow."
    >
      <PageHero
        eyebrow="Research"
        title="Where AI goes wrong in step-by-step math"
        lead="Grading worked answers means knowing where reasoning breaks down. LinAlg-Bench, our benchmark, measures exactly that across 10 frontier AI models."
      >
        <a href={b.arxiv} target="_blank" rel="noreferrer" className="lp-btn lp-btn-primary lp-btn-lg">
          Read the paper
        </a>
        <a href={b.huggingFace} target="_blank" rel="noreferrer" className="lp-btn lp-btn-secondary lp-btn-lg">
          Get the dataset
        </a>
      </PageHero>

      <section className="lp-section" style={{ paddingTop: 8 }}>
        <div className="lp-container">
          <Reveal className="lp-paper-card">
            <motion.span className="lp-eyebrow" variants={fadeUp}>
              Paper · {b.year}
            </motion.span>
            <motion.h2 variants={fadeUp}>{b.title}</motion.h2>
            <motion.p className="lp-paper-meta" variants={fadeUp}>
              {b.authors}. {b.venue}.
            </motion.p>
            <motion.p variants={fadeUp}>
              {b.name} tests 10 frontier language models on linear algebra problems that get harder in one controlled way: the
              matrices grow from 3×3 to 4×4 to 5×5. Every wrong answer is then classified by the kind of mistake, so the benchmark
              shows not just how often models fail but how.
            </motion.p>
            <motion.div className="lp-hero-ctas" variants={fadeUp}>
              {links.map((l) => (
                <a key={l.href} href={l.href} target="_blank" rel="noreferrer" className="lp-btn lp-btn-secondary">
                  <span className="material-icons" aria-hidden>
                    {l.icon}
                  </span>
                  {l.label}
                </a>
              ))}
            </motion.div>
          </Reveal>
        </div>
      </section>

      <section className="lp-section lp-section-soft" aria-labelledby="acc-title">
        <div className="lp-container">
          <SectionHead eyebrow="Key result" title={<span id="acc-title">Accuracy drops as matrices grow</span>} text="Share of correct answers across all 10 models, 2,200 responses per size." />
          <Reveal className="lp-acc">
            {SIZES.map((s) => (
              <motion.div key={s} className="lp-acc-tile" variants={fadeUp}>
                <span>{s} matrices</span>
                <b>{ACCURACY[s].percent}%</b>
                <small>
                  {ACCURACY[s].pass.toLocaleString('en-US')} of {ACCURACY[s].total.toLocaleString('en-US')} correct
                </small>
              </motion.div>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="lp-section">
        <Reveal className="lp-container lp-split lp-research-split">
          <motion.div variants={fadeUp}>
            <span className="lp-eyebrow">Findings</span>
            <h2>Mistakes follow a pattern</h2>
            <ul className="lp-checks lp-findings">
              {FINDINGS.map((f) => (
                <li key={f}>
                  <span className="material-icons" aria-hidden>
                    check_circle
                  </span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </motion.div>
          <motion.div variants={fadeUp}>
            <FailureChart />
          </motion.div>
        </Reveal>
      </section>

      <section className="lp-section lp-section-soft">
        <div className="lp-container">
          <SectionHead eyebrow="The benchmark" title="What is in LinAlg-Bench" />
          <Reveal className="lp-cards is-4">
            {DETAILS.map(([t, d]) => (
              <motion.article key={t} className="lp-feature lp-tone-accent" variants={fadeUp}>
                <h3>{t}</h3>
                <p>{d}</p>
              </motion.article>
            ))}
          </Reveal>
          <Reveal className="lp-limits">
            <motion.h3 variants={fadeUp}>Limitations</motion.h3>
            <motion.ul variants={fadeUp}>
              {LIMITS.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </motion.ul>
            <motion.p variants={fadeUp}>
              The benchmark is a diagnostic for research, not a ranking of models. The dataset is released under {b.license}.
            </motion.p>
          </Reveal>
        </div>
      </section>

      <section className="lp-section">
        <div className="lp-container lp-split">
          <div>
            <span className="lp-eyebrow">From research to product</span>
            <h2>Why it matters for grading</h2>
            <p>
              Sign slips, arithmetic errors and numbers copied wrongly are the same kinds of mistakes students make in handwritten
              working. Knowing where AI models also go wrong is why GradeAnt checks each step separately and sends anything it is
              unsure of to the teacher instead of guessing.
            </p>
          </div>
          <div>
            <span className="lp-eyebrow">Cite this work</span>
            <pre className="lp-bibtex">
              <code>{b.bibtex}</code>
            </pre>
          </div>
        </div>
      </section>

      <CtaBand />
    </LandingPage>
  )
}
