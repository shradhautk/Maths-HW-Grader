/** Contact details shown across the marketing pages. */
export const CONTACT = {
  email: 'contact@gradeant.com',
  phone: '+1-573-341-4720',
  // Rolla, MO is on Central time.
  hours: 'Mon–Fri, 9 AM–5 PM Central',
  address: '222 Fulton Hall, 301 W. 14th Street, Rolla MO 65409',
}

/** Public site URL: canonical links, sharing previews, sitemap. */
export const SITE_URL = 'https://grade-ant.com'

/** Social accounts. Only list accounts that exist (no "#" placeholders). */
export const SOCIAL = [{ label: 'LinkedIn', href: 'https://www.linkedin.com/company/gradeant/', icon: '/img/icon-linkedin.svg' }]

/** Research the product is built on (Research page + the home "Built on research" strip). */
export const RESEARCH = {
  linalgBench: {
    name: 'LinAlg-Bench',
    title: 'LinAlg-Bench: A Forensic Benchmark Revealing Structural Failure Modes in LLM Mathematical Reasoning',
    authors: 'Shradha Agarwal, Deepak Rajbhar, Tariq J',
    year: 2026,
    venue: 'arXiv preprint arXiv:2605.16675. Short version at the MATH-AI Workshop, NeurIPS 2026',
    arxiv: 'https://arxiv.org/abs/2605.16675',
    huggingFace: 'https://huggingface.co/datasets/LinAlgBench/linalg-bench',
    github: 'https://github.com/linalgbench2026/LinAlgBench',
    openReview: 'https://openreview.net/forum?id=fzVJyQ33Td',
    license: 'CC BY 4.0',
    bibtex: `@article{agarwal2026linalgbench,
  title={LinAlg-Bench: A Forensic Benchmark Revealing Structural Failure Modes in LLM Mathematical Reasoning},
  author={Agarwal, Shradha and Rajbhar, Deepak and J, Tariq},
  journal={arXiv preprint arXiv:2605.16675},
  year={2026}
}`,
  },
}

/**
 * Real product media. Each is null until the asset exists; until then the page
 * shows a drawn illustration of that screen instead (never a pitch-deck slide).
 * Drop files in public/img/product/ (WebP, under 300 KB, 1600 px wide, no student
 * names or faces without written consent) and set the path here.
 */
export const MEDIA: Record<'hero' | 'upload' | 'rubric' | 'grading' | 'review' | 'beforeAfter' | 'demoPoster', string | null> & {
  demoVideo: string | null
  demoCaptions: string | null
} = {
  hero: null, // split view: scanned answer (left) + step-by-step marks and feedback (right)
  upload: null,
  rubric: null,
  grading: null,
  review: null,
  beforeAfter: null, // messy handwritten derivation next to the feedback it produced
  demoVideo: null, // 30–60 s MP4, no sound
  demoPoster: null, // WebP poster frame for the video
  demoCaptions: null, // WebVTT captions for the video
}

/**
 * "Backed by" strip on the About page. Add a funder only after they have agreed
 * to be named, with a grey logo in public/img/funders/. No amounts.
 * Pending OK: Missouri S&T, IGI, CSTS, State of Missouri.
 */
export const FUNDERS: { name: string; logo: string }[] = []

/**
 * Open roles on the Careers page; each links to the GradeAnt LinkedIn job post.
 * Empty list → the page says there are no open roles right now.
 */
export const OPEN_ROLES: { title: string; type: string; location: string; href: string }[] = []
