import { API } from '../config'
import { apiRequest, unwrapLambda } from './api'
import { apiCache } from './apiCache'

/** One graded question as returned by the getgrades Lambda. */
export interface QuestionDetail {
  evaluation_id?: string
  question_number?: string | number
  subpart?: string
  question?: string
  student_answer?: string
  feedback?: string
  marks?: number | string
  max_marks?: number | string
  timestamp?: number
  [key: string]: unknown
}

export interface GradeSummary {
  student_id?: string
  assignment_id?: string
  total_marks?: number | string
  evaluation_id?: string
  qp_id?: string
  timestamp?: number
  [key: string]: unknown
}

export interface GradeResult {
  summary?: GradeSummary[]
  details?: QuestionDetail[]
  [key: string]: unknown
}

export type MarksStatus = 'correct' | 'partial' | 'incorrect' | 'unscored'

/**
 * POST to the getgrades Lambda (dispatches on `operation`). Reads are cached
 * per exact query for the tab, so re-searching or returning to a page
 * doesn't re-fetch.
 */
export async function gradesRequest<T = GradeResult>(
  payload: Record<string, unknown>,
  opts: { force?: boolean } = {},
): Promise<T> {
  const key = `grades:${JSON.stringify(payload)}`
  const data = await apiCache.withCache(key, !!opts.force, () => apiRequest<T>(API.grades,{ body: payload }))
  // Entries cached by the old static pages still hold the raw Lambda envelope.
  return unwrapLambda<T>(data)
}

export function toNumber(v: unknown): number {
  const n = parseFloat(String(v))
  return Number.isFinite(n) ? n : 0
}

/** Correct/partial/incorrect from marks vs max_marks the API already returns. */
export function getMarksStatus(marks: unknown, maxMarks: unknown): MarksStatus {
  const obtained = toNumber(marks)
  const max = toNumber(maxMarks)
  if (max <= 0) return 'unscored'
  if (obtained >= max) return 'correct'
  if (obtained <= 0) return 'incorrect'
  return 'partial'
}

export const STATUS_BADGE: Record<MarksStatus, string> = {
  correct: '✓ Correct',
  partial: '~ Partially correct',
  incorrect: '✕ Needs correction',
  unscored: '',
}

export function getGradeLetter(percent: number): { letter: string; cssVar: string } {
  if (percent >= 90) return { letter: 'A', cssVar: '--grade-a' }
  if (percent >= 80) return { letter: 'B', cssVar: '--grade-b' }
  if (percent >= 70) return { letter: 'C', cssVar: '--grade-c' }
  if (percent >= 60) return { letter: 'D', cssVar: '--grade-d' }
  return { letter: 'F', cssVar: '--grade-f' }
}

/** Sorts by question number, then subpart (questions without a subpart first). */
export function sortDetails(details: QuestionDetail[]): QuestionDetail[] {
  return [...details].sort((a, b) => {
    const diff = (parseInt(String(a.question_number)) || 0) - (parseInt(String(b.question_number)) || 0)
    if (diff !== 0) return diff
    if (a.subpart && b.subpart) return a.subpart.localeCompare(b.subpart)
    if (a.subpart) return 1
    if (b.subpart) return -1
    return 0
  })
}

export function maxMarksOf(details: QuestionDetail[] | undefined): number {
  return (details ?? []).reduce((sum, q) => sum + toNumber(q.max_marks), 0)
}

/** Unix seconds or milliseconds -> locale string ('' if missing/invalid). */
export function formatTimestamp(ts: unknown): string {
  const n = Number(ts)
  if (!ts || !Number.isFinite(n)) return ''
  const date = new Date(n < 1e12 ? n * 1000 : n)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString()
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

/**
 * Model output -> safe HTML. Handles the grader's LaTeX "\textbf{Heading:} \quad
 * \text{...}" sections and light Markdown (bold, ## headings, numbered bold
 * items). Everything is escaped before any markup is added; math delimiters are
 * left for MathJax.
 */
export function renderContent(text: string | undefined | null): string {
  if (!text) return ''

  if (text.includes('\\textbf{') && text.includes('}:} \\quad \\text{')) {
    const sections = text.match(/\[ \\textbf\{([^}]+)\}:} \\quad \\text\{([^}]+)\} \]/g) ?? []
    if (sections.length > 0) {
      return sections
        .map((section) => {
          const heading = escapeHtml(section.match(/\\textbf\{([^}]+)\}/)?.[1] ?? '')
          let content = escapeHtml(section.match(/\\quad \\text\{([^}]+)\}/)?.[1] ?? '')
          content = content.replace(/\\begin\{itemize\}(.*?)\\end\{itemize\}/gs, (_m, items: string) => {
            const li = items
              .split('\\item')
              .filter((i) => i.trim())
              .map((i) => `<li>${i.trim()}</li>`)
            return `<ul>${li.join('')}</ul>`
          })
          content = content.replace(/\\\(([^)]+)\\\)/g, '$$$1$$')
          return `<div class="feedback-section"><h4>${heading}:</h4><div class="feedback-content">${content}</div></div>`
        })
        .join('')
    }
  }

  return escapeHtml(text)
    .replace(/^(\d+)\.\s*\*\*(.*?)\*\*/gm, '<br>$1. <strong>$2</strong>')
    .replace(/^-\s*\*\*(.*?)\*\*/gm, '<br>- <strong>$1</strong>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/^## (.*?)$/gm, '<br><h2>$1</h2>')
    .replace(/^### (.*?)$/gm, '<br><h3>$1</h3>')
}
