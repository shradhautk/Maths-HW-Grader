/**
 * The two status chains from the wireframes (§20) — every workflow screen
 * reads one of these from the backend instead of assuming a UI flow.
 *
 * QUESTION_PAPERS (teacher artifact) and SUBMISSIONS (student artifact) only
 * meet at the fan-in gate: grading fires once a question is `rubric_reviewed`
 * on the teacher side AND `corrected` on the student side.
 */

export const QP_STATUSES = [
  'upload_requested',
  'qp_uploaded',
  'qp_parsed',
  'rubric_generating',
  'rubric_generated',
  'rubric_reviewed',
] as const
export type QpStatus = (typeof QP_STATUSES)[number]

export const SUBMISSION_STATUSES = [
  'upload_requested',
  'answer_uploaded',
  'ocr_done',
  'corrected',
  'grading_triggered',
  'ai_graded',
  'queued_for_review',
  'teacher_reviewed',
  'published',
  'published_revised',
] as const
export type SubmissionStatus = (typeof SUBMISSION_STATUSES)[number]

export type Tone = 'neutral' | 'info' | 'brand' | 'warning' | 'success' | 'danger'

export const STATUS_META: Record<QpStatus | SubmissionStatus, { label: string; tone: Tone }> = {
  upload_requested: { label: 'Awaiting upload', tone: 'neutral' },
  qp_uploaded: { label: 'Uploaded', tone: 'info' },
  qp_parsed: { label: 'Questions parsed', tone: 'info' },
  rubric_generating: { label: 'Drafting rubric…', tone: 'brand' },
  rubric_generated: { label: 'Rubric ready to review', tone: 'warning' },
  rubric_reviewed: { label: 'Rubric approved', tone: 'success' },
  answer_uploaded: { label: 'Submitted', tone: 'info' },
  ocr_done: { label: 'Handwriting read', tone: 'info' },
  corrected: { label: 'Answers confirmed', tone: 'success' },
  grading_triggered: { label: 'Grading…', tone: 'brand' },
  ai_graded: { label: 'AI graded', tone: 'brand' },
  queued_for_review: { label: 'Flagged for review', tone: 'warning' },
  teacher_reviewed: { label: 'Teacher reviewed', tone: 'success' },
  published: { label: 'Published', tone: 'success' },
  published_revised: { label: 'Revised', tone: 'success' },
}

export function statusMeta(status: string | undefined): { label: string; tone: Tone } {
  return (status && STATUS_META[status as SubmissionStatus]) || { label: status || 'Unknown', tone: 'neutral' }
}

const rank = (status: string | undefined) => SUBMISSION_STATUSES.indexOf(status as SubmissionStatus)

/** True once a submission has reached (or passed) `target` in the chain. */
export function reached(status: string | undefined, target: SubmissionStatus): boolean {
  const r = rank(status)
  return r >= 0 && r >= rank(target)
}

export const isPublished = (s: string | undefined) => s === 'published' || s === 'published_revised'

/**
 * The student's §14 timeline collapses the chain to what they care about:
 * submitted → handwriting read → grading → released.
 */
export function studentTimeline(status: string | undefined) {
  return [
    { label: 'Submitted', done: reached(status, 'answer_uploaded') },
    { label: 'Handwriting recognised', done: reached(status, 'ocr_done') },
    { label: 'GradeFlow™ grading', done: reached(status, 'ai_graded') },
    { label: 'Teacher review & release', done: isPublished(status) },
  ]
}

/** Assignment roll-ups (§20): contentStatus and gradesStatus. */
export type ContentStatus = 'draft' | 'published'
export type GradesStatus = 'not_started' | 'ai_graded' | 'teacher_reviewing' | 'published'

export const GRADES_STATUS_META: Record<GradesStatus, { label: string; tone: Tone }> = {
  not_started: { label: 'Not graded', tone: 'neutral' },
  ai_graded: { label: 'AI graded', tone: 'brand' },
  teacher_reviewing: { label: 'In review', tone: 'warning' },
  published: { label: 'Published', tone: 'success' },
}
