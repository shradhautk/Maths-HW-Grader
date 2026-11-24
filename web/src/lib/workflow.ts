import { WORKFLOW_PATH } from '../config'
import { apiRequest } from './api'
import { apiCache } from './apiCache'
import type { ContentStatus, GradesStatus, QpStatus, SubmissionStatus } from './status'

/**
 * Assignment workflow API (wireframes §13–§15, §19, §20, §04): one resource
 * dispatching on `operation`, like /admin and /getgrades. Reads are cached per
 * exact query; any write clears every cached workflow read.
 *
 * The operation names and payloads below are the contract the screens are
 * built against — see web/WIREFRAMES.md "Workflow API".
 */

export type GradingMethod = 'ai' | 'standard'
export type Persona = 'balanced' | 'strict' | 'lenient'

export interface Assignment {
  assignment_id: string
  qp_id: string
  title: string
  class_name?: string
  due_at?: string
  question_count?: number
  total_marks?: number
  method?: GradingMethod
  persona?: Persona
  partial_credit?: boolean
  qp_status?: QpStatus
  content_status?: ContentStatus
  grades_status?: GradesStatus
  submitted?: number
  enrolled?: number
}

export interface ParsedQuestion {
  question_number: string | number
  subpart?: string
  question?: string
  max_marks?: number
  ocr_confidence?: number
  question_reviewed?: boolean
}

export interface RubricItem {
  question_number: string | number
  subpart?: string
  title?: string
  max_marks: number
  criteria?: string
}

export interface Submission {
  student_id: string
  student_name?: string
  status: SubmissionStatus
  submitted_at?: string
  late?: boolean
  pages?: number
  evaluation_id?: string
  total_marks?: number
  confidence?: number
  /** Open re-check request from the student (§14 step 5). */
  recheck?: { question_number: string | number; subpart?: string; note: string }[]
}

export interface FlaggedItem {
  evaluation_id: string
  student_id: string
  student_name?: string
  question_number: string | number
  subpart?: string
  confidence: number
  reference_answer?: string
  student_answer?: string
  proposed_marks: number
  max_marks: number
}

export interface StudentAssignment extends Assignment {
  submission_status?: SubmissionStatus
  submitted_at?: string
  evaluation_id?: string
}

export interface AssistantMessage {
  role: 'user' | 'assistant'
  content: string
}

/** How long a workflow read may be reused (quick back/forward navigation stays instant). */
const READ_CACHE_MS = 15_000

const READS = new Set([
  'listAssignments',
  'getAssignment',
  'getParsedQuestions',
  'getRubric',
  'listSubmissions',
  'getGradingRun',
  'listFlagged',
  'listStudentAssignments',
  'getGradingQueue',
  'listStudents',
  'getTeacherAnalytics',
  'getTeacherSettings',
  'getProgress',
  'listAnnouncements',
])

export async function workflowRequest<T>(operation: string, payload: Record<string, unknown> = {}, force = false): Promise<T> {
  if (!READS.has(operation)) {
    const data = await apiRequest<T>(WORKFLOW_PATH, { body: { operation, ...payload } })
    apiCache.clearPrefix('workflow:')
    return data
  }
  // Short-lived: other people's actions (a teacher publishing, a student submitting)
  // change what these reads return, so don't keep them for the whole tab session.
  return apiCache.withCache(
    `workflow:${operation}:${JSON.stringify(payload)}`,
    force,
    () => apiRequest<T>(WORKFLOW_PATH, { body: { operation, ...payload } }),
    READ_CACHE_MS,
  )
}

// ---- Teacher: assignments (§13, §19) ----

export const listAssignments = (force = false) =>
  workflowRequest<{ assignments?: Assignment[] }>('listAssignments', {}, force).then((d) => d?.assignments ?? [])

export const getAssignment = (assignmentId: string, force = false) =>
  workflowRequest<Assignment>('getAssignment', { assignment_id: assignmentId }, force)

/** Creates the assignment row (contentStatus=draft) and returns the ids the QP upload is tagged with. */
export const createAssignment = (input: { title: string; class_name: string; due_at?: string }) =>
  workflowRequest<{ assignment_id: string; qp_id: string }>('createAssignment', input)

export const getParsedQuestions = (assignmentId: string, force = false) =>
  workflowRequest<{ qp_status?: QpStatus; questions?: ParsedQuestion[]; paper_total?: number }>(
    'getParsedQuestions',
    { assignment_id: assignmentId },
    force,
  )

export const confirmQuestions = (assignmentId: string) =>
  workflowRequest<{ ok: boolean }>('confirmQuestions', { assignment_id: assignmentId })

export const setGradingMethod = (
  assignmentId: string,
  method: GradingMethod,
  options: { persona?: Persona; partial_credit?: boolean } = {},
) => workflowRequest<{ qp_status?: QpStatus }>('setGradingMethod', { assignment_id: assignmentId, method, ...options })

export const getRubric = (assignmentId: string, force = false) =>
  workflowRequest<{ qp_status?: QpStatus; rubric?: RubricItem[] }>('getRubric', { assignment_id: assignmentId }, force)

/** Saving moves QUESTION_PAPERS to rubric_reviewed — the teacher's fan-in gate input. */
export const approveRubric = (assignmentId: string, rubric: RubricItem[]) =>
  workflowRequest<{ qp_status?: QpStatus }>('approveRubric', { assignment_id: assignmentId, rubric })

export const startGrading = (assignmentId: string) =>
  workflowRequest<{ ok: boolean }>('startGrading', { assignment_id: assignmentId })

// ---- Teacher: grading run & review (§15, §16, §19) ----

export interface GradingRun {
  counts?: Partial<Record<SubmissionStatus, number>>
  total?: number
  submissions?: Submission[]
}

export const getGradingRun = (assignmentId: string, force = false) =>
  workflowRequest<GradingRun>('getGradingRun', { assignment_id: assignmentId }, force)

export const listSubmissions = (assignmentId: string, force = false) =>
  workflowRequest<{ submissions?: Submission[]; enrolled?: number }>('listSubmissions', { assignment_id: assignmentId }, force)

export const listFlagged = (assignmentId: string, force = false) =>
  workflowRequest<{ items?: FlaggedItem[] }>('listFlagged', { assignment_id: assignmentId }, force).then(
    (d) => d?.items ?? [],
  )

export const acceptFlagged = (item: FlaggedItem) =>
  workflowRequest<{ ok: boolean }>('acceptFlagged', {
    evaluation_id: item.evaluation_id,
    question_number: item.question_number,
    subpart: item.subpart,
  })

export const publishGrades = (assignmentId: string) =>
  workflowRequest<{ notified?: number }>('publishGrades', { assignment_id: assignmentId })

export const remindStudent = (assignmentId: string, studentId: string) =>
  workflowRequest<{ ok: boolean }>('remindStudent', { assignment_id: assignmentId, student_id: studentId })

// ---- Student (§14) ----

export const listStudentAssignments = (force = false) =>
  workflowRequest<{ assignments?: StudentAssignment[] }>('listStudentAssignments', {}, force).then(
    (d) => d?.assignments ?? [],
  )

export const requestRecheck = (
  assignmentId: string,
  evaluationId: string,
  items: { question_number: string | number; subpart?: string; note: string }[],
) => workflowRequest<{ ok: boolean }>('requestRecheck', { assignment_id: assignmentId, evaluation_id: evaluationId, items })

// ---- GradeFlow assistant (§04) ----

export const askAssistant = (context: { assignment_id?: string; evaluation_id?: string; student_id?: string }, messages: AssistantMessage[]) =>
  workflowRequest<{ reply?: string }>('askAssistant', { ...context, messages }).then((d) => d?.reply ?? '')

// ---- Teacher: grading queue, students, analytics, settings (§02, §11) ----

export type QueueState = 'review' | 'ready' | 'grading' | 'collecting' | 'setup' | 'published'

export interface QueueItem {
  assignment_id: string
  qp_id: string
  title: string
  class_name?: string
  due_at?: string
  state: QueueState
  papers: number
  graded: number
  grading: number
  flagged: number
}

export const getGradingQueue = (force = false) =>
  workflowRequest<{ queue?: QueueItem[]; waiting?: number }>('getGradingQueue', {}, force).then((d) => ({
    queue: d?.queue ?? [],
    waiting: d?.waiting ?? 0,
  }))

export interface ClassStudent {
  student_id: string
  student_name?: string
  email?: string
  classes: string[]
  assigned: number
  submitted: number
  late: number
  average: number | null
  at_risk: boolean
  last_submitted_at: string | null
  latest: { assignment_id: string; qp_id: string } | null
}

export const listStudents = (force = false) =>
  workflowRequest<{ students?: ClassStudent[]; classes?: string[] }>('listStudents', {}, force).then((d) => ({
    students: d?.students ?? [],
    classes: d?.classes ?? [],
  }))

export interface TeacherAnalytics {
  classes: string[]
  class_name: string | null
  totals: {
    assignments: number
    enrolled: number
    submitted: number
    submission_rate: number | null
    graded: number
    average: number | null
    at_risk: number
  }
  distribution: Record<'A' | 'B' | 'C' | 'D' | 'F', number>
  weekly: { date: string; label: string; count: number }[]
  assignments: {
    assignment_id: string
    qp_id: string
    title: string
    class_name?: string
    enrolled: number
    submitted: number
    graded: number
    average: number | null
    at_risk: number
    grades_status?: GradesStatus
  }[]
  hardest: { assignment_id: string; title: string; label: string; average: number }[]
  insight: string | null
  upcoming: { assignment_id: string; title: string; class_name?: string; due_at: string }[]
}

export const getTeacherAnalytics = (className: string | null, force = false) =>
  workflowRequest<TeacherAnalytics>('getTeacherAnalytics', className ? { class_name: className } : {}, force)

export interface TeacherSettings {
  default_persona: Persona
  partial_credit: boolean
  notify_submissions: boolean
  notify_flagged: boolean
}

export const getTeacherSettings = (force = false) =>
  workflowRequest<{ settings: TeacherSettings }>('getTeacherSettings', {}, force).then((d) => d.settings)

export const updateTeacherSettings = (settings: Partial<TeacherSettings>) =>
  workflowRequest<{ settings: TeacherSettings }>('updateTeacherSettings', { settings }).then((d) => d.settings)

// ---- Student: progress + announcements (§03, §12) ----

export interface ProgressResult {
  assignment_id: string
  qp_id: string
  title: string
  class_name?: string
  marks: number
  max: number
  percent: number
  date?: string
  submitted_at?: string | null
  due_at?: string
  early?: boolean | null
  on_time?: boolean | null
}

export interface Progress {
  results: ProgressResult[]
  average: number | null
  by_class: { class_name: string; average: number; count: number }[]
  todo: { assignment_id: string; qp_id: string; title: string; class_name?: string; due_at?: string }[]
  streak: number
  insight: string | null
}

export const getProgress = (force = false) => workflowRequest<Progress>('getProgress', {}, force)

export interface Announcement {
  announcement_id: string
  class_name: string
  title: string
  message: string
  author_name?: string
  created_at: string
}

export const listAnnouncements = (force = false) =>
  workflowRequest<{ announcements?: Announcement[] }>('listAnnouncements', {}, force).then((d) => d?.announcements ?? [])

export const postAnnouncement = (input: { class_name: string; title: string; message: string }) =>
  workflowRequest<{ announcement: Announcement }>('postAnnouncement', input).then((d) => d.announcement)

export const deleteAnnouncement = (announcementId: string) =>
  workflowRequest<{ ok: boolean }>('deleteAnnouncement', { announcement_id: announcementId })
