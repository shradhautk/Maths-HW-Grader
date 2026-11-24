// Shared data helpers for the assignment workflow (port of workflow.py's
// "Plumbing" and "Data helpers" sections).
//
// A submission's status (wireframe §20 SUBMISSIONS chain) is derived from its
// assignment_grading rows (answer rows → ocr_done, every row corrected →
// corrected, a summary row → ai_graded, an unresolved low-confidence row →
// queued_for_review) and merged with the statuses the workflow records itself
// (grading_triggered, teacher_reviewed, published, published_revised). The
// later of the two wins.
import config from '../../config/index.js'
import { badRequest, forbidden, notFound } from '../../lib/errors.js'
import { isoUtc, num, pyMin, pyStr, sortBy, sum, truthy } from '../../lib/py.js'
import * as grading from '../../repositories/grading.repository.js'
import { listDirectory } from '../../repositories/users.repository.js'
import * as wf from '../../repositories/workflow.repository.js'

export const SUBMISSION_CHAIN = [
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
]
export const RANK = Object.fromEntries(SUBMISSION_CHAIN.map((s, i) => [s, i]))
export const rank = (status) => (Object.hasOwn(RANK, status) ? RANK[status] : 0)
export const PERSONAS = ['balanced', 'strict', 'lenient']
export const AT_RISK_PERCENT = 50
export const DEFAULT_SETTINGS = { default_persona: 'balanced', partial_credit: true, notify_submissions: true, notify_flagged: true }

export const now = () => isoUtc()

/** _require: the caller must have one of these roles. */
export function requireRole(caller, ...roles) {
  if (!roles.includes(caller.role)) throw forbidden("Your role can't do that.")
}

/** _required: values of fields that must not be missing/null/empty-string. */
export function required(body, ...fields) {
  const missing = fields.filter((f) => body[f] === undefined || body[f] === null || body[f] === '')
  if (missing.length) throw badRequest(`Missing required field(s): ${missing.join(', ')}`)
  return fields.map((f) => body[f])
}

/** Assignments the caller manages (all of them for an admin). */
export const myAssignments = (caller) =>
  caller.role === 'admin' ? wf.scanAssignments() : wf.assignmentsByTeacher(caller.user_id)

/**
 * Published assignments of the student's own classes (Users.classes). A student with no
 * classes sees none: falling back to every school's assignments leaked them across schools.
 */
export async function visibleAssignments(caller) {
  const classes = Array.isArray(caller.user.classes) ? caller.user.classes : []
  if (!classes.length) return []
  const published = await wf.publishedAssignments()
  return published.filter((a) => classes.includes(a.class_name))
}

export async function loadAssignment(assignmentId, caller) {
  const item = await wf.getAssignment(assignmentId)
  if (!item) throw notFound('Assignment not found.')
  if (caller && caller.role === 'teacher' && item.teacher_id !== caller.user_id) {
    throw forbidden('This assignment belongs to a different teacher.')
  }
  return item
}

export const studentDetails = (studentId, assignment) =>
  grading.studentRows(studentId, { assignmentId: assignment.assignment_id, qpId: assignment.qp_id, type: 'notSummary' })

/** Every assignment_grading row for this assignment, grouped by student: Map<student_id, {summary, details}>. */
export async function gradingRows(assignment, users) {
  const rows = await grading.assignmentRows(assignment.assignment_id, assignment.qp_id)
  if (rows !== null) {
    const byStudent = new Map()
    for (const r of rows) {
      if (!byStudent.has(r.student_id)) byStudent.set(r.student_id, { summary: null, details: [] })
      const entry = byStudent.get(r.student_id)
      if (r.type === 'summary') entry.summary = r
      else entry.details.push(r)
    }
    return byStudent
  }

  // No assignment_id index: find candidate students (graded, recorded by the
  // workflow, or enrolled via Users.classes) and read their rows.
  const summaries = await grading.summaryRows(assignment.assignment_id, assignment.qp_id)
  const candidates = new Set(summaries.map((s) => s.student_id))
  for (const s of await wf.submissionsForAssignment(assignment.assignment_id)) candidates.add(s.student_id)
  for (const u of enrolledStudents(assignment, users ?? (await usersById()))) candidates.add(u.user_id)
  const byStudent = new Map()
  for (const studentId of candidates) {
    const details = await studentDetails(studentId, assignment)
    const summary = summaries.find((s) => s.student_id === studentId) ?? null
    if (details.length || summary !== null) byStudent.set(studentId, { summary, details })
  }
  return byStudent
}

export function isFlagged(row) {
  const conf = row.confidence
  return conf !== undefined && conf !== null && num(conf, 1) < config.grading.flagConfidence && !truthy(row.flag_resolved_at)
}

export function derivedStatus(entry) {
  const { details, summary } = entry
  if (summary !== null) return details.some(isFlagged) ? 'queued_for_review' : 'ai_graded'
  if (details.length) return details.every((r) => r.status === 'corrected') ? 'corrected' : 'ocr_done'
  return null
}

export function mergeStatus(derived, stored) {
  const candidates = [derived, stored].filter((s) => typeof s === 'string' && Object.hasOwn(RANK, s))
  if (!candidates.length) return 'upload_requested'
  const best = candidates.reduce((a, b) => (RANK[b] > RANK[a] ? b : a))
  // A paper that's flagged again after the teacher reviewed it goes back to the queue.
  if (derived === 'queued_for_review' && ['teacher_reviewed', 'grading_triggered', 'ai_graded'].includes(stored)) {
    return 'queued_for_review'
  }
  return best
}

/** Map<user_id, user> for everyone in the Users table. */
export async function usersById() {
  const users = await listDirectory()
  return new Map(users.filter((u) => truthy(u.user_id)).map((u) => [u.user_id, u]))
}

/** Students whose Users.classes lists this assignment's class. */
export function enrolledStudents(assignment, users) {
  const cls = assignment.class_name
  return [...users.values()].filter(
    (u) => u.userRole === 'student' && truthy(cls) && (Array.isArray(u.classes) ? u.classes : []).includes(cls),
  )
}

/** One canonical copy of each parsed question (question text is denormalised onto every student's rows). */
export async function questions(assignment, byStudent) {
  const rows = byStudent ?? (await gradingRows(assignment))
  const out = new Map()
  for (const entry of rows.values()) {
    for (const r of entry.details) {
      const key = JSON.stringify([pyStr(r.question_number), r.subpart || ''])
      if (!out.has(key)) {
        out.set(key, {
          question_number: r.question_number ?? null,
          subpart: r.subpart ?? null,
          question: r.question ?? null,
          max_marks: r.max_marks ?? null,
          ocr_confidence: r.question_confidence ?? null,
          question_reviewed: truthy(r.question_reviewed_at),
        })
      }
    }
  }
  return sortBy([...out.values()], (q) => [num(q.question_number, 0), pyStr(q.subpart || '')])
}

export const marksTotal = (qs) => sum(qs.map((q) => num(q.max_marks)))

export async function assignmentView(item, byStudent) {
  const rows = byStudent ?? (await gradingRows(item))
  const qs = await questions(item, rows)
  let qpStatus = item.qp_status ?? 'qp_uploaded'
  if (qs.length && ['upload_requested', 'qp_uploaded'].includes(qpStatus)) qpStatus = 'qp_parsed'
  let submitted = 0
  for (const entry of rows.values()) if (entry.details.length || entry.summary !== null) submitted += 1
  const view = Object.fromEntries(Object.entries(item).filter(([k]) => k !== 'rubric' && k !== 'teacher_id'))
  return {
    ...view,
    qp_status: qpStatus,
    question_count: qs.length || (item.question_count ?? null),
    total_marks: marksTotal(qs) || (item.total_marks ?? null),
    submitted,
  }
}

/** assignment_grading timestamps are epoch seconds/ms; workflow ones are ISO strings. */
export function iso(value) {
  if (typeof value === 'number' || (typeof value === 'string' && /^\d+$/.test(value))) {
    const n = Number(value)
    return isoUtc(new Date(n > 1e12 ? n : n * 1000))
  }
  return pyStr(value)
}

/** Earliest row timestamp, or '' (min over timestamp/created_at, like workflow.py). */
export const firstRowTime = (rows) => pyMin(rows.map((r) => r.timestamp || r.created_at || ''), '')

/** Roster rows for one assignment, plus the grading rows and users they were built from. */
export async function submissions(item, users) {
  const people = users ?? (await usersById())
  const byStudent = await gradingRows(item, people)
  const stored = new Map((await wf.submissionsForAssignment(item.assignment_id)).map((s) => [s.student_id, s]))
  for (const u of enrolledStudents(item, people)) {
    if (!byStudent.has(u.user_id)) byStudent.set(u.user_id, { summary: null, details: [] })
  }
  const due = item.due_at
  const out = []
  for (const [studentId, entry] of byStudent) {
    const sub = stored.get(studentId) ?? {}
    const status = mergeStatus(derivedStatus(entry), sub.status)
    const rows = entry.details
    const submittedAt = sub.submitted_at || firstRowTime(rows) || null
    const row = {
      student_id: studentId,
      student_name: people.get(studentId)?.name ?? null,
      status,
      pages: sub.pages ?? null,
    }
    if (truthy(submittedAt)) {
      row.submitted_at = iso(submittedAt)
      if (truthy(due) && row.submitted_at.slice(0, 10) > pyStr(due).slice(0, 10)) row.late = true
    }
    if (entry.summary !== null) {
      row.evaluation_id = entry.summary.evaluation_id ?? null
      row.total_marks = entry.summary.total_marks ?? null
    }
    if (sub.recheck_status === 'open') row.recheck = sub.recheck ?? null
    const confs = rows.filter((r) => r.confidence !== undefined && r.confidence !== null).map((r) => num(r.confidence))
    if (confs.length) row.confidence = Math.min(...confs)
    out.push(row)
  }
  return { subs: sortBy(out, (r) => (r.student_name || r.student_id).toLowerCase()), byStudent, users: people }
}

export async function putSubmissionStatus(assignmentId, studentId, status, extra = {}) {
  let expression = 'SET #s = :s, updated_at = :t'
  const values = { ':s': status, ':t': now() }
  const names = { '#s': 'status' }
  Object.entries(extra).forEach(([k, v], i) => {
    names[`#x${i}`] = k
    values[`:x${i}`] = v
    expression += `, #x${i} = :x${i}`
  })
  await wf.updateSubmission(assignmentId, studentId, {
    UpdateExpression: expression,
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values,
  })
}

export const userSettings = (user) => ({ ...DEFAULT_SETTINGS, ...(truthy(user.gradeant_settings) ? user.gradeant_settings : {}) })

