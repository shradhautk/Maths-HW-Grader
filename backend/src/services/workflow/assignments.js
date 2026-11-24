// Teacher: create assignment → review parsed questions → choose AI or standard
// grading → rubric → verify → start grading (wireframe §13).
import { randomUUID } from 'node:crypto'
import { triggerGrading } from '../../aws/lambda.js'
import { badRequest, conflict } from '../../lib/errors.js'
import { num, pyStr, sortBy, trunc, truthy } from '../../lib/py.js'
import * as wf from '../../repositories/workflow.repository.js'
import { enqueueJob } from '../jobs.service.js'
import {
  PERSONAS,
  assignmentView,
  derivedStatus,
  enrolledStudents,
  gradingRows,
  loadAssignment,
  marksTotal,
  myAssignments,
  now,
  putSubmissionStatus,
  questions,
  required,
  requireRole,
  userSettings,
  usersById,
} from './context.js'

const shortId = () => randomUUID().replace(/-/g, '').slice(0, 10)

export async function listAssignments(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const items = await myAssignments(caller)
  const users = items.length ? await usersById() : new Map()
  const out = []
  for (const item of sortBy(items, (a) => a.created_at ?? '', { reverse: true })) {
    const view = await assignmentView(item)
    const enrolled = enrolledStudents(item, users).length
    if (enrolled) view.enrolled = enrolled
    out.push(view)
  }
  return { assignments: out }
}

export async function getAssignment(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [assignmentId] = required(body, 'assignment_id')
  const item = await loadAssignment(assignmentId, caller)
  const view = await assignmentView(item)
  const enrolled = enrolledStudents(item, await usersById()).length
  if (enrolled) view.enrolled = enrolled
  return view
}

export async function createAssignment(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [title, className] = required(body, 'title', 'class_name')
  const assignmentId = `A-${shortId()}`
  const item = {
    assignment_id: assignmentId,
    qp_id: `QP-${shortId()}`,
    title: trunc(pyStr(title), 200),
    class_name: trunc(pyStr(className), 100),
    teacher_id: caller.user_id,
    qp_status: 'qp_uploaded',
    content_status: 'published',
    grades_status: 'not_started',
    created_at: now(),
  }
  if (truthy(body.due_at)) item.due_at = pyStr(body.due_at)
  await wf.putAssignment(item)
  return { assignment_id: assignmentId, qp_id: item.qp_id }
}

export async function getParsedQuestions(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [assignmentId] = required(body, 'assignment_id')
  const item = await loadAssignment(assignmentId, caller)
  const qs = await questions(item)
  return {
    qp_status: qs.length ? 'qp_parsed' : (item.qp_status ?? 'qp_uploaded'),
    questions: qs,
    paper_total: marksTotal(qs),
  }
}

export async function confirmQuestions(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [assignmentId] = required(body, 'assignment_id')
  const item = await loadAssignment(assignmentId, caller)
  const qs = await questions(item)
  if (!qs.length) throw conflict('No parsed questions yet — they appear once the first answers are processed.')
  await wf.updateAssignment(assignmentId, {
    UpdateExpression: 'SET questions_confirmed_at = :t, question_count = :n, total_marks = :m, qp_status = if_not_exists(qp_status, :s)',
    ExpressionAttributeValues: { ':t': now(), ':n': qs.length, ':m': marksTotal(qs), ':s': 'qp_parsed' },
  })
  return { ok: true }
}

export async function setGradingMethod(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [assignmentId, method] = required(body, 'assignment_id', 'method')
  if (method !== 'ai' && method !== 'standard') throw badRequest("method must be 'ai' or 'standard'.")
  const item = await loadAssignment(assignmentId, caller)
  const defaults = userSettings(caller.user)
  const persona = [body.persona, item.persona, defaults.default_persona].find(truthy) ?? defaults.default_persona
  if (!PERSONAS.includes(persona)) throw badRequest(`persona must be one of ${PERSONAS.join(', ')}.`)
  const partial = Object.hasOwn(body, 'partial_credit')
    ? truthy(body.partial_credit)
    : Object.hasOwn(item, 'partial_credit')
      ? truthy(item.partial_credit)
      : defaults.partial_credit

  const changedMethod = item.method !== method
  let status = item.qp_status ?? 'qp_parsed'
  let expression = 'SET #m = :m, persona = :p, partial_credit = :pc'
  const values = { ':m': method, ':p': persona, ':pc': partial }
  if (changedMethod) {
    // A new method means a new rubric: AI drafts one in the background,
    // standard starts from the questions for the teacher to fill in.
    status = method === 'ai' ? 'rubric_generating' : 'rubric_generated'
    expression += ', qp_status = :s REMOVE rubric'
    values[':s'] = status
  }
  await wf.updateAssignment(assignmentId, {
    UpdateExpression: expression,
    ExpressionAttributeNames: { '#m': 'method' },
    ExpressionAttributeValues: values,
  })
  if (changedMethod && method === 'ai') await enqueueJob({ type: 'draftRubric', assignment_id: assignmentId })
  return { qp_status: status }
}

export function defaultRubric(qs) {
  return qs.map((q) => ({
    question_number: q.question_number,
    subpart: q.subpart ?? null,
    title: trunc(q.question || '', 80),
    max_marks: num(q.max_marks),
    criteria: '',
  }))
}

export async function getRubric(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [assignmentId] = required(body, 'assignment_id')
  const item = await loadAssignment(assignmentId, caller)
  const status = item.qp_status ?? null
  if (truthy(item.rubric)) return { qp_status: status, rubric: item.rubric }
  if (status === 'rubric_generating') return { qp_status: status, rubric: [] }
  return { qp_status: status, rubric: defaultRubric(await questions(item)) }
}

export async function approveRubric(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [assignmentId, rubric] = required(body, 'assignment_id', 'rubric')
  await loadAssignment(assignmentId, caller)
  if (!Array.isArray(rubric) || !rubric.length) throw badRequest('rubric must be a non-empty list.')
  const clean = rubric.map((r) => {
    if (!r || typeof r !== 'object' || Array.isArray(r) || r.question_number === undefined || r.question_number === null || r.question_number === '') {
      throw badRequest('Every rubric item needs a question_number.')
    }
    const marks = num(r.max_marks, -1)
    if (!(marks >= 0)) throw badRequest(`Invalid max_marks for Q${pyStr(r.question_number)}.`)
    // (subpart is dropped when empty, as workflow.py's _to_dynamo did.)
    const item = {
      question_number: r.question_number,
      title: trunc(pyStr(truthy(r.title) ? r.title : ''), 200),
      max_marks: marks,
      criteria: trunc(pyStr(truthy(r.criteria) ? r.criteria : ''), 4000),
    }
    if (truthy(r.subpart)) item.subpart = r.subpart
    return item
  })
  await wf.updateAssignment(assignmentId, {
    UpdateExpression: 'SET rubric = :r, qp_status = :s, rubric_reviewed_at = :t, rubric_reviewed_by = :u',
    ExpressionAttributeValues: { ':r': clean, ':s': 'rubric_reviewed', ':t': now(), ':u': caller.user_id },
  })
  return { qp_status: 'rubric_reviewed' }
}

/** Teacher side of the fan-in gate: every corrected submission is graded now. */
export async function startGrading(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [assignmentId] = required(body, 'assignment_id')
  const item = await loadAssignment(assignmentId, caller)
  if (item.qp_status !== 'rubric_reviewed') throw conflict('Approve the rubric before grading starts.')
  const byStudent = await gradingRows(item)
  let triggered = 0
  for (const [studentId, entry] of byStudent) {
    if (derivedStatus(entry) !== 'corrected') continue
    await putSubmissionStatus(assignmentId, studentId, 'grading_triggered')
    await triggerGrading({
      assignment_id: item.assignment_id,
      qp_id: item.qp_id,
      student_id: studentId,
      method: item.method ?? null,
      persona: item.persona ?? 'balanced',
      partial_credit: item.partial_credit ?? true,
      rubric: item.rubric ?? [],
    })
    triggered += 1
  }
  let expression = 'SET grading_started_at = :t'
  const values = { ':t': now() }
  // Re-grading after publish keeps the assignment "published"; students keep
  // their grades until the teacher publishes again (→ published_revised).
  if (item.grades_status !== 'published') {
    expression += ', grades_status = :g'
    values[':g'] = triggered ? 'teacher_reviewing' : 'ai_graded'
  }
  await wf.updateAssignment(assignmentId, { UpdateExpression: expression, ExpressionAttributeValues: values })
  return { ok: true, triggered }
}
