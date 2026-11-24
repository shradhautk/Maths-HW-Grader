// /correctanswer — port of lambda_python_files/correct_questions_answer.py:
// the post-upload correction workspace.
//   Student: getAnswersForCorrection / submitAnswerCorrections (OCR'd answer text)
//   Teacher: getQuestionsForCorrection / submitQuestionCorrections (parsed question
//            text, fanned out to every student's denormalised copy)
import { isConditionalCheckFailed, badRequest, forbidden, reply } from '../lib/errors.js'
import { isoNaive, pyStr, sortBy, truthy } from '../lib/py.js'
import { getRow, studentRows, summaryRows, updateRow } from '../repositories/grading.repository.js'
import * as wf from '../repositories/workflow.repository.js'
import { canManage, firstOwner, ownerResolver } from './ownership.js'
import { RANK, rank } from './workflow/context.js'

/**
 * Once a paper is graded (a summary row exists) or its submission has reached
 * grading_triggered, the answer text is what was marked: the student can no
 * longer rewrite it (e.g. to change an answer before asking for a re-check).
 */
async function answersLocked(studentId, assignmentId) {
  if (!truthy(assignmentId)) return false
  const sub = await wf.getSubmission(assignmentId, studentId)
  if (sub && rank(sub.status) >= RANK.grading_triggered) return true
  return (await studentRows(studentId, { assignmentId, type: 'summary' })).length > 0
}

const keyStr = (v) => pyStr(truthy(v) ? v : '')
const isMissing = (v) => v === undefined || v === null

export async function getAnswersForCorrection(body, caller) {
  const { student_id: studentId, assignment_id: assignmentId, qp_id: qpId } = body
  if (!truthy(studentId) || !truthy(assignmentId)) throw badRequest('student_id and assignment_id are required')
  if (caller.role === 'student' && studentId !== caller.user_id) throw forbidden('You can only view your own answers.')

  const items = await studentRows(studentId, {
    assignmentId,
    type: 'notSummary',
    ...(truthy(qpId) ? { qpId } : {}),
  })

  // A teacher (not admin) may only browse submissions for assignments they own.
  if (caller.role === 'teacher' && !(await canManage(caller, assignmentId, firstOwner(items)))) {
    throw forbidden("This student's answers belong to a different teacher's class.")
  }

  const answers = items.map((item) => ({
    evaluation_id: item.evaluation_id ?? null,
    question_number: item.question_number ?? null,
    subpart: item.subpart ?? null,
    question: item.question ?? null,
    student_answer: item.student_answer ?? null,
    is_latex: item.is_latex ?? false,
    max_marks: item.max_marks ?? null,
    status: item.status ?? null,
    corrected_by: item.corrected_by ?? null,
    corrected_at: item.corrected_at ?? null,
  }))
  return {
    student_id: studentId,
    assignment_id: assignmentId,
    qp_id: qpId ?? null,
    answers: sortBy(answers, (a) => [keyStr(a.question_number), keyStr(a.subpart)]),
  }
}

/** Student submits corrected answer text; status='corrected' feeds the grading fan-in gate. */
export async function submitAnswerCorrections(body, caller) {
  if (caller.role !== 'student') throw forbidden('Only the student who owns this submission can correct it.')
  const { corrections } = body
  if (!Array.isArray(corrections) || !corrections.length) throw badRequest('corrections (non-empty array) is required')

  const updated = []
  const errors = []
  const now = isoNaive()
  const locked = new Map() // assignment_id → answersLocked
  for (const correction of corrections) {
    const c = correction && typeof correction === 'object' ? correction : {}
    const evaluationId = c.evaluation_id
    const questionNumber = c.question_number
    const answer = c.corrected_answer
    const isLatex = truthy(c.is_latex ?? false)

    if (!truthy(evaluationId)) {
      errors.push('Missing evaluation_id in correction')
      continue
    }
    if (isMissing(questionNumber)) {
      errors.push(`Missing question_number for evaluation_id ${evaluationId}`)
      continue
    }
    if (isMissing(answer) || pyStr(answer).trim() === '') {
      errors.push(`corrected_answer is required for evaluation_id ${evaluationId}`)
      continue
    }
    try {
      // Only the owning student may correct their own answer.
      const existing = await getRow(evaluationId)
      if (!existing) {
        errors.push(`No record found with evaluation_id ${evaluationId}`)
        continue
      }
      if (existing.student_id !== caller.user_id) {
        errors.push(`evaluation_id ${evaluationId} belongs to a different student.`)
        continue
      }
      const aId = existing.assignment_id
      if (!locked.has(aId)) locked.set(aId, await answersLocked(caller.user_id, aId))
      if (locked.get(aId)) {
        errors.push(`evaluation_id ${evaluationId} has already been graded and can no longer be changed.`)
        continue
      }
      let expression =
        'SET student_answer = :sa, is_latex = :il, #st = :status, corrected_by = :cb, corrected_at = :ca, updated_at = :ua'
      const values = {
        ':sa': answer,
        ':il': isLatex,
        ':status': 'corrected',
        ':cb': caller.user_id,
        ':ca': now,
        ':ua': now,
        ':qn': questionNumber,
      }
      if (typeof existing.student_answer === 'string') {
        // Keep the text as read from the paper (once, never overwritten), so grading can tell a small
        // OCR fix from a rewritten answer (the grader flags big changes for teacher review).
        // Same as python-backend/functions/corrections.py.
        expression += ', ocr_answer = if_not_exists(ocr_answer, :ocr)'
        values[':ocr'] = existing.student_answer
      }
      await updateRow(evaluationId, {
        UpdateExpression: expression,
        ConditionExpression: '#qn = :qn',
        ExpressionAttributeNames: { '#st': 'status', '#qn': 'question_number' },
        ExpressionAttributeValues: values,
      })
      updated.push({ evaluation_id: evaluationId, question_number: questionNumber, updated: true, is_latex: isLatex })
    } catch (err) {
      if (isConditionalCheckFailed(err)) errors.push(`No matching question_number ${pyStr(questionNumber)} for evaluation_id ${evaluationId}`)
      else errors.push(`Failed to correct evaluation_id ${evaluationId}: ${err?.message}`)
    }
  }

  const response = {
    message: errors.length ? 'Correction completed with some errors' : 'Correction completed',
    updated_items: updated,
    total_updated: updated.length,
    success: errors.length === 0,
  }
  if (errors.length) response.errors = errors
  if (!updated.length) {
    response.message = 'No answers were corrected'
    return reply(404, response)
  }
  return reply(errors.length ? 207 : 200, response)
}

/** Roster for a question paper — every student's summary row (type-index). */
const classRoster = (qpId, assignmentId) => summaryRows(assignmentId, qpId)

/** One canonical copy of each question's parsed text, so the teacher reviews it once. */
export async function getQuestionsForCorrection(body, caller) {
  if (!['teacher', 'admin'].includes(caller.role)) throw forbidden('Only a teacher or admin can review question text.')
  const { qp_id: qpId, assignment_id: assignmentId } = body
  if (!truthy(qpId) || !truthy(assignmentId)) throw badRequest('qp_id and assignment_id are required')

  let roster = await classRoster(qpId, assignmentId)
  if (caller.role === 'teacher') {
    const resolve = ownerResolver()
    const allowed = await Promise.all(roster.map((g) => canManage(caller, assignmentId, g.teacher_id, resolve)))
    roster = roster.filter((_, i) => allowed[i])
  }
  if (!roster.length) return { qp_id: qpId, assignment_id: assignmentId, questions: [] }

  const byKey = new Map()
  for (const student of roster) {
    if (!truthy(student.student_id)) continue
    const details = await studentRows(student.student_id, { assignmentId, qpId, type: 'notSummary' })
    for (const item of details) {
      const key = JSON.stringify([pyStr(item.question_number), item.subpart || ''])
      if (!byKey.has(key)) {
        byKey.set(key, {
          question_number: item.question_number ?? null,
          subpart: item.subpart ?? null,
          question: item.question ?? null,
          is_latex: item.question_is_latex ?? false,
          max_marks: item.max_marks ?? null,
          question_reviewed: truthy(item.question_reviewed_at),
          question_reviewed_by: item.question_reviewed_by ?? null,
          question_reviewed_at: item.question_reviewed_at ?? null,
        })
      }
    }
    // One representative student's rows are enough: question text is identical across the class.
    if (byKey.size) break
  }
  const questions = sortBy([...byKey.values()], (q) => [keyStr(q.question_number), keyStr(q.subpart)])
  return { qp_id: qpId, assignment_id: assignmentId, questions }
}

/** Teacher corrects question text; each correction fans out to every student's copy. */
export async function submitQuestionCorrections(body, caller) {
  if (!['teacher', 'admin'].includes(caller.role)) throw forbidden('Only a teacher or admin can correct question text.')
  const { qp_id: qpId, assignment_id: assignmentId, corrections } = body
  if (!truthy(qpId) || !truthy(assignmentId)) throw badRequest('qp_id and assignment_id are required')
  if (!Array.isArray(corrections) || !corrections.length) throw badRequest('corrections (non-empty array) is required')

  const roster = await classRoster(qpId, assignmentId)
  const resolve = ownerResolver()
  const now = isoNaive()
  const updatedQuestions = []
  const errors = []

  for (const correction of corrections) {
    const c = correction && typeof correction === 'object' ? correction : {}
    const questionNumber = c.question_number
    const subpart = c.subpart ?? null
    const text = c.corrected_question
    const isLatex = truthy(c.is_latex ?? false)
    if (isMissing(questionNumber)) {
      errors.push('Missing question_number in correction')
      continue
    }
    if (isMissing(text) || pyStr(text).trim() === '') {
      errors.push(`corrected_question is required for question_number ${pyStr(questionNumber)}`)
      continue
    }

    let recordsUpdated = 0
    for (const student of roster) {
      const studentId = student.student_id
      if (!truthy(studentId)) continue
      // Rows of an assignment the caller doesn't own are skipped, not overwritten.
      if (!(await canManage(caller, assignmentId, student.teacher_id, resolve))) continue
      try {
        const details = await studentRows(studentId, { assignmentId, qpId, questionNumber, type: 'notSummary' })
        for (const item of details) {
          if (truthy(subpart) && item.subpart !== subpart) continue
          await updateRow(item.evaluation_id, {
            UpdateExpression:
              'SET question = :q, question_is_latex = :il, question_reviewed_by = :rb, question_reviewed_at = :ra, updated_at = :ua',
            ConditionExpression: '#qn = :qn',
            ExpressionAttributeNames: { '#qn': 'question_number' },
            ExpressionAttributeValues: {
              ':q': text,
              ':il': isLatex,
              ':rb': caller.user_id,
              ':ra': now,
              ':ua': now,
              ':qn': questionNumber,
            },
          })
          recordsUpdated += 1
        }
      } catch (err) {
        errors.push(`Failed to update student ${studentId}, question ${pyStr(questionNumber)}: ${err?.message}`)
      }
    }
    updatedQuestions.push({ question_number: questionNumber, subpart, records_updated: recordsUpdated })
  }

  const total = updatedQuestions.reduce((n, q) => n + q.records_updated, 0)
  const response = {
    message: errors.length ? 'Question correction completed with some errors' : 'Question correction completed',
    updated_questions: updatedQuestions,
    total_records_updated: total,
    success: errors.length === 0,
  }
  if (errors.length) response.errors = errors
  if (total === 0) {
    response.message = 'No question records were updated'
    return reply(404, response)
  }
  return reply(errors.length ? 207 : 200, response)
}

export const operations = {
  getAnswersForCorrection,
  submitAnswerCorrections,
  getQuestionsForCorrection,
  submitQuestionCorrections,
}
