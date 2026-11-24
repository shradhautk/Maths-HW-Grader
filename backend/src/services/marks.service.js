// /updatedb — port of lambda_python_files/update_db.py: teacher/admin edits
// to marks. A teacher may only edit rows of assignments they own (see
// ownership.js; admins excepted). An edit stamps teacher_id only when the row
// had none, so it never moves ownership.
import { isConditionalCheckFailed, badRequest, forbidden, notFound, reply, HttpError } from '../lib/errors.js'
import { isoNaive, pyStr, truthy } from '../lib/py.js'
import { getRow, updateRow } from '../repositories/grading.repository.js'
import { canManage, ownerResolver } from './ownership.js'

/** Decimal(str(value)): a finite number, or throws like Python's Decimal would. */
function decimal(value) {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN
  if (!Number.isFinite(n)) throw Object.assign(new Error(`Invalid number: ${pyStr(value)}`), { name: 'ValueError' })
  return n
}

/** True when `row` exists and the caller may not edit it. */
const ownedByOther = async (row, caller, resolve) => Boolean(row) && !(await canManage(caller, row.assignment_id, row.teacher_id, resolve))

export async function updateMarks(body, caller) {
  if (!['teacher', 'admin'].includes(caller.role)) throw forbidden('Only a teacher or admin can update grades.')
  const resolve = ownerResolver()
  const summaryRaw = body.summary
  const details = Array.isArray(body.details) ? body.details : []
  if (!truthy(summaryRaw) && !details.length) throw badRequest('Either summary or details data is required')

  const updated = []
  const errors = []

  // Summary: the frontend sends a one-element array; only [0] is applied (as before).
  // A details-only request skips this step (update_db.py crashed with a 500 there).
  const summary = Array.isArray(summaryRaw) ? summaryRaw[0] : undefined
  if (Array.isArray(summaryRaw) ? summaryRaw.length : truthy(summaryRaw)) {
    const evaluationId = summary?.evaluation_id
    try {
      if (!summary || typeof summary !== 'object') throw new Error('summary must be a list of objects')
      const totalMarks = decimal(summary.total_marks)
      const existing = truthy(evaluationId) ? await getRow(evaluationId) : undefined
      if (!truthy(evaluationId)) throw new Error('Missing evaluation_id in summary record')
      if (await ownedByOther(existing, caller, resolve)) {
        errors.push(`evaluation_id ${evaluationId} belongs to a different teacher.`)
      } else {
        await updateRow(evaluationId, {
          UpdateExpression: 'SET total_marks = :tm, updated_at = :ua, teacher_id = if_not_exists(teacher_id, :tid)',
          ConditionExpression: '#type = :summary',
          ExpressionAttributeNames: { '#type': 'type' },
          ExpressionAttributeValues: { ':tm': totalMarks, ':ua': isoNaive(), ':tid': caller.user_id, ':summary': 'summary' },
        })
        updated.push({
          type: 'summary',
          evaluation_id: evaluationId,
          updated: true,
          total_marks: totalMarks,
          student_id: summary.student_id ?? null,
          assignment_id: summary.assignment_id ?? null,
          qp_id: summary.qp_id ?? null,
        })
      }
    } catch (err) {
      if (isConditionalCheckFailed(err)) errors.push(`No summary record found with evaluation_id ${pyStr(evaluationId)}`)
      else if (err?.name === 'ValueError') errors.push(`Invalid total_marks value for evaluation_id ${pyStr(evaluationId)}: ${err.message}`)
      else errors.push(`Failed to update summary for evaluation_id ${pyStr(evaluationId)}: ${err?.message}`)
    }
  }

  for (const record of details) {
    const d = record && typeof record === 'object' ? record : {}
    const { evaluation_id: evaluationId, question_number: questionNumber, subpart, marks, max_marks: maxMarks } = d
    try {
      if (!truthy(evaluationId)) {
        errors.push('Missing evaluation_id in detail record')
        continue
      }
      if (questionNumber === undefined || questionNumber === null) {
        errors.push(`Missing question_number for evaluation_id ${evaluationId}`)
        continue
      }
      if (marks === undefined || marks === null) {
        errors.push(`Missing marks for evaluation_id ${evaluationId}`)
        continue
      }
      const marksValue = decimal(marks)
      const existing = await getRow(evaluationId)
      if (await ownedByOther(existing, caller, resolve)) {
        errors.push(`evaluation_id ${evaluationId} belongs to a different teacher.`)
        continue
      }
      let expression = 'SET marks = :m, updated_at = :ua, teacher_id = if_not_exists(teacher_id, :tid)'
      const values = { ':m': marksValue, ':ua': isoNaive(), ':tid': caller.user_id, ':qn': questionNumber }
      const hasMax = maxMarks !== undefined && maxMarks !== null
      if (hasMax) {
        expression += ', max_marks = :mm'
        values[':mm'] = decimal(maxMarks)
      }
      await updateRow(evaluationId, {
        UpdateExpression: expression,
        ConditionExpression: '#qn = :qn',
        ExpressionAttributeNames: { '#qn': 'question_number' },
        ExpressionAttributeValues: values,
      })
      const row = {
        type: 'details',
        evaluation_id: evaluationId,
        updated: true,
        marks: marksValue,
        question_number: pyStr(questionNumber),
        subpart: subpart ?? null,
      }
      if (hasMax) row.max_marks = values[':mm']
      updated.push(row)
    } catch (err) {
      if (isConditionalCheckFailed(err)) {
        errors.push(
          `No detail record found with evaluation_id ${evaluationId}, question_number ${pyStr(questionNumber)}, subpart ${pyStr(subpart)}`,
        )
      } else if (err?.name === 'ValueError') {
        errors.push(`Invalid marks value for evaluation_id ${pyStr(evaluationId)}: ${err.message}`)
      } else {
        errors.push(`Failed to update details for evaluation_id ${pyStr(evaluationId)}: ${err?.message}`)
      }
    }
  }

  const response = {
    message: 'Update completed',
    updated_items: updated,
    total_updated: updated.length,
    success: errors.length === 0,
    records_processed: { summary: 1, details: details.length },
  }
  if (errors.length) {
    response.errors = errors
    response.message = 'Update completed with some errors'
  }
  if (!updated.length) {
    response.message = 'No records were updated'
    return reply(404, response)
  }
  return reply(errors.length ? 207 : 200, response)
}

/** Total marks only, by evaluation_id; summary rows only. */
export async function updateStudentGrade(body, caller) {
  if (!['teacher', 'admin'].includes(caller.role)) throw forbidden('Only a teacher or admin can update grades.')
  const { evaluation_id: evaluationId, total_marks: totalMarks } = body
  if (!truthy(evaluationId)) throw badRequest('evaluation_id is required')
  if (totalMarks === undefined || totalMarks === null) throw badRequest('total_marks is required')

  const existing = await getRow(evaluationId)
  if (await ownedByOther(existing, caller)) throw forbidden('This assignment belongs to a different teacher.')
  let value
  try {
    value = decimal(totalMarks)
  } catch (err) {
    throw badRequest(`Invalid number format: ${err.message}`)
  }
  try {
    const res = await updateRow(evaluationId, {
      UpdateExpression: 'SET total_marks = :tm, updated_at = :ua, teacher_id = if_not_exists(teacher_id, :tid)',
      ConditionExpression: '#type = :summary',
      ExpressionAttributeNames: { '#type': 'type' },
      ExpressionAttributeValues: { ':tm': value, ':ua': isoNaive(), ':tid': caller.user_id, ':summary': 'summary' },
      ReturnValues: 'ALL_NEW',
    })
    return {
      message: 'Total marks updated successfully',
      evaluation_id: evaluationId,
      updated_attributes: res.Attributes ?? {},
      total_marks: value,
    }
  } catch (err) {
    if (isConditionalCheckFailed(err)) throw notFound(`No summary record found with evaluation_id ${evaluationId}`)
    throw new HttpError(500, `Database error: ${err?.name ?? 'Error'}`)
  }
}

export const operations = { updateMarks, updateStudentGrade }
