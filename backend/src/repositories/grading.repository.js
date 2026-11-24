// assignment_grading: one row per question per student plus one
// type="summary" row per student, keyed by evaluation_id. GSIs:
// student_id-index, type-index and (optional, recommended) assignment_id-index.
import { isMissingIndexError, table } from '../data/index.js'
import config from '../config/index.js'

const grading = table(() => config.tables.grading)

export const getRow = (evaluationId) => grading.get({ evaluation_id: evaluationId })

export const updateRow = (evaluationId, params) => grading.update({ Key: { evaluation_id: evaluationId }, ...params })

/**
 * Rows for one student. `filters` (all ANDed):
 *   assignmentId, qpId, questionNumber  → equality
 *   type: 'summary' | 'notSummary'
 */
export function studentRows(studentId, { assignmentId, qpId, questionNumber, type } = {}) {
  const names = {}
  const values = { ':sid': studentId }
  const conditions = []
  if (assignmentId !== undefined) {
    names['#aid'] = 'assignment_id'
    values[':aid'] = assignmentId
    conditions.push('#aid = :aid')
  }
  if (qpId !== undefined) {
    names['#qp'] = 'qp_id'
    values[':qp'] = qpId
    conditions.push('#qp = :qp')
  }
  if (questionNumber !== undefined) {
    names['#qn'] = 'question_number'
    values[':qn'] = questionNumber
    conditions.push('#qn = :qn')
  }
  if (type) {
    names['#type'] = 'type'
    values[':summary'] = 'summary'
    conditions.push(type === 'summary' ? '#type = :summary' : '#type <> :summary')
  }
  return grading.queryAll({
    IndexName: 'student_id-index',
    KeyConditionExpression: 'student_id = :sid',
    ...(conditions.length ? { FilterExpression: conditions.join(' AND ') } : {}),
    ...(Object.keys(names).length ? { ExpressionAttributeNames: names } : {}),
    ExpressionAttributeValues: values,
  })
}

/** Every student's summary row for one assignment / question paper (type-index). */
export const summaryRows = (assignmentId, qpId) =>
  grading.queryAll({
    IndexName: 'type-index',
    KeyConditionExpression: '#type = :summary',
    FilterExpression: '#aid = :aid AND #qp = :qp',
    ExpressionAttributeNames: { '#type': 'type', '#aid': 'assignment_id', '#qp': 'qp_id' },
    ExpressionAttributeValues: { ':summary': 'summary', ':aid': assignmentId, ':qp': qpId },
  })

/** All rows via the assignment_id GSI, or null when that index doesn't exist. */
export async function assignmentRows(assignmentId, qpId) {
  try {
    const rows = await grading.queryAll({
      IndexName: config.tables.gradingAssignmentIndex,
      KeyConditionExpression: '#aid = :aid',
      FilterExpression: '#qp = :qp',
      ExpressionAttributeNames: { '#aid': 'assignment_id', '#qp': 'qp_id' },
      ExpressionAttributeValues: { ':aid': assignmentId, ':qp': qpId },
    })
    return rows.filter((r) => r.student_id)
  } catch (err) {
    if (isMissingIndexError(err)) return null
    throw err
  }
}
