// /getgrades — port of lambda_python_files/get_grades.py.
//   getSingleStudentGrade   a student may only read their own grades
//   getAllStudentGrades     class roster, teacher/admin only
//   updateStudentGrade      total marks, teacher/admin only
// A teacher only reaches rows of assignments they own (see ownership.js);
// rows with no owner are refused to teachers.
import { badRequest, forbidden, notFound } from '../lib/errors.js'
import { truthy } from '../lib/py.js'
import { studentRows, summaryRows, updateRow } from '../repositories/grading.repository.js'
import { canManage, firstOwner, ownerResolver } from './ownership.js'

export async function getSingleStudentGrade(body, caller) {
  const { student_id: studentId, assignment_id: assignmentId } = body
  if (!truthy(studentId) || !truthy(assignmentId)) throw badRequest('Missing required parameters: student_id and assignment_id')
  if (caller.role === 'student' && studentId !== caller.user_id) throw forbidden('You can only view your own grades.')

  // (The field names are swapped relative to their contents, as the frontend expects.)
  const details = await studentRows(studentId, { assignmentId, type: 'notSummary' })
  const summary = await studentRows(studentId, { assignmentId, type: 'summary' })

  if (caller.role === 'teacher' && !(await canManage(caller, assignmentId, firstOwner([...summary, ...details])))) {
    throw forbidden("This student's results belong to a different teacher's class.")
  }
  return { student_id: studentId, assignment_id: assignmentId, details, summary }
}

export async function getAllStudentGrades(body, caller) {
  if (!['teacher', 'admin'].includes(caller.role)) throw forbidden('Only a teacher or admin can view a class roster.')
  const { qp_id: qpId, assignment_id: assignmentId } = body
  if (!truthy(qpId) || !truthy(assignmentId)) throw badRequest('Missing required parameters: qp_id and assignment_id')

  let grades = await summaryRows(assignmentId, qpId)
  if (caller.role === 'teacher') {
    const resolve = ownerResolver()
    const allowed = await Promise.all(grades.map((g) => canManage(caller, g.assignment_id ?? assignmentId, g.teacher_id, resolve)))
    grades = grades.filter((_, i) => allowed[i])
  }
  return { qp_id: qpId, assignment_id: assignmentId, student_count: grades.length, grades }
}

export async function updateStudentGrade(body, caller) {
  if (!['teacher', 'admin'].includes(caller.role)) throw forbidden('Only a teacher or admin can update grades.')
  const { student_id: studentId, qp_id: qpId, total_marks: totalMarks } = body
  if (![studentId, qpId, totalMarks].every(truthy)) {
    throw badRequest('student_id, qp_id, type, and total_marks are required')
  }
  const rows = await studentRows(studentId, { type: 'summary' })
  const item = rows.find((i) => i.qp_id === qpId && i.type === 'summary')
  // (get_grades.py returned nothing here, which API Gateway turned into a 502.)
  if (!item) throw notFound('No summary record found for that student and question paper.')

  if (!(await canManage(caller, item.assignment_id, item.teacher_id))) {
    throw forbidden('This assignment belongs to a different teacher.')
  }
  // An edit never moves ownership: teacher_id is only filled in when it was empty.
  await updateRow(item.evaluation_id, {
    UpdateExpression: 'SET total_marks = :tm, teacher_id = if_not_exists(teacher_id, :tid)',
    ExpressionAttributeValues: { ':tm': totalMarks, ':tid': caller.user_id },
  })
  // Same (pre-update) rows the Python handler echoed back.
  return { message: 'Total marks updated successfully', updatedAttributes: rows }
}

export const operations = { getSingleStudentGrade, getAllStudentGrades, updateStudentGrade }
