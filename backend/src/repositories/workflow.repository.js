// Tables owned by the assignment workflow:
//   GradeAnt_Assignments    PK assignment_id; GSI teacher_id-index
//   GradeAnt_Submissions    PK assignment_id, SK student_id; GSI student_id-index
//   GradeAnt_Announcements  PK announcement_id
import { table } from '../data/index.js'
import config from '../config/index.js'

const assignments = table(() => config.tables.assignments)
const submissions = table(() => config.tables.submissions)
const announcements = table(() => config.tables.announcements)

// ---- Assignments ----

export const getAssignment = (assignmentId) => assignments.get({ assignment_id: assignmentId })
export const putAssignment = (item) => assignments.put(item)
export const updateAssignment = (assignmentId, params) => assignments.update({ Key: { assignment_id: assignmentId }, ...params })
export const scanAssignments = () => assignments.scanAll()

export const assignmentsByTeacher = (teacherId) =>
  assignments.queryAll({
    IndexName: 'teacher_id-index',
    KeyConditionExpression: 'teacher_id = :t',
    ExpressionAttributeValues: { ':t': teacherId },
  })

export const publishedAssignments = () =>
  assignments.scanAll({
    FilterExpression: 'content_status = :p',
    ExpressionAttributeValues: { ':p': 'published' },
  })

// ---- Submissions ----

export const getSubmission = (assignmentId, studentId) => submissions.get({ assignment_id: assignmentId, student_id: studentId })

export const submissionsForAssignment = (assignmentId) =>
  submissions.queryAll({
    KeyConditionExpression: 'assignment_id = :a',
    ExpressionAttributeValues: { ':a': assignmentId },
  })

export const updateSubmission = (assignmentId, studentId, params) =>
  submissions.update({ Key: { assignment_id: assignmentId, student_id: studentId }, ...params })

// ---- Announcements ----

export const getAnnouncement = (id) => announcements.get({ announcement_id: id })
export const putAnnouncement = (item) => announcements.put(item)
export const deleteAnnouncement = (id) => announcements.delete({ announcement_id: id })
export const scanAnnouncements = () => announcements.scanAll()

export const announcementsByAuthor = (authorId) =>
  announcements.scanAll({
    FilterExpression: 'author_id = :a',
    ExpressionAttributeValues: { ':a': authorId },
  })

export function announcementsForClasses(classes) {
  const values = {}
  classes.forEach((c, i) => (values[`:c${i}`] = c))
  return announcements.scanAll({
    FilterExpression: `class_name IN (${Object.keys(values).join(', ')})`,
    ExpressionAttributeValues: values,
  })
}
