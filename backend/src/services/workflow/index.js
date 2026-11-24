// /workflow — port of lambda_python_files/workflow.py: the assignment
// workflow behind the wireframe screens §04, §13–§15, §19, §20. One resource
// dispatching on `operation`; the contract is in web/WIREFRAMES.md.
import { listAnnouncements, deleteAnnouncement, postAnnouncement } from './announcements.js'
import {
  approveRubric,
  confirmQuestions,
  createAssignment,
  getAssignment,
  getParsedQuestions,
  getRubric,
  listAssignments,
  setGradingMethod,
  startGrading,
} from './assignments.js'
import { askAssistant } from './assistant.js'
import { acceptFlagged, getGradingRun, listFlagged, listSubmissions, publishGrades, remindStudent } from './grading.js'
import { getGradingQueue, getTeacherAnalytics, getTeacherSettings, listStudents, updateTeacherSettings } from './insights.js'
import { getProgress, listStudentAssignments, requestRecheck } from './student.js'

export const operations = {
  listAssignments,
  getAssignment,
  createAssignment,
  getParsedQuestions,
  confirmQuestions,
  setGradingMethod,
  getRubric,
  approveRubric,
  startGrading,
  getGradingRun,
  listSubmissions,
  listFlagged,
  acceptFlagged,
  publishGrades,
  remindStudent,
  listStudentAssignments,
  requestRecheck,
  askAssistant,
  getGradingQueue,
  listStudents,
  getTeacherAnalytics,
  getTeacherSettings,
  updateTeacherSettings,
  getProgress,
  postAnnouncement,
  listAnnouncements,
  deleteAnnouncement,
}
