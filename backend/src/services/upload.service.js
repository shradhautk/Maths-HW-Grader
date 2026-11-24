// /uploadpdf — presigned S3 PUT URLs for answer sheets and question papers.
// Response: { urls, fileIds, uploadId }. The endpoint stays off (503) until UPLOAD_BUCKET is set.
//
// UPLOAD_LAYOUT picks the object keys (same rules as python-backend/functions/upload.py):
//   template (default)  keys follow UPLOAD_KEY_TEMPLATE
//   pipeline            the grading pipeline's layout, with the file names the pipeline reads,
//                       chosen by the server:
//                         student → <root>/students/assignments/<aid>__<qp>/original_pdf/<studentId>.pdf
//                         teacher → <root>/teachers/QP_Rubric/<aid>__<qp>/QP_pdfs/<qp>.pdf
//                         answer  → <root>/teachers/QPRubricWithAnswers/<aid>__<qp>/solution_pdfs/<aid>_<qp>_answer.pdf
//                       Exactly one file; the assignment must exist; UPLOAD_BUCKET must be PIPELINE_BUCKET.
// Question papers and solutions (and, in pipeline layout, any upload by a teacher) need the
// teacher who owns the assignment (admins pass). The body may be wrapped as {"metadata": {...}}.
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { storage } from '../aws/s3.js'
import config from '../config/index.js'
import { HttpError, badRequest, forbidden } from '../lib/errors.js'
import { requirePermission } from '../lib/permissions.js'
import * as wf from '../repositories/workflow.repository.js'
import { canManage } from './ownership.js'

const ALLOWED_TYPES = new Set(['application/pdf'])
const UPLOAD_TYPES = new Set(['student', 'teacher', 'answer'])
const MAX_FILES = 20
const ID = /^[A-Za-z0-9._-]{1,200}$/

const safe = (v) => String(v ?? '').replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 200) || '_'
const validId = (v) => typeof v === 'string' && ID.test(v)

export function objectKey(meta, fileName) {
  const values = {
    uploadType: safe(meta.uploadType),
    assignmentId: safe(meta.assignmentId),
    qpId: safe(meta.qpId),
    studentId: safe(meta.studentId),
    courseId: safe(meta.courseId),
    fileName: safe(fileName),
    uuid: randomUUID(),
  }
  return config.storage.uploadKeyTemplate.replace(/\{(\w+)\}/g, (m, name) => values[name] ?? m)
}

/** The pipeline's key layout (python-backend gradeant_common/pipeline/paths.py Layout). */
export function pipelineKey(uploadType, { root, assignmentId, qpId, studentId }) {
  const folder = `${assignmentId}__${qpId}`
  if (uploadType === 'student') return `${root}/students/assignments/${folder}/original_pdf/${studentId}.pdf`
  if (uploadType === 'teacher') return `${root}/teachers/QP_Rubric/${folder}/QP_pdfs/${qpId}.pdf`
  return `${root}/teachers/QPRubricWithAnswers/${folder}/solution_pdfs/${assignmentId}_${qpId}_answer.pdf`
}

function layoutMode() {
  const mode = config.storage.uploadLayout
  if (mode !== 'template' && mode !== 'pipeline') throw new HttpError(503, 'File uploads are misconfigured on this server.')
  return mode
}

function pipelineKeys(body, caller, uploadType, assignment, files) {
  if (config.storage.pipelineBucket !== config.storage.uploadBucket) {
    throw new HttpError(503, 'File uploads are misconfigured on this server.')
  }
  if (files.length !== 1) throw badRequest('Send exactly one PDF per upload.')
  if (!assignment) throw new HttpError(404, 'Assignment not found.')
  const assignmentId = body.assignmentId
  const clientQp = body.qpId || body.courseId
  const qpId = assignment.qp_id || clientQp
  if (clientQp && assignment.qp_id && clientQp !== assignment.qp_id) {
    throw badRequest("qpId doesn't match the assignment's question paper.")
  }
  if (!validId(assignmentId) || !validId(qpId)) {
    throw badRequest('assignmentId and qpId may only contain letters, digits, ".", "_" and "-".')
  }
  let studentId
  if (uploadType === 'student') {
    studentId = caller.role === 'student' ? caller.user_id : body.studentId
    if (!validId(studentId)) throw badRequest('studentId is required (letters, digits, ".", "_" and "-").')
  }
  return [pipelineKey(uploadType, { root: config.storage.pipelineRoot, assignmentId, qpId, studentId })]
}

export async function presignUploads(rawBody, caller) {
  const bucket = config.storage.uploadBucket
  if (!bucket) throw new HttpError(503, 'File uploads are not configured on this server.')
  // The original upload Lambda's wrapped form.
  const body = rawBody?.metadata && typeof rawBody.metadata === 'object' && !Array.isArray(rawBody.metadata) ? rawBody.metadata : rawBody
  const mode = layoutMode()
  if (!UPLOAD_TYPES.has(body.uploadType)) throw badRequest('uploadType must be student, teacher or answer.')
  if (caller.role === 'student' && body.uploadType !== 'student') throw forbidden('Students can only upload answer sheets.')
  // config/app.yaml permissions.upload.<uploadType>.
  requirePermission('upload', body.uploadType, caller.role, "Your role can't upload this kind of file.")
  if (caller.role === 'student' && body.studentId && body.studentId !== caller.user_id) {
    throw forbidden('You can only upload your own answers.')
  }
  const assignment = typeof body.assignmentId === 'string' && body.assignmentId ? await wf.getAssignment(body.assignmentId) : null
  if (caller.role === 'student') {
    // Only for a published assignment of one of the student's own classes.
    const classes = Array.isArray(caller.user?.classes) ? caller.user.classes : []
    if (!assignment || assignment.content_status !== 'published' || !classes.includes(assignment.class_name)) {
      throw forbidden('You can only upload answers for assignments in your own classes.')
    }
  } else if (body.uploadType === 'teacher' || body.uploadType === 'answer' || mode === 'pipeline') {
    // Question papers, solutions (and, in pipeline layout, sheets uploaded for a student) belong to
    // the assignment's teacher. Legacy flows may upload before the assignment exists: template only.
    if (assignment && !(await canManage(caller, body.assignmentId))) {
      throw forbidden('This assignment belongs to a different teacher.')
    }
  }
  const files = Array.isArray(body.files) ? body.files : []
  if (!files.length || files.length > MAX_FILES) throw badRequest(`Send between 1 and ${MAX_FILES} files.`)
  for (const f of files) {
    if (!f || typeof f.name !== 'string' || !f.name) throw badRequest('Every file needs a name.')
    if (!ALLOWED_TYPES.has(f.type)) throw badRequest(`${f.name}: only PDF files can be uploaded.`)
    if (!(Number(f.size) > 0) || Number(f.size) > config.storage.uploadMaxBytes) {
      throw badRequest(`${f.name}: files must be under ${Math.round(config.storage.uploadMaxBytes / 1048576)} MB.`)
    }
  }
  let keys
  if (mode === 'pipeline') {
    keys = pipelineKeys(body, caller, body.uploadType, assignment, files)
  } else {
    const meta = { ...body, studentId: body.studentId ?? (caller.role === 'student' ? caller.user_id : undefined) }
    keys = files.map((f) => objectKey(meta, f.name))
  }
  const store = storage()
  const urls = await Promise.all(
    // The validated size is signed into each URL, so a client can't declare a small
    // file and then upload a huge one.
    files.map((f, i) => store.presignPut(bucket, keys[i], f.type, config.storage.uploadUrlTtlSeconds, Math.floor(Number(f.size)))),
  )
  return { urls, fileIds: keys.map((k) => path.posix.basename(k)), uploadId: `upload-${randomUUID()}` }
}
