// /uploadpdf with UPLOAD_LAYOUT=pipeline: the same keys and rules as
// python-backend/functions/upload.py (tests/test_upload_pipeline.py), so either backend
// can sit in front of the grading pipeline.
import assert from 'node:assert/strict'
import { after, before, describe, test } from 'node:test'
import { config, post, putUsers, startServer, stopServer, table, tokenFor } from './helpers.js'

const USERS = [
  { email: 's@x.edu', user_id: 'S1', name: 'S', userRole: 'student', classes: ['PHY'] },
  { email: 't@x.edu', user_id: 'T1', name: 'T', userRole: 'teacher' },
  { email: 't2@x.edu', user_id: 'T2', name: 'T2', userRole: 'teacher' },
  { email: 'a@x.edu', user_id: 'AD', name: 'A', userRole: 'admin' },
]
const PDF = { name: 'my answers.pdf', type: 'application/pdf', size: 2048 }
const as = (uid) => ({ token: tokenFor(USERS.find((u) => u.user_id === uid)) })
const upload = (body, uid) => post('/uploadpdf', body, as(uid))
const keyOf = (url) => decodeURIComponent(new URL(url).pathname).replace(/^\/_local-storage\/[^/]+\//, '')

let saved
before(async () => {
  await startServer()
  await putUsers(USERS)
  await table(config.tables.assignments).put({
    assignment_id: 'A-1', qp_id: 'QP-1', teacher_id: 'T1', class_name: 'PHY', title: 'HW', content_status: 'published',
  })
  saved = { ...config.storage }
  Object.assign(config.storage, { uploadLayout: 'pipeline', pipelineBucket: config.storage.uploadBucket, pipelineRoot: 'Data_warehouse' })
})
after(async () => {
  Object.assign(config.storage, saved)
  await stopServer()
})

describe('/uploadpdf in the pipeline layout', () => {
  test('a student key uses the caller and the assignment\'s question paper', async () => {
    let [status, body] = await upload({ uploadType: 'student', assignmentId: 'A-1', courseId: 'QP-1', files: [PDF] }, 'S1')
    assert.equal(status, 200)
    assert.equal(keyOf(body.urls[0]), 'Data_warehouse/students/assignments/A-1__QP-1/original_pdf/S1.pdf')
    assert.deepEqual(body.fileIds, ['S1.pdf'])
    assert.match(body.uploadId, /^upload-[0-9a-f-]{36}$/)
    ;[status, body] = await upload({ uploadType: 'student', assignmentId: 'A-1', files: [PDF] }, 'S1') // qp from the record
    assert.equal(status, 200)
    assert.equal(keyOf(body.urls[0]), 'Data_warehouse/students/assignments/A-1__QP-1/original_pdf/S1.pdf')
  })

  test('student rules', async () => {
    const base = { uploadType: 'student', assignmentId: 'A-1', files: [PDF] }
    assert.equal((await upload({ ...base, qpId: 'QP-OTHER' }, 'S1'))[0], 400)
    assert.equal((await upload({ ...base, files: [PDF, PDF] }, 'S1'))[0], 400)
    assert.equal((await upload({ ...base, studentId: 'S9' }, 'S1'))[0], 403)
    assert.equal((await upload({ ...base, assignmentId: 'A-404' }, 'S1'))[0], 403)
  })

  test('question paper, solution, and a teacher uploading for a student', async () => {
    let [status, body] = await upload({ uploadType: 'teacher', assignmentId: 'A-1', courseId: 'QP-1', files: [PDF] }, 'T1')
    assert.equal(status, 200)
    assert.equal(keyOf(body.urls[0]), 'Data_warehouse/teachers/QP_Rubric/A-1__QP-1/QP_pdfs/QP-1.pdf')
    ;[status, body] = await upload({ uploadType: 'answer', assignmentId: 'A-1', qpId: 'QP-1', files: [PDF] }, 'T1')
    assert.equal(keyOf(body.urls[0]), 'Data_warehouse/teachers/QPRubricWithAnswers/A-1__QP-1/solution_pdfs/A-1_QP-1_answer.pdf')
    assert.equal((await upload({ uploadType: 'student', assignmentId: 'A-1', files: [PDF] }, 'T1'))[0], 400) // must name the student
    ;[status, body] = await upload({ uploadType: 'student', assignmentId: 'A-1', studentId: 'S1', files: [PDF] }, 'T1')
    assert.equal(status, 200)
    assert.match(keyOf(body.urls[0]), /\/original_pdf\/S1\.pdf$/)
  })

  test('ownership and the {"metadata": …} wrapper', async () => {
    const body = { uploadType: 'teacher', assignmentId: 'A-1', courseId: 'QP-1', files: [PDF] }
    assert.equal((await upload(body, 'T2'))[0], 403)
    assert.equal((await upload(body, 'AD'))[0], 200)
    assert.equal((await upload({ metadata: body }, 'T1'))[0], 200)
    assert.equal((await upload({ ...body, assignmentId: 'A-404' }, 'T1'))[0], 404)
  })

  test('misconfiguration answers 503; the template layout is unchanged', async () => {
    const body = { uploadType: 'teacher', assignmentId: 'A-1', courseId: 'QP-1', files: [PDF] }
    config.storage.pipelineBucket = 'some-other-bucket'
    assert.equal((await upload(body, 'T1'))[0], 503)
    config.storage.pipelineBucket = config.storage.uploadBucket
    config.storage.uploadLayout = 'template'
    try {
      const [status, out] = await upload(body, 'T1')
      assert.equal(status, 200)
      assert.match(keyOf(out.urls[0]), /^uploads\/teacher\/A-1\/my_answers\.pdf$/)
      assert.equal((await upload({ ...body, assignmentId: 'A-NEW' }, 'T1'))[0], 200) // legacy: before the assignment exists
      assert.equal((await upload(body, 'T2'))[0], 403)
    } finally {
      config.storage.uploadLayout = 'pipeline'
    }
  })
})
