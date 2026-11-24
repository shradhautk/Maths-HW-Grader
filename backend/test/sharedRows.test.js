// The Python grader (AWS Lambda) and the Node API (Render) share the Supabase table
// gradeant.assignment_grading. Rows written exactly the way the grader writes them,
// through the two SQL functions it calls (sql/002_pipeline_api.sql), must read correctly
// through the Node services: correction, roster status, Start grading → regrade, grades.
import assert from 'node:assert/strict'
import { after, before, describe, test } from 'node:test'

process.env.DB_DRIVER = 'pglite'
const { config, post, putUsers, startServer, stopServer, tokenFor } = await import('./helpers.js')
const { sql } = await import('../src/data/postgres.js')

const USERS = [
  { email: 't@x.edu', user_id: 'T1', name: 'Priya', userRole: 'teacher' },
  { email: 'a@x.edu', user_id: 'S1', name: 'Okafor', userRole: 'student', classes: ['PHY-204'] },
]
const as = (email) => ({ token: tokenFor(USERS.find((u) => u.email === email)) })
const workflow = (email, operation, payload = {}) => post('/workflow', { operation, ...payload }, as(email))

/** python-backend gradeant_common.pipeline.grading.detail_rows(..., graded=False): no `type`, `ocr_answer` kept. */
const pythonRow = (a, q, answer) => ({
  question: `Question ${q}`,
  student_answer: answer,
  ocr_answer: answer,
  rubric: 'Mentions mass defect.',
  max_marks: 5,
  question_number: q,
  answer_with: 'without_standard_answer',
  evaluation_id: `py-${a.assignment_id}-${q}`,
  assignment_id: a.assignment_id,
  qp_id: a.qp_id,
  student_id: 'S1',
  timestamp: 1794931265,
})

const replaceRows = (rows, deleteIds = []) =>
  sql('select gradeant.grading_replace_rows($1::jsonb, $2::text[]) as n', [JSON.stringify(rows), deleteIds])
const patchRow = (id, set, remove = []) => sql('select gradeant.grading_patch($1, $2::jsonb, $3::text[])', [id, JSON.stringify(set), remove])

describe('rows written by the Python grader, read by the Node API (Postgres)', () => {
  before(async () => {
    assert.equal(config.drivers.db, 'pglite')
    await startServer()
    await putUsers(USERS)
  })
  after(stopServer)

  test('upload → correction → Start grading → regrade → grades, all on shared rows', async () => {
    const [, a] = await workflow('t@x.edu', 'createAssignment', { title: 'Nuclear HW', class_name: 'PHY-204' })

    // 1. The grader stores the parsed sheet ungraded (GRADE_ON_UPLOAD=false).
    assert.equal((await replaceRows([pythonRow(a, 1, 'E = mc^2'), pythonRow(a, 2, '492 MeV')]))[0].n, 2)

    // 2. The student sees their transcribed answers and corrects one (Node writes ocr_answer once, keeps it).
    const [st, answers] = await post('/correctanswer', { operation: 'getAnswersForCorrection', student_id: 'S1', assignment_id: a.assignment_id }, as('a@x.edu'))
    assert.equal(st, 200)
    assert.equal(JSON.stringify(answers).includes('492 MeV'), true)
    for (const [q, text] of [[1, 'E = Δm c^2'], [2, '492 MeV']]) {
      const [cs] = await post(
        '/correctanswer',
        { operation: 'submitAnswerCorrections', corrections: [{ evaluation_id: `py-${a.assignment_id}-${q}`, question_number: q, corrected_answer: text }] },
        as('a@x.edu'),
      )
      assert.equal(cs, 200)
    }
    const [[row1]] = [await sql('select item from gradeant.assignment_grading where evaluation_id = $1', [`py-${a.assignment_id}-1`])]
    assert.equal(row1.item.student_answer, 'E = Δm c^2')
    assert.equal(row1.item.ocr_answer, 'E = mc^2')
    assert.equal(row1.item.status, 'corrected')

    // 3. The teacher's roster shows the student as corrected (detail rows have no `type`).
    let [, roster] = await workflow('t@x.edu', 'listSubmissions', { assignment_id: a.assignment_id })
    assert.equal(roster.submissions.find((s) => s.student_id === 'S1').status, 'corrected')

    // 4. The grader regrades in place (grading_patch) and writes the summary last.
    for (const [q, marks] of [[1, 4], [2, 5]]) {
      await patchRow(`py-${a.assignment_id}-${q}`, { marks, reasoning: 'ok', feedback: 'Good', graded_at: '2026-11-18T10:00:00+00:00', status: 'graded' }, [
        'flag_resolved_at',
        'flag_resolved_by',
        'flag_resolution',
      ])
    }
    await replaceRows([
      { type: 'summary', evaluation_id: `py-${a.assignment_id}-summary`, assignment_id: a.assignment_id, qp_id: a.qp_id, student_id: 'S1', total_marks: 9, timestamp: 1794931265, answer_with: 'without_standard_answer', graded_at: '2026-11-18T10:00:00+00:00', method: 'ai', persona: 'balanced' },
    ])

    ;[, roster] = await workflow('t@x.edu', 'listSubmissions', { assignment_id: a.assignment_id })
    assert.equal(roster.submissions.find((s) => s.student_id === 'S1').status, 'ai_graded')

    // 5. Teacher and student read the grades.
    const [gs, grade] = await post('/getgrades', { operation: 'getSingleStudentGrade', student_id: 'S1', assignment_id: a.assignment_id }, as('t@x.edu'))
    assert.equal(gs, 200)
    const text = JSON.stringify(grade)
    assert.ok(text.includes('"total_marks":9'), text)
    const [, all] = await post('/getgrades', { operation: 'getAllStudentGrades', qp_id: a.qp_id, assignment_id: a.assignment_id }, as('t@x.edu'))
    assert.equal(JSON.stringify(all).includes('"total_marks":9'), true)

    // 6. A graded paper is locked: further corrections are refused (the summary row exists).
    const [, late] = await post(
      '/correctanswer',
      { operation: 'submitAnswerCorrections', corrections: [{ evaluation_id: `py-${a.assignment_id}-1`, question_number: 1, corrected_answer: 'rewritten' }] },
      as('a@x.edu'),
    )
    assert.match(JSON.stringify(late), /already been graded/)
  })

  test('grading_replace_rows deletes stale rows and replaces in one call; grading_patch upserts', async () => {
    await replaceRows([{ evaluation_id: 'old-1', student_id: 'S9', assignment_id: 'AX', qp_id: 'QX' }])
    await replaceRows([{ evaluation_id: 'new-1', student_id: 'S9', assignment_id: 'AX', qp_id: 'QX', marks: 1 }], ['old-1'])
    const ids = (await sql("select evaluation_id from gradeant.assignment_grading where student_id = 'S9' order by 1")).map((r) => r.evaluation_id)
    assert.deepEqual(ids, ['new-1'])
    await patchRow('patched-1', { marks: 2 }, ['nothing'])
    const [{ item }] = await sql("select item from gradeant.assignment_grading where evaluation_id = 'patched-1'")
    assert.deepEqual(item, { evaluation_id: 'patched-1', marks: 2 })
    await assert.rejects(sql('select gradeant.grading_replace_rows($1::jsonb)', [JSON.stringify({ not: 'an array' })]), /p_rows must be a JSON array/)
  })
})
