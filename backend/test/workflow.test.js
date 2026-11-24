// Port of lambda_python_files/tests/test_workflow.py: the same scenarios and
// assertions, run against both table layouts (with and without the
// assignment_id GSI on assignment_grading).
import assert from 'node:assert/strict'
import { after, before, beforeEach, describe, test } from 'node:test'
import { config, post, putUsers, startServer, stopServer, table, tokenFor } from './helpers.js'

const { setAIOverride: setClaudeOverride } = await import('../src/ai/index.js')

const USERS = [
  { email: 't@x.edu', user_id: 'T1', name: 'Priya', userRole: 'teacher' },
  { email: 't2@x.edu', user_id: 'T2', name: 'Other', userRole: 'teacher' },
  { email: 'a@x.edu', user_id: 'S1', name: 'Okafor', userRole: 'student', classes: ['PHY-204'] },
  { email: 'b@x.edu', user_id: 'S2', name: 'Marin', userRole: 'student', classes: ['PHY-204'] },
  { email: 'c@x.edu', user_id: 'S3', name: 'Tanaka', userRole: 'student', classes: ['PHY-204'] },
  { email: 'd@x.edu', user_id: 'S4', name: 'Nobody', userRole: 'student' }, // no classes recorded
  { email: 'e@x.edu', user_id: 'S5', name: 'Chem', userRole: 'student', classes: ['CHEM-101'] },
]

let calls = []

async function fakeClaude({ system, messages, effort, outputFormat }) {
  calls.push({ system, messages, effort, format: outputFormat })
  if (outputFormat) {
    return JSON.stringify({ items: [{ question_number: '1', subpart: '', title: 'Internal energy', max_marks: 5, criteria: 'Defines U; unit J' }] })
  }
  return 'Q2 was the hardest.'
}

async function call(email, op, payload = {}) {
  const user = USERS.find((u) => u.email === email)
  const [status, body] = await post('/workflow', { operation: op, ...payload }, { token: tokenFor(user) })
  return [status, body]
}

async function seedAnswers(a, student, { corrected = true, graded = false, confidence, sfx = '' } = {}) {
  const t = table(config.tables.grading)
  for (const [q, mm] of [
    [1, 5],
    [2, 5],
  ]) {
    const item = {
      evaluation_id: `${student}-${q}${sfx}`,
      student_id: student,
      assignment_id: a.assignment_id,
      qp_id: a.qp_id,
      type: 'detail',
      question_number: q,
      question: `Question ${q}`,
      max_marks: mm,
      student_answer: 'ans',
      timestamp: Math.floor(Date.now() / 1000),
    }
    if (corrected) item.status = 'corrected'
    if (graded) item.marks = 4
    if (confidence !== undefined && q === 2) item.confidence = confidence
    await t.put(item)
  }
  if (graded) {
    await t.put({
      evaluation_id: `${student}-sum${sfx}`,
      student_id: student,
      assignment_id: a.assignment_id,
      qp_id: a.qp_id,
      type: 'summary',
      total_marks: 8,
    })
  }
}

async function gradedAssignment({ title = 'Thermo HW', due, sfx = '' } = {}) {
  const [, a] = await call('t@x.edu', 'createAssignment', { title, class_name: 'PHY-204', ...(due ? { due_at: due } : {}) })
  await seedAnswers(a, 'S1', { graded: true, sfx }) // 8/10 = 80%
  await seedAnswers(a, 'S2', { graded: true, confidence: 0.6, sfx }) // 8/10, one answer flagged
  await call('t@x.edu', 'approveRubric', {
    assignment_id: a.assignment_id,
    rubric: [
      { question_number: 1, max_marks: 5 },
      { question_number: 2, max_marks: 5 },
    ],
  })
  return a
}

for (const assignmentIndex of [false, true]) {
  describe(`workflow (${assignmentIndex ? 'assignment-index' : 'no-assignment-index'})`, () => {
    before(() => setClaudeOverride(fakeClaude))
    beforeEach(async () => {
      calls = []
      await startServer({ assignmentIndex })
      await putUsers(USERS)
    })
    after(async () => {
      setClaudeOverride(undefined)
      await stopServer()
    })

    test('auth required', async () => {
      const [status, , headers] = await post('/workflow', { operation: 'listAssignments' })
      assert.equal(status, 401)
      assert.equal(headers.get('access-control-allow-origin'), '*')
    })

    test('internal rubric job not reachable from the API', async () => {
      assert.equal((await call('t@x.edu', 'draftRubric', { assignment_id: 'x' }))[0], 400)
    })

    test('full teacher and student flow', async () => {
      let [code, a] = await call('t@x.edu', 'createAssignment', { title: 'Thermo HW', class_name: 'PHY-204', due_at: '2030-01-01' })
      assert.equal(code, 200)
      assert.ok(a.assignment_id && a.qp_id)

      // Students see it straight away.
      let [, inbox] = await call('a@x.edu', 'listStudentAssignments')
      assert.equal(inbox.assignments[0].submission_status, 'upload_requested')

      // Another teacher can't touch it; a student can't create.
      assert.equal((await call('t2@x.edu', 'getAssignment', { assignment_id: a.assignment_id }))[0], 403)
      assert.equal((await call('a@x.edu', 'createAssignment', { title: 'x', class_name: 'y' }))[0], 403)

      // No parsed questions until answers arrive.
      assert.equal((await call('t@x.edu', 'confirmQuestions', { assignment_id: a.assignment_id }))[0], 409)
      await seedAnswers(a, 'S1', { corrected: true })
      await seedAnswers(a, 'S2', { corrected: false })

      let [, view] = await call('t@x.edu', 'getAssignment', { assignment_id: a.assignment_id })
      assert.equal(view.qp_status, 'qp_parsed')
      assert.equal(view.question_count, 2)
      assert.equal(view.total_marks, 10)
      assert.equal(view.enrolled, 3)
      assert.equal(view.submitted, 2)
      assert.equal((await call('t@x.edu', 'confirmQuestions', { assignment_id: a.assignment_id }))[0], 200)

      // AI method drafts a rubric (inline in tests).
      let [status, res] = await call('t@x.edu', 'setGradingMethod', { assignment_id: a.assignment_id, method: 'ai', persona: 'strict' })
      assert.equal(status, 200)
      assert.equal(res.qp_status, 'rubric_generating')
      let [, rub] = await call('t@x.edu', 'getRubric', { assignment_id: a.assignment_id })
      assert.equal(rub.qp_status, 'rubric_generated')
      assert.equal(rub.rubric[0].criteria, 'Defines U; unit J')
      assert.equal(rub.rubric[0].max_marks, 5)
      assert.equal(calls.at(-1).format.type, 'json_schema')

      assert.equal((await call('t@x.edu', 'startGrading', { assignment_id: a.assignment_id }))[0], 409) // rubric not approved
      ;[, res] = await call('t@x.edu', 'approveRubric', { assignment_id: a.assignment_id, rubric: rub.rubric })
      assert.equal(res.qp_status, 'rubric_reviewed')

      ;[, res] = await call('t@x.edu', 'startGrading', { assignment_id: a.assignment_id })
      assert.equal(res.triggered, 1) // only S1 has confirmed answers

      let [, run] = await call('t@x.edu', 'getGradingRun', { assignment_id: a.assignment_id })
      assert.deepEqual(Object.fromEntries(run.submissions.map((s) => [s.student_id, s.status])), {
        S1: 'grading_triggered',
        S2: 'ocr_done',
        S3: 'upload_requested',
      })

      // Pipeline grades S1 with one low-confidence answer → spot-check.
      await seedAnswers(a, 'S1', { corrected: true, graded: true, confidence: 0.6 })
      ;[, run] = await call('t@x.edu', 'getGradingRun', { assignment_id: a.assignment_id })
      assert.equal(run.submissions.find((s) => s.student_id === 'S1').status, 'queued_for_review')
      let [, flagged] = await call('t@x.edu', 'listFlagged', { assignment_id: a.assignment_id })
      assert.equal(flagged.items.length, 1)
      assert.equal(flagged.items[0].confidence, 0.6)
      assert.equal((await call('t@x.edu', 'publishGrades', { assignment_id: a.assignment_id }))[0], 409)
      assert.equal((await call('t2@x.edu', 'acceptFlagged', { evaluation_id: flagged.items[0].evaluation_id }))[0], 403)
      assert.equal((await call('t@x.edu', 'acceptFlagged', { evaluation_id: flagged.items[0].evaluation_id }))[0], 200)
      assert.deepEqual((await call('t@x.edu', 'listFlagged', { assignment_id: a.assignment_id }))[1].items, [])

      // Students don't see unpublished grades.
      let entry = (await call('a@x.edu', 'listStudentAssignments'))[1].assignments[0]
      assert.equal(entry.submission_status, 'grading_triggered')
      assert.ok(!('evaluation_id' in entry))
      assert.equal(
        (await call('a@x.edu', 'requestRecheck', { assignment_id: a.assignment_id, items: [{ question_number: 2, note: 'sign' }] }))[0],
        409,
      )

      ;[code, res] = await call('t@x.edu', 'publishGrades', { assignment_id: a.assignment_id })
      assert.equal(code, 200)
      assert.equal(res.notified, 1)
      entry = (await call('a@x.edu', 'listStudentAssignments'))[1].assignments[0]
      assert.equal(entry.submission_status, 'published')
      assert.equal(entry.evaluation_id, 'S1-sum')

      // Re-check loops back to the teacher's roster.
      assert.equal(
        (await call('a@x.edu', 'requestRecheck', { assignment_id: a.assignment_id, items: [{ question_number: 2, note: '' }] }))[0],
        400,
      )
      assert.equal(
        (await call('a@x.edu', 'requestRecheck', { assignment_id: a.assignment_id, items: [{ question_number: 2, note: 'sign was right' }] }))[0],
        200,
      )
      const [, roster] = await call('t@x.edu', 'listSubmissions', { assignment_id: a.assignment_id })
      const s1 = roster.submissions.find((s) => s.student_id === 'S1')
      assert.equal(s1.recheck[0].note, 'sign was right')
      assert.equal(s1.student_name, 'Okafor')
      assert.equal(roster.enrolled, 3)

      // Republishing marks it revised.
      await call('t@x.edu', 'publishGrades', { assignment_id: a.assignment_id })
      assert.equal((await call('a@x.edu', 'listStudentAssignments'))[1].assignments[0].submission_status, 'published_revised')

      const [, lst] = await call('t@x.edu', 'listAssignments')
      assert.equal(lst.assignments[0].grades_status, 'published')
      assert.ok(!('rubric' in lst.assignments[0]))

      // Starting grading again after publishing doesn't un-publish the assignment.
      assert.equal((await call('t@x.edu', 'startGrading', { assignment_id: a.assignment_id }))[0], 200)
      assert.equal((await call('t@x.edu', 'getAssignment', { assignment_id: a.assignment_id }))[1].grades_status, 'published')
    })

    test('class boundaries: reminders, inbox and announcements stay within a class', async () => {
      const [, a] = await call('t@x.edu', 'createAssignment', { title: 'HW', class_name: 'PHY-204' })
      await call('t@x.edu', 'postAnnouncement', { class_name: 'PHY-204', title: 'Hi', message: 'PHY only' })
      // A student of another class, or with none, can't be reminded through this assignment.
      for (const sid of ['S5', 'S4']) {
        assert.equal((await call('t@x.edu', 'remindStudent', { assignment_id: a.assignment_id, student_id: sid }))[0], 404, sid)
      }
      // A student with no classes sees no assignments or announcements (not every school's).
      assert.deepEqual((await call('d@x.edu', 'listStudentAssignments'))[1].assignments, [])
      assert.deepEqual((await call('d@x.edu', 'listAnnouncements'))[1].announcements, [])
      assert.deepEqual((await call('d@x.edu', 'getProgress'))[1].todo, [])
      // A student of another class doesn't see PHY-204's work either.
      assert.deepEqual((await call('e@x.edu', 'listStudentAssignments'))[1].assignments, [])
      assert.deepEqual((await call('e@x.edu', 'listAnnouncements'))[1].announcements, [])
      // The enrolled student does.
      assert.ok((await call('c@x.edu', 'listStudentAssignments'))[1].assignments.some((x) => x.assignment_id === a.assignment_id))
    })

    test('reminder and assistant', async () => {
      const [, a] = await call('t@x.edu', 'createAssignment', { title: 'HW', class_name: 'PHY-204' })
      let [code, res] = await call('t@x.edu', 'remindStudent', { assignment_id: a.assignment_id, student_id: 'S3' })
      assert.equal(code, 200)
      assert.equal(res.emailed, false) // no SES sender configured
      assert.equal((await call('t@x.edu', 'remindStudent', { assignment_id: a.assignment_id, student_id: 'T2' }))[0], 404)

      await seedAnswers(a, 'S1', { graded: true })
      ;[code, res] = await call('t@x.edu', 'askAssistant', {
        assignment_id: a.assignment_id,
        messages: [
          { role: 'assistant', content: 'Hi' },
          { role: 'user', content: 'Who struggled?' },
        ],
      })
      assert.equal(code, 200)
      assert.equal(res.reply, 'Q2 was the hardest.')
      const sent = calls.at(-1)
      assert.equal(sent.messages[0].role, 'user')
      assert.ok(sent.system.includes('Okafor'))
      assert.equal(sent.effort, 'low')
      assert.equal((await call('a@x.edu', 'askAssistant', { messages: [{ role: 'user', content: 'x' }] }))[0], 403)
      assert.equal((await call('t@x.edu', 'askAssistant', { messages: [] }))[0], 400)
    })

    test('AI rubric drafting failure falls back to the question-based rubric', async () => {
      setClaudeOverride(async () => {
        throw new Error('Bedrock unavailable')
      })
      try {
        const [, a] = await call('t@x.edu', 'createAssignment', { title: 'HW', class_name: 'PHY-204' })
        await seedAnswers(a, 'S1')
        assert.equal((await call('t@x.edu', 'setGradingMethod', { assignment_id: a.assignment_id, method: 'ai' }))[0], 200)
        const [, rub] = await call('t@x.edu', 'getRubric', { assignment_id: a.assignment_id })
        assert.equal(rub.qp_status, 'rubric_generated')
        assert.deepEqual(
          rub.rubric.map((r) => [r.question_number, r.criteria]),
          [
            [1, ''],
            [2, ''],
          ],
        )
      } finally {
        setClaudeOverride(fakeClaude)
      }
    })

    test('standard method rubric from questions', async () => {
      const [, a] = await call('t@x.edu', 'createAssignment', { title: 'HW', class_name: 'PHY-204' })
      await seedAnswers(a, 'S1')
      const [, res] = await call('t@x.edu', 'setGradingMethod', { assignment_id: a.assignment_id, method: 'standard' })
      assert.equal(res.qp_status, 'rubric_generated')
      const rub = (await call('t@x.edu', 'getRubric', { assignment_id: a.assignment_id }))[1].rubric
      assert.deepEqual(
        rub.map((r) => r.question_number),
        [1, 2],
      )
      assert.equal(rub[0].criteria, '')
      assert.equal(
        (await call('t@x.edu', 'approveRubric', { assignment_id: a.assignment_id, rubric: [{ question_number: 1, max_marks: -1 }] }))[0],
        400,
      )
      assert.equal((await call('t@x.edu', 'setGradingMethod', { assignment_id: a.assignment_id, method: 'magic' }))[0], 400)
    })

    test('grading queue and students', async () => {
      const a = await gradedAssignment()
      const [, b] = await call('t@x.edu', 'createAssignment', { title: 'Empty', class_name: 'PHY-204' })
      let [code, q] = await call('t@x.edu', 'getGradingQueue')
      assert.equal(code, 200)
      const rows = Object.fromEntries(q.queue.map((r) => [r.assignment_id, r]))
      assert.equal(rows[a.assignment_id].state, 'review')
      assert.equal(rows[a.assignment_id].flagged, 1)
      assert.equal(rows[a.assignment_id].papers, 2)
      assert.equal(rows[b.assignment_id].state, 'setup')
      assert.equal(q.queue[0].assignment_id, a.assignment_id)
      assert.equal(q.waiting, 1)
      assert.deepEqual((await call('t2@x.edu', 'getGradingQueue'))[1].queue, [])
      assert.equal((await call('a@x.edu', 'getGradingQueue'))[0], 403)

      await call('t@x.edu', 'acceptFlagged', { evaluation_id: 'S2-2' })
      assert.equal((await call('t@x.edu', 'getGradingQueue'))[1].queue[0].state, 'ready')

      const [, st] = await call('t@x.edu', 'listStudents')
      const names = Object.fromEntries(st.students.map((s) => [s.student_name, s]))
      assert.deepEqual(new Set(Object.keys(names)), new Set(['Okafor', 'Marin', 'Tanaka']))
      assert.deepEqual(st.classes, ['PHY-204'])
      assert.equal(names.Okafor.submitted, 1)
      assert.equal(names.Okafor.assigned, 2)
      assert.equal(Math.round(names.Okafor.average), 80)
      assert.equal(names.Tanaka.submitted, 0)
      assert.equal(names.Tanaka.average, null)
      assert.equal(names.Tanaka.latest, null)
      assert.equal(names.Okafor.latest.assignment_id, a.assignment_id)
      assert.equal(names.Okafor.at_risk, false)
      assert.equal((await call('a@x.edu', 'listStudents'))[0], 403)
    })

    test('teacher analytics', async () => {
      await gradedAssignment()
      const [code, an] = await call('t@x.edu', 'getTeacherAnalytics')
      assert.equal(code, 200)
      assert.deepEqual(an.classes, ['PHY-204'])
      assert.equal(an.totals.assignments, 1)
      assert.equal(an.totals.graded, 2)
      assert.equal(Math.round(an.totals.average), 80)
      assert.equal(an.totals.submitted, 2)
      assert.equal(an.totals.enrolled, 3)
      assert.equal(Math.round(an.totals.submission_rate), 67)
      assert.deepEqual(an.distribution, { A: 0, B: 2, C: 0, D: 0, F: 0 })
      assert.equal(an.weekly.length, 7)
      assert.equal(
        an.weekly.reduce((t, w) => t + w.count, 0),
        2,
      )
      assert.equal(an.assignments[0].title, 'Thermo HW')
      assert.equal(an.hardest[0].label, 'Q1')
      assert.ok(an.insight.includes('recap'))
      assert.equal((await call('t@x.edu', 'getTeacherAnalytics', { class_name: 'NOPE' }))[1].totals.assignments, 0)
      assert.equal((await call('t@x.edu', 'getTeacherAnalytics', { class_name: 'PHY-204' }))[1].totals.assignments, 1)
      await gradedAssignment({ title: 'Future HW', due: '2099-01-01', sfx: '-f' })
      assert.deepEqual(
        (await call('t@x.edu', 'getTeacherAnalytics'))[1].upcoming.map((u) => u.title),
        ['Future HW'],
      )
    })

    test('teacher settings drive AI defaults', async () => {
      let [, res] = await call('t@x.edu', 'getTeacherSettings')
      assert.deepEqual(res.settings, { default_persona: 'balanced', partial_credit: true, notify_submissions: true, notify_flagged: true })
      assert.equal((await call('t@x.edu', 'updateTeacherSettings', { settings: { default_persona: 'wild' } }))[0], 400)
      assert.equal((await call('t@x.edu', 'updateTeacherSettings', { settings: { partial_credit: 'yes' } }))[0], 400)
      let code
      ;[code, res] = await call('t@x.edu', 'updateTeacherSettings', { settings: { default_persona: 'strict', partial_credit: false } })
      assert.equal(code, 200)
      assert.equal(res.settings.default_persona, 'strict')
      assert.equal((await call('t@x.edu', 'getTeacherSettings'))[1].settings.partial_credit, false)
      assert.equal((await call('t2@x.edu', 'getTeacherSettings'))[1].settings.default_persona, 'balanced') // per user

      const [, a] = await call('t@x.edu', 'createAssignment', { title: 'HW', class_name: 'PHY-204' })
      await seedAnswers(a, 'S1')
      await call('t@x.edu', 'setGradingMethod', { assignment_id: a.assignment_id, method: 'ai' })
      const view = (await call('t@x.edu', 'getAssignment', { assignment_id: a.assignment_id }))[1]
      assert.equal(view.persona, 'strict')
      assert.equal(view.partial_credit, false)
      assert.ok(calls.at(-1).messages[0].content.includes('Grading persona: strict'))
      assert.ok(calls.at(-1).messages[0].content.includes('only for correct final answers'))
      await call('t@x.edu', 'setGradingMethod', { assignment_id: a.assignment_id, method: 'ai', persona: 'lenient' })
      assert.equal((await call('a@x.edu', 'getTeacherSettings'))[0], 403)
    })

    test('student progress', async () => {
      const a = await gradedAssignment({ due: '2030-01-10' })
      await gradedAssignment({ title: 'Second HW', due: '2030-02-10', sfx: '-b' })
      await call('t@x.edu', 'createAssignment', { title: 'Not yet', class_name: 'PHY-204', due_at: '2030-03-01' })
      assert.equal((await call('t@x.edu', 'getProgress'))[0], 403)
      let [code, pr] = await call('a@x.edu', 'getProgress')
      assert.equal(code, 200)
      assert.deepEqual(pr.results, [])
      assert.equal(pr.average, null)
      assert.deepEqual(new Set(pr.todo.map((t) => t.title)), new Set(['Not yet'])) // graded-but-unpublished work isn't shown yet
      await call('t@x.edu', 'acceptFlagged', { evaluation_id: 'S2-2' })
      await call('t@x.edu', 'acceptFlagged', { evaluation_id: 'S2-2-b' })
      await call('t@x.edu', 'publishGrades', { assignment_id: a.assignment_id })
      ;[code, pr] = await call('a@x.edu', 'getProgress')
      assert.deepEqual(
        pr.results.map((r) => r.title),
        ['Thermo HW'],
      )
      assert.equal(Math.round(pr.results[0].percent), 80)
      assert.equal(pr.results[0].marks, 8)
      assert.equal(pr.results[0].max, 10)
      assert.equal(Math.round(pr.average), 80)
      assert.equal(pr.by_class[0].class_name, 'PHY-204')
      assert.equal(pr.by_class[0].count, 1)
      assert.deepEqual(
        pr.todo.map((t) => t.title),
        ['Not yet'],
      ) // 'Second HW' is already submitted (graded, unpublished)
      // S3 never submitted anything: nothing to show, everything to do.
      const pr3 = (await call('c@x.edu', 'getProgress'))[1]
      assert.deepEqual(pr3.results, [])
      assert.equal(pr3.todo.length, 3)
    })

    test('announcements', async () => {
      assert.equal((await call('a@x.edu', 'postAnnouncement', { class_name: 'PHY-204', title: 'x', message: 'y' }))[0], 403)
      assert.equal((await call('t@x.edu', 'postAnnouncement', { class_name: 'PHY-204', title: 'x', message: 'y' }))[0], 400) // no assignment yet
      await call('t@x.edu', 'createAssignment', { title: 'HW', class_name: 'PHY-204' })
      await call('t2@x.edu', 'createAssignment', { title: 'Other', class_name: 'CHM-110' })
      assert.equal((await call('t@x.edu', 'postAnnouncement', { class_name: 'PHY-204', title: '', message: 'y' }))[0], 400)
      const [code, res] = await call('t@x.edu', 'postAnnouncement', {
        class_name: 'PHY-204',
        title: 'Quiz Friday',
        message: 'Chapters 3–4. Bring a calculator.',
      })
      assert.equal(code, 200)
      assert.equal(res.announcement.author_name, 'Priya')
      await call('t2@x.edu', 'postAnnouncement', { class_name: 'CHM-110', title: 'Lab moved', message: 'Room 12.' })
      const seen = (await call('a@x.edu', 'listAnnouncements'))[1].announcements.map((n) => n.title)
      assert.deepEqual(seen, ['Quiz Friday']) // S1 is enrolled in PHY-204 only
      assert.deepEqual(
        (await call('t@x.edu', 'listAnnouncements'))[1].announcements.map((n) => n.title),
        ['Quiz Friday'],
      )
      const nid = res.announcement.announcement_id
      assert.equal((await call('t2@x.edu', 'deleteAnnouncement', { announcement_id: nid }))[0], 403)
      assert.equal((await call('a@x.edu', 'deleteAnnouncement', { announcement_id: nid }))[0], 403)
      assert.equal((await call('t@x.edu', 'deleteAnnouncement', { announcement_id: nid }))[0], 200)
      assert.deepEqual((await call('a@x.edu', 'listAnnouncements'))[1].announcements, [])
      assert.equal((await call('t@x.edu', 'deleteAnnouncement', { announcement_id: nid }))[0], 404)
    })
  })
}
