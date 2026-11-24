// Regression tests for the security review fixes: teacher-to-teacher isolation
// of grading rows, locked answers after grading, account suspension, admin
// session revocation, signup password rules and signed upload sizes.
import assert from 'node:assert/strict'
import { after, beforeEach, describe, test } from 'node:test'
import { config, post, putUsers, startServer, stopServer, table, tokenFor } from './helpers.js'

const { hashLoginPassword } = await import('../src/lib/passwords.js')
const { makeSessionToken } = await import('../src/lib/tokens.js')
const { sentMail } = await import('../src/aws/ses.js')

const PASSWORD = 'Password123!'
const USERS = [
  { email: 'admin@x.edu', user_id: 'A1', name: 'Alex', userRole: 'admin' },
  { email: 't@x.edu', user_id: 'T1', name: 'Priya', userRole: 'teacher' },
  { email: 't2@x.edu', user_id: 'T2', name: 'Other', userRole: 'teacher' },
  { email: 'a@x.edu', user_id: 'S1', name: 'Okafor', userRole: 'student' },
]
const tok = (email) => tokenFor(USERS.find((u) => u.email === email))

const grading = () => table(config.tables.grading)

/** T1's assignment with S1's rows. Like rows written by the grading pipeline, they carry no teacher_id. */
async function seed({ graded = false } = {}) {
  await table(config.tables.assignments).put({ assignment_id: 'A-1', qp_id: 'QP-1', teacher_id: 'T1', class_name: 'PHY', title: 'HW' })
  await grading().put({
    evaluation_id: 'S1-1',
    student_id: 'S1',
    assignment_id: 'A-1',
    qp_id: 'QP-1',
    type: 'detail',
    question_number: 1,
    question: 'Q1',
    student_answer: 'ans',
    max_marks: 5,
  })
  if (graded) {
    await grading().put({ evaluation_id: 'S1-sum', student_id: 'S1', assignment_id: 'A-1', qp_id: 'QP-1', type: 'summary', total_marks: 4 })
  }
}

beforeEach(async () => {
  await startServer()
  const password = await hashLoginPassword(PASSWORD)
  await putUsers(USERS.map((u) => ({ ...u, password })))
})
after(stopServer)

describe('teacher isolation for ownerless grading rows', () => {
  test("another teacher can't read or change them; the assignment's teacher and admins can", async () => {
    await seed({ graded: true })
    const read = { operation: 'getSingleStudentGrade', student_id: 'S1', assignment_id: 'A-1' }
    assert.equal((await post('/getgrades', read, { token: tok('t2@x.edu') }))[0], 403)
    assert.equal((await post('/getgrades', read, { token: tok('t@x.edu') }))[0], 200)
    assert.equal((await post('/getgrades', read, { token: tok('admin@x.edu') }))[0], 200)

    const roster = { operation: 'getAllStudentGrades', qp_id: 'QP-1', assignment_id: 'A-1' }
    assert.equal((await post('/getgrades', roster, { token: tok('t2@x.edu') }))[1].student_count, 0)
    assert.equal((await post('/getgrades', roster, { token: tok('t@x.edu') }))[1].student_count, 1)

    const answers = { operation: 'getAnswersForCorrection', student_id: 'S1', assignment_id: 'A-1' }
    assert.equal((await post('/correctanswer', answers, { token: tok('t2@x.edu') }))[0], 403)

    const edit = { operation: 'updateStudentGrade', evaluation_id: 'S1-sum', total_marks: 10 }
    assert.equal((await post('/updatedb', edit, { token: tok('t2@x.edu') }))[0], 403)
    assert.equal((await grading().get({ evaluation_id: 'S1-sum' })).total_marks, 4)
    assert.equal((await post('/updatedb', edit, { token: tok('t@x.edu') }))[0], 200)

    const question = { operation: 'submitQuestionCorrections', qp_id: 'QP-1', assignment_id: 'A-1', corrections: [{ question_number: 1, corrected_question: 'Hacked' }] }
    assert.equal((await post('/correctanswer', question, { token: tok('t2@x.edu') }))[0], 404)
    assert.equal((await grading().get({ evaluation_id: 'S1-1' })).question, 'Q1')
  })

  test('an admin edit does not move ownership away from the teacher', async () => {
    await seed({ graded: true })
    await post('/updatedb', { operation: 'updateStudentGrade', evaluation_id: 'S1-sum', total_marks: 9 }, { token: tok('t@x.edu') })
    await post('/updatedb', { operation: 'updateStudentGrade', evaluation_id: 'S1-sum', total_marks: 7 }, { token: tok('admin@x.edu') })
    assert.equal((await grading().get({ evaluation_id: 'S1-sum' })).teacher_id, 'T1')
  })

  test('rows with no owner at all are refused to every teacher', async () => {
    await grading().put({ evaluation_id: 'X-sum', student_id: 'S1', assignment_id: 'GONE', qp_id: 'QP-9', type: 'summary', total_marks: 1 })
    const edit = { operation: 'updateStudentGrade', evaluation_id: 'X-sum', total_marks: 10 }
    assert.equal((await post('/updatedb', edit, { token: tok('t@x.edu') }))[0], 403)
    assert.equal((await post('/updatedb', edit, { token: tok('admin@x.edu') }))[0], 200)
  })
})

describe('student answer corrections', () => {
  const correction = { operation: 'submitAnswerCorrections', corrections: [{ evaluation_id: 'S1-1', question_number: 1, corrected_answer: 'fixed' }] }

  test('allowed before grading', async () => {
    await seed()
    assert.equal((await post('/correctanswer', correction, { token: tok('a@x.edu') }))[0], 200)
  })

  test('refused once the paper is graded', async () => {
    await seed({ graded: true })
    const [status, body] = await post('/correctanswer', correction, { token: tok('a@x.edu') })
    assert.equal(status, 404)
    assert.match(body.errors[0], /already been graded/)
    assert.equal((await grading().get({ evaluation_id: 'S1-1' })).student_answer, 'ans')
  })
})

describe('accounts', () => {
  test('signup enforces the password rules', async () => {
    const [status, body] = await post('/auth/signup', { email: 'n@x.edu', password: 'short', role: 'student' })
    assert.equal(status, 400)
    assert.match(body.message, /at least 8/)
    assert.equal((await post('/auth/signup', { email: 'n@x.edu', password: PASSWORD, role: 'student' }))[0], 201)
  })

  test('suspending a user in the admin console blocks login and existing sessions', async () => {
    await table(config.tables.adminUsers).put({ user_id: 'G-S1', email: 'a@x.edu', role: 'student', status: 'active' })
    const session = (await post('/auth/login', { email: 'a@x.edu', password: PASSWORD, userType: 'student' }))[1].token
    const adminToken = tok('admin@x.edu')
    assert.equal((await post('/admin', { operation: 'updateUserStatus', userId: 'G-S1', status: 'suspended' }, { token: adminToken }))[0], 200)

    assert.equal((await post('/auth/login', { email: 'a@x.edu', password: PASSWORD, userType: 'student' }))[0], 403)
    assert.equal((await post('/getgrades', { operation: 'x' }, { token: session }))[0], 401)
    assert.equal((await post('/auth/me', {}, { token: session }))[0], 401)

    await post('/admin', { operation: 'updateUserStatus', userId: 'G-S1', status: 'active' }, { token: adminToken })
    assert.equal((await post('/auth/login', { email: 'a@x.edu', password: PASSWORD, userType: 'student' }))[0], 200)
  })

  test('a deleted admin loses their console session at once', async () => {
    await table(config.tables.adminUsers).put({ user_id: 'G-A9', email: 'x@x.edu', role: 'admin', status: 'active' })
    const session = await makeSessionToken({ sub: 'G-A9', role: 'admin' }, 3600)
    assert.equal((await post('/admin', { operation: 'listUsers' }, { token: session }))[0], 200)

    assert.equal((await post('/admin', { operation: 'deleteUser', userId: 'G-A9' }, { token: tok('admin@x.edu') }))[0], 200)
    assert.equal((await post('/admin', { operation: 'listUsers' }, { token: session }))[0], 401)
  })

  test('app.yaml decides which roles can sign themselves up', async () => {
    // Admins never sign themselves up; unknown roles are refused.
    assert.equal((await post('/auth/signup', { email: 'admin2@x.edu', password: PASSWORD, role: 'admin' }))[0], 403)
    assert.equal((await post('/auth/signup', { email: 'x@x.edu', password: PASSWORD, role: 'owner' }))[0], 400)

    const saved = config.app.roles.teacher.signup
    try {
      // roles.teacher.signup: [invite] → teachers only by admin invite.
      config.app.roles.teacher.signup = ['invite']
      const [status, body] = await post('/auth/signup', { email: 'tt2@x.edu', password: PASSWORD, role: 'teacher' })
      assert.equal(status, 403)
      assert.match(body.message, /Teacher accounts are invite-only/)
      assert.equal(await table(config.tables.users).get({ email: 'tt2@x.edu' }), undefined)

      // roles.teacher.signup: [self, invite] (shipped) → teachers can register themselves.
      config.app.roles.teacher.signup = ['self', 'invite']
      assert.equal((await post('/auth/signup', { email: 'tt@x.edu', password: PASSWORD, role: 'teacher' }))[0], 201)
    } finally {
      config.app.roles.teacher.signup = saved
    }
  })

  test('app.yaml permissions can turn an operation off', async () => {
    await seed({ graded: true })
    const read = { operation: 'getSingleStudentGrade', student_id: 'S1', assignment_id: 'A-1' }
    const saved = config.app.permissions.grades.getSingleStudentGrade
    config.app.permissions.grades.getSingleStudentGrade = ['teacher', 'admin'] // students off
    try {
      assert.equal((await post('/getgrades', read, { token: tok('a@x.edu') }))[0], 403)
      assert.equal((await post('/getgrades', read, { token: tok('t@x.edu') }))[0], 200)
    } finally {
      config.app.permissions.grades.getSingleStudentGrade = saved
    }
  })

  test('a role that is not invitable is refused by inviteUsers', async () => {
    const saved = config.app.roles.teacher.signup
    config.app.roles.teacher.signup = ['self']
    try {
      const invitees = [{ email: 'z@x.edu', name: 'Z', role: 'teacher' }]
      assert.equal((await post('/admin', { operation: 'inviteUsers', invitees }, { token: tok('admin@x.edu') }))[0], 400)
    } finally {
      config.app.roles.teacher.signup = saved
    }
  })

  test("an invited student's account records the invite's class", async () => {
    const before = sentMail.length
    const invitees = [{ email: 'news@x.edu', name: 'New Student', role: 'student', className: 'PHY-204' }]
    assert.equal((await post('/admin', { operation: 'inviteUsers', invitees }, { token: tok('admin@x.edu') }))[0], 200)
    const link = sentMail.slice(before)[0].html.match(/href="([^"]+)"/)[1]
    const token = new URL(link.replace(/&amp;/g, '&')).searchParams.get('token')
    assert.equal((await post('/admin', { operation: 'completeFirstLogin', token, password: 'Welcome123' }))[0], 200)
    assert.deepEqual((await table(config.tables.users).get({ email: 'news@x.edu' })).classes, ['PHY-204'])
  })

  test('an invited teacher gets a working app account', async () => {
    const before = sentMail.length
    const invitees = [{ email: 'newt@x.edu', name: 'New Teacher', role: 'teacher' }]
    assert.equal((await post('/admin', { operation: 'inviteUsers', invitees }, { token: tok('admin@x.edu') }))[0], 200)
    const link = sentMail.slice(before)[0].html.match(/href="([^"]+)"/)[1]
    assert.match(link, /\/first-login\?token=/)
    const token = new URL(link.replace(/&amp;/g, '&')).searchParams.get('token')

    assert.equal((await post('/admin', { operation: 'completeFirstLogin', token, password: 'longenough1' }))[0], 400) // no uppercase
    const [status, done] = await post('/admin', { operation: 'completeFirstLogin', token, password: 'Welcome123' })
    assert.equal(status, 200)
    assert.equal(done.role, 'teacher')
    // The returned token works on the teacher app APIs, and the teacher can log in normally.
    assert.equal((await post('/workflow', { operation: 'listAssignments' }, { token: done.token }))[0], 200)
    const [loginStatus, session] = await post('/auth/login', { email: 'newt@x.edu', password: 'Welcome123', userType: 'teacher' })
    assert.equal(loginStatus, 200)
    assert.equal(session.user_id, done.userId)
    // The invite can't be used twice.
    assert.equal((await post('/admin', { operation: 'completeFirstLogin', token, password: 'Welcome123' }))[0], 400)
  })

  test("an invite never takes over an existing login account", async () => {
    const before = sentMail.length
    const invitees = [{ email: 'a@x.edu', name: 'Okafor', role: 'teacher' }] // already a student
    await post('/admin', { operation: 'inviteUsers', invitees }, { token: tok('admin@x.edu') })
    const link = sentMail.slice(before)[0].html.match(/href="([^"]+)"/)[1]
    const token = new URL(link.replace(/&amp;/g, '&')).searchParams.get('token')
    assert.equal((await post('/admin', { operation: 'completeFirstLogin', token, password: 'Welcome123' }))[0], 409)
    const student = await table(config.tables.users).get({ email: 'a@x.edu' })
    assert.equal(student.userRole, 'student')
    assert.equal((await post('/auth/login', { email: 'a@x.edu', password: PASSWORD, userType: 'student' }))[0], 200)
  })

  test('a demoted admin loses admin access even with an admin-role session token', async () => {
    await table(config.tables.adminUsers).put({ user_id: 'G-A8', email: 'y@x.edu', role: 'teacher', status: 'active' })
    const session = await makeSessionToken({ sub: 'G-A8', role: 'admin' }, 3600)
    assert.equal((await post('/admin', { operation: 'listUsers' }, { token: session }))[0], 403)
  })
})

describe('uploads', () => {
  /** A published PHY assignment, and S1 enrolled in PHY. */
  async function enrolled() {
    await table(config.tables.assignments).put({ assignment_id: 'A-UP', qp_id: 'QP-UP', teacher_id: 'T1', class_name: 'PHY', title: 'HW', content_status: 'published' })
    await table(config.tables.users).update({ Key: { email: 'a@x.edu' }, UpdateExpression: 'SET classes = :c', ExpressionAttributeValues: { ':c': ['PHY'] } })
  }

  test('a student can only upload answers for assignments of their own classes', async () => {
    await enrolled()
    const file = [{ name: 'p.pdf', type: 'application/pdf', size: 8 }]
    const as = { token: tok('a@x.edu') }
    await table(config.tables.assignments).put({ assignment_id: 'A-OTHER', qp_id: 'QP-O', teacher_id: 'T2', class_name: 'CHEM', title: 'X', content_status: 'published' })
    await table(config.tables.assignments).put({ assignment_id: 'A-DRAFT', qp_id: 'QP-D', teacher_id: 'T1', class_name: 'PHY', title: 'Y', content_status: 'draft' })
    for (const assignmentId of ['A-OTHER', 'A-DRAFT', 'A-MISSING', undefined]) {
      const [status, body] = await post('/uploadpdf', { uploadType: 'student', assignmentId, files: file }, as)
      assert.equal(status, 403, String(assignmentId))
      assert.match(body.message, /your own classes/)
    }
    assert.equal((await post('/uploadpdf', { uploadType: 'student', assignmentId: 'A-UP', files: file }, as))[0], 200)
  })

  test('the presigned URL only accepts the size that was validated', async () => {
    await enrolled()
    const pdf = Buffer.from('%PDF-1.4')
    const meta = { uploadType: 'student', assignmentId: 'A-UP', files: [{ name: 'p.pdf', type: 'application/pdf', size: pdf.length }] }
    const [, { urls }] = await post('/uploadpdf', meta, { token: tok('a@x.edu') })
    const big = Buffer.concat([pdf, Buffer.alloc(1024)])
    assert.equal((await fetch(urls[0], { method: 'PUT', headers: { 'Content-Type': 'application/pdf' }, body: big })).status, 403)
    assert.equal((await fetch(urls[0], { method: 'PUT', headers: { 'Content-Type': 'application/pdf' }, body: pdf })).status, 200)
  })
})
