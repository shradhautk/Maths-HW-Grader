// Demo data for local development (same as the Python local_dev_server.py):
// an admin, a teacher, three students in PHY-204 and one graded assignment.
// Password for every demo account: Password123!
import { randomUUID } from 'node:crypto'
import { table } from '../data/index.js'
import config from '../config/index.js'
import { hashLoginPassword } from '../lib/passwords.js'
import { isoNaive } from '../lib/py.js'

export const DEMO_PASSWORD = 'Password123!'

const PEOPLE = [
  ['A1', 'admin@gradeant.dev', 'Alex Morgan', 'admin', null],
  ['T1', 'teacher@gradeant.dev', 'Dr. Priya Nair', 'teacher', null],
  ['S1', 'riya@gradeant.dev', 'Riya Sharma', 'student', ['PHY-204']],
  ['S2', 'sam@gradeant.dev', 'Sam Okafor', 'student', ['PHY-204']],
  ['S3', 'li@gradeant.dev', 'Li Wei', 'student', ['PHY-204']],
]

const QUESTIONS = [
  [1, 'Define internal energy and state its SI unit.', 5],
  [2, 'State the first law of thermodynamics for a closed system.', 5],
  [3, 'Calculate the work done in an isothermal expansion of 1 mol of ideal gas from 1 L to 2 L at 300 K.', 10],
]

// student -> per-question [marks, answer, feedback, confidence]
const RESULTS = {
  S1: [
    [5, 'U is the total energy of the system; unit J.', 'Correct.', 0.97],
    [4, 'dU = Q - W', 'Right relation; state the sign convention.', 0.95],
    [9, 'W = nRT ln(V2/V1) = 1729 J', 'Good — units clear.', 0.93],
  ],
  S2: [
    [3, 'energy inside', 'Missing the SI unit.', 0.9],
    [5, 'ΔU = Q − W for a closed system', 'Correct.', 0.96],
    [4, 'W = 8.314*300*ln2 = 1.7 kJ (sign?)', 'Method right; sign of work unclear.', 0.61],
  ],
}

export async function seedDemoData() {
  const users = table(config.tables.users)
  if (await users.get({ email: 'teacher@gradeant.dev' })) return // already seeded

  const now = isoNaive().slice(0, 19)
  const password = await hashLoginPassword(DEMO_PASSWORD)
  for (const [uid, email, name, role, classes] of PEOPLE) {
    const item = { user_id: uid, email, name, userRole: role, password, created_at: now, updated_at: now }
    if (classes) item.classes = classes
    await users.put(item)
    await table(config.tables.adminUsers).put({
      user_id: uid,
      email,
      name,
      role,
      status: 'active',
      createdAt: now,
      school: 'Lincoln HS',
    })
  }

  const aId = 'A-thermo'
  const qp = 'QP-thermo'
  const due = new Date(Date.now() + 3 * 86400000).toISOString().slice(0, 10)
  await table(config.tables.assignments).put({
    assignment_id: aId,
    qp_id: qp,
    title: 'Thermodynamics HW',
    class_name: 'PHY-204',
    teacher_id: 'T1',
    method: 'standard',
    persona: 'balanced',
    partial_credit: true,
    qp_status: 'rubric_reviewed',
    content_status: 'published',
    grades_status: 'teacher_reviewing',
    created_at: now,
    rubric: [
      { question_number: 1, max_marks: 5, title: 'Internal energy', criteria: 'Defines U as the total energy of the system; unit J.' },
      { question_number: 2, max_marks: 5, title: 'First law', criteria: 'ΔU = Q − W with a stated sign convention.' },
      { question_number: 3, max_marks: 10, title: 'Isothermal work', criteria: 'W = nRT ln(V2/V1) ≈ 1729 J; correct sign and units.' },
    ],
    due_at: due,
  })

  await table(config.tables.announcements).put({
    announcement_id: 'N-welcome',
    class_name: 'PHY-204',
    title: 'Welcome to PHY-204',
    message: 'Thermodynamics HW is open. Photograph every page of your working and upload it before the due date.',
    author_id: 'T1',
    author_name: 'Dr. Priya Nair',
    created_at: now,
  })

  const grading = table(config.tables.grading)
  const ts = Math.floor(Date.now() / 1000) - 3600
  for (const [sid, rows] of Object.entries(RESULTS)) {
    for (let i = 0; i < QUESTIONS.length; i++) {
      const [qn, question, maxMarks] = QUESTIONS[i]
      const [marks, answer, feedback, confidence] = rows[i]
      await grading.put({
        evaluation_id: `${sid}-${qn}-${randomUUID().replace(/-/g, '').slice(0, 6)}`,
        student_id: sid,
        assignment_id: aId,
        qp_id: qp,
        type: 'detail',
        question_number: qn,
        question,
        max_marks: maxMarks,
        marks,
        student_answer: answer,
        feedback,
        confidence,
        status: 'corrected',
        teacher_id: 'T1',
        timestamp: ts,
      })
    }
    await grading.put({
      evaluation_id: `${sid}-summary`,
      student_id: sid,
      assignment_id: aId,
      qp_id: qp,
      type: 'summary',
      total_marks: rows.reduce((total, r) => total + r[0], 0),
      teacher_id: 'T1',
      timestamp: ts,
    })
  }
  console.log(JSON.stringify({ level: 'info', msg: `seeded demo data (password for demo users: ${DEMO_PASSWORD})` }))
}
