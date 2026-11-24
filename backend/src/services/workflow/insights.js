// Teacher: grading queue, students, analytics and settings (wireframes §02, §11).
import { badRequest } from '../../lib/errors.js'
import { isoDate, mean, num, pyRound, pyStr, sortBy, truthy } from '../../lib/py.js'
import { setGradingSettings } from '../../repositories/users.repository.js'
import {
  AT_RISK_PERCENT,
  DEFAULT_SETTINGS,
  PERSONAS,
  RANK,
  enrolledStudents,
  isFlagged,
  marksTotal,
  myAssignments,
  questions,
  rank,
  requireRole,
  submissions,
  userSettings,
  usersById,
} from './context.js'

const letter = (p) => (p >= 90 ? 'A' : p >= 80 ? 'B' : p >= 70 ? 'C' : p >= 60 ? 'D' : 'F')
const isSubmitted = (s) => rank(s.status) >= RANK.answer_uploaded

/** Submissions, parsed questions and per-student percent for one assignment. */
async function stats(item, users) {
  const { subs, byStudent } = await submissions(item, users)
  const qs = await questions(item, byStudent)
  const maxMarks = marksTotal(qs) || num(item.total_marks)
  const percents = new Map()
  if (maxMarks > 0) {
    for (const [sid, entry] of byStudent) {
      if (entry.summary !== null) percents.set(sid, Math.min(100, (num(entry.summary.total_marks) / maxMarks) * 100))
    }
  }
  return { subs, byStudent, maxMarks, percents }
}

const QUEUE_ORDER = { review: 0, ready: 1, grading: 2, collecting: 3, setup: 4, published: 5 }

/** §02 "batches waiting to review": every assignment with where it is in grading. */
export async function getGradingQueue(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const items = await myAssignments(caller)
  const users = items.length ? await usersById() : new Map()
  const out = []
  for (const item of items) {
    const st = await stats(item, users)
    const submitted = st.subs.filter(isSubmitted)
    const graded = submitted.filter((s) => rank(s.status) >= RANK.ai_graded)
    const gradingCount = st.subs.filter((s) => s.status === 'grading_triggered' || s.status === 'corrected').length
    let flagged = 0
    for (const e of st.byStudent.values()) if (e.summary !== null) flagged += e.details.filter(isFlagged).length
    let state
    if (item.qp_status !== 'rubric_reviewed' && !(graded.length || gradingCount)) state = 'setup'
    else if (item.grades_status === 'published' && !flagged && !gradingCount) state = 'published'
    else if (flagged) state = 'review'
    else if (gradingCount) state = 'grading'
    else if (graded.length) state = 'ready'
    else state = 'collecting'
    out.push({
      assignment_id: item.assignment_id,
      qp_id: item.qp_id,
      title: item.title ?? null,
      class_name: item.class_name ?? null,
      due_at: item.due_at ?? null,
      state,
      papers: submitted.length,
      graded: graded.length,
      grading: gradingCount,
      flagged,
    })
  }
  const queue = sortBy(out, (r) => [QUEUE_ORDER[r.state], -r.flagged, r.due_at || '9'])
  return { queue, waiting: queue.filter((r) => r.state === 'review' || r.state === 'ready').length }
}

/** §02 Students: everyone in the teacher's classes with how they're doing. */
export async function listStudents(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const items = sortBy(await myAssignments(caller), (a) => a.created_at ?? '')
  const users = items.length ? await usersById() : new Map()
  const students = new Map()
  const classes = new Set()
  for (const item of items) {
    const st = await stats(item, users)
    classes.add(item.class_name)
    for (const s of st.subs) {
      const sid = s.student_id
      if (!students.has(sid)) {
        students.set(sid, {
          student_id: sid,
          student_name: s.student_name,
          email: users.get(sid)?.email ?? null,
          classes: new Set(),
          assigned: 0,
          submitted: 0,
          late: 0,
          pcts: [],
          last_submitted_at: null,
          latest: null,
        })
      }
      const rec = students.get(sid)
      rec.classes.add(item.class_name)
      rec.assigned += 1
      if (isSubmitted(s)) {
        rec.submitted += 1
        rec.late += s.late ? 1 : 0
        const at = s.submitted_at
        if (at && (rec.last_submitted_at === null || at >= rec.last_submitted_at)) {
          rec.last_submitted_at = at
          rec.latest = { assignment_id: item.assignment_id, qp_id: item.qp_id }
        }
      }
      if (st.percents.has(sid)) rec.pcts.push(st.percents.get(sid))
    }
  }
  const out = [...students.values()].map(({ pcts, classes: cls, ...rec }) => {
    const average = mean(pcts)
    return {
      ...rec,
      classes: sortBy([...cls].filter(truthy), (c) => c),
      average,
      at_risk: average !== null && average < AT_RISK_PERCENT,
    }
  })
  return {
    students: sortBy(out, (r) => (r.student_name || r.student_id).toLowerCase()),
    classes: sortBy([...classes].filter(truthy), (c) => c),
  }
}

const DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

/** §11 class analytics across assignments (optionally one class). */
export async function getTeacherAnalytics(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const everything = await myAssignments(caller)
  const classFilter = truthy(body.class_name) ? body.class_name : null
  const items = everything.filter((a) => !classFilter || a.class_name === classFilter)
  const users = items.length ? await usersById() : new Map()

  const today = new Date().toISOString().slice(0, 10)
  const days = Array.from({ length: 7 }, (_, i) => new Date(Date.parse(`${today}T00:00:00Z`) - (6 - i) * 86400000))
  const weekly = new Map(days.map((d) => [d.toISOString().slice(0, 10), 0]))
  const dist = { A: 0, B: 0, C: 0, D: 0, F: 0 }
  const rows = []
  const allPcts = []
  const risk = new Set()
  const hardestByKey = new Map()
  const upcoming = []
  let submittedTotal = 0
  let enrolledTotal = 0

  for (const item of items) {
    const st = await stats(item, users)
    for (const [sid, p] of st.percents) {
      dist[letter(p)] += 1
      allPcts.push(p)
      if (p < AT_RISK_PERCENT) risk.add(sid)
    }
    const submitted = st.subs.filter(isSubmitted)
    for (const s of submitted) {
      const d = isoDate(s.submitted_at)
      if (d && weekly.has(d)) weekly.set(d, weekly.get(d) + 1)
    }
    const enrolled = enrolledStudents(item, users).length || st.subs.length
    submittedTotal += submitted.length
    enrolledTotal += enrolled
    const pcts = [...st.percents.values()]
    rows.push({
      assignment_id: item.assignment_id,
      qp_id: item.qp_id,
      title: item.title ?? null,
      class_name: item.class_name ?? null,
      enrolled,
      submitted: submitted.length,
      graded: pcts.length,
      average: mean(pcts),
      at_risk: pcts.filter((p) => p < AT_RISK_PERCENT).length,
      grades_status: item.grades_status ?? null,
    })
    for (const entry of st.byStudent.values()) {
      if (entry.summary === null) continue
      for (const r of entry.details) {
        const max = num(r.max_marks)
        if (max > 0 && r.marks !== undefined && r.marks !== null) {
          const key = JSON.stringify([item.assignment_id, pyStr(r.question_number), r.subpart || ''])
          if (!hardestByKey.has(key)) {
            hardestByKey.set(key, {
              assignment_id: item.assignment_id,
              title: item.title ?? null,
              label: `Q${pyStr(r.question_number ?? null)}${r.subpart || ''}`,
              got: 0,
              max: 0,
            })
          }
          const q = hardestByKey.get(key)
          q.got += num(r.marks)
          q.max += max
        }
      }
    }
    if (truthy(item.due_at) && pyStr(item.due_at).slice(0, 10) >= today && item.grades_status !== 'published') {
      upcoming.push({ assignment_id: item.assignment_id, title: item.title ?? null, class_name: item.class_name ?? null, due_at: item.due_at })
    }
  }

  const hardest = sortBy(
    [...hardestByKey.values()].filter((q) => q.max > 0).map(({ got, max, ...q }) => ({ ...q, average: (got / max) * 100 })),
    (q) => q.average,
  ).slice(0, 5)
  let insight = null
  if (hardest.length && hardest[0].average < 100) {
    const h = hardest[0]
    insight = `${h.label} in "${pyStr(h.title)}" cost the most marks (${pyRound(h.average)}% average) — consider a recap before the next assessment.`
  }
  return {
    classes: sortBy([...new Set(everything.map((a) => a.class_name).filter(truthy))], (c) => c),
    class_name: classFilter,
    totals: {
      assignments: items.length,
      enrolled: enrolledTotal,
      submitted: submittedTotal,
      submission_rate: enrolledTotal ? (submittedTotal / enrolledTotal) * 100 : null,
      graded: allPcts.length,
      average: mean(allPcts),
      at_risk: risk.size,
    },
    distribution: dist,
    weekly: [...weekly.entries()].map(([date, count]) => ({ date, label: DAY_LABELS[new Date(`${date}T00:00:00Z`).getUTCDay()], count })),
    assignments: sortBy(rows, (r) => r.title || ''),
    hardest,
    insight,
    upcoming: sortBy(upcoming, (u) => pyStr(u.due_at)).slice(0, 5),
  }
}

export async function getTeacherSettings(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  return { settings: userSettings(caller.user) }
}

export async function updateTeacherSettings(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const incoming = body.settings
  if (!incoming || typeof incoming !== 'object' || Array.isArray(incoming)) throw badRequest('settings must be an object.')
  const merged = userSettings(caller.user)
  for (const key of Object.keys(DEFAULT_SETTINGS)) if (Object.hasOwn(incoming, key)) merged[key] = incoming[key]
  if (!PERSONAS.includes(merged.default_persona)) throw badRequest(`default_persona must be one of ${PERSONAS.join(', ')}.`)
  for (const key of ['partial_credit', 'notify_submissions', 'notify_flagged']) {
    if (typeof merged[key] !== 'boolean') throw badRequest(`${key} must be true or false.`)
  }
  await setGradingSettings(caller.email, merged)
  return { settings: merged }
}
