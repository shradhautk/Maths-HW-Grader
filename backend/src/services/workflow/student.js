// Student: assignment inbox, re-check requests (§14) and progress (§03, §12).
import { badRequest, conflict } from '../../lib/errors.js'
import { daysBetween, isoDate, mean, num, pyRound, pyStr, sortBy, trunc, truthy } from '../../lib/py.js'
import * as grading from '../../repositories/grading.repository.js'
import * as wf from '../../repositories/workflow.repository.js'
import {
  RANK,
  derivedStatus,
  firstRowTime,
  iso,
  mergeStatus,
  now,
  rank,
  required,
  requireRole,
  studentDetails,
  visibleAssignments,
} from './context.js'

async function studentEntry(studentId, assignment) {
  const details = await studentDetails(studentId, assignment)
  const summaries = await grading.studentRows(studentId, { assignmentId: assignment.assignment_id, type: 'summary' })
  const sub = (await wf.getSubmission(assignment.assignment_id, studentId)) ?? {}
  const entry = { summary: summaries[0] ?? null, details }
  return { entry, sub, status: mergeStatus(derivedStatus(entry), sub.status) }
}

export async function listStudentAssignments(body, caller) {
  requireRole(caller, 'student')
  const published = await visibleAssignments(caller)
  const out = []
  for (const a of sortBy(published, (x) => x.due_at || x.created_at || '')) {
    const { entry, status: merged } = await studentEntry(caller.user_id, a)
    // Students only ever see published grades; teacher-side states read as "grading".
    const status = ['ai_graded', 'queued_for_review', 'teacher_reviewed'].includes(merged) ? 'grading_triggered' : merged
    const view = {
      assignment_id: a.assignment_id,
      qp_id: a.qp_id,
      title: a.title,
      class_name: a.class_name,
      due_at: a.due_at,
      question_count: a.question_count,
      total_marks: a.total_marks,
      submission_status: status,
    }
    if (entry.summary !== null && (status === 'published' || status === 'published_revised')) {
      view.evaluation_id = entry.summary.evaluation_id
    }
    out.push(Object.fromEntries(Object.entries(view).filter(([, v]) => v !== undefined && v !== null)))
  }
  return { assignments: out }
}

export async function requestRecheck(body, caller) {
  requireRole(caller, 'student')
  const [assignmentId, items] = required(body, 'assignment_id', 'items')
  if (!Array.isArray(items) || !items.length) throw badRequest('Pick at least one question.')
  const clean = items.map((raw) => {
    const i = raw && typeof raw === 'object' ? raw : {}
    const note = pyStr(truthy(i.note) ? i.note : '').trim()
    if (i.question_number === undefined || i.question_number === null || i.question_number === '' || !note) {
      throw badRequest('Each item needs a question_number and a note.')
    }
    // (subpart is dropped when empty, as workflow.py's _to_dynamo did.)
    const item = { question_number: i.question_number, note: trunc(note, 1000) }
    if (truthy(i.subpart)) item.subpart = i.subpart
    return item
  })
  const sub = await wf.getSubmission(assignmentId, caller.user_id)
  if (!sub || !['published', 'published_revised'].includes(sub.status)) {
    throw conflict('You can request a re-check once your grades are published.')
  }
  await wf.updateSubmission(assignmentId, caller.user_id, {
    UpdateExpression: 'SET recheck = :r, recheck_requested_at = :t, recheck_status = :s',
    ExpressionAttributeValues: { ':r': clean, ':t': now(), ':s': 'open' },
  })
  return { ok: true }
}

/** Grade trend, by-class averages, due soon and a personal insight — published work only. */
export async function getProgress(body, caller) {
  requireRole(caller, 'student')
  const uid = caller.user_id
  const results = []
  const todo = []
  for (const a of await visibleAssignments(caller)) {
    const { entry, sub, status } = await studentEntry(uid, a)
    const { details } = entry
    if ((status === 'published' || status === 'published_revised') && entry.summary !== null) {
      const max = details.reduce((t, r) => t + num(r.max_marks), 0)
      if (max <= 0) continue
      const marks = num(entry.summary.total_marks)
      const first = firstRowTime(details)
      const submittedAt = truthy(sub.submitted_at) ? iso(sub.submitted_at) : first !== '' ? iso(first) : null
      const publishedAt =
        sub.published_at || (truthy(entry.summary.timestamp) ? iso(entry.summary.timestamp) : (a.created_at ?? null))
      let early = null
      let onTime = null
      if (truthy(a.due_at) && submittedAt) {
        const due = isoDate(a.due_at)
        const submitted = isoDate(submittedAt)
        if (due && submitted) {
          const gap = daysBetween(due, submitted)
          early = gap >= 1
          onTime = gap >= 0
        }
      }
      results.push({
        assignment_id: a.assignment_id,
        qp_id: a.qp_id,
        title: a.title ?? null,
        class_name: a.class_name ?? null,
        marks,
        max,
        percent: Math.min(100, (marks / max) * 100),
        date: publishedAt,
        submitted_at: submittedAt,
        due_at: a.due_at ?? null,
        early,
        on_time: onTime,
      })
    } else if (rank(status) < RANK.answer_uploaded) {
      todo.push({
        assignment_id: a.assignment_id,
        qp_id: a.qp_id,
        title: a.title ?? null,
        class_name: a.class_name ?? null,
        due_at: a.due_at ?? null,
      })
    }
  }
  const sortedResults = sortBy(results, (r) => pyStr(r.date))
  const sortedTodo = sortBy(todo, (t) => pyStr(t.due_at || '9'))

  const byClass = new Map()
  for (const r of sortedResults) {
    const key = r.class_name || 'Other'
    if (!byClass.has(key)) byClass.set(key, [])
    byClass.get(key).push(r.percent)
  }
  let streak = 0
  for (const r of [...sortedResults].reverse()) {
    if (r.on_time === false || r.on_time === null) break
    streak += 1
  }
  const early = sortedResults.filter((r) => r.early).map((r) => r.percent)
  const late = sortedResults.filter((r) => r.early === false).map((r) => r.percent)
  let insight = null
  if (early.length && late.length && mean(early) > mean(late)) {
    insight = `You score about ${pyRound(mean(early) - mean(late))} points higher when you submit a day or more early. Keep the streak going!`
  } else if (streak >= 2) {
    insight = `${streak} on-time submissions in a row — keep the streak going!`
  }
  return {
    results: sortedResults,
    average: mean(sortedResults.map((r) => r.percent)),
    by_class: sortBy([...byClass.entries()], ([c]) => c).map(([c, v]) => ({ class_name: c, average: mean(v), count: v.length })),
    todo: sortedTodo.slice(0, 5),
    streak,
    insight,
  }
}
