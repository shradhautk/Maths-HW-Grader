// GradeFlow assistant (§04) and AI rubric drafting — Gemini (src/ai/index.js), prompts unchanged.
import { askAI } from '../../ai/index.js'
import { badRequest } from '../../lib/errors.js'
import { num, pyStr, sortBy, trunc, truthy } from '../../lib/py.js'
import * as wf from '../../repositories/workflow.repository.js'
import { defaultRubric } from './assignments.js'
import { gradingRows, isFlagged, loadAssignment, myAssignments, questions, requireRole, usersById } from './context.js'

async function gradingContext(assignment, studentId) {
  let byStudent = await gradingRows(assignment)
  if (truthy(studentId)) byStudent = new Map([...byStudent].filter(([k]) => k === studentId))
  const users = await usersById()
  const lines = [
    `Assignment: ${pyStr(assignment.title ?? null)} (${pyStr(assignment.class_name ?? null)})`,
    `Grading method: ${pyStr(assignment.method ?? 'not chosen')}, persona: ${pyStr(assignment.persona ?? 'balanced')}`,
  ]
  if (truthy(assignment.rubric)) {
    lines.push('Rubric:')
    for (const r of assignment.rubric) {
      lines.push(`  Q${pyStr(r.question_number)}${r.subpart || ''} (${pyStr(r.max_marks ?? null)} marks): ${r.criteria || r.title || ''}`)
    }
  }
  lines.push('Results:')
  for (const [sid, entry] of byStudent) {
    const name = users.get(sid)?.name || sid
    const total = entry.summary ? pyStr(entry.summary.total_marks ?? null) : 'not graded'
    lines.push(`- Student ${name} [${sid}]: total ${total}`)
    for (const r of sortBy(entry.details, (x) => [num(x.question_number), pyStr(x.subpart || '')])) {
      lines.push(
        `    Q${pyStr(r.question_number ?? null)}${r.subpart || ''}: ${pyStr(r.marks ?? '?')}/${pyStr(r.max_marks ?? '?')}` +
          `${isFlagged(r) ? ' (low confidence)' : ''}` +
          ` | answer: ${String(r.student_answer || '').trim()}` +
          ` | feedback: ${String(r.feedback || '').trim()}`,
      )
    }
  }
  return lines.join('\n')
}

const ASSISTANT_SYSTEM = `You are the GradeFlow assistant inside GradeAnt, an AI grading tool for STEM teachers.
You help the teacher understand and act on grading results: explain why an answer earned its marks against the rubric, \
find patterns across the class, identify students who are struggling, and draft feedback or re-teaching notes.
Ground every claim in the grading data provided below; if the data doesn't contain the answer, say so. \
Refer to students by the name shown. Keep replies concise and practical; use short lists where they help. \
You can't change grades yourself — when a change looks warranted, tell the teacher which score to override in the correction workspace.

Grading data:
`

export async function askAssistant(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const { messages } = body
  if (!Array.isArray(messages) || !messages.length) throw badRequest('messages is required.')
  const clean = messages.slice(-20).map((m) => {
    const role = m?.role
    const content = m?.content
    if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string' || !content.trim()) {
      throw badRequest('Each message needs role user|assistant and text content.')
    }
    return { role, content: trunc(content, 4000) }
  })
  while (clean.length && clean[0].role !== 'user') clean.shift()
  if (!clean.length || clean[clean.length - 1].role !== 'user') throw badRequest('The last message must be from the user.')

  let context
  if (truthy(body.assignment_id)) {
    const assignment = await loadAssignment(body.assignment_id, caller)
    context = await gradingContext(assignment, body.student_id)
  } else {
    const parts = []
    for (const a of await myAssignments(caller)) parts.push(await gradingContext(a))
    context = parts.join('\n\n') || 'No assignments yet.'
  }

  // Chat should answer quickly, so it uses low effort.
  const reply = await askAI({ system: ASSISTANT_SYSTEM + context, messages: clean, effort: 'low' })
  return {
    reply: reply ?? "I can't help with that request. Try asking about the grades, rubric or feedback for this assignment.",
  }
}

const RUBRIC_SCHEMA = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          question_number: { type: 'string' },
          subpart: { type: 'string' },
          title: { type: 'string' },
          max_marks: { type: 'number' },
          criteria: { type: 'string' },
        },
        required: ['question_number', 'subpart', 'title', 'max_marks', 'criteria'],
        additionalProperties: false,
      },
    },
  },
  required: ['items'],
  additionalProperties: false,
}

/** The drafted items from the model's reply (JSON, possibly wrapped in prose or a code fence), or null. */
export function parseRubricItems(text) {
  if (!text) return null
  const start = text.indexOf('{')
  const end = text.lastIndexOf('}')
  if (start < 0 || end <= start) return null
  try {
    const items = JSON.parse(text.slice(start, end + 1)).items
    return Array.isArray(items) ? items.filter((r) => r && typeof r === 'object') : null
  } catch {
    return null
  }
}

/** Background job: GradeFlow drafts a marking rubric from the reviewed questions. */
export async function draftRubric(assignmentId) {
  const item = await wf.getAssignment(assignmentId)
  if (!item || item.qp_status !== 'rubric_generating') return
  const qs = await questions(item)
  const rubric = defaultRubric(qs)
  if (qs.length) {
    const listing = qs
      .map((q) => `Q${pyStr(q.question_number)}${q.subpart || ''} [${num(q.max_marks)} marks]: ${q.question || ''}`)
      .join('\n')
    const persona = item.persona ?? 'balanced'
    const partial =
      !Object.hasOwn(item, 'partial_credit') || truthy(item.partial_credit)
        ? 'Award method marks (partial credit) for correct working.'
        : 'Award marks only for correct final answers.'
    let text = null
    try {
      text = await askAI({
        system:
          'You write marking rubrics for STEM homework. For each question, give a short title and concise criteria ' +
          'saying exactly what earns the marks (key steps, expected result with units/sign, common errors to penalise). ' +
          "Keep each question's max_marks exactly as given. Use an empty string for subpart when there is none.",
        messages: [{ role: 'user', content: `Grading persona: ${pyStr(persona)}. ${partial}\n\nQuestions:\n${listing}` }],
        effort: 'medium',
        outputFormat: { type: 'json_schema', schema: RUBRIC_SCHEMA },
      })
    } catch (err) {
      // Don't leave the assignment stuck in "Drafting rubric…": the teacher gets
      // the question-based rubric to fill in, as with the standard method.
      console.error(
        JSON.stringify({ level: 'error', msg: 'rubric drafting failed; using the default rubric', assignmentId, error: err?.name, detail: err?.message }),
      )
    }
    const items = parseRubricItems(text)
    if (items) {
      const drafted = new Map(items.map((r) => [JSON.stringify([pyStr(r?.question_number), r?.subpart || '']), r]))
      for (const r of rubric) {
        const d = drafted.get(JSON.stringify([pyStr(r.question_number), r.subpart || '']))
        if (d) {
          r.title = d.title || r.title
          r.criteria = d.criteria || ''
        }
      }
    }
  }
  try {
    await wf.updateAssignment(assignmentId, {
      UpdateExpression: 'SET rubric = :r, qp_status = :s',
      ConditionExpression: 'qp_status = :generating',
      // (null subparts are dropped, as workflow.py's _to_dynamo did.)
      ExpressionAttributeValues: {
        ':r': rubric.map(({ subpart, ...r }) => (subpart === null || subpart === undefined ? r : { ...r, subpart })),
        ':s': 'rubric_generated',
        ':generating': 'rubric_generating',
      },
    })
  } catch (err) {
    // The teacher switched method (or approved) while the draft was running.
    if (err?.name !== 'ConditionalCheckFailedException') throw err
  }
}
