// Teacher-to-teacher isolation for assignment_grading rows.
//
// A row belongs to the teacher who owns its assignment (Assignments.teacher_id).
// The row's own teacher_id is only a fallback for legacy rows whose assignment
// has no workflow record. A row with no owner either way is refused to every
// teacher (fail closed) — before, any teacher could read it, edit it, and so
// claim it. Admins can reach every row.
import * as wf from '../repositories/workflow.repository.js'

/** Memoised owner lookup, so a request over many rows reads each assignment once. */
export function ownerResolver() {
  const cache = new Map()
  return async (assignmentId, rowOwner) => {
    let assignmentOwner
    if (typeof assignmentId === 'string' && assignmentId) {
      if (!cache.has(assignmentId)) cache.set(assignmentId, (await wf.getAssignment(assignmentId))?.teacher_id)
      assignmentOwner = cache.get(assignmentId)
    }
    return assignmentOwner || rowOwner || null
  }
}

/** True when the caller may read or change grading data for this assignment/row. */
export async function canManage(caller, assignmentId, rowOwner, resolve = ownerResolver()) {
  if (caller.role === 'admin') return true
  if (caller.role !== 'teacher') return false
  return (await resolve(assignmentId, rowOwner)) === caller.user_id
}

/** The first teacher_id stamped on any of these rows (legacy ownership fallback). */
export const firstOwner = (rows) => rows.map((r) => r?.teacher_id).find(Boolean)
