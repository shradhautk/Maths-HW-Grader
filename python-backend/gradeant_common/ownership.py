"""Teacher-to-teacher isolation for assignment_grading rows (backend/src/services/ownership.js).

A row belongs to the teacher who owns its assignment (GradeAnt_Assignments.teacher_id).
The row's own teacher_id is only a fallback for legacy rows whose assignment has no
record. A row with no owner either way is refused to every teacher (fail closed).
Admins can reach every row.
"""
from . import dynamo


def owner_resolver():
    """Memoised owner lookup, so a request over many rows reads each assignment once."""
    cache = {}

    def resolve(assignment_id, row_owner=None):
        assignment_owner = None
        if isinstance(assignment_id, str) and assignment_id:
            if assignment_id not in cache:
                item = dynamo.table('assignments').get_item(Key={'assignment_id': assignment_id}).get('Item') or {}
                cache[assignment_id] = item.get('teacher_id')
            assignment_owner = cache[assignment_id]
        return assignment_owner or row_owner or None

    return resolve


def can_manage(caller, assignment_id, row_owner=None, resolve=None):
    """True when the caller may read or change grading data for this assignment/row."""
    if caller['role'] == 'admin':
        return True
    if caller['role'] != 'teacher':
        return False
    return (resolve or owner_resolver())(assignment_id, row_owner) == caller['user_id']


def owned_by_other(row, caller, resolve=None):
    """True when the row exists and the caller may not edit it."""
    return bool(row) and not can_manage(caller, row.get('assignment_id'), row.get('teacher_id'), resolve)


def first_owner(rows):
    """The first teacher_id stamped on any of these rows (legacy ownership fallback)."""
    return next((r.get('teacher_id') for r in rows if r and r.get('teacher_id')), None)


# Submission lifecycle, in order (backend/src/services/workflow/context.js SUBMISSION_CHAIN).
SUBMISSION_CHAIN = [
    'upload_requested', 'answer_uploaded', 'ocr_done', 'corrected', 'grading_triggered',
    'ai_graded', 'queued_for_review', 'teacher_reviewed', 'published', 'published_revised',
]
RANK = {s: i for i, s in enumerate(SUBMISSION_CHAIN)}


def rank(status):
    return RANK.get(status, 0)
