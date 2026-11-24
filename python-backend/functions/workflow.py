"""
GradeAnt — assignment workflow Lambda  (POST /workflow)
=======================================================
Backs the React screens built from the website wireframes §04, §13–§15,
§19 and §20: create assignment → review parsed questions → choose AI or
standard grading → rubric → verify → grading run → spot-check → publish,
the student inbox / re-check request, and the GradeFlow assistant.

Same conventions as get_grades.py / correct_questions_answer.py: one API
Gateway resource with Lambda **proxy** integration, body
`{"operation": "...", ...}`, `Authorization: Bearer <JWT>` issued by
authetication.py (same JWT_SECRET), role looked up fresh from the Users
table on every request.

Data
----
Existing (read, plus a few flag fields written):
  assignment_grading   PK evaluation_id. One row per question per student
                       plus one type="summary" row per student. GSIs
                       student_id-index, type-index. Written by the grading
                       pipeline and the correction / grades Lambdas.
  Users                PK email. user_id, name, userRole, optional
                       `classes` (list of class names) for enrolment.
New (owned by this Lambda):
  GradeAnt_Assignments PK assignment_id (S). GSI teacher_id-index (PK teacher_id).
  GradeAnt_Submissions PK assignment_id (S), SK student_id (S).
                       GSI student_id-index (PK student_id).
  GradeAnt_Announcements PK announcement_id (S). Teacher → class notices
                       (read by scan; volumes are small).

A submission's status (wireframe §20 SUBMISSIONS chain) is derived from
its assignment_grading rows (answer rows → ocr_done, every row
status="corrected" → corrected, a summary row → ai_graded, an unresolved
low-confidence row → queued_for_review) and combined with the workflow
statuses this Lambda records itself (grading_triggered, teacher_reviewed,
published, published_revised). The later of the two wins.

Students see an assignment as soon as it's created (content_status
"published"): the question paper is only parsed once answers come in, so
the teacher's question review necessarily follows the first submissions.

Environment
-----------
  JWT_SECRET            must match authetication.py (REQUIRED in production)
  USER_TABLE_NAME       default "Users"
  GRADING_TABLE         default "assignment_grading"
  GRADING_ASSIGNMENT_INDEX default "assignment_id-index" — recommended GSI on
                        assignment_grading (PK assignment_id). Without it, a
                        student shows up once graded, once this Lambda has a
                        record for them, or via Users.classes enrolment.
  ASSIGNMENTS_TABLE     default "GradeAnt_Assignments"
  SUBMISSIONS_TABLE     default "GradeAnt_Submissions"
  ANNOUNCEMENTS_TABLE   default "GradeAnt_Announcements"
  FLAG_CONFIDENCE       default 0.75 — rows with a lower `confidence` are spot-checked
  GEMINI_MODEL          default gemini-3.5-flash-preview — the one model for every AI call
  GEMINI_API_KEY(_ARN)  the Gemini key (value, or a Secrets Manager secret)
  GRADING_FUNCTION_NAME optional — Lambda invoked (async) per submission on startGrading
  SNS_TOPIC_ARN         optional — "grades published" notification
  SES_SENDER_EMAIL      optional — From address for submission reminders

Dependencies (bundle in a layer): PyJWT, google-genai. boto3 ships
with the Lambda runtime. Timeout: 60 s (askAssistant calls Gemini).
IAM: DynamoDB on the four tables, the Gemini secret (if an ARN is used),
lambda:InvokeFunction on itself (+ GRADING_FUNCTION_NAME), sns:Publish,
ses:SendEmail as configured.
"""
import datetime
import json
import os
import uuid
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from gradeant_common import ai, config, dynamo
from gradeant_common.dynamo import query_all as _query_all, scan_all as _scan_all, to_dynamo as _to_dynamo
from gradeant_common.dispatch import operations
from gradeant_common.http import ApiError, create_response, json_default as _json_default

GRADING_ASSIGNMENT_INDEX = config.grading_assignment_index()
FLAG_CONFIDENCE = config.env_float('FLAG_CONFIDENCE', 0.75)
INTERNAL_SOURCE = 'gradeant.workflow'

# Table names: gradeant_common.config (USERS_TABLE, GRADING_TABLE, ... as in backend/.env.example).
users_table = dynamo.table('users')
grading_table = dynamo.table('grading')
assignments_table = dynamo.table('assignments')
submissions_table = dynamo.table('submissions')
announcements_table = dynamo.table('announcements')

SUBMISSION_CHAIN = [
    'upload_requested', 'answer_uploaded', 'ocr_done', 'corrected', 'grading_triggered',
    'ai_graded', 'queued_for_review', 'teacher_reviewed', 'published', 'published_revised',
]
RANK = {s: i for i, s in enumerate(SUBMISSION_CHAIN)}
PERSONAS = ('balanced', 'strict', 'lenient')
AT_RISK_PERCENT = 50
DEFAULT_SETTINGS = {'default_persona': 'balanced', 'partial_credit': True, 'notify_submissions': True, 'notify_flagged': True}


# ----------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------

def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _require(caller, *roles):
    if caller['role'] not in roles:
        raise ApiError(403, "Your role can't do that.")


def _required(body, *fields):
    missing = [f for f in fields if body.get(f) in (None, '')]
    if missing:
        raise ApiError(400, f"Missing required field(s): {', '.join(missing)}")
    return [body[f] for f in fields]


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------

def _my_assignments(caller):
    if caller['role'] == 'admin':
        return _scan_all(assignments_table)
    return _query_all(assignments_table, IndexName='teacher_id-index', KeyConditionExpression=Key('teacher_id').eq(caller['user_id']))


def _visible_assignments(caller):
    """Published assignments of the student's own classes (Users.classes). A student with no
    classes sees none: falling back to every school's assignments leaked them across schools."""
    classes = caller['user'].get('classes') or []
    if not classes:
        return []
    published = _scan_all(assignments_table, FilterExpression=Attr('content_status').eq('published'))
    return [a for a in published if a.get('class_name') in classes]


def _load_assignment(assignment_id, caller=None):
    item = assignments_table.get_item(Key={'assignment_id': assignment_id}).get('Item')
    if not item:
        raise ApiError(404, 'Assignment not found.')
    if caller and caller['role'] == 'teacher' and item.get('teacher_id') != caller['user_id']:
        raise ApiError(403, 'This assignment belongs to a different teacher.')
    return item


def _grading_rows(assignment, users=None):
    """Every assignment_grading row for this assignment, grouped by student."""
    rows = _assignment_rows(assignment)
    if rows is not None:
        by_student = {}
        for r in rows:
            entry = by_student.setdefault(r['student_id'], {'summary': None, 'details': []})
            if r.get('type') == 'summary':
                entry['summary'] = r
            else:
                entry['details'].append(r)
        return by_student

    # No assignment_id index: find candidate students (graded, recorded by
    # this Lambda, or enrolled via Users.classes) and read their rows.
    summaries = _query_all(
        grading_table,
        IndexName='type-index',
        KeyConditionExpression=Key('type').eq('summary'),
        FilterExpression=Attr('assignment_id').eq(assignment['assignment_id']) & Attr('qp_id').eq(assignment['qp_id']),
    )
    candidates = {s['student_id'] for s in summaries}
    candidates |= {s['student_id'] for s in _query_all(submissions_table, KeyConditionExpression=Key('assignment_id').eq(assignment['assignment_id']))}
    candidates |= {u['user_id'] for u in _enrolled_students(assignment, users if users is not None else _users_by_id())}
    by_student = {}
    for student_id in candidates:
        details = _student_details(student_id, assignment)
        summary = next((s for s in summaries if s['student_id'] == student_id), None)
        if details or summary is not None:
            by_student[student_id] = {'summary': summary, 'details': details}
    return by_student


def _assignment_rows(assignment):
    """All rows via GRADING_ASSIGNMENT_INDEX (PK assignment_id) — None if that GSI doesn't exist."""
    try:
        rows = _query_all(
            grading_table,
            IndexName=GRADING_ASSIGNMENT_INDEX,
            KeyConditionExpression=Key('assignment_id').eq(assignment['assignment_id']),
            FilterExpression=Attr('qp_id').eq(assignment['qp_id']),
        )
    except ClientError as e:
        if e.response['Error']['Code'] in ('ValidationException', 'ResourceNotFoundException'):
            return None
        raise
    return [r for r in rows if r.get('student_id')]


def _student_details(student_id, assignment):
    return _query_all(
        grading_table,
        IndexName='student_id-index',
        KeyConditionExpression=Key('student_id').eq(student_id),
        FilterExpression=Attr('assignment_id').eq(assignment['assignment_id'])
        & Attr('qp_id').eq(assignment['qp_id'])
        & Attr('type').ne('summary'),
    )


def _is_flagged(row):
    conf = row.get('confidence')
    return conf is not None and _num(conf, 1.0) < FLAG_CONFIDENCE and not row.get('flag_resolved_at')


def _derived_status(entry):
    details, summary = entry['details'], entry['summary']
    if summary is not None:
        return 'queued_for_review' if any(_is_flagged(r) for r in details) else 'ai_graded'
    if details:
        if all(r.get('status') == 'corrected' for r in details):
            return 'corrected'
        return 'ocr_done'
    return None


def _merge_status(derived, stored):
    candidates = [s for s in (derived, stored) if s in RANK]
    if not candidates:
        return 'upload_requested'
    best = max(candidates, key=lambda s: RANK[s])
    # A paper that's flagged again after the teacher reviewed it goes back to the queue.
    if derived == 'queued_for_review' and stored in ('teacher_reviewed', 'grading_triggered', 'ai_graded'):
        return 'queued_for_review'
    return best


def _users_by_id():
    users = _scan_all(users_table, ProjectionExpression='user_id, email, #n, userRole, classes', ExpressionAttributeNames={'#n': 'name'})
    return {u['user_id']: u for u in users if u.get('user_id')}


def _enrolled_students(assignment, users):
    """Students whose Users.classes lists this assignment's class (if anyone uses `classes`)."""
    cls = assignment.get('class_name')
    return [u for u in users.values() if u.get('userRole') == 'student' and cls and cls in (u.get('classes') or [])]


def _questions(assignment, by_student=None):
    """One canonical copy of each parsed question (question text is denormalised onto every student's rows)."""
    by_student = by_student if by_student is not None else _grading_rows(assignment)
    questions = {}
    for entry in by_student.values():
        for r in entry['details']:
            key = (str(r.get('question_number')), r.get('subpart') or '')
            if key not in questions:
                questions[key] = {
                    'question_number': r.get('question_number'),
                    'subpart': r.get('subpart'),
                    'question': r.get('question'),
                    'max_marks': r.get('max_marks'),
                    'ocr_confidence': r.get('question_confidence'),
                    'question_reviewed': bool(r.get('question_reviewed_at')),
                }
    return sorted(questions.values(), key=lambda q: (_num(q['question_number'], 0), str(q.get('subpart') or '')))


def _assignment_view(item, by_student=None):
    by_student = by_student if by_student is not None else _grading_rows(item)
    questions = _questions(item, by_student)
    qp_status = item.get('qp_status', 'qp_uploaded')
    if questions and qp_status in ('upload_requested', 'qp_uploaded'):
        qp_status = 'qp_parsed'
    submitted = 0
    for entry in by_student.values():
        if entry['details'] or entry['summary'] is not None:
            submitted += 1
    view = {k: v for k, v in item.items() if k not in ('rubric', 'teacher_id')}
    view.update({
        'qp_status': qp_status,
        'question_count': len(questions) or item.get('question_count'),
        'total_marks': sum(_num(q.get('max_marks')) for q in questions) or item.get('total_marks'),
        'submitted': submitted,
    })
    return view


# ----------------------------------------------------------------------
# Teacher: assignments (§13)
# ----------------------------------------------------------------------

def list_assignments(body, caller):
    _require(caller, 'teacher', 'admin')
    items = _my_assignments(caller)
    users = _users_by_id() if items else {}
    out = []
    for item in sorted(items, key=lambda a: a.get('created_at', ''), reverse=True):
        view = _assignment_view(item)
        enrolled = len(_enrolled_students(item, users))
        if enrolled:
            view['enrolled'] = enrolled
        out.append(view)
    return {'assignments': out}


def get_assignment(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    view = _assignment_view(item)
    enrolled = len(_enrolled_students(item, _users_by_id()))
    if enrolled:
        view['enrolled'] = enrolled
    return view


def create_assignment(body, caller):
    _require(caller, 'teacher', 'admin')
    title, class_name = _required(body, 'title', 'class_name')
    assignment_id = f"A-{uuid.uuid4().hex[:10]}"
    item = {
        'assignment_id': assignment_id,
        'qp_id': f"QP-{uuid.uuid4().hex[:10]}",
        'title': str(title)[:200],
        'class_name': str(class_name)[:100],
        'teacher_id': caller['user_id'],
        'qp_status': 'qp_uploaded',
        'content_status': 'published',
        'grades_status': 'not_started',
        'created_at': _now(),
    }
    if body.get('due_at'):
        item['due_at'] = str(body['due_at'])
    assignments_table.put_item(Item=item)
    return {'assignment_id': assignment_id, 'qp_id': item['qp_id']}


def get_parsed_questions(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    questions = _questions(item)
    return {
        'qp_status': 'qp_parsed' if questions else item.get('qp_status', 'qp_uploaded'),
        'questions': questions,
        'paper_total': sum(_num(q.get('max_marks')) for q in questions),
    }


def confirm_questions(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    questions = _questions(item)
    if not questions:
        raise ApiError(409, 'No parsed questions yet — they appear once the first answers are processed.')
    assignments_table.update_item(
        Key={'assignment_id': assignment_id},
        UpdateExpression='SET questions_confirmed_at = :t, question_count = :n, total_marks = :m, qp_status = if_not_exists(qp_status, :s)',
        ExpressionAttributeValues={
            ':t': _now(), ':n': len(questions),
            ':m': _to_dynamo(sum(_num(q.get('max_marks')) for q in questions)), ':s': 'qp_parsed',
        },
    )
    return {'ok': True}


def set_grading_method(body, caller):
    _require(caller, 'teacher', 'admin')
    assignment_id, method = _required(body, 'assignment_id', 'method')
    if method not in ('ai', 'standard'):
        raise ApiError(400, "method must be 'ai' or 'standard'.")
    item = _load_assignment(assignment_id, caller)
    defaults = _user_settings(caller['user'])
    persona = body.get('persona') or item.get('persona') or defaults['default_persona']
    if persona not in PERSONAS:
        raise ApiError(400, f"persona must be one of {', '.join(PERSONAS)}.")
    if 'partial_credit' in body:
        partial = bool(body['partial_credit'])
    else:
        partial = bool(item['partial_credit']) if 'partial_credit' in item else defaults['partial_credit']

    changed_method = item.get('method') != method
    status = item.get('qp_status', 'qp_parsed')
    expr = 'SET #m = :m, persona = :p, partial_credit = :pc'
    values = {':m': method, ':p': persona, ':pc': partial}
    if changed_method:
        # A new method means a new rubric: AI drafts one in the background,
        # standard starts from the questions for the teacher to fill in.
        status = 'rubric_generating' if method == 'ai' else 'rubric_generated'
        expr += ', qp_status = :s REMOVE rubric'
        values[':s'] = status
    assignments_table.update_item(
        Key={'assignment_id': assignment_id}, UpdateExpression=expr,
        ExpressionAttributeNames={'#m': 'method'}, ExpressionAttributeValues=values,
    )
    if changed_method and method == 'ai':
        _invoke_self({'operation': 'draftRubric', 'assignment_id': assignment_id})
    return {'qp_status': status}


def _default_rubric(questions):
    return [{
        'question_number': q['question_number'],
        'subpart': q.get('subpart'),
        'title': (q.get('question') or '')[:80],
        'max_marks': _num(q.get('max_marks')),
        'criteria': '',
    } for q in questions]


def get_rubric(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    status = item.get('qp_status')
    if item.get('rubric'):
        return {'qp_status': status, 'rubric': item['rubric']}
    if status == 'rubric_generating':
        return {'qp_status': status, 'rubric': []}
    return {'qp_status': status, 'rubric': _default_rubric(_questions(item))}


def approve_rubric(body, caller):
    _require(caller, 'teacher', 'admin')
    assignment_id, rubric = _required(body, 'assignment_id', 'rubric')
    _load_assignment(assignment_id, caller)
    if not isinstance(rubric, list) or not rubric:
        raise ApiError(400, 'rubric must be a non-empty list.')
    clean = []
    for r in rubric:
        if not isinstance(r, dict) or r.get('question_number') in (None, ''):
            raise ApiError(400, 'Every rubric item needs a question_number.')
        marks = _num(r.get('max_marks'), -1)
        if marks < 0:
            raise ApiError(400, f"Invalid max_marks for Q{r.get('question_number')}.")
        clean.append({
            'question_number': r['question_number'],
            'subpart': r.get('subpart') or None,
            'title': str(r.get('title') or '')[:200],
            'max_marks': marks,
            'criteria': str(r.get('criteria') or '')[:4000],
        })
    assignments_table.update_item(
        Key={'assignment_id': assignment_id},
        UpdateExpression='SET rubric = :r, qp_status = :s, rubric_reviewed_at = :t, rubric_reviewed_by = :u',
        ExpressionAttributeValues={':r': _to_dynamo(clean), ':s': 'rubric_reviewed', ':t': _now(), ':u': caller['user_id']},
    )
    return {'qp_status': 'rubric_reviewed'}


def start_grading(body, caller):
    """Teacher side of the fan-in gate: every corrected submission is graded now."""
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    if item.get('qp_status') != 'rubric_reviewed':
        raise ApiError(409, 'Approve the rubric before grading starts.')
    by_student = _grading_rows(item)
    triggered = 0
    for student_id, entry in by_student.items():
        if _derived_status(entry) != 'corrected':
            continue
        _put_submission_status(assignment_id, student_id, 'grading_triggered')
        _trigger_grading(item, student_id)
        triggered += 1
    expr, values = 'SET grading_started_at = :t', {':t': _now()}
    # Re-grading after publish keeps the assignment "published"; students keep
    # their grades until the teacher publishes again (→ published_revised).
    if item.get('grades_status') != 'published':
        expr += ', grades_status = :g'
        values[':g'] = 'teacher_reviewing' if triggered else 'ai_graded'
    assignments_table.update_item(Key={'assignment_id': assignment_id}, UpdateExpression=expr, ExpressionAttributeValues=values)
    return {'ok': True, 'triggered': triggered}


def _trigger_grading(assignment, student_id):
    fn = os.environ.get('GRADING_FUNCTION_NAME')
    if not fn:
        return
    boto3.client('lambda', region_name=config.data_region()).invoke(
        FunctionName=fn,
        InvocationType='Event',
        Payload=json.dumps({
            'assignment_id': assignment['assignment_id'],
            'qp_id': assignment['qp_id'],
            'student_id': student_id,
            'method': assignment.get('method'),
            'persona': assignment.get('persona', 'balanced'),
            'partial_credit': assignment.get('partial_credit', True),
            'rubric': assignment.get('rubric', []),
        }, default=_json_default).encode(),
    )


def _put_submission_status(assignment_id, student_id, status, extra=None):
    expr = 'SET #s = :s, updated_at = :t'
    values = {':s': status, ':t': _now()}
    for k, v in (extra or {}).items():
        expr += f', {k} = :{k}'
        values[f':{k}'] = _to_dynamo(v)
    submissions_table.update_item(
        Key={'assignment_id': assignment_id, 'student_id': student_id},
        UpdateExpression=expr, ExpressionAttributeNames={'#s': 'status'}, ExpressionAttributeValues=values,
    )


# ----------------------------------------------------------------------
# Teacher: grading run, spot-check, publish, roster (§15, §19)
# ----------------------------------------------------------------------

def _submissions(item, users=None):
    users = users if users is not None else _users_by_id()
    by_student = _grading_rows(item, users)
    stored = {s['student_id']: s for s in _query_all(submissions_table, KeyConditionExpression=Key('assignment_id').eq(item['assignment_id']))}
    for u in _enrolled_students(item, users):
        by_student.setdefault(u['user_id'], {'summary': None, 'details': []})
    due = item.get('due_at')
    out = []
    for student_id, entry in by_student.items():
        sub = stored.get(student_id, {})
        status = _merge_status(_derived_status(entry), sub.get('status'))
        rows = entry['details']
        submitted_at = sub.get('submitted_at') or min((r.get('timestamp') or r.get('created_at') or '' for r in rows), default='') or None
        row = {
            'student_id': student_id,
            'student_name': (users.get(student_id) or {}).get('name'),
            'status': status,
            'pages': sub.get('pages'),
        }
        if submitted_at:
            row['submitted_at'] = _iso(submitted_at)
            if due and row['submitted_at'][:10] > str(due)[:10]:
                row['late'] = True
        if entry['summary'] is not None:
            row['evaluation_id'] = entry['summary'].get('evaluation_id')
            row['total_marks'] = entry['summary'].get('total_marks')
        if sub.get('recheck_status') == 'open':
            row['recheck'] = sub.get('recheck')
        confs = [_num(r['confidence']) for r in rows if r.get('confidence') is not None]
        if confs:
            row['confidence'] = min(confs)
        out.append(row)
    out.sort(key=lambda r: (r.get('student_name') or r['student_id']).lower())
    return out, by_student, users


def _iso(value):
    """assignment_grading timestamps are epoch seconds/ms; workflow ones are ISO strings."""
    if isinstance(value, (int, float, Decimal)) or (isinstance(value, str) and value.isdigit()):
        n = float(value)
        return datetime.datetime.fromtimestamp(n / 1000 if n > 1e12 else n, datetime.timezone.utc).isoformat()
    return str(value)


def get_grading_run(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    subs, _, _ = _submissions(item)
    counts = {}
    for s in subs:
        counts[s['status']] = counts.get(s['status'], 0) + 1
    return {'total': len(subs), 'counts': counts, 'submissions': subs}


def list_submissions(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    subs, _, users = _submissions(item)
    enrolled = len(_enrolled_students(item, users)) or len(subs)
    return {'submissions': subs, 'enrolled': enrolled}


def list_flagged(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    users = _users_by_id()
    rubric = {(str(r['question_number']), r.get('subpart') or ''): r for r in item.get('rubric') or []}
    items = []
    for student_id, entry in _grading_rows(item).items():
        if entry['summary'] is None:
            continue
        for r in entry['details']:
            if not _is_flagged(r):
                continue
            key = (str(r.get('question_number')), r.get('subpart') or '')
            items.append({
                'evaluation_id': r['evaluation_id'],
                'student_id': student_id,
                'student_name': (users.get(student_id) or {}).get('name'),
                'question_number': r.get('question_number'),
                'subpart': r.get('subpart'),
                'confidence': _num(r.get('confidence')),
                'reference_answer': (rubric.get(key) or {}).get('criteria') or None,
                'student_answer': r.get('student_answer'),
                'proposed_marks': _num(r.get('marks')),
                'max_marks': _num(r.get('max_marks')),
            })
    items.sort(key=lambda f: f['confidence'])
    return {'items': items}


def accept_flagged(body, caller):
    _require(caller, 'teacher', 'admin')
    (evaluation_id,) = _required(body, 'evaluation_id')
    row = grading_table.get_item(Key={'evaluation_id': evaluation_id}).get('Item')
    if not row or row.get('type') == 'summary':
        raise ApiError(404, 'Answer not found.')
    # Owned by the assignment's teacher (the row's teacher_id is a legacy
    # fallback); an answer with no owner either way is refused to teachers.
    assignment_id = row.get('assignment_id')
    assignment = assignments_table.get_item(Key={'assignment_id': assignment_id}).get('Item') if assignment_id else None
    owner = (assignment or {}).get('teacher_id') or row.get('teacher_id')
    if caller['role'] == 'teacher' and owner != caller['user_id']:
        raise ApiError(403, 'This answer belongs to a different teacher.')
    grading_table.update_item(
        Key={'evaluation_id': evaluation_id},
        UpdateExpression='SET flag_resolved_at = :t, flag_resolved_by = :u, flag_resolution = :r',
        ExpressionAttributeValues={':t': _now(), ':u': caller['user_id'], ':r': 'accepted'},
    )
    return {'ok': True}


def publish_grades(body, caller):
    _require(caller, 'teacher', 'admin')
    (assignment_id,) = _required(body, 'assignment_id')
    item = _load_assignment(assignment_id, caller)
    subs, _, users = _submissions(item)
    if any(s['status'] == 'queued_for_review' for s in subs):
        raise ApiError(409, 'Clear the flagged answers before publishing.')
    if any(s['status'] in ('grading_triggered', 'corrected') for s in subs):
        raise ApiError(409, 'Some papers are still grading.')
    now = _now()
    published = []
    for s in subs:
        if s.get('total_marks') is None:
            continue
        status = 'published_revised' if s['status'] in ('published', 'published_revised') else 'published'
        _put_submission_status(assignment_id, s['student_id'], status, {'published_at': now})
        published.append(s['student_id'])
    if not published:
        raise ApiError(409, 'No graded papers to publish yet.')
    assignments_table.update_item(
        Key={'assignment_id': assignment_id},
        UpdateExpression='SET grades_status = :g, published_at = :t',
        ExpressionAttributeValues={':g': 'published', ':t': now},
    )
    topic = os.environ.get('SNS_TOPIC_ARN')
    if topic:
        boto3.client('sns', region_name=config.data_region()).publish(
            TopicArn=topic,
            Subject=f"Grades published: {item.get('title', assignment_id)}"[:100],
            Message=json.dumps({'assignment_id': assignment_id, 'student_ids': published}),
        )
    return {'notified': len(published)}


def remind_student(body, caller):
    _require(caller, 'teacher', 'admin')
    assignment_id, student_id = _required(body, 'assignment_id', 'student_id')
    item = _load_assignment(assignment_id, caller)
    subs, _, users = _submissions(item)
    student = users.get(student_id)
    # Only students on this assignment's roster (enrolled in its class, or with a submission/answers
    # for it): otherwise any teacher could make GradeAnt email any student.
    if not student or student.get('userRole') != 'student' or student_id not in {s['student_id'] for s in subs}:
        raise ApiError(404, 'Student not found.')
    sender = os.environ.get('SES_SENDER_EMAIL')
    emailed = False
    if sender and student.get('email'):
        due = f" It's due {str(item['due_at'])[:10]}." if item.get('due_at') else ''
        boto3.client('ses', region_name=config.data_region()).send_email(
            Source=sender,
            Destination={'ToAddresses': [student['email']]},
            Message={
                'Subject': {'Data': f"Reminder: {item.get('title', 'assignment')}"},
                'Body': {'Text': {'Data': f"Hi {student.get('name') or ''},\n\nPlease submit your answers for "
                                          f"\"{item.get('title')}\" ({item.get('class_name')}).{due}\n\n— GradeAnt"}},
            },
        )
        emailed = True
    submissions_table.update_item(
        Key={'assignment_id': assignment_id, 'student_id': student_id},
        UpdateExpression='SET reminded_at = :t, #s = if_not_exists(#s, :s)',
        ExpressionAttributeNames={'#s': 'status'},
        ExpressionAttributeValues={':t': _now(), ':s': 'upload_requested'},
    )
    return {'ok': True, 'emailed': emailed}


# ----------------------------------------------------------------------
# Student (§14)
# ----------------------------------------------------------------------

def list_student_assignments(body, caller):
    _require(caller, 'student')
    published = _visible_assignments(caller)
    out = []
    for a in sorted(published, key=lambda x: x.get('due_at') or x.get('created_at', '')):
        entry = {'summary': None, 'details': _student_details(caller['user_id'], a)}
        summary = _query_all(
            grading_table, IndexName='student_id-index',
            KeyConditionExpression=Key('student_id').eq(caller['user_id']),
            FilterExpression=Attr('assignment_id').eq(a['assignment_id']) & Attr('type').eq('summary'),
        )
        entry['summary'] = summary[0] if summary else None
        sub = submissions_table.get_item(Key={'assignment_id': a['assignment_id'], 'student_id': caller['user_id']}).get('Item') or {}
        status = _merge_status(_derived_status(entry), sub.get('status'))
        # Students only ever see published grades; teacher-side states read as "grading".
        if status in ('ai_graded', 'queued_for_review', 'teacher_reviewed'):
            status = 'grading_triggered'
        view = {
            'assignment_id': a['assignment_id'], 'qp_id': a['qp_id'], 'title': a.get('title'),
            'class_name': a.get('class_name'), 'due_at': a.get('due_at'),
            'question_count': a.get('question_count'), 'total_marks': a.get('total_marks'),
            'submission_status': status,
        }
        if entry['summary'] is not None and status in ('published', 'published_revised'):
            view['evaluation_id'] = entry['summary'].get('evaluation_id')
        out.append({k: v for k, v in view.items() if v is not None})
    return {'assignments': out}


def request_recheck(body, caller):
    _require(caller, 'student')
    assignment_id, items = _required(body, 'assignment_id', 'items')
    if not isinstance(items, list) or not items:
        raise ApiError(400, 'Pick at least one question.')
    clean = []
    for i in items:
        note = str((i or {}).get('note') or '').strip()
        if (i or {}).get('question_number') in (None, '') or not note:
            raise ApiError(400, 'Each item needs a question_number and a note.')
        clean.append({'question_number': i['question_number'], 'subpart': i.get('subpart') or None, 'note': note[:1000]})
    sub = submissions_table.get_item(Key={'assignment_id': assignment_id, 'student_id': caller['user_id']}).get('Item')
    if not sub or sub.get('status') not in ('published', 'published_revised'):
        raise ApiError(409, 'You can request a re-check once your grades are published.')
    submissions_table.update_item(
        Key={'assignment_id': assignment_id, 'student_id': caller['user_id']},
        UpdateExpression='SET recheck = :r, recheck_requested_at = :t, recheck_status = :s',
        ExpressionAttributeValues={':r': _to_dynamo(clean), ':t': _now(), ':s': 'open'},
    )
    return {'ok': True}



# ----------------------------------------------------------------------
# Teacher: grading queue, students, analytics, settings (wireframes §02, §11)
# ----------------------------------------------------------------------

def _letter(percent):
    return 'A' if percent >= 90 else 'B' if percent >= 80 else 'C' if percent >= 70 else 'D' if percent >= 60 else 'F'


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _submitted(sub):
    return RANK.get(sub['status'], 0) >= RANK['answer_uploaded']


def _stats(item, users):
    """Submissions, parsed questions and per-student percent for one assignment."""
    subs, by_student, _ = _submissions(item, users)
    questions = _questions(item, by_student)
    max_marks = sum(_num(q.get('max_marks')) for q in questions) or _num(item.get('total_marks'))
    percents = {}
    if max_marks > 0:
        for sid, entry in by_student.items():
            if entry['summary'] is not None:
                percents[sid] = min(100.0, _num(entry['summary'].get('total_marks')) / max_marks * 100)
    return {'subs': subs, 'by_student': by_student, 'max_marks': max_marks, 'percents': percents}


QUEUE_ORDER = {'review': 0, 'ready': 1, 'grading': 2, 'collecting': 3, 'setup': 4, 'published': 5}


def get_grading_queue(body, caller):
    """§02 "batches waiting to review": every assignment with where it is in grading."""
    _require(caller, 'teacher', 'admin')
    items = _my_assignments(caller)
    users = _users_by_id() if items else {}
    out = []
    for item in items:
        st = _stats(item, users)
        submitted = [s for s in st['subs'] if _submitted(s)]
        graded = [s for s in submitted if RANK.get(s['status'], 0) >= RANK['ai_graded']]
        grading = sum(1 for s in st['subs'] if s['status'] in ('grading_triggered', 'corrected'))
        flagged = sum(1 for e in st['by_student'].values() if e['summary'] is not None for r in e['details'] if _is_flagged(r))
        if item.get('qp_status') != 'rubric_reviewed' and not (graded or grading):
            state = 'setup'  # (papers already being graded means setup was finished)
        elif item.get('grades_status') == 'published' and not flagged and not grading:
            state = 'published'
        elif flagged:
            state = 'review'
        elif grading:
            state = 'grading'
        elif graded:
            state = 'ready'
        else:
            state = 'collecting'
        out.append({
            'assignment_id': item['assignment_id'], 'qp_id': item['qp_id'], 'title': item.get('title'),
            'class_name': item.get('class_name'), 'due_at': item.get('due_at'), 'state': state,
            'papers': len(submitted), 'graded': len(graded), 'grading': grading, 'flagged': flagged,
        })
    out.sort(key=lambda r: (QUEUE_ORDER[r['state']], -r['flagged'], r.get('due_at') or '9'))
    return {'queue': out, 'waiting': sum(1 for r in out if r['state'] in ('review', 'ready'))}


def list_students(body, caller):
    """§02 Students: everyone in the teacher's classes with how they're doing."""
    _require(caller, 'teacher', 'admin')
    items = sorted(_my_assignments(caller), key=lambda a: a.get('created_at', ''))
    users = _users_by_id() if items else {}
    students, classes = {}, set()
    for item in items:
        st = _stats(item, users)
        classes.add(item.get('class_name'))
        for s in st['subs']:
            sid = s['student_id']
            rec = students.setdefault(sid, {
                'student_id': sid, 'student_name': s.get('student_name'), 'email': (users.get(sid) or {}).get('email'),
                'classes': set(), 'assigned': 0, 'submitted': 0, 'late': 0, '_pcts': [], 'last_submitted_at': None, 'latest': None,
            })
            rec['classes'].add(item.get('class_name'))
            rec['assigned'] += 1
            if _submitted(s):
                rec['submitted'] += 1
                rec['late'] += 1 if s.get('late') else 0
                at = s.get('submitted_at')
                if at and (rec['last_submitted_at'] is None or at >= rec['last_submitted_at']):
                    rec['last_submitted_at'] = at
                    rec['latest'] = {'assignment_id': item['assignment_id'], 'qp_id': item['qp_id']}
            if sid in st['percents']:
                rec['_pcts'].append(st['percents'][sid])
    out = []
    for rec in students.values():
        avg = _mean(rec.pop('_pcts'))
        rec['classes'] = sorted(c for c in rec['classes'] if c)
        rec['average'] = avg
        rec['at_risk'] = avg is not None and avg < AT_RISK_PERCENT
        out.append(rec)
    out.sort(key=lambda r: (r.get('student_name') or r['student_id']).lower())
    return {'students': out, 'classes': sorted(c for c in classes if c)}


def get_teacher_analytics(body, caller):
    """§11 class analytics across assignments (optionally one class)."""
    _require(caller, 'teacher', 'admin')
    everything = _my_assignments(caller)
    class_filter = body.get('class_name') or None
    items = [a for a in everything if not class_filter or a.get('class_name') == class_filter]
    users = _users_by_id() if items else {}
    today = datetime.datetime.now(datetime.timezone.utc).date()
    days = [today - datetime.timedelta(days=i) for i in range(6, -1, -1)]
    weekly = {d: 0 for d in days}
    dist = {k: 0 for k in 'ABCDF'}
    rows, all_pcts, risk, questions, upcoming = [], [], set(), {}, []
    submitted_total = enrolled_total = 0
    for item in items:
        st = _stats(item, users)
        for sid, p in st['percents'].items():
            dist[_letter(p)] += 1
            all_pcts.append(p)
            if p < AT_RISK_PERCENT:
                risk.add(sid)
        submitted = [s for s in st['subs'] if _submitted(s)]
        for s in submitted:
            try:
                d = datetime.date.fromisoformat(str(s.get('submitted_at'))[:10])
            except ValueError:
                continue
            if d in weekly:
                weekly[d] += 1
        enrolled = len(_enrolled_students(item, users)) or len(st['subs'])
        submitted_total += len(submitted)
        enrolled_total += enrolled
        pcts = list(st['percents'].values())
        rows.append({
            'assignment_id': item['assignment_id'], 'qp_id': item['qp_id'], 'title': item.get('title'),
            'class_name': item.get('class_name'), 'enrolled': enrolled, 'submitted': len(submitted), 'graded': len(pcts),
            'average': _mean(pcts), 'at_risk': sum(1 for p in pcts if p < AT_RISK_PERCENT), 'grades_status': item.get('grades_status'),
        })
        for entry in st['by_student'].values():
            if entry['summary'] is None:
                continue
            for r in entry['details']:
                mx = _num(r.get('max_marks'))
                if mx > 0 and r.get('marks') is not None:
                    q = questions.setdefault((item['assignment_id'], str(r.get('question_number')), r.get('subpart') or ''), {
                        'assignment_id': item['assignment_id'], 'title': item.get('title'),
                        'label': f"Q{r.get('question_number')}{r.get('subpart') or ''}", 'got': 0.0, 'max': 0.0,
                    })
                    q['got'] += _num(r['marks'])
                    q['max'] += mx
        if item.get('due_at') and str(item['due_at'])[:10] >= today.isoformat() and item.get('grades_status') != 'published':
            upcoming.append({'assignment_id': item['assignment_id'], 'title': item.get('title'), 'class_name': item.get('class_name'), 'due_at': item['due_at']})
    hardest = sorted(
        ({**{k: v for k, v in q.items() if k not in ('got', 'max')}, 'average': q['got'] / q['max'] * 100} for q in questions.values() if q['max'] > 0),
        key=lambda q: q['average'],
    )[:5]
    insight = None
    if hardest and hardest[0]['average'] < 100:
        h = hardest[0]
        insight = f"{h['label']} in \"{h['title']}\" cost the most marks ({round(h['average'])}% average) — consider a recap before the next assessment."
    rows.sort(key=lambda r: r['title'] or '')
    return {
        'classes': sorted({a.get('class_name') for a in everything if a.get('class_name')}),
        'class_name': class_filter,
        'totals': {
            'assignments': len(items), 'enrolled': enrolled_total, 'submitted': submitted_total,
            'submission_rate': (submitted_total / enrolled_total * 100) if enrolled_total else None,
            'graded': len(all_pcts), 'average': _mean(all_pcts), 'at_risk': len(risk),
        },
        'distribution': dist,
        'weekly': [{'date': d.isoformat(), 'label': d.strftime('%a'), 'count': n} for d, n in weekly.items()],
        'assignments': rows,
        'hardest': hardest,
        'insight': insight,
        'upcoming': sorted(upcoming, key=lambda u: str(u['due_at']))[:5],
    }


def _user_settings(user):
    return {**DEFAULT_SETTINGS, **(user.get('gradeant_settings') or {})}


def get_teacher_settings(body, caller):
    _require(caller, 'teacher', 'admin')
    return {'settings': _user_settings(caller['user'])}


def update_teacher_settings(body, caller):
    _require(caller, 'teacher', 'admin')
    incoming = body.get('settings')
    if not isinstance(incoming, dict):
        raise ApiError(400, 'settings must be an object.')
    merged = _user_settings(caller['user'])
    for key in DEFAULT_SETTINGS:
        if key in incoming:
            merged[key] = incoming[key]
    if merged['default_persona'] not in PERSONAS:
        raise ApiError(400, f"default_persona must be one of {', '.join(PERSONAS)}.")
    for key in ('partial_credit', 'notify_submissions', 'notify_flagged'):
        if not isinstance(merged[key], bool):
            raise ApiError(400, f'{key} must be true or false.')
    users_table.update_item(
        Key={'email': caller['email']}, UpdateExpression='SET gradeant_settings = :s',
        ExpressionAttributeValues={':s': merged},
    )
    return {'settings': merged}


# ----------------------------------------------------------------------
# Student: progress (§03, §12) and announcements
# ----------------------------------------------------------------------

def get_progress(body, caller):
    """Grade trend, by-class averages, due soon and a personal insight — published work only."""
    _require(caller, 'student')
    uid = caller['user_id']
    results, todo = [], []
    for a in _visible_assignments(caller):
        details = _student_details(uid, a)
        summaries = _query_all(
            grading_table, IndexName='student_id-index', KeyConditionExpression=Key('student_id').eq(uid),
            FilterExpression=Attr('assignment_id').eq(a['assignment_id']) & Attr('type').eq('summary'),
        )
        sub = submissions_table.get_item(Key={'assignment_id': a['assignment_id'], 'student_id': uid}).get('Item') or {}
        entry = {'summary': summaries[0] if summaries else None, 'details': details}
        status = _merge_status(_derived_status(entry), sub.get('status'))
        if status in ('published', 'published_revised') and entry['summary'] is not None:
            mx = sum(_num(r.get('max_marks')) for r in details)
            if mx <= 0:
                continue
            marks = _num(entry['summary'].get('total_marks'))
            first = min((r.get('timestamp') or r.get('created_at') or '' for r in details), default='')
            submitted_at = _iso(sub['submitted_at']) if sub.get('submitted_at') else (_iso(first) if first != '' else None)
            published_at = sub.get('published_at') or (_iso(entry['summary']['timestamp']) if entry['summary'].get('timestamp') else a.get('created_at'))
            early = on_time = None
            if a.get('due_at') and submitted_at:
                try:
                    gap = (datetime.date.fromisoformat(str(a['due_at'])[:10]) - datetime.date.fromisoformat(submitted_at[:10])).days
                    early, on_time = gap >= 1, gap >= 0
                except ValueError:
                    pass
            results.append({
                'assignment_id': a['assignment_id'], 'qp_id': a['qp_id'], 'title': a.get('title'), 'class_name': a.get('class_name'),
                'marks': marks, 'max': mx, 'percent': min(100.0, marks / mx * 100), 'date': published_at,
                'submitted_at': submitted_at, 'due_at': a.get('due_at'), 'early': early, 'on_time': on_time,
            })
        elif RANK.get(status, 0) < RANK['answer_uploaded']:
            todo.append({'assignment_id': a['assignment_id'], 'qp_id': a['qp_id'], 'title': a.get('title'),
                         'class_name': a.get('class_name'), 'due_at': a.get('due_at')})
    results.sort(key=lambda r: str(r['date']))
    todo.sort(key=lambda t: str(t.get('due_at') or '9'))
    by_class = {}
    for r in results:
        by_class.setdefault(r['class_name'] or 'Other', []).append(r['percent'])
    streak = 0
    for r in reversed(results):
        if r['on_time'] is False or r['on_time'] is None:
            break
        streak += 1
    early = [r['percent'] for r in results if r['early']]
    late = [r['percent'] for r in results if r['early'] is False]
    insight = None
    if early and late and _mean(early) > _mean(late):
        insight = f"You score about {round(_mean(early) - _mean(late))} points higher when you submit a day or more early. Keep the streak going!"
    elif streak >= 2:
        insight = f"{streak} on-time submissions in a row — keep the streak going!"
    return {
        'results': results,
        'average': _mean(r['percent'] for r in results),
        'by_class': [{'class_name': c, 'average': _mean(v), 'count': len(v)} for c, v in sorted(by_class.items())],
        'todo': todo[:5],
        'streak': streak,
        'insight': insight,
    }


def post_announcement(body, caller):
    _require(caller, 'teacher', 'admin')
    class_name, title, message = _required(body, 'class_name', 'title', 'message')
    mine = {a.get('class_name') for a in _my_assignments(caller)}
    if caller['role'] != 'admin' and class_name not in mine:
        raise ApiError(400, 'Create an assignment for that class first, then you can post to it.')
    item = {
        'announcement_id': f"N-{uuid.uuid4().hex[:10]}", 'class_name': str(class_name)[:100], 'title': str(title).strip()[:150],
        'message': str(message).strip()[:2000], 'author_id': caller['user_id'], 'author_name': caller['user'].get('name') or 'Teacher',
        'created_at': _now(),
    }
    if not item['title'] or not item['message']:
        raise ApiError(400, 'A title and a message are required.')
    announcements_table.put_item(Item=item)
    return {'announcement': item}


def list_announcements(body, caller):
    if caller['role'] == 'student':
        classes = caller['user'].get('classes') or sorted({a.get('class_name') for a in _visible_assignments(caller) if a.get('class_name')})
        items = _scan_all(announcements_table, FilterExpression=Attr('class_name').is_in(classes)) if classes else []
    else:
        _require(caller, 'teacher', 'admin')
        items = _scan_all(announcements_table) if caller['role'] == 'admin' else _scan_all(announcements_table, FilterExpression=Attr('author_id').eq(caller['user_id']))
    items.sort(key=lambda a: a.get('created_at', ''), reverse=True)
    return {'announcements': items[:50]}


def delete_announcement(body, caller):
    _require(caller, 'teacher', 'admin')
    (announcement_id,) = _required(body, 'announcement_id')
    item = announcements_table.get_item(Key={'announcement_id': announcement_id}).get('Item')
    if not item:
        raise ApiError(404, 'Announcement not found.')
    if caller['role'] != 'admin' and item.get('author_id') != caller['user_id']:
        raise ApiError(403, 'You can only delete your own announcements.')
    announcements_table.delete_item(Key={'announcement_id': announcement_id})
    return {'ok': True}


# ----------------------------------------------------------------------
# GradeFlow assistant + rubric drafting (Gemini, GEMINI_MODEL; gradeant_common/ai.py)
# ----------------------------------------------------------------------

def _ask_ai(system, messages, *, effort, max_tokens=16000, output_format=None):
    """The original _ask_claude contract on Gemini: the reply text, or None when the model declines.
    The prompts are the callers' own, unchanged from lambda_python_files/workflow.py. `effort` was a
    Claude setting with no Gemini equivalent; it's accepted so the call sites stay as they were."""
    del effort
    return ai.ask(system, messages, max_tokens=max_tokens, output_format=output_format)


def _grading_context(assignment, student_id=None):
    by_student = _grading_rows(assignment)
    if student_id:
        by_student = {k: v for k, v in by_student.items() if k == student_id}
    users = _users_by_id()
    lines = [
        f"Assignment: {assignment.get('title')} ({assignment.get('class_name')})",
        f"Grading method: {assignment.get('method', 'not chosen')}, persona: {assignment.get('persona', 'balanced')}",
    ]
    if assignment.get('rubric'):
        lines.append('Rubric:')
        for r in assignment['rubric']:
            lines.append(f"  Q{r['question_number']}{r.get('subpart') or ''} ({r.get('max_marks')} marks): {r.get('criteria') or r.get('title') or ''}")
    lines.append('Results:')
    for sid, entry in by_student.items():
        name = (users.get(sid) or {}).get('name') or sid
        total = entry['summary'].get('total_marks') if entry['summary'] else 'not graded'
        lines.append(f"- Student {name} [{sid}]: total {total}")
        for r in sorted(entry['details'], key=lambda x: (_num(x.get('question_number')), str(x.get('subpart') or ''))):
            lines.append(
                f"    Q{r.get('question_number')}{r.get('subpart') or ''}: {r.get('marks', '?')}/{r.get('max_marks', '?')}"
                f"{' (low confidence)' if _is_flagged(r) else ''}"
                f" | answer: {(r.get('student_answer') or '').strip()}"
                f" | feedback: {(r.get('feedback') or '').strip()}"
            )
    return '\n'.join(lines)


ASSISTANT_SYSTEM = """You are the GradeFlow assistant inside GradeAnt, an AI grading tool for STEM teachers.
You help the teacher understand and act on grading results: explain why an answer earned its marks against the rubric, \
find patterns across the class, identify students who are struggling, and draft feedback or re-teaching notes.
Ground every claim in the grading data provided below; if the data doesn't contain the answer, say so. \
Refer to students by the name shown. Keep replies concise and practical; use short lists where they help. \
You can't change grades yourself — when a change looks warranted, tell the teacher which score to override in the correction workspace.

Grading data:
"""


def ask_assistant(body, caller):
    _require(caller, 'teacher', 'admin')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, 'messages is required.')
    clean = []
    for m in messages[-20:]:
        role, content = (m or {}).get('role'), (m or {}).get('content')
        if role not in ('user', 'assistant') or not isinstance(content, str) or not content.strip():
            raise ApiError(400, 'Each message needs role user|assistant and text content.')
        clean.append({'role': role, 'content': content[:4000]})
    while clean and clean[0]['role'] != 'user':
        clean.pop(0)
    if not clean or clean[-1]['role'] != 'user':
        raise ApiError(400, 'The last message must be from the user.')

    if body.get('assignment_id'):
        assignment = _load_assignment(body['assignment_id'], caller)
        context = _grading_context(assignment, body.get('student_id'))
    else:
        items = _my_assignments(caller)
        context = '\n\n'.join(_grading_context(a) for a in items) or 'No assignments yet.'

    # Chat runs behind API Gateway's 29 s limit, so it uses low effort.
    reply = _ask_ai(ASSISTANT_SYSTEM + context, clean, effort='low')
    if reply is None:
        reply = "I can't help with that request. Try asking about the grades, rubric or feedback for this assignment."
    return {'reply': reply}


RUBRIC_SCHEMA = {
    'type': 'object',
    'properties': {
        'items': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'question_number': {'type': 'string'},
                    'subpart': {'type': 'string'},
                    'title': {'type': 'string'},
                    'max_marks': {'type': 'number'},
                    'criteria': {'type': 'string'},
                },
                'required': ['question_number', 'subpart', 'title', 'max_marks', 'criteria'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['items'],
    'additionalProperties': False,
}


def draft_rubric(assignment_id):
    """Internal (async) — GradeFlow drafts a marking rubric from the reviewed questions."""
    item = assignments_table.get_item(Key={'assignment_id': assignment_id}).get('Item')
    if not item or item.get('qp_status') != 'rubric_generating':
        return
    questions = _questions(item)
    rubric = _default_rubric(questions)
    if questions:
        listing = '\n'.join(
            f"Q{q['question_number']}{q.get('subpart') or ''} [{_num(q.get('max_marks'))} marks]: {q.get('question') or ''}"
            for q in questions
        )
        persona = item.get('persona', 'balanced')
        partial = 'Award method marks (partial credit) for correct working.' if item.get('partial_credit', True) else 'Award marks only for correct final answers.'
        text = _ask_ai(
            'You write marking rubrics for STEM homework. For each question, give a short title and concise criteria '
            'saying exactly what earns the marks (key steps, expected result with units/sign, common errors to penalise). '
            'Keep each question\'s max_marks exactly as given. Use an empty string for subpart when there is none.',
            [{'role': 'user', 'content': f"Grading persona: {persona}. {partial}\n\nQuestions:\n{listing}"}],
            effort='medium',
            output_format={'type': 'json_schema', 'schema': RUBRIC_SCHEMA},
        )
        if text:
            drafted = {(str(r['question_number']), r.get('subpart') or ''): r for r in json.loads(text).get('items', [])}
            for r in rubric:
                d = drafted.get((str(r['question_number']), r.get('subpart') or ''))
                if d:
                    r['title'] = d.get('title') or r['title']
                    r['criteria'] = d.get('criteria') or ''
    assignments_table.update_item(
        Key={'assignment_id': assignment_id},
        UpdateExpression='SET rubric = :r, qp_status = :s',
        ConditionExpression=Attr('qp_status').eq('rubric_generating'),
        ExpressionAttributeValues={':r': _to_dynamo(rubric), ':s': 'rubric_generated'},
    )


def _invoke_self(payload):
    name = os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
    if not name:  # local run / tests: do it inline
        draft_rubric(payload['assignment_id'])
        return
    boto3.client('lambda', region_name=config.data_region()).invoke(
        FunctionName=name, InvocationType='Event',
        Payload=json.dumps({'source': INTERNAL_SOURCE, **payload}).encode(),
    )


# ----------------------------------------------------------------------
# Handler
# ----------------------------------------------------------------------

OPERATIONS = {
    'listAssignments': list_assignments,
    'getAssignment': get_assignment,
    'createAssignment': create_assignment,
    'getParsedQuestions': get_parsed_questions,
    'confirmQuestions': confirm_questions,
    'setGradingMethod': set_grading_method,
    'getRubric': get_rubric,
    'approveRubric': approve_rubric,
    'startGrading': start_grading,
    'getGradingRun': get_grading_run,
    'listSubmissions': list_submissions,
    'listFlagged': list_flagged,
    'acceptFlagged': accept_flagged,
    'publishGrades': publish_grades,
    'remindStudent': remind_student,
    'listStudentAssignments': list_student_assignments,
    'requestRecheck': request_recheck,
    'askAssistant': ask_assistant,
    'getGradingQueue': get_grading_queue,
    'listStudents': list_students,
    'getTeacherAnalytics': get_teacher_analytics,
    'getTeacherSettings': get_teacher_settings,
    'updateTeacherSettings': update_teacher_settings,
    'getProgress': get_progress,
    'postAnnouncement': post_announcement,
    'listAnnouncements': list_announcements,
    'deleteAnnouncement': delete_announcement,
}


def lambda_handler(event, context):
    # Async self-invocation (rubric drafting). API Gateway events always carry
    # requestContext, so this path can't be reached from outside.
    if event.get('source') == INTERNAL_SOURCE and 'requestContext' not in event:
        if event.get('operation') == 'draftRubric':
            draft_rubric(event['assignment_id'])
        return {'ok': True}

    return operations(event, context, _HTTP_OPERATIONS, area='workflow', internal='Internal error')


# Operations return plain data; the dispatcher wants responses.
_HTTP_OPERATIONS = {name: (lambda fn: lambda body, caller: create_response(200, fn(body, caller)))(fn)
                    for name, fn in OPERATIONS.items()}
