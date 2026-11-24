"""Lambda gradeant-grader (handler grader.lambda_handler). Not behind API Gateway.

Port of lambda_python_files/grader_lambda.py. Two entry shapes:

  SQS (S3 event for <root>/students/assignments/<aid>__<qp>/Parsed_pdfs/JSON/<student>.json)
      Loads the question paper, rubric and parsed answers, builds the QA pairs and
      - GRADE_ON_UPLOAD=true  (default, as the original): grades and writes rows + summary
      - GRADE_ON_UPLOAD=false: writes the rows ungraded (no marks, no summary) and waits for
        the teacher's "Start grading".
      Returns {"batchItemFailures": [...]}: only failed messages go back to the queue.

  Asynchronous invoke from gradeant-workflow startGrading (payload pinned by
  tests/test_grader_contract.py): {assignment_id, qp_id, student_id, method, persona,
  partial_credit, rubric}. Regrades the student's (student-corrected) rows in place with the
  payload's rubric and persona, then writes the summary last. Raises on failure, so Lambda
  retries once and then sends the event to the gradeant-grader-failures queue.

  SQS message whose body is that same payload (grading: sqs, "Start grading" sent through the
  grader's queue by the Node API): handled exactly like the invoke, failures go back to the queue.

Database: config.database() — DynamoDB (GRADING_TABLE) or, with database: supabase, Supabase
through PostgREST (gradeant_common/pg_rest.py, SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY).

Settings: PIPELINE_BUCKET, PIPELINE_ROOT, GEMINI_API_KEY(_ARN), GEMINI_MODEL, GRADE_ON_UPLOAD,
GRADER_MAX_WORKERS, DATABASE_PROVIDER. No answer text is ever logged.
"""
from gradeant_common import grading as grading_table
from gradeant_common.config import env
from gradeant_common.http import log
from gradeant_common.pipeline import PipelineInputError, llm, paths, s3io, sqs
from gradeant_common.pipeline import grading


def grade_on_upload():
    return env('GRADE_ON_UPLOAD', 'true').lower() != 'false'


# ---------------------------------------------------------------------------
# SQS: parsed answer sheet uploaded
# ---------------------------------------------------------------------------

def _read_required(bucket, key, what):
    if not s3io.exists(bucket, key):
        raise PipelineInputError(f'{what} not found: {key}')
    return s3io.read_json(bucket, key)


def progressed_rows(student_id, assignment_id, qp_id):
    """Why this student's rows must not be rewritten from the parsed sheet, or '' when they may:
    graded (a summary exists), corrected by the student, or edited by a teacher."""
    for row in grading_table.student_rows(student_id, assignment_id=assignment_id, qp_id=qp_id):
        if row.get('type') == 'summary':
            return 'graded'
        if row.get('status') in ('corrected', 'graded') or row.get('corrected_at'):
            return 'corrected'
        if row.get('updated_at'):
            return 'edited by a teacher'
    return ''


def grade_upload(bucket, key):
    """Returns a short outcome string; raises on anything that should be retried."""
    s3io.check_bucket(bucket)
    if '/Parsed_pdfs/JSON/' not in key or not key.lower().endswith('.json'):
        return 'skipped (not a parsed answer sheet)'
    layout, kind, student_id = paths.parse_student_key(key)  # a bad name in the JSON folder fails loudly
    if kind != 'parsed_json':
        return 'skipped (not a parsed answer sheet)'
    progressed = progressed_rows(student_id, layout.assignment_id, layout.qp_id)
    if progressed:
        # SQS delivers at least once (and the legacy pipeline may still be wired up). Rewriting rows
        # that moved on would erase the student's corrections, a teacher's mark edits or the grade itself.
        return f'skipped (answers already {progressed})'

    has_solution = s3io.exists(bucket, layout.solution_pdf)
    qp = _read_required(bucket, layout.qp_schema_json(has_solution), 'Question-paper schema')
    rubric_key = layout.rubric_json(has_solution)
    rubric = s3io.read_json(bucket, rubric_key) if s3io.exists(bucket, rubric_key) else None
    answers = s3io.read_json(bucket, key)

    pairs = grading.extract_qa_pairs(qp, answers, rubric, include_std_answer=has_solution)
    if not pairs:
        raise PipelineInputError(f'No questions found for {key}')
    answer_with = 'standard_answer' if has_solution else 'without_standard_answer'
    aid, qp_id = layout.assignment_id, layout.qp_id

    if grade_on_upload():
        results = grading.grade_pairs(pairs, llm.get_client('grade'))
        rows = grading.detail_rows(results, aid, qp_id, student_id, graded=True)
        summary = grading.summary_row(rows, aid, qp_id, student_id, answer_with=answer_with)
    else:
        rows = grading.detail_rows(pairs, aid, qp_id, student_id, graded=False)
        summary = None
    outcome = grading.write_student(aid, qp_id, student_id, rows, summary)
    log('info', 'grader: answer sheet processed', assignment_id=aid, qp_id=qp_id, student_id=student_id,
        graded=summary is not None, questions=len(rows), total_marks=summary and summary['total_marks'], **outcome)
    return 'graded' if summary else 'stored ungraded'


def _sqs(event, request_id):
    failures = []
    for message_id, record in sqs.messages(event):
        try:
            job = sqs.regrade_job(record)
            if job is not None:  # "Start grading" sent through the queue (grading: sqs)
                regrade(job)
                continue
            for bucket, key in sqs.s3_objects(record):
                outcome = grade_upload(bucket, key)
                if outcome.startswith('skipped'):
                    log('info', 'grader: ' + outcome, key=key, messageId=message_id, requestId=request_id)
        except Exception as exc:  # noqa: BLE001 — every failure goes back to the queue, then the DLQ
            log('error', 'grader: message failed', messageId=message_id, requestId=request_id,
                error=type(exc).__name__, detail=str(exc)[:500])
            failures.append(message_id)
    return sqs.batch_response(failures)


# ---------------------------------------------------------------------------
# Invoke from gradeant-workflow startGrading: regrade corrected rows
# ---------------------------------------------------------------------------

def _qkey(question_number, subpart):
    """Rubric ↔ row match key: question 1 == '1' == Decimal('1'); subpart None == ''."""
    sub = None if subpart in (None, '') else str(subpart).strip() or None
    return grading.id_part(question_number), sub


def regrade(payload):
    aid, qp_id, sid = payload.get('assignment_id'), payload.get('qp_id'), payload.get('student_id')
    if not all(paths.valid_id(x) for x in (aid, qp_id, sid)):
        raise PipelineInputError('assignment_id, qp_id and student_id must be plain ids')
    persona = payload.get('persona') or 'balanced'
    partial_credit = payload.get('partial_credit', True) is not False
    rubric = {_qkey(r.get('question_number'), r.get('subpart')): r
              for r in payload.get('rubric') or [] if isinstance(r, dict)}

    rows = grading_table.student_rows(sid, assignment_id=aid, qp_id=qp_id, type='notSummary')
    if not rows:
        raise PipelineInputError(f'No answer rows for student {sid} on {aid}/{qp_id}')

    pairs = []
    for row in rows:
        spec = rubric.get(_qkey(row.get('question_number'), row.get('subpart')), {})
        pair = {
            'question': row.get('question', ''),
            'student_answer': row.get('student_answer', ''),
            'rubric': spec.get('criteria') or row.get('rubric'),
            'max_marks': spec.get('max_marks', row.get('max_marks')),
            'question_number': row.get('question_number'),
            'subpart': row.get('subpart'),
        }
        if row.get('standard_answer'):
            pair['standard_answer'] = row['standard_answer']
        pairs.append(pair)

    # Grade everything first: a failure raises before any write, so the rows stay 'corrected'.
    results = grading.grade_pairs(pairs, llm.get_client('grade'), persona=persona, partial_credit=partial_credit)

    graded_at = grading.now_iso()
    flagged = 0
    for row, result in zip(rows, results):
        fields = {'marks': result['marks'], 'reasoning': result['reasoning'], 'feedback': result['feedback'],
                  'graded_at': graded_at, 'status': 'graded'}
        remove = ()
        # A correction that rewrote the answer (not just fixed OCR) goes to the teacher's review queue:
        # a low `confidence` is what gradeant-workflow flags (FLAG_CONFIDENCE), until the teacher accepts it.
        review, similarity = grading.needs_review(row.get('ocr_answer'), row.get('student_answer'))
        if review:
            flagged += 1
            fields.update({'confidence': round(similarity, 2), 'flag_reason': grading.CORRECTION_FLAG_REASON})
            remove = ('flag_resolved_at', 'flag_resolved_by', 'flag_resolution')
        grading_table.patch_row(row['evaluation_id'], fields, remove)  # DynamoDB UpdateItem or Supabase RPC
    answer_with = next((r['answer_with'] for r in rows if r.get('answer_with')), 'without_standard_answer')
    summary = grading.summary_row(results, aid, qp_id, sid, answer_with=answer_with,
                                  method=payload.get('method'), persona=persona)
    grading_table.replace_rows([summary])  # last: readers treat the summary as "graded"
    log('info', 'grader: regraded', assignment_id=aid, qp_id=qp_id, student_id=sid,
        questions=len(rows), total_marks=summary['total_marks'], persona=persona, flagged_for_review=flagged)
    return {'ok': True, 'student_id': sid, 'total_marks': summary['total_marks']}


def lambda_handler(event, context):
    request_id = getattr(context, 'aws_request_id', 'local')
    if isinstance(event, dict) and 'Records' in event:
        return _sqs(event, request_id)
    if isinstance(event, dict) and {'assignment_id', 'qp_id', 'student_id'} <= event.keys():
        return regrade(event)
    raise ValueError('unsupported event shape')
