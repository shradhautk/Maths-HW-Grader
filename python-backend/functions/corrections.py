"""POST /correctanswer — Lambda gradeant-corrections (handler corrections.lambda_handler).

The post-upload correction workspace:
    Student: getAnswersForCorrection / submitAnswerCorrections (OCR'd answer text)
    Teacher: getQuestionsForCorrection / submitQuestionCorrections (parsed question
             text, fanned out to every student's denormalised copy)

Same behaviour as backend/src/services/correction.service.js.
"""
import datetime
import json

from gradeant_common import dynamo, grading
from gradeant_common.dispatch import operations
from gradeant_common.http import ApiError, create_response
from gradeant_common.ownership import RANK, can_manage, first_owner, owner_resolver, rank


def _now():
    """Naive UTC ISO timestamp, as the original Lambdas stored."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()


def _key_str(value):
    return str(value) if value else ''


def answers_locked(student_id, assignment_id):
    """Once a paper is graded (a summary row exists) or its submission has reached
    grading_triggered, the answer text is what was marked: the student can no longer
    rewrite it (e.g. to change an answer before asking for a re-check)."""
    if not assignment_id:
        return False
    sub = dynamo.table('submissions').get_item(Key={'assignment_id': assignment_id, 'student_id': student_id}).get('Item')
    if sub and rank(sub.get('status')) >= RANK['grading_triggered']:
        return True
    return len(grading.student_rows(student_id, assignment_id=assignment_id, type='summary')) > 0


def get_answers_for_correction(body, caller):
    student_id, assignment_id, qp_id = body.get('student_id'), body.get('assignment_id'), body.get('qp_id')
    if not student_id or not assignment_id:
        raise ApiError(400, 'student_id and assignment_id are required')
    if caller['role'] == 'student' and student_id != caller['user_id']:
        raise ApiError(403, 'You can only view your own answers.')

    items = grading.student_rows(student_id, assignment_id=assignment_id, type='notSummary', qp_id=qp_id if qp_id else None)

    # A teacher (not admin) may only browse submissions for assignments they own.
    if caller['role'] == 'teacher' and not can_manage(caller, assignment_id, first_owner(items)):
        raise ApiError(403, "This student's answers belong to a different teacher's class.")

    answers = [{
        'evaluation_id': item.get('evaluation_id'),
        'question_number': item.get('question_number'),
        'subpart': item.get('subpart'),
        'question': item.get('question'),
        'student_answer': item.get('student_answer'),
        'is_latex': item.get('is_latex', False),
        'max_marks': item.get('max_marks'),
        'status': item.get('status'),
        'corrected_by': item.get('corrected_by'),
        'corrected_at': item.get('corrected_at'),
    } for item in items]
    answers.sort(key=lambda a: (_key_str(a['question_number']), _key_str(a['subpart'])))
    return create_response(200, {'student_id': student_id, 'assignment_id': assignment_id, 'qp_id': qp_id, 'answers': answers})


def submit_answer_corrections(body, caller):
    """Student submits corrected answer text; status='corrected' feeds the grading fan-in gate."""
    if caller['role'] != 'student':
        raise ApiError(403, 'Only the student who owns this submission can correct it.')
    corrections = body.get('corrections')
    if not isinstance(corrections, list) or not corrections:
        raise ApiError(400, 'corrections (non-empty array) is required')

    updated, errors, now, locked = [], [], _now(), {}
    for correction in corrections:
        c = correction if isinstance(correction, dict) else {}
        evaluation_id = c.get('evaluation_id')
        question_number = c.get('question_number')
        answer = c.get('corrected_answer')
        is_latex = bool(c.get('is_latex', False))

        if not evaluation_id:
            errors.append('Missing evaluation_id in correction')
            continue
        if question_number is None:
            errors.append(f'Missing question_number for evaluation_id {evaluation_id}')
            continue
        if answer is None or str(answer).strip() == '':
            errors.append(f'corrected_answer is required for evaluation_id {evaluation_id}')
            continue
        try:
            # Only the owning student may correct their own answer.
            existing = grading.get_row(evaluation_id)
            if not existing:
                errors.append(f'No record found with evaluation_id {evaluation_id}')
                continue
            if existing.get('student_id') != caller['user_id']:
                errors.append(f'evaluation_id {evaluation_id} belongs to a different student.')
                continue
            assignment_id = existing.get('assignment_id')
            if assignment_id not in locked:
                locked[assignment_id] = answers_locked(caller['user_id'], assignment_id)
            if locked[assignment_id]:
                errors.append(f'evaluation_id {evaluation_id} has already been graded and can no longer be changed.')
                continue
            expression = 'SET student_answer = :sa, is_latex = :il, #st = :status, corrected_by = :cb, corrected_at = :ca, updated_at = :ua'
            values = {':sa': answer, ':il': is_latex, ':status': 'corrected', ':cb': caller['user_id'],
                      ':ca': now, ':ua': now, ':qn': question_number}
            if isinstance(existing.get('student_answer'), str):
                # Keep the text as read from the paper (once, never overwritten), so grading can tell a
                # small OCR fix from a rewritten answer (the grader flags big changes for teacher review).
                expression += ', ocr_answer = if_not_exists(ocr_answer, :ocr)'
                values[':ocr'] = existing['student_answer']
            grading.update_row(
                evaluation_id,
                UpdateExpression=expression,
                ConditionExpression='#qn = :qn',
                ExpressionAttributeNames={'#st': 'status', '#qn': 'question_number'},
                ExpressionAttributeValues=values,
            )
            updated.append({'evaluation_id': evaluation_id, 'question_number': question_number, 'updated': True, 'is_latex': is_latex})
        except Exception as e:  # noqa: BLE001 — reported per correction, like the original
            if dynamo.is_conditional_check_failed(e):
                errors.append(f'No matching question_number {question_number} for evaluation_id {evaluation_id}')
            else:
                errors.append(f'Failed to correct evaluation_id {evaluation_id}: {e}')

    response = {
        'message': 'Correction completed with some errors' if errors else 'Correction completed',
        'updated_items': updated,
        'total_updated': len(updated),
        'success': not errors,
    }
    if errors:
        response['errors'] = errors
    if not updated:
        response['message'] = 'No answers were corrected'
        return create_response(404, response)
    return create_response(207 if errors else 200, response)


def get_questions_for_correction(body, caller):
    """One canonical copy of each question's parsed text, so the teacher reviews it once."""
    if caller['role'] not in ('teacher', 'admin'):
        raise ApiError(403, 'Only a teacher or admin can review question text.')
    qp_id, assignment_id = body.get('qp_id'), body.get('assignment_id')
    if not qp_id or not assignment_id:
        raise ApiError(400, 'qp_id and assignment_id are required')

    roster = grading.summary_rows(assignment_id, qp_id)
    if caller['role'] == 'teacher':
        resolve = owner_resolver()
        roster = [g for g in roster if can_manage(caller, assignment_id, g.get('teacher_id'), resolve)]
    if not roster:
        return create_response(200, {'qp_id': qp_id, 'assignment_id': assignment_id, 'questions': []})

    by_key = {}
    for student in roster:
        if not student.get('student_id'):
            continue
        for item in grading.student_rows(student['student_id'], assignment_id=assignment_id, qp_id=qp_id, type='notSummary'):
            key = json.dumps([str(item.get('question_number')), item.get('subpart') or ''])
            if key not in by_key:
                by_key[key] = {
                    'question_number': item.get('question_number'),
                    'subpart': item.get('subpart'),
                    'question': item.get('question'),
                    'is_latex': item.get('question_is_latex', False),
                    'max_marks': item.get('max_marks'),
                    'question_reviewed': bool(item.get('question_reviewed_at')),
                    'question_reviewed_by': item.get('question_reviewed_by'),
                    'question_reviewed_at': item.get('question_reviewed_at'),
                }
        # One representative student's rows are enough: question text is identical across the class.
        if by_key:
            break
    questions = sorted(by_key.values(), key=lambda q: (_key_str(q['question_number']), _key_str(q['subpart'])))
    return create_response(200, {'qp_id': qp_id, 'assignment_id': assignment_id, 'questions': questions})


def submit_question_corrections(body, caller):
    """Teacher corrects question text; each correction fans out to every student's copy."""
    if caller['role'] not in ('teacher', 'admin'):
        raise ApiError(403, 'Only a teacher or admin can correct question text.')
    qp_id, assignment_id, corrections = body.get('qp_id'), body.get('assignment_id'), body.get('corrections')
    if not qp_id or not assignment_id:
        raise ApiError(400, 'qp_id and assignment_id are required')
    if not isinstance(corrections, list) or not corrections:
        raise ApiError(400, 'corrections (non-empty array) is required')

    roster = grading.summary_rows(assignment_id, qp_id)
    resolve = owner_resolver()
    now = _now()
    updated_questions, errors = [], []

    for correction in corrections:
        c = correction if isinstance(correction, dict) else {}
        question_number = c.get('question_number')
        subpart = c.get('subpart')
        text = c.get('corrected_question')
        is_latex = bool(c.get('is_latex', False))
        if question_number is None:
            errors.append('Missing question_number in correction')
            continue
        if text is None or str(text).strip() == '':
            errors.append(f'corrected_question is required for question_number {question_number}')
            continue

        records_updated = 0
        for student in roster:
            student_id = student.get('student_id')
            if not student_id:
                continue
            # Rows of an assignment the caller doesn't own are skipped, not overwritten.
            if not can_manage(caller, assignment_id, student.get('teacher_id'), resolve):
                continue
            try:
                for item in grading.student_rows(student_id, assignment_id=assignment_id, qp_id=qp_id,
                                                 question_number=question_number, type='notSummary'):
                    if subpart and item.get('subpart') != subpart:
                        continue
                    grading.update_row(
                        item['evaluation_id'],
                        UpdateExpression='SET question = :q, question_is_latex = :il, question_reviewed_by = :rb, question_reviewed_at = :ra, updated_at = :ua',
                        ConditionExpression='#qn = :qn',
                        ExpressionAttributeNames={'#qn': 'question_number'},
                        ExpressionAttributeValues={':q': text, ':il': is_latex, ':rb': caller['user_id'], ':ra': now,
                                                   ':ua': now, ':qn': question_number},
                    )
                    records_updated += 1
            except Exception as e:  # noqa: BLE001 — reported per student, like the original
                errors.append(f'Failed to update student {student_id}, question {question_number}: {e}')
        updated_questions.append({'question_number': question_number, 'subpart': subpart, 'records_updated': records_updated})

    total = sum(q['records_updated'] for q in updated_questions)
    response = {
        'message': 'Question correction completed with some errors' if errors else 'Question correction completed',
        'updated_questions': updated_questions,
        'total_records_updated': total,
        'success': not errors,
    }
    if errors:
        response['errors'] = errors
    if total == 0:
        response['message'] = 'No question records were updated'
        return create_response(404, response)
    return create_response(207 if errors else 200, response)


OPERATIONS = {
    'getAnswersForCorrection': get_answers_for_correction,
    'submitAnswerCorrections': submit_answer_corrections,
    'getQuestionsForCorrection': get_questions_for_correction,
    'submitQuestionCorrections': submit_question_corrections,
}


def lambda_handler(event, context):
    return operations(event, context, OPERATIONS, area='corrections')
