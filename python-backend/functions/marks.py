"""POST /updatedb — Lambda gradeant-marks (handler marks.lambda_handler).

Teacher/admin edits to marks:
    updateMarks          summary total and/or per-question marks
    updateStudentGrade   total marks only, by evaluation_id

A teacher may only edit rows of assignments they own (admins excepted). An edit
stamps teacher_id only when the row had none, so it never moves ownership.
Same behaviour as backend/src/services/marks.service.js.
"""
import datetime
from decimal import Decimal, InvalidOperation

from gradeant_common import dynamo, grading
from gradeant_common.dispatch import operations
from gradeant_common.http import ApiError, create_response
from gradeant_common.ownership import owned_by_other, owner_resolver


def _now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()


def _py_str(value):
    return 'None' if value is None else str(value)


def _decimal(value):
    """A finite number (int, float or numeric string) as Decimal, else ValueError('Invalid number: ...')."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)) or (isinstance(value, str) and not value.strip()):
        raise ValueError(f'Invalid number: {_py_str(value)}')
    try:
        d = Decimal(str(value).strip())
    except InvalidOperation:
        raise ValueError(f'Invalid number: {_py_str(value)}') from None
    if not d.is_finite():
        raise ValueError(f'Invalid number: {_py_str(value)}')
    return d


def update_marks(body, caller):
    if caller['role'] not in ('teacher', 'admin'):
        raise ApiError(403, 'Only a teacher or admin can update grades.')
    resolve = owner_resolver()
    summary_raw = body.get('summary')
    details = body.get('details') if isinstance(body.get('details'), list) else []
    if not summary_raw and not details:
        raise ApiError(400, 'Either summary or details data is required')

    updated, errors = [], []

    # Summary: the frontend sends a one-element list; only [0] is applied.
    # A details-only request skips this step (the original update_db.py crashed with a 500 there).
    if summary_raw:
        summary = summary_raw[0] if isinstance(summary_raw, list) else None
        evaluation_id = summary.get('evaluation_id') if isinstance(summary, dict) else None
        try:
            if not isinstance(summary, dict):
                raise RuntimeError('summary must be a list of objects')
            total_marks = _decimal(summary.get('total_marks'))
            if not evaluation_id:
                raise RuntimeError('Missing evaluation_id in summary record')
            existing = grading.get_row(evaluation_id)
            if owned_by_other(existing, caller, resolve):
                errors.append(f'evaluation_id {evaluation_id} belongs to a different teacher.')
            else:
                grading.update_row(
                    evaluation_id,
                    UpdateExpression='SET total_marks = :tm, updated_at = :ua, teacher_id = if_not_exists(teacher_id, :tid)',
                    ConditionExpression='#type = :summary',
                    ExpressionAttributeNames={'#type': 'type'},
                    ExpressionAttributeValues={':tm': total_marks, ':ua': _now(), ':tid': caller['user_id'], ':summary': 'summary'},
                )
                updated.append({
                    'type': 'summary',
                    'evaluation_id': evaluation_id,
                    'updated': True,
                    'total_marks': total_marks,
                    'student_id': summary.get('student_id'),
                    'assignment_id': summary.get('assignment_id'),
                    'qp_id': summary.get('qp_id'),
                })
        except Exception as e:  # noqa: BLE001 — reported in the response, like the original
            if dynamo.is_conditional_check_failed(e):
                errors.append(f'No summary record found with evaluation_id {_py_str(evaluation_id)}')
            elif isinstance(e, ValueError):
                errors.append(f'Invalid total_marks value for evaluation_id {_py_str(evaluation_id)}: {e}')
            else:
                errors.append(f'Failed to update summary for evaluation_id {_py_str(evaluation_id)}: {e}')

    for record in details:
        d = record if isinstance(record, dict) else {}
        evaluation_id = d.get('evaluation_id')
        question_number = d.get('question_number')
        subpart = d.get('subpart')
        marks = d.get('marks')
        max_marks = d.get('max_marks')
        try:
            if not evaluation_id:
                errors.append('Missing evaluation_id in detail record')
                continue
            if question_number is None:
                errors.append(f'Missing question_number for evaluation_id {evaluation_id}')
                continue
            if marks is None:
                errors.append(f'Missing marks for evaluation_id {evaluation_id}')
                continue
            marks_value = _decimal(marks)
            existing = grading.get_row(evaluation_id)
            if owned_by_other(existing, caller, resolve):
                errors.append(f'evaluation_id {evaluation_id} belongs to a different teacher.')
                continue
            expression = 'SET marks = :m, updated_at = :ua, teacher_id = if_not_exists(teacher_id, :tid)'
            values = {':m': marks_value, ':ua': _now(), ':tid': caller['user_id'], ':qn': question_number}
            if max_marks is not None:
                expression += ', max_marks = :mm'
                values[':mm'] = _decimal(max_marks)
            grading.update_row(
                evaluation_id,
                UpdateExpression=expression,
                ConditionExpression='#qn = :qn',
                ExpressionAttributeNames={'#qn': 'question_number'},
                ExpressionAttributeValues=values,
            )
            row = {
                'type': 'details',
                'evaluation_id': evaluation_id,
                'updated': True,
                'marks': marks_value,
                'question_number': _py_str(question_number),
                'subpart': subpart,
            }
            if max_marks is not None:
                row['max_marks'] = values[':mm']
            updated.append(row)
        except Exception as e:  # noqa: BLE001
            if dynamo.is_conditional_check_failed(e):
                errors.append(f'No detail record found with evaluation_id {evaluation_id}, '
                              f'question_number {_py_str(question_number)}, subpart {_py_str(subpart)}')
            elif isinstance(e, ValueError):
                errors.append(f'Invalid marks value for evaluation_id {_py_str(evaluation_id)}: {e}')
            else:
                errors.append(f'Failed to update details for evaluation_id {_py_str(evaluation_id)}: {e}')

    response = {
        'message': 'Update completed',
        'updated_items': updated,
        'total_updated': len(updated),
        'success': not errors,
        'records_processed': {'summary': 1, 'details': len(details)},
    }
    if errors:
        response['errors'] = errors
        response['message'] = 'Update completed with some errors'
    if not updated:
        response['message'] = 'No records were updated'
        return create_response(404, response)
    return create_response(207 if errors else 200, response)


def update_student_grade(body, caller):
    """Total marks only, by evaluation_id; summary rows only."""
    if caller['role'] not in ('teacher', 'admin'):
        raise ApiError(403, 'Only a teacher or admin can update grades.')
    evaluation_id, total_marks = body.get('evaluation_id'), body.get('total_marks')
    if not evaluation_id:
        raise ApiError(400, 'evaluation_id is required')
    if total_marks is None:
        raise ApiError(400, 'total_marks is required')

    if owned_by_other(grading.get_row(evaluation_id), caller):
        raise ApiError(403, 'This assignment belongs to a different teacher.')
    try:
        value = _decimal(total_marks)
    except ValueError as e:
        raise ApiError(400, f'Invalid number format: {e}') from None
    try:
        res = grading.update_row(
            evaluation_id,
            UpdateExpression='SET total_marks = :tm, updated_at = :ua, teacher_id = if_not_exists(teacher_id, :tid)',
            ConditionExpression='#type = :summary',
            ExpressionAttributeNames={'#type': 'type'},
            ExpressionAttributeValues={':tm': value, ':ua': _now(), ':tid': caller['user_id'], ':summary': 'summary'},
            ReturnValues='ALL_NEW',
        )
    except Exception as e:  # noqa: BLE001
        if dynamo.is_conditional_check_failed(e):
            raise ApiError(404, f'No summary record found with evaluation_id {evaluation_id}') from None
        code = getattr(e, 'response', {}).get('Error', {}).get('Code') or type(e).__name__
        raise ApiError(500, f'Database error: {code}') from None
    return create_response(200, {
        'message': 'Total marks updated successfully',
        'evaluation_id': evaluation_id,
        'updated_attributes': res.get('Attributes') or {},
        'total_marks': value,
    })


OPERATIONS = {'updateMarks': update_marks, 'updateStudentGrade': update_student_grade}


def lambda_handler(event, context):
    return operations(event, context, OPERATIONS, area='marks')
