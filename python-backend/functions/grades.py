"""POST /getgrades — Lambda gradeant-grades (handler grades.lambda_handler).

    getSingleStudentGrade   a student may only read their own grades
    getAllStudentGrades     class roster, teacher/admin only
    updateStudentGrade      total marks, teacher/admin only

A teacher only reaches rows of assignments they own (gradeant_common.ownership);
rows with no owner are refused to teachers. Same behaviour as
backend/src/services/grades.service.js.
"""
from gradeant_common import grading
from gradeant_common.dispatch import operations
from gradeant_common.http import ApiError, create_response
from gradeant_common.ownership import can_manage, first_owner, owner_resolver


def get_single_student_grade(body, caller):
    student_id, assignment_id = body.get('student_id'), body.get('assignment_id')
    if not student_id or not assignment_id:
        raise ApiError(400, 'Missing required parameters: student_id and assignment_id')
    if caller['role'] == 'student' and student_id != caller['user_id']:
        raise ApiError(403, 'You can only view your own grades.')

    # (The field names are swapped relative to their contents, as the frontend expects.)
    details = grading.student_rows(student_id, assignment_id=assignment_id, type='notSummary')
    summary = grading.student_rows(student_id, assignment_id=assignment_id, type='summary')

    if caller['role'] == 'teacher' and not can_manage(caller, assignment_id, first_owner(summary + details)):
        raise ApiError(403, "This student's results belong to a different teacher's class.")
    return _ok({'student_id': student_id, 'assignment_id': assignment_id, 'details': details, 'summary': summary})


def get_all_student_grades(body, caller):
    if caller['role'] not in ('teacher', 'admin'):
        raise ApiError(403, 'Only a teacher or admin can view a class roster.')
    qp_id, assignment_id = body.get('qp_id'), body.get('assignment_id')
    if not qp_id or not assignment_id:
        raise ApiError(400, 'Missing required parameters: qp_id and assignment_id')

    grades = grading.summary_rows(assignment_id, qp_id)
    if caller['role'] == 'teacher':
        resolve = owner_resolver()
        grades = [g for g in grades if can_manage(caller, g.get('assignment_id') or assignment_id, g.get('teacher_id'), resolve)]
    return _ok({'qp_id': qp_id, 'assignment_id': assignment_id, 'student_count': len(grades), 'grades': grades})


def update_student_grade(body, caller):
    if caller['role'] not in ('teacher', 'admin'):
        raise ApiError(403, 'Only a teacher or admin can update grades.')
    student_id, qp_id, total_marks = body.get('student_id'), body.get('qp_id'), body.get('total_marks')
    if not all([student_id, qp_id, total_marks]):
        raise ApiError(400, 'student_id, qp_id, type, and total_marks are required')

    rows = grading.student_rows(student_id, type='summary')
    item = next((i for i in rows if i.get('qp_id') == qp_id and i.get('type') == 'summary'), None)
    # (The original get_grades.py returned nothing here, which API Gateway turned into a 502.)
    if not item:
        raise ApiError(404, 'No summary record found for that student and question paper.')
    if not can_manage(caller, item.get('assignment_id'), item.get('teacher_id')):
        raise ApiError(403, 'This assignment belongs to a different teacher.')

    # An edit never moves ownership: teacher_id is only filled in when it was empty.
    grading.update_row(
        item['evaluation_id'],
        UpdateExpression='SET total_marks = :tm, teacher_id = if_not_exists(teacher_id, :tid)',
        ExpressionAttributeValues={':tm': total_marks, ':tid': caller['user_id']},
    )
    # Same (pre-update) rows the original handler echoed back.
    return _ok({'message': 'Total marks updated successfully', 'updatedAttributes': rows})


def _ok(body):
    return create_response(200, body)


OPERATIONS = {
    'getSingleStudentGrade': get_single_student_grade,
    'getAllStudentGrades': get_all_student_grades,
    'updateStudentGrade': update_student_grade,
}


def lambda_handler(event, context):
    return operations(event, context, OPERATIONS, area='grades')
