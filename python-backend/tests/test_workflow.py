"""End-to-end tests for workflow.py against moto's in-memory DynamoDB.

Run:  pip install boto3 "moto[dynamodb]" pyjwt pytest  &&  pytest python-backend/tests
"""
import json
import os
import sys
import time
from decimal import Decimal

import boto3
import jwt
import pytest
from moto import mock_aws

SECRET = 'test-secret-that-is-at-least-32-bytes-long'
os.environ.update({'JWT_SECRET': SECRET, 'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x'})
os.environ.pop('AWS_LAMBDA_FUNCTION_NAME', None)
sys.path[:0] = [os.path.join(os.path.dirname(__file__), '..', 'functions'), os.path.join(os.path.dirname(__file__), '..')]


def _tables(ddb, assignment_index=False):
    ddb.create_table(TableName='Users', BillingMode='PAY_PER_REQUEST',
                     KeySchema=[{'AttributeName': 'email', 'KeyType': 'HASH'}],
                     AttributeDefinitions=[{'AttributeName': 'email', 'AttributeType': 'S'}])
    ddb.create_table(
        TableName='assignment_grading', BillingMode='PAY_PER_REQUEST',
        KeySchema=[{'AttributeName': 'evaluation_id', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': n, 'AttributeType': 'S'} for n in ('evaluation_id', 'student_id', 'type')
                              + (('assignment_id',) if assignment_index else ())],
        GlobalSecondaryIndexes=[
            {'IndexName': 'student_id-index', 'KeySchema': [{'AttributeName': 'student_id', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}},
            {'IndexName': 'type-index', 'KeySchema': [{'AttributeName': 'type', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}},
        ] + ([{'IndexName': 'assignment_id-index', 'KeySchema': [{'AttributeName': 'assignment_id', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}}]
             if assignment_index else []))
    ddb.create_table(
        TableName='GradeAnt_Assignments', BillingMode='PAY_PER_REQUEST',
        KeySchema=[{'AttributeName': 'assignment_id', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': n, 'AttributeType': 'S'} for n in ('assignment_id', 'teacher_id')],
        GlobalSecondaryIndexes=[{'IndexName': 'teacher_id-index', 'KeySchema': [{'AttributeName': 'teacher_id', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}}])
    ddb.create_table(TableName='GradeAnt_Announcements', BillingMode='PAY_PER_REQUEST',
                     KeySchema=[{'AttributeName': 'announcement_id', 'KeyType': 'HASH'}],
                     AttributeDefinitions=[{'AttributeName': 'announcement_id', 'AttributeType': 'S'}])
    ddb.create_table(
        TableName='GradeAnt_Submissions', BillingMode='PAY_PER_REQUEST',
        KeySchema=[{'AttributeName': 'assignment_id', 'KeyType': 'HASH'}, {'AttributeName': 'student_id', 'KeyType': 'RANGE'}],
        AttributeDefinitions=[{'AttributeName': n, 'AttributeType': 'S'} for n in ('assignment_id', 'student_id')],
        GlobalSecondaryIndexes=[{'IndexName': 'student_id-index', 'KeySchema': [{'AttributeName': 'student_id', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}}])


USERS = [
    {'email': 't@x.edu', 'user_id': 'T1', 'name': 'Priya', 'userRole': 'teacher'},
    {'email': 't2@x.edu', 'user_id': 'T2', 'name': 'Other', 'userRole': 'teacher'},
    {'email': 'a@x.edu', 'user_id': 'S1', 'name': 'Okafor', 'userRole': 'student', 'classes': ['PHY-204']},
    {'email': 'b@x.edu', 'user_id': 'S2', 'name': 'Marin', 'userRole': 'student', 'classes': ['PHY-204']},
    {'email': 'c@x.edu', 'user_id': 'S3', 'name': 'Tanaka', 'userRole': 'student', 'classes': ['PHY-204']},
]


@pytest.fixture(params=[False, True], ids=['no-assignment-index', 'assignment-index'])
def wf(monkeypatch, request):
    with mock_aws():
        ddb = boto3.resource('dynamodb', region_name='us-east-1')
        _tables(ddb, assignment_index=request.param)
        for u in USERS:
            ddb.Table('Users').put_item(Item=u)
        sys.modules.pop('workflow', None)
        import workflow
        calls = []

        def fake_ai(system, messages, *, effort, max_tokens=16000, output_format=None):
            calls.append({'system': system, 'messages': messages, 'effort': effort, 'format': output_format})
            if output_format:
                return json.dumps({'items': [{'question_number': '1', 'subpart': '', 'title': 'Internal energy', 'max_marks': 5, 'criteria': 'Defines U; unit J'}]})
            return 'Q2 was the hardest.'

        monkeypatch.setattr(workflow, '_ask_ai', fake_ai)
        workflow._test_calls = calls
        workflow._ddb = ddb
        yield workflow


def call(wf, user_email, op, **payload):
    user = next(u for u in USERS if u['email'] == user_email)
    token = jwt.encode({'email': user['email'], 'user_id': user['user_id'], 'exp': int(time.time()) + 600}, SECRET, algorithm='HS256')
    res = wf.lambda_handler({'httpMethod': 'POST', 'requestContext': {}, 'headers': {'Authorization': f'Bearer {token}'},
                             'body': json.dumps({'operation': op, **payload})}, None)
    return res['statusCode'], json.loads(res['body'])


def seed_answers(wf, a, student, corrected=True, graded=False, confidence=None, sfx=''):
    t = wf._ddb.Table('assignment_grading')
    for q, mm in ((1, 5), (2, 5)):
        item = {'evaluation_id': f'{student}-{q}{sfx}', 'student_id': student, 'assignment_id': a['assignment_id'], 'qp_id': a['qp_id'],
                'type': 'detail', 'question_number': q, 'question': f'Question {q}', 'max_marks': mm, 'student_answer': 'ans',
                'timestamp': int(time.time())}
        if corrected:
            item['status'] = 'corrected'
        if graded:
            item['marks'] = 4
        if confidence is not None and q == 2:
            item['confidence'] = Decimal(str(confidence))
        t.put_item(Item=item)
    if graded:
        t.put_item(Item={'evaluation_id': f'{student}-sum{sfx}', 'student_id': student, 'assignment_id': a['assignment_id'],
                         'qp_id': a['qp_id'], 'type': 'summary', 'total_marks': 8})


def test_auth_required(wf):
    res = wf.lambda_handler({'httpMethod': 'POST', 'requestContext': {}, 'headers': {}, 'body': '{"operation":"listAssignments"}'}, None)
    assert res['statusCode'] == 401
    assert res['headers']['Access-Control-Allow-Origin'] == '*'


def test_internal_event_not_reachable_from_api(wf):
    # An API Gateway event can't trigger the internal rubric job.
    code, _ = call(wf, 't@x.edu', 'draftRubric', assignment_id='x')
    assert code == 400


def test_full_teacher_and_student_flow(wf):
    code, a = call(wf, 't@x.edu', 'createAssignment', title='Thermo HW', class_name='PHY-204', due_at='2030-01-01')
    assert code == 200 and a['assignment_id'] and a['qp_id']

    # Students see it straight away.
    code, inbox = call(wf, 'a@x.edu', 'listStudentAssignments')
    assert code == 200 and inbox['assignments'][0]['submission_status'] == 'upload_requested'

    # Another teacher can't touch it; a student can't create.
    assert call(wf, 't2@x.edu', 'getAssignment', assignment_id=a['assignment_id'])[0] == 403
    assert call(wf, 'a@x.edu', 'createAssignment', title='x', class_name='y')[0] == 403

    # No parsed questions until answers arrive.
    assert call(wf, 't@x.edu', 'confirmQuestions', assignment_id=a['assignment_id'])[0] == 409
    seed_answers(wf, a, 'S1', corrected=True)
    seed_answers(wf, a, 'S2', corrected=False)

    code, view = call(wf, 't@x.edu', 'getAssignment', assignment_id=a['assignment_id'])
    assert view['qp_status'] == 'qp_parsed' and view['question_count'] == 2 and view['total_marks'] == 10
    assert view['enrolled'] == 3 and view['submitted'] == 2
    assert call(wf, 't@x.edu', 'confirmQuestions', assignment_id=a['assignment_id'])[0] == 200

    # AI method drafts a rubric (inline when not running in Lambda).
    code, res = call(wf, 't@x.edu', 'setGradingMethod', assignment_id=a['assignment_id'], method='ai', persona='strict')
    assert code == 200 and res['qp_status'] == 'rubric_generating'
    code, rub = call(wf, 't@x.edu', 'getRubric', assignment_id=a['assignment_id'])
    assert rub['qp_status'] == 'rubric_generated'
    assert rub['rubric'][0]['criteria'] == 'Defines U; unit J' and rub['rubric'][0]['max_marks'] == 5
    assert wf._test_calls[-1]['format']['type'] == 'json_schema'

    assert call(wf, 't@x.edu', 'startGrading', assignment_id=a['assignment_id'])[0] == 409  # rubric not approved
    code, res = call(wf, 't@x.edu', 'approveRubric', assignment_id=a['assignment_id'], rubric=rub['rubric'])
    assert res['qp_status'] == 'rubric_reviewed'

    code, res = call(wf, 't@x.edu', 'startGrading', assignment_id=a['assignment_id'])
    assert res['triggered'] == 1  # only S1 has confirmed answers

    code, run = call(wf, 't@x.edu', 'getGradingRun', assignment_id=a['assignment_id'])
    status = {s['student_id']: s['status'] for s in run['submissions']}
    assert status == {'S1': 'grading_triggered', 'S2': 'ocr_done', 'S3': 'upload_requested'}

    # Pipeline grades S1 with one low-confidence answer → spot-check.
    seed_answers(wf, a, 'S1', corrected=True, graded=True, confidence=0.6)
    code, run = call(wf, 't@x.edu', 'getGradingRun', assignment_id=a['assignment_id'])
    assert {s['student_id']: s['status'] for s in run['submissions']}['S1'] == 'queued_for_review'
    code, flagged = call(wf, 't@x.edu', 'listFlagged', assignment_id=a['assignment_id'])
    assert len(flagged['items']) == 1 and flagged['items'][0]['confidence'] == 0.6
    assert call(wf, 't@x.edu', 'publishGrades', assignment_id=a['assignment_id'])[0] == 409
    assert call(wf, 't2@x.edu', 'acceptFlagged', evaluation_id=flagged['items'][0]['evaluation_id'])[0] == 403
    assert call(wf, 't@x.edu', 'acceptFlagged', evaluation_id=flagged['items'][0]['evaluation_id'])[0] == 200
    assert call(wf, 't@x.edu', 'listFlagged', assignment_id=a['assignment_id'])[1]['items'] == []

    # Students don't see unpublished grades.
    inbox = call(wf, 'a@x.edu', 'listStudentAssignments')[1]['assignments'][0]
    assert inbox['submission_status'] == 'grading_triggered' and 'evaluation_id' not in inbox
    assert call(wf, 'a@x.edu', 'requestRecheck', assignment_id=a['assignment_id'], items=[{'question_number': 2, 'note': 'sign'}])[0] == 409

    code, res = call(wf, 't@x.edu', 'publishGrades', assignment_id=a['assignment_id'])
    assert code == 200 and res['notified'] == 1
    inbox = call(wf, 'a@x.edu', 'listStudentAssignments')[1]['assignments'][0]
    assert inbox['submission_status'] == 'published' and inbox['evaluation_id'] == 'S1-sum'

    # Re-check loops back to the teacher's roster.
    assert call(wf, 'a@x.edu', 'requestRecheck', assignment_id=a['assignment_id'], items=[{'question_number': 2, 'note': ''}])[0] == 400
    assert call(wf, 'a@x.edu', 'requestRecheck', assignment_id=a['assignment_id'], items=[{'question_number': 2, 'note': 'sign was right'}])[0] == 200
    code, roster = call(wf, 't@x.edu', 'listSubmissions', assignment_id=a['assignment_id'])
    s1 = next(s for s in roster['submissions'] if s['student_id'] == 'S1')
    assert s1['recheck'][0]['note'] == 'sign was right' and s1['student_name'] == 'Okafor'
    assert roster['enrolled'] == 3

    # Republishing marks it revised.
    call(wf, 't@x.edu', 'publishGrades', assignment_id=a['assignment_id'])
    assert call(wf, 'a@x.edu', 'listStudentAssignments')[1]['assignments'][0]['submission_status'] == 'published_revised'

    code, lst = call(wf, 't@x.edu', 'listAssignments')
    assert lst['assignments'][0]['grades_status'] == 'published' and 'rubric' not in lst['assignments'][0]

    # Starting grading again after publishing doesn't un-publish the assignment.
    assert call(wf, 't@x.edu', 'startGrading', assignment_id=a['assignment_id'])[0] == 200
    assert call(wf, 't@x.edu', 'getAssignment', assignment_id=a['assignment_id'])[1]['grades_status'] == 'published'


def test_reminder_and_assistant(wf):
    a = call(wf, 't@x.edu', 'createAssignment', title='HW', class_name='PHY-204')[1]
    code, res = call(wf, 't@x.edu', 'remindStudent', assignment_id=a['assignment_id'], student_id='S3')
    assert code == 200 and res['emailed'] is False  # no SES sender configured
    assert call(wf, 't@x.edu', 'remindStudent', assignment_id=a['assignment_id'], student_id='T2')[0] == 404

    seed_answers(wf, a, 'S1', graded=True)
    code, res = call(wf, 't@x.edu', 'askAssistant', assignment_id=a['assignment_id'],
                     messages=[{'role': 'assistant', 'content': 'Hi'}, {'role': 'user', 'content': 'Who struggled?'}])
    assert code == 200 and res['reply'] == 'Q2 was the hardest.'
    sent = wf._test_calls[-1]
    assert sent['messages'][0]['role'] == 'user' and 'Okafor' in sent['system'] and sent['effort'] == 'low'
    assert call(wf, 'a@x.edu', 'askAssistant', messages=[{'role': 'user', 'content': 'x'}])[0] == 403
    assert call(wf, 't@x.edu', 'askAssistant', messages=[])[0] == 400


def test_standard_method_rubric_from_questions(wf):
    a = call(wf, 't@x.edu', 'createAssignment', title='HW', class_name='PHY-204')[1]
    seed_answers(wf, a, 'S1')
    code, res = call(wf, 't@x.edu', 'setGradingMethod', assignment_id=a['assignment_id'], method='standard')
    assert res['qp_status'] == 'rubric_generated'
    rub = call(wf, 't@x.edu', 'getRubric', assignment_id=a['assignment_id'])[1]['rubric']
    assert [r['question_number'] for r in rub] == [1, 2] and rub[0]['criteria'] == ''
    assert call(wf, 't@x.edu', 'approveRubric', assignment_id=a['assignment_id'], rubric=[{'question_number': 1, 'max_marks': -1}])[0] == 400
    assert call(wf, 't@x.edu', 'setGradingMethod', assignment_id=a['assignment_id'], method='magic')[0] == 400


def _graded_assignment(wf, title='Thermo HW', due=None, sfx=''):
    a = call(wf, 't@x.edu', 'createAssignment', title=title, class_name='PHY-204', **({'due_at': due} if due else {}))[1]
    seed_answers(wf, a, 'S1', graded=True, sfx=sfx)                  # 8/10 = 80%
    seed_answers(wf, a, 'S2', graded=True, confidence=0.6, sfx=sfx)  # 8/10, one answer flagged
    call(wf, 't@x.edu', 'approveRubric', assignment_id=a['assignment_id'], rubric=[{'question_number': 1, 'max_marks': 5}, {'question_number': 2, 'max_marks': 5}])
    return a


def test_grading_queue_and_students(wf):
    a = _graded_assignment(wf)
    b = call(wf, 't@x.edu', 'createAssignment', title='Empty', class_name='PHY-204')[1]
    code, q = call(wf, 't@x.edu', 'getGradingQueue')
    assert code == 200
    rows = {r['assignment_id']: r for r in q['queue']}
    assert rows[a['assignment_id']]['state'] == 'review' and rows[a['assignment_id']]['flagged'] == 1 and rows[a['assignment_id']]['papers'] == 2
    assert rows[b['assignment_id']]['state'] == 'setup'
    assert q['queue'][0]['assignment_id'] == a['assignment_id'] and q['waiting'] == 1
    assert call(wf, 't2@x.edu', 'getGradingQueue')[1]['queue'] == []
    assert call(wf, 'a@x.edu', 'getGradingQueue')[0] == 403

    call(wf, 't@x.edu', 'acceptFlagged', evaluation_id='S2-2')
    assert call(wf, 't@x.edu', 'getGradingQueue')[1]['queue'][0]['state'] == 'ready'

    code, st = call(wf, 't@x.edu', 'listStudents')
    names = {s['student_name']: s for s in st['students']}
    assert set(names) == {'Okafor', 'Marin', 'Tanaka'} and st['classes'] == ['PHY-204']
    assert names['Okafor']['submitted'] == 1 and names['Okafor']['assigned'] == 2 and round(names['Okafor']['average']) == 80
    assert names['Tanaka']['submitted'] == 0 and names['Tanaka']['average'] is None and names['Tanaka']['latest'] is None
    assert names['Okafor']['latest']['assignment_id'] == a['assignment_id'] and names['Okafor']['at_risk'] is False
    assert call(wf, 'a@x.edu', 'listStudents')[0] == 403


def test_teacher_analytics(wf):
    _graded_assignment(wf)
    code, an = call(wf, 't@x.edu', 'getTeacherAnalytics')
    assert code == 200 and an['classes'] == ['PHY-204']
    assert an['totals']['assignments'] == 1 and an['totals']['graded'] == 2 and round(an['totals']['average']) == 80
    assert an['totals']['submitted'] == 2 and an['totals']['enrolled'] == 3 and round(an['totals']['submission_rate']) == 67
    assert an['distribution'] == {'A': 0, 'B': 2, 'C': 0, 'D': 0, 'F': 0}
    assert len(an['weekly']) == 7 and sum(w['count'] for w in an['weekly']) == 2
    assert an['assignments'][0]['title'] == 'Thermo HW' and an['hardest'][0]['label'] == 'Q1' and 'recap' in an['insight']
    assert call(wf, 't@x.edu', 'getTeacherAnalytics', class_name='NOPE')[1]['totals']['assignments'] == 0
    assert call(wf, 't@x.edu', 'getTeacherAnalytics', class_name='PHY-204')[1]['totals']['assignments'] == 1
    _graded_assignment(wf, 'Future HW', due='2099-01-01', sfx='-f')
    assert [u['title'] for u in call(wf, 't@x.edu', 'getTeacherAnalytics')[1]['upcoming']] == ['Future HW']


def test_teacher_settings_drive_ai_defaults(wf):
    code, res = call(wf, 't@x.edu', 'getTeacherSettings')
    assert res['settings'] == {'default_persona': 'balanced', 'partial_credit': True, 'notify_submissions': True, 'notify_flagged': True}
    assert call(wf, 't@x.edu', 'updateTeacherSettings', settings={'default_persona': 'wild'})[0] == 400
    assert call(wf, 't@x.edu', 'updateTeacherSettings', settings={'partial_credit': 'yes'})[0] == 400
    code, res = call(wf, 't@x.edu', 'updateTeacherSettings', settings={'default_persona': 'strict', 'partial_credit': False})
    assert code == 200 and res['settings']['default_persona'] == 'strict'
    assert call(wf, 't@x.edu', 'getTeacherSettings')[1]['settings']['partial_credit'] is False
    assert call(wf, 't2@x.edu', 'getTeacherSettings')[1]['settings']['default_persona'] == 'balanced'  # per user

    a = call(wf, 't@x.edu', 'createAssignment', title='HW', class_name='PHY-204')[1]
    seed_answers(wf, a, 'S1')
    call(wf, 't@x.edu', 'setGradingMethod', assignment_id=a['assignment_id'], method='ai')
    view = call(wf, 't@x.edu', 'getAssignment', assignment_id=a['assignment_id'])[1]
    assert view['persona'] == 'strict' and view['partial_credit'] is False
    assert 'Grading persona: strict' in wf._test_calls[-1]['messages'][0]['content'] and 'only for correct final answers' in wf._test_calls[-1]['messages'][0]['content']
    call(wf, 't@x.edu', 'setGradingMethod', assignment_id=a['assignment_id'], method='ai', persona='lenient')  # explicit still wins (no change in method → kept)
    assert call(wf, 'a@x.edu', 'getTeacherSettings')[0] == 403


def test_student_progress(wf):
    a = _graded_assignment(wf, due='2030-01-10')
    _graded_assignment(wf, 'Second HW', due='2030-02-10', sfx='-b')
    call(wf, 't@x.edu', 'createAssignment', title='Not yet', class_name='PHY-204', due_at='2030-03-01')
    assert call(wf, 't@x.edu', 'getProgress')[0] == 403
    code, pr = call(wf, 'a@x.edu', 'getProgress')
    assert code == 200 and pr['results'] == [] and pr['average'] is None
    assert {t['title'] for t in pr['todo']} == {'Not yet'}  # graded-but-unpublished work isn't shown yet
    call(wf, 't@x.edu', 'acceptFlagged', evaluation_id='S2-2')
    call(wf, 't@x.edu', 'acceptFlagged', evaluation_id='S2-2-b')
    call(wf, 't@x.edu', 'publishGrades', assignment_id=a['assignment_id'])
    code, pr = call(wf, 'a@x.edu', 'getProgress')
    assert [r['title'] for r in pr['results']] == ['Thermo HW'] and round(pr['results'][0]['percent']) == 80
    assert pr['results'][0]['marks'] == 8 and pr['results'][0]['max'] == 10 and round(pr['average']) == 80
    assert pr['by_class'][0]['class_name'] == 'PHY-204' and pr['by_class'][0]['count'] == 1
    assert [t['title'] for t in pr['todo']] == ['Not yet']  # 'Second HW' is already submitted (graded, unpublished)
    # S3 never submitted anything: nothing to show, everything to do.
    pr3 = call(wf, 'c@x.edu', 'getProgress')[1]
    assert pr3['results'] == [] and len(pr3['todo']) == 3


def test_announcements(wf):
    assert call(wf, 'a@x.edu', 'postAnnouncement', class_name='PHY-204', title='x', message='y')[0] == 403
    assert call(wf, 't@x.edu', 'postAnnouncement', class_name='PHY-204', title='x', message='y')[0] == 400  # no assignment for that class yet
    call(wf, 't@x.edu', 'createAssignment', title='HW', class_name='PHY-204')
    call(wf, 't2@x.edu', 'createAssignment', title='Other', class_name='CHM-110')
    assert call(wf, 't@x.edu', 'postAnnouncement', class_name='PHY-204', title='', message='y')[0] == 400
    code, res = call(wf, 't@x.edu', 'postAnnouncement', class_name='PHY-204', title='Quiz Friday', message='Chapters 3–4. Bring a calculator.')
    assert code == 200 and res['announcement']['author_name'] == 'Priya'
    call(wf, 't2@x.edu', 'postAnnouncement', class_name='CHM-110', title='Lab moved', message='Room 12.')
    seen = [n['title'] for n in call(wf, 'a@x.edu', 'listAnnouncements')[1]['announcements']]
    assert seen == ['Quiz Friday']                    # S1 is enrolled in PHY-204 only
    assert [n['title'] for n in call(wf, 't@x.edu', 'listAnnouncements')[1]['announcements']] == ['Quiz Friday']
    nid = res['announcement']['announcement_id']
    assert call(wf, 't2@x.edu', 'deleteAnnouncement', announcement_id=nid)[0] == 403
    assert call(wf, 'a@x.edu', 'deleteAnnouncement', announcement_id=nid)[0] == 403
    assert call(wf, 't@x.edu', 'deleteAnnouncement', announcement_id=nid)[0] == 200
    assert call(wf, 'a@x.edu', 'listAnnouncements')[1]['announcements'] == []
    assert call(wf, 't@x.edu', 'deleteAnnouncement', announcement_id=nid)[0] == 404
