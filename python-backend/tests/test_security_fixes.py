"""Regression tests for the 2026-11-16 security review (same rules as the Node backend's tests):

    1. with roles.teacher.signup: [invite], teachers can't sign themselves up; with [self, invite] they can
    2. reminders only reach students on the assignment's roster
    3. a student with no classes, or another class, sees no other class's work
    4. an invited student's account records the invite's class
    5. students upload answers only for assignments of their own classes
    6. auth replies never include exception text
    7. the profile-photo size is required and signed into the upload URL
"""
import base64
import hashlib
import importlib
import json
import os
import sys
import time
from urllib.parse import parse_qs, urlparse

import boto3
import jwt
import pytest
from moto import mock_aws

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT]
SECRET = 'test-secret-that-is-at-least-32-bytes-long'
PASSWORD = 'Correct-horse-9'


def _table(ddb, name, keys, gsis=()):
    attrs = {k for k, _ in keys} | {k for _, k in gsis}
    kw = dict(TableName=name, BillingMode='PAY_PER_REQUEST',
              KeySchema=[{'AttributeName': k, 'KeyType': t} for k, t in keys],
              AttributeDefinitions=[{'AttributeName': a, 'AttributeType': 'S'} for a in sorted(attrs)])
    if gsis:
        kw['GlobalSecondaryIndexes'] = [{'IndexName': n, 'KeySchema': [{'AttributeName': k, 'KeyType': 'HASH'}],
                                         'Projection': {'ProjectionType': 'ALL'}} for n, k in gsis]
    ddb.create_table(**kw)


def _login_hash(password):
    salt = os.urandom(32)
    return base64.b64encode(salt + hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)).decode()


USERS = [
    ('A1', 'admin@x.edu', 'admin', None),
    ('T1', 't@x.edu', 'teacher', None),
    ('S1', 'a@x.edu', 'student', ['PHY-204']),
    ('S4', 'd@x.edu', 'student', None),        # no classes recorded
    ('S5', 'e@x.edu', 'student', ['CHEM-101']),
]


@pytest.fixture()
def env(monkeypatch):
    for k, v in {'JWT_SECRET': SECRET, 'SESSION_TOKEN_SECRET': 'session-secret-for-tests-only-0123',
                 'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x',
                 'UPLOAD_BUCKET': 'uploads-test', 'PROFILE_PHOTO_BUCKET': 'photos-test', 'MAIL_DRIVER': 'log'}.items():
        monkeypatch.setenv(k, v)
    for k in ('AWS_LAMBDA_FUNCTION_NAME', 'APP_ENV', 'CORS_ALLOWED_ORIGINS', 'JWT_SECRET_ARN', 'SES_SENDER_EMAIL'):
        monkeypatch.delenv(k, raising=False)
    with mock_aws():
        from gradeant_common import config, dynamo, secrets
        dynamo.reset()
        secrets.clear_cache()
        config.app_config.cache_clear()
        ddb = boto3.resource('dynamodb', region_name='us-east-1')
        _table(ddb, 'Users', [('email', 'HASH')])
        _table(ddb, 'assignment_grading', [('evaluation_id', 'HASH')],
               [('student_id-index', 'student_id'), ('type-index', 'type'), ('assignment_id-index', 'assignment_id')])
        _table(ddb, 'GradeAnt_Assignments', [('assignment_id', 'HASH')], [('teacher_id-index', 'teacher_id')])
        _table(ddb, 'GradeAnt_Submissions', [('assignment_id', 'HASH'), ('student_id', 'RANGE')], [('student_id-index', 'student_id')])
        _table(ddb, 'GradeAnt_Announcements', [('announcement_id', 'HASH')])
        for name, pk in [('GradeAnt_Users', 'user_id'), ('GradeAnt_Invites', 'token'), ('GradeAnt_AuditLogs', 'log_id'),
                         ('GradeAnt_ResetRequests', 'request_id')]:
            _table(ddb, name, [(pk, 'HASH')])
        boto3.client('s3', region_name='us-east-1').create_bucket(Bucket='photos-test')
        stored = _login_hash(PASSWORD)
        for uid, email, role, classes in USERS:
            item = {'email': email, 'user_id': uid, 'userRole': role, 'name': uid, 'password': stored}
            if classes:
                item['classes'] = classes
            ddb.Table('Users').put_item(Item=item)
        ddb.Table('GradeAnt_Users').put_item(Item={'user_id': 'A1', 'email': 'admin@x.edu', 'role': 'admin', 'status': 'active'})
        mods = {m: importlib.reload(importlib.import_module(m)) for m in ('auth', 'workflow', 'admin', 'upload')}
        yield ddb, mods


def token(email):
    uid = next(u[0] for u in USERS if u[1] == email)
    return jwt.encode({'user_id': uid, 'email': email, 'exp': int(time.time()) + 600}, SECRET, algorithm='HS256')


def call(mod, path, body, email=None, raw=None):
    headers = {'Content-Type': 'application/json'}
    if email:
        headers['Authorization'] = f'Bearer {token(email)}'
    r = mod.lambda_handler({'httpMethod': 'POST', 'path': path, 'headers': headers, 'requestContext': {},
                            'body': raw if raw is not None else json.dumps(body)}, None)
    return r['statusCode'], (json.loads(r['body']) if r.get('body') else None)


def wf(m, email, op, **payload):
    return call(m['workflow'], '/workflow', {'operation': op, **payload}, email)


# 1 -------------------------------------------------------------------------------------------
def test_teacher_signup_follows_app_yaml(env):
    ddb, m = env
    from gradeant_common.config import app_config
    teacher = app_config()['roles']['teacher']
    saved = teacher['signup']
    try:
        teacher['signup'] = ['self', 'invite']  # shipped setting
        assert call(m['auth'], '/auth/signup', {'email': 'self@x.edu', 'password': 'Password123!', 'role': 'teacher'})[0] == 201
        teacher['signup'] = ['invite']  # invite-only
        _invite_only(ddb, m)
    finally:
        teacher['signup'] = saved


def _invite_only(ddb, m):
    status, body = call(m['auth'], '/auth/signup', {'email': 'tt@x.edu', 'password': 'Password123!', 'role': 'teacher'})
    assert status == 403 and 'Teacher accounts are invite-only' in body['message']
    assert 'Item' not in ddb.Table('Users').get_item(Key={'email': 'tt@x.edu'})
    assert call(m['auth'], '/auth/signup', {'email': 'ss@x.edu', 'password': 'Password123!', 'role': 'student'})[0] == 201


# 2 + 3 ---------------------------------------------------------------------------------------
def test_class_boundaries(env):
    _, m = env
    status, a = wf(m, 't@x.edu', 'createAssignment', title='HW', class_name='PHY-204')
    assert status == 200
    wf(m, 't@x.edu', 'postAnnouncement', class_name='PHY-204', title='Hi', message='PHY only')
    for sid in ('S5', 'S4'):  # another class / no class: not on this assignment's roster
        assert wf(m, 't@x.edu', 'remindStudent', assignment_id=a['assignment_id'], student_id=sid)[0] == 404, sid
    assert wf(m, 't@x.edu', 'remindStudent', assignment_id=a['assignment_id'], student_id='S1')[0] == 200

    assert wf(m, 'd@x.edu', 'listStudentAssignments')[1]['assignments'] == []
    assert wf(m, 'd@x.edu', 'listAnnouncements')[1]['announcements'] == []
    assert wf(m, 'd@x.edu', 'getProgress')[1]['todo'] == []
    assert wf(m, 'e@x.edu', 'listStudentAssignments')[1]['assignments'] == []
    assert wf(m, 'e@x.edu', 'listAnnouncements')[1]['announcements'] == []
    assert any(x['assignment_id'] == a['assignment_id'] for x in wf(m, 'a@x.edu', 'listStudentAssignments')[1]['assignments'])


# 4 -------------------------------------------------------------------------------------------
def test_invited_student_gets_the_invite_class(env):
    ddb, m = env
    ddb.Table('GradeAnt_Users').put_item(Item={'user_id': 'N1', 'email': 'news@x.edu', 'role': 'student', 'status': 'invited'})
    ddb.Table('GradeAnt_Invites').put_item(Item={'token': 'tok-1', 'user_id': 'N1', 'full_name': 'New', 'role': 'student',
                                                 'class_name': 'PHY-204', 'expires_at': int(time.time()) + 600, 'consumed': False})
    status, _ = call(m['admin'], '/admin', {'operation': 'completeFirstLogin', 'token': 'tok-1', 'password': 'Welcome123'})
    assert status == 200
    assert ddb.Table('Users').get_item(Key={'email': 'news@x.edu'})['Item']['classes'] == ['PHY-204']


# 5 -------------------------------------------------------------------------------------------
def test_students_upload_only_for_their_own_classes(env):
    ddb, m = env
    t = ddb.Table('GradeAnt_Assignments')
    t.put_item(Item={'assignment_id': 'A-PHY', 'qp_id': 'Q1', 'teacher_id': 'T1', 'class_name': 'PHY-204', 'content_status': 'published'})
    t.put_item(Item={'assignment_id': 'A-CHEM', 'qp_id': 'Q2', 'teacher_id': 'T1', 'class_name': 'CHEM-101', 'content_status': 'published'})
    t.put_item(Item={'assignment_id': 'A-DRAFT', 'qp_id': 'Q3', 'teacher_id': 'T1', 'class_name': 'PHY-204', 'content_status': 'draft'})
    files = [{'name': 'p.pdf', 'type': 'application/pdf', 'size': 8}]
    for aid in ('A-CHEM', 'A-DRAFT', 'A-MISSING', None):
        status, body = call(m['upload'], '/uploadpdf', {'uploadType': 'student', 'assignmentId': aid, 'files': files}, 'a@x.edu')
        assert status == 403 and 'your own classes' in body['message'], aid
    assert call(m['upload'], '/uploadpdf', {'uploadType': 'student', 'assignmentId': 'A-PHY', 'files': files}, 'a@x.edu')[0] == 200


# 6 -------------------------------------------------------------------------------------------
def test_auth_errors_carry_no_exception_text(env):
    _, m = env
    # A non-string password makes the hashing code raise: the reply must not echo the exception.
    status, body = call(m['auth'], '/auth/login', {'email': 'a@x.edu', 'password': 12345, 'userType': 'student'})
    assert status == 500 and body == {'message': 'Error during login.'}


# 7 -------------------------------------------------------------------------------------------
def test_profile_photo_size_is_required_and_signed(env):
    _, m = env
    path = '/auth/profile-photo'
    assert call(m['auth'], path, {'action': 'upload-url', 'contentType': 'image/png'}, 't@x.edu')[0] == 400
    assert call(m['auth'], path, {'action': 'upload-url', 'contentType': 'image/png', 'size': 5 * 1024 * 1024 + 1}, 't@x.edu')[0] == 400
    assert call(m['auth'], path, {'action': 'upload-url', 'contentType': 'image/png', 'size': True}, 't@x.edu')[0] == 400
    status, body = call(m['auth'], path, {'action': 'upload-url', 'contentType': 'image/png', 'size': 2048}, 't@x.edu')
    assert status == 200 and body['key'] == 'profile-photos/T1.png'
    assert 'content-length' in parse_qs(urlparse(body['uploadUrl']).query)['X-Amz-SignedHeaders'][0].split(';')
