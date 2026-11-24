"""contact, upload and health functions, plus the shared request layer (CORS, preflight,
secrets). Mirrors backend/test/api.test.js for the same endpoints.

Run:  pip install -r python-backend/requirements-dev.txt  &&  pytest python-backend/tests
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

ROOT = os.path.join(os.path.dirname(__file__), '..')  # python-backend/
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT]
SECRET = 'test-secret-that-is-at-least-32-bytes-long'
PASSWORD = 'Correct-horse-9'


@pytest.fixture()
def env(monkeypatch):
    for k, v in {'JWT_SECRET': SECRET, 'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_ACCESS_KEY_ID': 'x',
                 'AWS_SECRET_ACCESS_KEY': 'x', 'UPLOAD_BUCKET': 'uploads-test', 'MAIL_DRIVER': 'log'}.items():
        monkeypatch.setenv(k, v)
    for k in ('AWS_LAMBDA_FUNCTION_NAME', 'APP_ENV', 'CORS_ALLOWED_ORIGINS', 'JWT_SECRET_ARN'):
        monkeypatch.delenv(k, raising=False)
    with mock_aws():
        from gradeant_common import config, dynamo, mail, secrets
        dynamo.reset()
        secrets.clear_cache()
        config.app_config.cache_clear()
        mail.sent_mail.clear()
        ddb = boto3.resource('dynamodb', region_name='us-east-1')
        ddb.create_table(TableName='Users', BillingMode='PAY_PER_REQUEST',
                         KeySchema=[{'AttributeName': 'email', 'KeyType': 'HASH'}],
                         AttributeDefinitions=[{'AttributeName': 'email', 'AttributeType': 'S'}])
        salt = os.urandom(32)
        stored = base64.b64encode(salt + hashlib.pbkdf2_hmac('sha256', PASSWORD.encode(), salt, 100000)).decode()
        for uid, email, role in [('S1', 's@x.edu', 'student'), ('T1', 't@x.edu', 'teacher')]:
            item = {'email': email, 'user_id': uid, 'userRole': role, 'name': uid, 'password': stored}
            if role == 'student':
                item['classes'] = ['PHY']
            ddb.Table('Users').put_item(Item=item)
        ddb.create_table(TableName='GradeAnt_Assignments', BillingMode='PAY_PER_REQUEST',
                         KeySchema=[{'AttributeName': 'assignment_id', 'KeyType': 'HASH'}],
                         AttributeDefinitions=[{'AttributeName': 'assignment_id', 'AttributeType': 'S'}])
        ddb.Table('GradeAnt_Assignments').put_item(Item={'assignment_id': 'A-1', 'qp_id': 'QP-1', 'teacher_id': 'T1', 'class_name': 'PHY',
                                                          'title': 'HW', 'content_status': 'published'})
        mods = {m: importlib.reload(importlib.import_module(m)) for m in ('contact', 'upload', 'health', 'grades')}
        yield ddb, mods


def token(uid, email):
    return jwt.encode({'user_id': uid, 'email': email, 'exp': int(time.time()) + 600}, SECRET, algorithm='HS256')


def event(method, path, body=None, headers=None):
    return {'httpMethod': method, 'path': path, 'headers': {'Content-Type': 'application/json', **(headers or {})},
            'requestContext': {'identity': {'sourceIp': '203.0.113.7'}},
            'body': None if body is None else json.dumps(body)}


def call(mod, *args, **kw):
    r = mod.lambda_handler(event(*args, **kw), None)
    return r['statusCode'], (json.loads(r['body']) if r.get('body') else None), r.get('headers', {})


# ---- health ------------------------------------------------------------------

def test_health_and_readiness(env):
    _, m = env
    status, body, _ = call(m['health'], 'GET', '/health')
    assert status == 200 and body['status'] == 'ok' and body['runtime'] == 'python'
    status, body, _ = call(m['health'], 'GET', '/health/ready')
    assert status == 200 and body['checks']['dynamodb']['ok'] and body['checks']['jwtSecret']['ok']
    assert call(m['health'], 'GET', '/getgrades')[0] == 404


def test_readiness_degraded_without_secret(env, monkeypatch):
    _, m = env
    monkeypatch.delenv('JWT_SECRET')
    status, body, _ = call(m['health'], 'GET', '/health/ready')
    assert status == 503 and body['status'] == 'degraded' and not body['checks']['jwtSecret']['ok']


# ---- contact -----------------------------------------------------------------

def test_contact_message_and_pilot(env):
    _, m = env
    from gradeant_common import mail
    status, body, _ = call(m['contact'], 'POST', '/contact',
                           {'name': 'Ada', 'email': 'ada@school.edu', 'subject': 'Support', 'message': 'Hello\nthere'})
    assert status == 200 and body['message'] == 'Thanks, your message has been sent.'
    sent = mail.sent_mail[-1]
    assert sent['to'] == 'contact@gradeant.com' and sent['replyTo'] == 'ada@school.edu'
    assert sent['subject'] == '[Website message] Support — Ada' and 'Hello\nthere' in sent['text']
    status, body, _ = call(m['contact'], 'POST', '/contact', {'kind': 'pilot', 'name': 'Ada', 'email': 'ada@school.edu',
                                                              'school': 'Rolla High', 'subject': 'Physics', 'students': '120'})
    assert status == 200 and 'pilot request' in body['message']
    assert mail.sent_mail[-1]['subject'].startswith('[Pilot request] Rolla High')


def test_contact_validation_honeypot_and_rate_limit(env):
    _, m = env
    c = m['contact']
    assert call(c, 'POST', '/contact', {'name': '', 'email': 'a@b.co', 'message': 'x'})[0] == 400
    assert call(c, 'POST', '/contact', {'name': 'A', 'email': 'nope', 'message': 'x'})[0] == 400
    assert call(c, 'POST', '/contact', {'kind': 'pilot', 'name': 'A', 'email': 'a@b.co', 'school': 'S', 'subject': 'M', 'students': 'lots'})[0] == 400
    from gradeant_common import mail
    before = len(mail.sent_mail)
    assert call(c, 'POST', '/contact', {'name': 'Bot', 'email': 'b@b.co', 'message': 'spam', 'website': 'http://spam'})[0] == 200
    assert len(mail.sent_mail) == before  # honeypot: nothing sent
    c._recent.clear()
    codes = [call(c, 'POST', '/contact', {'name': 'A', 'email': 'a@b.co', 'message': f'm{i}'})[0] for i in range(6)]
    assert codes == [200] * 5 + [429]
    status, body, _ = call(c, 'POST', '/contact', 'not used', headers={})
    # (a non-object body is treated as {} → missing name)
    assert status == 400


# ---- upload ------------------------------------------------------------------

def test_upload_rules_and_presigned_urls(env):
    _, m = env
    u = m['upload']
    meta = {'uploadType': 'student', 'assignmentId': 'A-1', 'files': [{'name': 'answers.pdf', 'type': 'application/pdf', 'size': 2048}]}
    student = {'Authorization': f"Bearer {token('S1', 's@x.edu')}"}
    assert call(u, 'POST', '/uploadpdf', meta)[0] == 401
    assert call(u, 'POST', '/uploadpdf', {**meta, 'studentId': 'S9'}, headers=student)[0] == 403
    assert call(u, 'POST', '/uploadpdf', {**meta, 'uploadType': 'teacher'}, headers=student)[0] == 403
    assert call(u, 'POST', '/uploadpdf', {**meta, 'files': [{'name': 'x.exe', 'type': 'application/x-msdownload', 'size': 10}]}, headers=student)[0] == 400
    assert call(u, 'POST', '/uploadpdf', {**meta, 'files': [{'name': 'big.pdf', 'type': 'application/pdf', 'size': 50 * 1024 * 1024}]}, headers=student)[0] == 400
    status, body, _ = call(u, 'POST', '/uploadpdf', meta, headers=student)
    assert status == 200 and len(body['urls']) == 1
    url = urlparse(body['urls'][0])
    assert url.path.endswith('/uploads/student/A-1/answers.pdf') or '/uploads/student/A-1/answers.pdf' in url.path
    signed = parse_qs(url.query)['X-Amz-SignedHeaders'][0].split(';')
    assert 'content-length' in signed  # the validated size is part of the signature


def test_upload_off_without_bucket(env, monkeypatch):
    _, m = env
    monkeypatch.delenv('UPLOAD_BUCKET')
    student = {'Authorization': f"Bearer {token('S1', 's@x.edu')}"}
    status, body, _ = call(m['upload'], 'POST', '/uploadpdf', {'uploadType': 'student', 'files': []}, headers=student)
    assert status == 503 and body['message'] == 'File uploads are not configured on this server.'


# ---- shared request layer ----------------------------------------------------

def test_preflight_and_cors_allowlist(env, monkeypatch):
    _, m = env
    status, _, headers = call(m['grades'], 'OPTIONS', '/getgrades', headers={'Origin': 'https://app.gradeant.io'})
    assert status == 204 and headers['Access-Control-Allow-Origin'] == '*'  # development: app.yaml corsOrigins ["*"]
    monkeypatch.setenv('CORS_ALLOWED_ORIGINS', 'https://app.gradeant.io')
    _, _, headers = call(m['grades'], 'OPTIONS', '/getgrades', headers={'Origin': 'https://app.gradeant.io'})
    assert headers['Access-Control-Allow-Origin'] == 'https://app.gradeant.io'
    _, _, headers = call(m['grades'], 'OPTIONS', '/getgrades', headers={'Origin': 'https://evil.example'})
    assert 'Access-Control-Allow-Origin' not in headers and headers['Vary'] == 'Origin'


def test_errors_never_leak_in_production(env, monkeypatch):
    _, m = env
    monkeypatch.setenv('APP_ENV', 'production')
    monkeypatch.setenv('CORS_ALLOWED_ORIGINS', 'https://app.gradeant.io')
    from gradeant_common import config
    config.app_config.cache_clear()
    teacher = {'Authorization': f"Bearer {token('T1', 't@x.edu')}"}
    # the grading table doesn't exist in this fixture → an unexpected boto error inside the operation
    status, body, _ = call(m['grades'], 'POST', '/getgrades', {'operation': 'getAllStudentGrades', 'qp_id': 'Q', 'assignment_id': 'A'}, headers=teacher)
    assert status == 500 and body == {'error': 'Internal server error'}


def test_secret_from_secrets_manager(env, monkeypatch):
    _, m = env
    sm = boto3.client('secretsmanager', region_name='us-east-1')
    arn = sm.create_secret(Name='gradeant/jwt', SecretString=json.dumps({'JWT_SECRET': SECRET}))['ARN']
    monkeypatch.delenv('JWT_SECRET')
    monkeypatch.setenv('JWT_SECRET_ARN', arn)
    from gradeant_common import secrets
    secrets.clear_cache()
    assert secrets.jwt_secret() == SECRET
    status, body, _ = call(m['health'], 'GET', '/health/ready')
    assert status == 200 and body['checks']['jwtSecret']['ok']
