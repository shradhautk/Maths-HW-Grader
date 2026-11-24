"""Runs the parity cases against the ORIGINAL Python Lambda handlers (moto
in-memory DynamoDB) and prints {case: {status, body}} as JSON on stdout.

    python backend/scripts/parity/run_python.py   (needs lambda_python_files/requirements-dev.txt)
"""
import base64
import hashlib
import json
import os
import re
import secrets
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SECRET = 'parity-jwt-secret-parity-jwt-secret-0123'
SESSION_SECRET = 'parity-session-secret-0123456789abcdef'

os.environ.update({
    'JWT_SECRET': SECRET, 'SESSION_TOKEN_SECRET': SESSION_SECRET,
    'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x',
})
if os.environ.get('PARITY_PYTHON') != 'python-backend':
    # The original admin Lambda names its tables differently; python-backend uses the Node names (defaults).
    os.environ.update({'USERS_TABLE': 'GradeAnt_Users', 'LOGIN_USERS_TABLE': 'Users'})
for k in ('AWS_LAMBDA_FUNCTION_NAME', 'PROFILE_PHOTO_BUCKET', 'SES_SENDER_EMAIL', 'SNS_TOPIC_ARN', 'GRADING_FUNCTION_NAME'):
    os.environ.pop(k, None)

# The handlers print() diagnostics; keep stdout for the JSON result only.
REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

from moto import mock_aws  # noqa: E402

mock = mock_aws()
mock.start()
import boto3  # noqa: E402
import jwt  # noqa: E402
from decimal import Decimal  # noqa: E402

ddb = boto3.resource('dynamodb', region_name='us-east-1')


def table(name, *keys, gsis=()):
    attrs = {k for k, _ in keys} | {k for _, k in gsis}
    ddb.create_table(
        TableName=name, BillingMode='PAY_PER_REQUEST',
        KeySchema=[{'AttributeName': k, 'KeyType': t} for k, t in keys],
        AttributeDefinitions=[{'AttributeName': a, 'AttributeType': 'S'} for a in sorted(attrs)],
        **({'GlobalSecondaryIndexes': [{'IndexName': n, 'KeySchema': [{'AttributeName': k, 'KeyType': 'HASH'}],
                                        'Projection': {'ProjectionType': 'ALL'}} for n, k in gsis]} if gsis else {}))


table('Users', ('email', 'HASH'))
table('assignment_grading', ('evaluation_id', 'HASH'),
      gsis=[('student_id-index', 'student_id'), ('type-index', 'type'), ('assignment_id-index', 'assignment_id')])
table('GradeAnt_Assignments', ('assignment_id', 'HASH'), gsis=[('teacher_id-index', 'teacher_id')])
table('GradeAnt_Submissions', ('assignment_id', 'HASH'), ('student_id', 'RANGE'), gsis=[('student_id-index', 'student_id')])
table('GradeAnt_Announcements', ('announcement_id', 'HASH'))
for name, pk in [('GradeAnt_Users', 'user_id'), ('GradeAnt_ResetRequests', 'request_id'), ('GradeAnt_Content', 'post_id'),
                 ('GradeAnt_Tickets', 'ticket_id'), ('GradeAnt_AuditLogs', 'log_id'), ('GradeAnt_Settings', 'setting_key'),
                 ('GradeAnt_Invites', 'token'), ('GradeAnt_Analytics', 'metric_key')]:
    table(name, (pk, 'HASH'))


def resolve_value(v):
    if isinstance(v, dict) and '$loginHash' in v:
        salt = os.urandom(32)
        return base64.b64encode(salt + hashlib.pbkdf2_hmac('sha256', v['$loginHash'].encode(), salt, 100000)).decode()
    if isinstance(v, dict) and '$adminHash' in v:
        salt = secrets.token_bytes(16)
        return f"{salt.hex()}:{hashlib.pbkdf2_hmac('sha256', v['$adminHash'].encode(), salt, 200000).hex()}"
    if isinstance(v, float):
        return Decimal(str(v))
    if isinstance(v, dict):
        return {k: resolve_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [resolve_value(x) for x in v]
    return v


fixture = json.load(open(os.path.join(HERE, 'fixture.json')))
for name, items in fixture['tables'].items():
    for item in items:
        ddb.Table(name).put_item(Item=resolve_value(item))

import importlib  # noqa: E402

# PARITY_PYTHON=python-backend checks the updated Python 3.13 backend
# (python-backend/functions + its gradeant_common layer); the default is the
# original Lambdas in lambda_python_files/ and admin/.
if os.environ.get('PARITY_PYTHON') == 'python-backend':
    sys.path[:0] = [os.path.join(REPO, 'python-backend', 'functions'), os.path.join(REPO, 'python-backend')]
    MODULES = {'/auth': 'auth', '/getgrades': 'grades', '/correctanswer': 'corrections', '/updatedb': 'marks',
               '/admin': 'admin', '/workflow': 'workflow'}
else:
    sys.path[:0] = [os.path.join(REPO, 'lambda_python_files'), os.path.join(REPO, 'admin')]
    MODULES = {'/auth': 'authetication', '/getgrades': 'get_grades', '/correctanswer': 'correct_questions_answer',
               '/updatedb': 'update_db', '/admin': 'lambda_admin', '/workflow': 'workflow'}
HANDLERS = {prefix: importlib.import_module(name) for prefix, name in MODULES.items()}

claude_calls = []


def fake_claude(system, messages, *, effort, max_tokens=16000, output_format=None):
    claude_calls.append({'system': system, 'messages': messages, 'effort': effort, 'format': output_format})
    return None if output_format else 'CANNED'


# python-backend calls the model through _ask_ai (Gemini); the original Lambdas through _ask_claude.
setattr(HANDLERS['/workflow'], '_ask_ai' if hasattr(HANDLERS['/workflow'], '_ask_ai') else '_ask_claude', fake_claude)
users = {u['email']: u for u in fixture['tables']['Users']}


def token_for(spec):
    if spec == '@bad':
        return 'not-a-token'
    if spec == '@ghost':
        return jwt.encode({'email': 'ghost@x.edu', 'user_id': 'G1', 'exp': int(time.time()) + 600}, SECRET, algorithm='HS256')
    expired = spec.startswith('@expired:')
    email = spec.split(':', 1)[1] if expired else spec
    u = users[email]
    exp = int(time.time()) + (-10 if expired else 600)
    return jwt.encode({'email': u['email'], 'user_id': u['user_id'], 'exp': exp}, SECRET, algorithm='HS256')


results = {}
REF = re.compile(r'\$\{ref:([^}]+)\}')


def ref(expr):
    case, _, path = expr.partition('.')
    # case names contain dots: find the longest case-name prefix we have
    parts = expr.split('.')
    for i in range(len(parts), 0, -1):
        name = '.'.join(parts[:i])
        if name in results:
            value = results[name]['body']
            for key in parts[i:]:
                value = (value or {}).get(key, '<missing>')
            return value
    raise KeyError(expr)


def subst(v):
    if isinstance(v, str):
        return REF.sub(lambda m: str(ref(m.group(1))), v)
    if isinstance(v, dict):
        return {k: subst(x) for k, x in v.items()}
    if isinstance(v, list):
        return [subst(x) for x in v]
    return v


for case in json.load(open(os.path.join(HERE, 'cases.json'))):
    headers = {'Content-Type': 'application/json'}  # browsers always send headers
    if case.get('as'):
        headers['Authorization'] = f"Bearer {token_for(case['as'])}"
    if case.get('token'):
        headers['Authorization'] = f"Bearer {subst(case['token'])}"
    path = case['path']
    module = next(m for prefix, m in HANDLERS.items() if path == prefix or path.startswith(prefix + '/'))
    event = {'httpMethod': 'POST', 'path': path, 'resource': path, 'headers': headers, 'requestContext': {},
             'body': case['raw'] if 'raw' in case else json.dumps(subst(case.get('body', {})))}
    try:
        res = module.lambda_handler(event, None)
        if res is None:
            results[case['name']] = {'status': 502, 'body': {'message': 'Internal server error'}}
            continue
        body = res.get('body')
        results[case['name']] = {'status': int(res['statusCode']), 'body': json.loads(body) if body else None}
    except Exception as e:  # an uncaught exception = Lambda error = 502 from API Gateway
        results[case['name']] = {'status': 502, 'body': {'message': 'Internal server error', 'exception': repr(e)}}

results['__claude_calls__'] = {'status': 0, 'body': claude_calls}
json.dump(results, REAL_STDOUT, default=lambda o: int(o) if isinstance(o, Decimal) and o % 1 == 0 else float(o))
