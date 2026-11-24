"""Suspension and teacher-ownership rules in the Python Lambdas (same as the Node API), against moto.

Run:  pip install -r python-backend/requirements-dev.txt  &&  pytest python-backend/tests
"""
import base64, hashlib, importlib, json, os, sys, time
import boto3, jwt, pytest
from moto import mock_aws

SECRET = 'test-secret-that-is-at-least-32-bytes-long'
os.environ.update({'JWT_SECRET': SECRET, 'AWS_DEFAULT_REGION': 'us-east-1', 'AWS_ACCESS_KEY_ID': 'x',
                   'AWS_SECRET_ACCESS_KEY': 'x', 'SESSION_TOKEN_SECRET': 's' * 40})
ROOT = os.path.join(os.path.dirname(__file__), '..')  # python-backend/
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT]


def table(ddb, name, key, gsis=()):
    attrs = {key} | {g for g in gsis}
    kw = dict(TableName=name, BillingMode='PAY_PER_REQUEST', KeySchema=[{'AttributeName': key, 'KeyType': 'HASH'}],
              AttributeDefinitions=[{'AttributeName': a, 'AttributeType': 'S'} for a in attrs])
    if gsis:
        kw['GlobalSecondaryIndexes'] = [{'IndexName': f'{g}-index', 'KeySchema': [{'AttributeName': g, 'KeyType': 'HASH'}],
                                         'Projection': {'ProjectionType': 'ALL'}} for g in gsis]
    ddb.create_table(**kw)


def pw(p):
    salt = os.urandom(32)
    return base64.b64encode(salt + hashlib.pbkdf2_hmac('sha256', p.encode(), salt, 100000)).decode()


USERS = {'T1': ('t1@x.dev', 'teacher'), 'T2': ('t2@x.dev', 'teacher'), 'S1': ('s1@x.dev', 'student'),
         'AD': ('ad@x.dev', 'admin'), 'TS': ('ts@x.dev', 'teacher')}


@pytest.fixture
def env():
    with mock_aws():
        ddb = boto3.resource('dynamodb')
        table(ddb, 'Users', 'email')
        table(ddb, 'GradeAnt_Users', 'user_id')
        table(ddb, 'GradeAnt_AuditLogs', 'log_id')
        table(ddb, 'GradeAnt_Assignments', 'assignment_id')
        table(ddb, 'assignment_grading', 'evaluation_id', gsis=('student_id', 'type'))
        for uid, (email, role) in USERS.items():
            ddb.Table('Users').put_item(Item={'email': email, 'user_id': uid, 'userRole': role, 'name': uid,
                                              'password': pw('Password123!'),
                                              **({'status': 'suspended'} if uid == 'TS' else {})})
            ddb.Table('GradeAnt_Users').put_item(Item={'user_id': uid, 'email': email, 'role': role, 'status': 'active'})
        # A1 is T1's assignment, but its rows were (wrongly) stamped by T2 — the assignment wins.
        ddb.Table('GradeAnt_Assignments').put_item(Item={'assignment_id': 'A1', 'teacher_id': 'T1'})
        g = ddb.Table('assignment_grading')
        g.put_item(Item={'evaluation_id': 'E1s', 'student_id': 'S1', 'assignment_id': 'A1', 'qp_id': 'Q1', 'type': 'summary', 'total_marks': 5, 'teacher_id': 'T2'})
        g.put_item(Item={'evaluation_id': 'E1d', 'student_id': 'S1', 'assignment_id': 'A1', 'qp_id': 'Q1', 'type': 'details', 'question_number': '1', 'marks': 5})
        # A2 has no assignment record and no teacher_id: nobody owns it.
        g.put_item(Item={'evaluation_id': 'E2s', 'student_id': 'S1', 'assignment_id': 'A2', 'qp_id': 'Q2', 'type': 'summary', 'total_marks': 3})
        g.put_item(Item={'evaluation_id': 'E2d', 'student_id': 'S1', 'assignment_id': 'A2', 'qp_id': 'Q2', 'type': 'details', 'question_number': '1', 'marks': 3})
        mods = {m: importlib.reload(importlib.import_module(m)) for m in
                ('grades', 'marks', 'corrections', 'auth', 'admin')}
        yield ddb, mods


def call(mod, uid, body):
    email = USERS[uid][0]
    tok = jwt.encode({'user_id': uid, 'email': email, 'exp': int(time.time()) + 600}, SECRET, algorithm='HS256')
    r = mod.lambda_handler({'httpMethod': 'POST', 'headers': {'Authorization': f'Bearer {tok}'}, 'body': json.dumps(body)}, None)
    return r['statusCode'], json.loads(r['body'])


# ---- ownership ---------------------------------------------------------------

def test_get_grades_ownership(env):
    _, m = env
    gg = m['grades']
    single = lambda a: {'operation': 'getSingleStudentGrade', 'student_id': 'S1', 'assignment_id': a}
    assert call(gg, 'T1', single('A1'))[0] == 200          # owns the assignment
    assert call(gg, 'T2', single('A1'))[0] == 403          # row stamp alone no longer grants access
    assert call(gg, 'T1', single('A2'))[0] == 403          # ownerless: fail closed
    assert call(gg, 'AD', single('A2'))[0] == 200
    roster = lambda a, q: {'operation': 'getAllStudentGrades', 'assignment_id': a, 'qp_id': q}
    assert call(gg, 'T1', roster('A1', 'Q1'))[1]['student_count'] == 1
    assert call(gg, 'T2', roster('A1', 'Q1'))[1]['student_count'] == 0
    assert call(gg, 'T1', roster('A2', 'Q2'))[1]['student_count'] == 0
    upd = lambda: {'operation': 'updateStudentGrade', 'student_id': 'S1', 'qp_id': 'Q1', 'total_marks': 9}
    assert call(gg, 'T2', upd())[0] == 403
    assert call(gg, 'T1', upd())[0] == 200


def test_update_db_ownership_and_no_restamp(env):
    ddb, m = env
    ud = m['marks']
    g = ddb.Table('assignment_grading')
    marks = {'operation': 'updateMarks', 'summary': [{'evaluation_id': 'E1s', 'total_marks': 7}],
             'details': [{'evaluation_id': 'E1d', 'question_number': '1', 'marks': 4}]}
    s, b = call(ud, 'T2', marks)
    assert s == 404 and len(b['errors']) == 2               # both rows refused to T2
    assert call(ud, 'T1', marks)[0] == 200
    assert g.get_item(Key={'evaluation_id': 'E1s'})['Item']['teacher_id'] == 'T2'   # not moved
    assert g.get_item(Key={'evaluation_id': 'E1d'})['Item']['teacher_id'] == 'T1'   # filled in when empty
    one = {'operation': 'updateStudentGrade', 'evaluation_id': 'E2s', 'total_marks': 1}
    assert call(ud, 'T1', one)[0] == 403                    # ownerless
    assert call(ud, 'AD', one)[0] == 200


def test_correction_ownership(env):
    _, m = env
    cq = m['corrections']
    ans = lambda a: {'operation': 'getAnswersForCorrection', 'student_id': 'S1', 'assignment_id': a}
    assert call(cq, 'T1', ans('A1'))[0] == 200
    assert call(cq, 'T2', ans('A1'))[0] == 403
    assert call(cq, 'T1', ans('A2'))[0] == 403
    q = {'operation': 'getQuestionsForCorrection', 'qp_id': 'Q1', 'assignment_id': 'A1'}
    assert len(call(cq, 'T1', q)[1]['questions']) == 1
    assert call(cq, 'T2', q)[1]['questions'] == []
    fix = {'operation': 'submitQuestionCorrections', 'qp_id': 'Q1', 'assignment_id': 'A1',
           'corrections': [{'question_number': '1', 'corrected_question': 'New text'}]}
    call(cq, 'T2', fix)
    assert 'question' not in env[0].Table('assignment_grading').get_item(Key={'evaluation_id': 'E1d'})['Item']
    call(cq, 'T1', fix)
    assert env[0].Table('assignment_grading').get_item(Key={'evaluation_id': 'E1d'})['Item']['question'] == 'New text'


# ---- suspension --------------------------------------------------------------

def test_suspended_blocked_everywhere(env):
    _, m = env
    s, b = call(m['grades'], 'TS', {'operation': 'getAllStudentGrades', 'assignment_id': 'A1', 'qp_id': 'Q1'})
    assert (s, b['error']) == (401, 'This account is suspended.')
    assert call(m['marks'], 'TS', {'operation': 'updateStudentGrade'})[0] == 401
    assert call(m['corrections'], 'TS', {'operation': 'x'})[0] == 401
    auth = m['auth']
    r = auth.lambda_handler({'httpMethod': 'POST', 'path': '/auth/login', 'resource': '/auth/login', 'body': json.dumps(
        {'email': 'ts@x.dev', 'password': 'Password123!', 'userType': 'teacher'})}, None)
    assert r['statusCode'] == 403, r
    r = auth.lambda_handler({'httpMethod': 'POST', 'path': '/auth/login', 'resource': '/auth/login', 'body': json.dumps(
        {'email': 'ts@x.dev', 'password': 'wrong', 'userType': 'teacher'})}, None)
    assert r['statusCode'] == 401                           # wrong password doesn't reveal the suspension
    tok = jwt.encode({'user_id': 'TS', 'email': 'ts@x.dev', 'exp': int(time.time()) + 600}, SECRET, algorithm='HS256')
    assert auth.authenticated_user({'headers': {'Authorization': f'Bearer {tok}'}}) is None


def test_admin_console_mirrors_suspension(env):
    ddb, m = env
    la = m['admin']
    users = ddb.Table('Users')
    st = lambda: users.get_item(Key={'email': 't1@x.dev'})['Item'].get('status')
    assert call(la, 'AD', {'operation': 'updateUserStatus', 'userId': 'T1', 'status': 'suspended'})[0] == 200
    assert st() == 'suspended'
    assert call(m['grades'], 'T1', {'operation': 'getAllStudentGrades', 'assignment_id': 'A1', 'qp_id': 'Q1'})[0] == 401
    assert call(la, 'AD', {'operation': 'updateUserStatus', 'userId': 'T1', 'status': 'active'})[0] == 200
    assert st() == 'active'
    assert call(la, 'AD', {'operation': 'deleteUser', 'userId': 'T1'})[0] == 200
    assert st() == 'suspended' and 'Item' not in ddb.Table('GradeAnt_Users').get_item(Key={'user_id': 'T1'})
    # an invited user with no login account yet: no error
    ddb.Table('GradeAnt_Users').put_item(Item={'user_id': 'N1', 'email': 'new@x.dev', 'role': 'teacher', 'status': 'invited'})
    assert call(la, 'AD', {'operation': 'updateUserStatus', 'userId': 'N1', 'status': 'suspended'})[0] == 200
    assert 'Item' not in users.get_item(Key={'email': 'new@x.dev'})
    # a suspended admin loses the console at once
    users.update_item(Key={'email': 'ad@x.dev'}, UpdateExpression='SET #s = :s',
                      ExpressionAttributeNames={'#s': 'status'}, ExpressionAttributeValues={':s': 'suspended'})
    assert call(la, 'AD', {'operation': 'listUsers'})[0] == 401


def test_admin_session_token_rechecked(env):
    ddb, m = env
    la = m['admin']
    claims = {'sub': 'AD', 'role': 'admin', 'exp': int(time.time()) + 600}
    payload = la._b64url_encode(json.dumps(claims).encode())
    import hmac
    sig = hmac.new(la._session_key().encode(), payload.encode('ascii'), hashlib.sha256).hexdigest()
    ev = {'headers': {'Authorization': f'Bearer {payload}.{sig}'}}
    assert la._get_bearer_claims(ev)['role'] == 'admin'
    ddb.Table('GradeAnt_Users').update_item(Key={'user_id': 'AD'}, UpdateExpression='SET #r = :r',
                                            ExpressionAttributeNames={'#r': 'role'}, ExpressionAttributeValues={':r': 'teacher'})
    assert la._get_bearer_claims(ev)['role'] == 'teacher'   # demotion applies at once
    ddb.Table('GradeAnt_Users').delete_item(Key={'user_id': 'AD'})
    with pytest.raises(la.AuthError):
        la._get_bearer_claims(ev)
