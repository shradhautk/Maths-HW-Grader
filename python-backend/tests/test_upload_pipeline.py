"""Group C: /uploadpdf in the pipeline layout (port of lambda_python_files/upload_pdf.py), and the
whole chain end to end with moto: presigned key → PDF put → jsonifier → grader (rows, ungraded) →
workflow "Start grading" payload → grader regrade → summary.
"""
import base64
import hashlib
import importlib
import json
import os
import sys
import time
from urllib.parse import unquote, urlparse

import boto3
import jwt
import pytest
from moto import mock_aws

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT, os.path.dirname(__file__)]
SECRET = 'test-secret-that-is-at-least-32-bytes-long'
BUCKET = 'assignmentgrader-test'
PDF = {'name': 'my answers.pdf', 'type': 'application/pdf', 'size': 2048}


@pytest.fixture()
def env(monkeypatch):
    for k, v in {'JWT_SECRET': SECRET, 'AWS_DEFAULT_REGION': 'eu-north-1', 'AWS_ACCESS_KEY_ID': 'x',
                 'AWS_SECRET_ACCESS_KEY': 'x', 'UPLOAD_BUCKET': BUCKET, 'PIPELINE_BUCKET': BUCKET,
                 'UPLOAD_LAYOUT': 'pipeline', 'MAIL_DRIVER': 'log'}.items():
        monkeypatch.setenv(k, v)
    for k in ('AWS_LAMBDA_FUNCTION_NAME', 'APP_ENV', 'CORS_ALLOWED_ORIGINS', 'JWT_SECRET_ARN', 'PIPELINE_ROOT',
              'GRADE_ON_UPLOAD', 'GEMINI_API_KEY', 'GEMINI_API_KEY_ARN', 'AWS_DATA_REGION', 'PIPELINE_BUCKET_REGION'):
        monkeypatch.delenv(k, raising=False)
    with mock_aws():
        from gradeant_common import config, dynamo, secrets
        from gradeant_common.pipeline import llm, s3io
        dynamo.reset()
        s3io.reset()
        secrets.clear_cache()
        config.app_config.cache_clear()
        llm.set_client(None)
        ddb = boto3.resource('dynamodb', region_name='eu-north-1')
        ddb.create_table(TableName='Users', BillingMode='PAY_PER_REQUEST',
                         KeySchema=[{'AttributeName': 'email', 'KeyType': 'HASH'}],
                         AttributeDefinitions=[{'AttributeName': 'email', 'AttributeType': 'S'}])
        salt = os.urandom(32)
        stored = base64.b64encode(salt + hashlib.pbkdf2_hmac('sha256', b'Correct-horse-9', salt, 100000)).decode()
        for uid, email, role in [('S1', 's@x.edu', 'student'), ('T1', 't@x.edu', 'teacher'),
                                 ('T2', 't2@x.edu', 'teacher'), ('AD', 'a@x.edu', 'admin')]:
            item = {'email': email, 'user_id': uid, 'userRole': role, 'name': uid, 'password': stored}
            if role == 'student':
                item['classes'] = ['PHY']
            ddb.Table('Users').put_item(Item=item)
        ddb.create_table(TableName='GradeAnt_Assignments', BillingMode='PAY_PER_REQUEST',
                         KeySchema=[{'AttributeName': 'assignment_id', 'KeyType': 'HASH'}],
                         AttributeDefinitions=[{'AttributeName': 'assignment_id', 'AttributeType': 'S'}])
        ddb.Table('GradeAnt_Assignments').put_item(Item={'assignment_id': 'A-1', 'qp_id': 'QP-1', 'teacher_id': 'T1',
                                                          'class_name': 'PHY', 'title': 'HW', 'content_status': 'published'})
        boto3.client('s3', region_name='eu-north-1').create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})
        upload = importlib.reload(importlib.import_module('upload'))
        yield ddb, upload
        llm.set_client(None)
        dynamo.reset()
        s3io.reset()


USERS = {'S1': 's@x.edu', 'T1': 't@x.edu', 'T2': 't2@x.edu', 'AD': 'a@x.edu'}


def call(upload, body, uid):
    tok = jwt.encode({'user_id': uid, 'email': USERS[uid], 'exp': int(time.time()) + 600}, SECRET, algorithm='HS256')
    r = upload.lambda_handler({'httpMethod': 'POST', 'path': '/uploadpdf', 'body': json.dumps(body),
                               'headers': {'Content-Type': 'application/json', 'Authorization': f'Bearer {tok}'}}, None)
    return r['statusCode'], json.loads(r['body'])


def key_of(url):
    return unquote(urlparse(url).path).split(f'/{BUCKET}/', 1)[-1].lstrip('/').removeprefix(f'{BUCKET}/')


def test_student_key_uses_caller_and_assignment_qp(env):
    _, upload = env
    status, body = call(upload, {'uploadType': 'student', 'assignmentId': 'A-1', 'courseId': 'QP-1', 'files': [PDF]}, 'S1')
    assert status == 200
    assert key_of(body['urls'][0]).endswith('Data_warehouse/students/assignments/A-1__QP-1/original_pdf/S1.pdf')
    assert body['fileIds'] == ['S1.pdf'] and body['uploadId'].startswith('upload-')
    # No qpId at all → taken from the assignment record
    status, body = call(upload, {'uploadType': 'student', 'assignmentId': 'A-1', 'files': [PDF]}, 'S1')
    assert status == 200 and 'A-1__QP-1/original_pdf/S1.pdf' in key_of(body['urls'][0])


def test_student_rules(env):
    _, upload = env
    base = {'uploadType': 'student', 'assignmentId': 'A-1', 'files': [PDF]}
    assert call(upload, {**base, 'qpId': 'QP-OTHER'}, 'S1')[0] == 400            # qp mismatch
    assert call(upload, {**base, 'files': [PDF, PDF]}, 'S1')[0] == 400           # one file per upload
    assert call(upload, {**base, 'studentId': 'S9'}, 'S1')[0] == 403             # someone else's sheet
    assert call(upload, {**base, 'assignmentId': 'A-404'}, 'S1')[0] == 403       # not an assignment of theirs


def test_teacher_and_answer_keys(env):
    _, upload = env
    status, body = call(upload, {'uploadType': 'teacher', 'assignmentId': 'A-1', 'courseId': 'QP-1', 'files': [PDF]}, 'T1')
    assert status == 200 and key_of(body['urls'][0]).endswith('Data_warehouse/teachers/QP_Rubric/A-1__QP-1/QP_pdfs/QP-1.pdf')
    status, body = call(upload, {'uploadType': 'answer', 'assignmentId': 'A-1', 'qpId': 'QP-1', 'files': [PDF]}, 'T1')
    assert status == 200 and key_of(body['urls'][0]).endswith(
        'Data_warehouse/teachers/QPRubricWithAnswers/A-1__QP-1/solution_pdfs/A-1_QP-1_answer.pdf')
    # A teacher uploading a student's sheet must name the student
    assert call(upload, {'uploadType': 'student', 'assignmentId': 'A-1', 'files': [PDF]}, 'T1')[0] == 400
    status, body = call(upload, {'uploadType': 'student', 'assignmentId': 'A-1', 'studentId': 'S1', 'files': [PDF]}, 'T1')
    assert status == 200 and key_of(body['urls'][0]).endswith('/original_pdf/S1.pdf')


def test_ownership_and_metadata_wrapper(env):
    _, upload = env
    body = {'uploadType': 'teacher', 'assignmentId': 'A-1', 'courseId': 'QP-1', 'files': [PDF]}
    assert call(upload, body, 'T2')[0] == 403                                    # another teacher's assignment
    assert call(upload, body, 'AD')[0] == 200                                     # admins pass
    assert call(upload, {'metadata': body}, 'T1')[0] == 200                       # original's wrapped form
    assert call(upload, {**body, 'assignmentId': 'A-404'}, 'T1')[0] == 404        # pipeline layout needs the assignment


def test_misconfigured_bucket_and_template_layout_unchanged(env, monkeypatch):
    _, upload = env
    body = {'uploadType': 'teacher', 'assignmentId': 'A-1', 'courseId': 'QP-1', 'files': [PDF]}
    monkeypatch.setenv('PIPELINE_BUCKET', 'some-other-bucket')
    assert call(upload, body, 'T1')[0] == 503
    monkeypatch.setenv('UPLOAD_LAYOUT', 'template')
    status, out = call(upload, body, 'T1')
    assert status == 200 and key_of(out['urls'][0]).endswith('uploads/teacher/A-1/my_answers.pdf')
    # Template layout: a teacher may still upload before the assignment exists (legacy flows), not for someone else's
    assert call(upload, {**body, 'assignmentId': 'A-NEW'}, 'T1')[0] == 200
    assert call(upload, body, 'T2')[0] == 403


def test_end_to_end_upload_to_summary(env, monkeypatch):
    """Check-in C demo: presign → PUT → jsonifier → grader (ungraded) → Start grading → regrade → summary."""
    pytest.importorskip('fitz')
    import test_pipeline_grading as tg
    import test_pipeline_jsonifier as tj
    from gradeant_common.pipeline import llm, paths, s3io
    ddb, upload = env
    monkeypatch.setenv('GRADE_ON_UPLOAD', 'false')
    tg_table = boto3.client('dynamodb', region_name='eu-north-1')
    tg_table.create_table(
        TableName='assignment_grading', BillingMode='PAY_PER_REQUEST',
        KeySchema=[{'AttributeName': 'evaluation_id', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': n, 'AttributeType': 'S'} for n in ('evaluation_id', 'student_id', 'type')],
        GlobalSecondaryIndexes=[{'IndexName': i, 'KeySchema': [{'AttributeName': k, 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}}
                                for i, k in (('student_id-index', 'student_id'), ('type-index', 'type'))])
    layout = paths.Layout('Data_warehouse', 'A-1', 'QP-1')

    # Teacher side: question paper PDF, its answer schema, its schema JSON and rubric
    s3io.s3().put_object(Bucket=BUCKET, Key=layout.qp_pdf, Body=tj.make_pdf('Q1. Define binding energy. [5 marks]'))
    s3io.write_text(BUCKET, layout.qp_schema_answer_txt, tj.SCHEMA)
    s3io.write_json(BUCKET, layout.qp_schema_json(False), {'questions': [{'question_number': 1, 'description': 'Define binding energy.', 'max_marks': 5}]})
    s3io.write_json(BUCKET, layout.rubric_json(False), {'questions': [{'rubric': 'Mentions mass defect.'}]})

    # 1. The student asks for an upload URL; the browser PUTs the PDF to that key
    status, body = call(upload, {'uploadType': 'student', 'assignmentId': 'A-1', 'files': [PDF]}, 'S1')
    assert status == 200
    pdf_key = key_of(body['urls'][0])
    s3io.s3().put_object(Bucket=BUCKET, Key=pdf_key, Body=tj.make_pdf('1) Binding energy = energy to split a nucleus'))

    # 2. S3 → SQS → jsonifier
    jsonifier = importlib.reload(importlib.import_module('jsonifier'))
    llm.set_client(tj.FakeParseLLM())
    assert jsonifier.lambda_handler(tj.sqs_event(pdf_key, bucket=BUCKET), None) == {'batchItemFailures': []}
    json_key = layout.parsed_json('S1')

    # 3. S3 → SQS → grader: GRADE_ON_UPLOAD=false → ungraded rows, no summary
    grader = importlib.reload(importlib.import_module('grader'))
    assert grader.lambda_handler(tg.sqs_event(json_key, bucket=BUCKET), None) == {'batchItemFailures': []}
    table = ddb.Table('assignment_grading')
    rows = table.scan()['Items']
    assert len(rows) == 1 and 'marks' not in rows[0] and rows[0]['student_id'] == 'S1'

    # 4. The student confirms the transcription; the teacher presses Start grading → workflow invokes the grader
    table.update_item(Key={'evaluation_id': rows[0]['evaluation_id']}, UpdateExpression='SET #s = :s',
                      ExpressionAttributeNames={'#s': 'status'}, ExpressionAttributeValues={':s': 'corrected'})
    assignment = {'assignment_id': 'A-1', 'qp_id': 'QP-1', 'method': 'ai', 'persona': 'balanced', 'partial_credit': True,
                  'rubric': [{'question_number': 1, 'subpart': None, 'max_marks': 5, 'criteria': 'Mentions mass defect.'}]}
    workflow, payload = tg._workflow_payload(monkeypatch, assignment, 'S1')
    llm.set_client(tg.FakeLLM(grade=lambda p: '{"marks_awarded": 4, "reasoning": "good"}'))
    assert grader.lambda_handler(payload, None) == {'ok': True, 'student_id': 'S1', 'total_marks': 4}

    details = [r for r in table.scan()['Items'] if r.get('type') != 'summary']
    summary = next(r for r in table.scan()['Items'] if r.get('type') == 'summary')
    assert details[0]['marks'] == 4 and details[0]['status'] == 'graded' and summary['total_marks'] == 4
    assert workflow._derived_status({'details': details, 'summary': summary}) == 'ai_graded'
