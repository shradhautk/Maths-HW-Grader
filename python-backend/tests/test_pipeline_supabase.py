"""The grader on Supabase (database: supabase): every database call goes through PostgREST.

A fake PostgREST (httpx2.MockTransport) keeps rows in memory with the same semantics as
backend/sql/002_pipeline_api.sql (grading_replace_rows / grading_patch) and the generated
columns of gradeant.assignment_grading. S3 is moto; the model is a fake.
The live test at the end runs only with SUPABASE_TEST_URL / SUPABASE_TEST_KEY set.
"""
import importlib
import json
import os
import sys
import uuid
from urllib.parse import parse_qsl

import boto3
import httpx2
import pytest
from moto import mock_aws

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT, os.path.dirname(__file__)]

import test_pipeline_grading as tg  # noqa: E402  (fixtures: question paper, answers, FakeLLM, events)
from gradeant_common import dynamo, grading as grading_table, pg_rest, secrets  # noqa: E402
from gradeant_common.pipeline import grading, llm, s3io  # noqa: E402

URL = 'https://test-project.supabase.co'
KEY = 'service-role-test-key'


class FakePostgREST:
    """In-memory gradeant.assignment_grading behind PostgREST's URL shapes."""

    def __init__(self, max_rows=1000):
        self.rows = {}              # evaluation_id → item
        self.requests = []
        self.max_rows = max_rows
        self.fail_next = []         # status codes to answer before behaving normally

    def __call__(self, request):
        self.requests.append(request)
        if self.fail_next:
            return httpx2.Response(self.fail_next.pop(0), json={'message': 'unavailable'})
        assert request.headers['apikey'] == KEY and request.headers['authorization'] == f'Bearer {KEY}'
        assert request.headers['accept-profile'] == 'gradeant' and request.headers['content-profile'] == 'gradeant'
        path = request.url.path
        if request.method == 'GET' and path == '/rest/v1/assignment_grading':
            q = dict(parse_qsl(request.url.query.decode()))
            assert q.pop('select') == 'item' and q.pop('order') == 'evaluation_id'
            limit, offset = int(q.pop('limit')), int(q.pop('offset'))
            filters = {col: value[3:] for col, value in q.items()}
            assert all(value.startswith('eq.') for value in q.values()) and set(filters) <= {'student_id', 'assignment_id', 'qp_id'}
            matched = [self.rows[k] for k in sorted(self.rows) if all(self.rows[k].get(c) == v for c, v in filters.items())]
            page = matched[offset:offset + min(limit, self.max_rows)]
            return httpx2.Response(200, json=[{'item': item} for item in page])
        body = json.loads(request.content or b'{}')
        if path == '/rest/v1/rpc/grading_replace_rows':
            assert isinstance(body['p_rows'], list)
            for evaluation_id in body.get('p_delete', []):
                self.rows.pop(evaluation_id, None)
            for row in body['p_rows']:
                self.rows[row['evaluation_id']] = row
            return httpx2.Response(200, json=len(body['p_rows']))
        if path == '/rest/v1/rpc/grading_patch':
            item = dict(self.rows.get(body['p_id'], {'evaluation_id': body['p_id']}))
            item.update(body['p_set'])
            for key in body.get('p_remove', []):
                item.pop(key, None)
            self.rows[body['p_id']] = item
            return httpx2.Response(204)
        return httpx2.Response(404, json={'message': f'no route {request.method} {path}'})


@pytest.fixture()
def supabase(monkeypatch):
    for k, v in {'AWS_DEFAULT_REGION': 'eu-north-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x',
                 'PIPELINE_BUCKET': tg.BUCKET, 'DATABASE_PROVIDER': 'supabase', 'SUPABASE_URL': URL,
                 'SUPABASE_SERVICE_ROLE_KEY': KEY, 'JWT_SECRET': 'test-secret-that-is-at-least-32-bytes-long'}.items():
        monkeypatch.setenv(k, v)
    for k in ('GEMINI_API_KEY', 'GEMINI_API_KEY_ARN', 'PIPELINE_ROOT', 'PIPELINE_BUCKET_REGION', 'AWS_DATA_REGION',
              'GRADE_ON_UPLOAD', 'GRADER_MAX_WORKERS', 'SUPABASE_SERVICE_ROLE_KEY_ARN'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pg_rest.time, 'sleep', lambda s: None)
    secrets.clear_cache()
    fake = FakePostgREST()
    pg_rest.set_transport(httpx2.MockTransport(fake))
    llm.set_client(None)
    with mock_aws():  # S3 only: there is deliberately NO DynamoDB table here
        boto3.client('s3', region_name='eu-north-1').create_bucket(
            Bucket=tg.BUCKET, CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})
        s3io.reset()
        dynamo.reset()
        yield fake
        s3io.reset()
        dynamo.reset()  # building the payload reloads workflow inside this mock: drop its cached client
    pg_rest.set_transport(None)
    llm.set_client(None)
    secrets.clear_cache()


@pytest.fixture()
def grader():
    return importlib.reload(importlib.import_module('grader'))


def _details(fake):
    return sorted((r for r in fake.rows.values() if r.get('type') != 'summary'), key=lambda r: (str(r['question_number']), str(r.get('subpart'))))


def _summary(fake):
    return next((r for r in fake.rows.values() if r.get('type') == 'summary'), None)


def test_requests_have_the_postgrest_shape(supabase):
    supabase.rows = {'a': {'evaluation_id': 'a', 'student_id': 'S1', 'assignment_id': 'A1', 'qp_id': 'QP1', 'question_number': 1},
                     'b': {'evaluation_id': 'b', 'student_id': 'S1', 'assignment_id': 'A1', 'qp_id': 'QP1', 'type': 'summary'},
                     'c': {'evaluation_id': 'c', 'student_id': 'S2', 'assignment_id': 'A1', 'qp_id': 'QP1'}}
    assert [r['evaluation_id'] for r in grading_table.student_rows('S1', assignment_id='A1', qp_id='QP1')] == ['a', 'b']
    assert [r['evaluation_id'] for r in grading_table.student_rows('S1', assignment_id='A1', type='notSummary')] == ['a']
    assert [r['evaluation_id'] for r in grading_table.student_rows('S1', type='summary')] == ['b']
    assert [r['evaluation_id'] for r in grading_table.student_rows('S1', question_number=1.0)] == ['a']   # 1 == 1.0
    assert grading_table.student_rows('S1', question_number='1') == []                                    # "1" != 1
    first = supabase.requests[0]
    assert first.method == 'GET' and first.url.path == '/rest/v1/assignment_grading'
    assert dict(parse_qsl(first.url.query.decode()))['student_id'] == 'eq.S1'


def test_paging_reads_every_row_when_postgrest_caps_pages(supabase):
    supabase.max_rows = 2  # PostgREST max_rows lower than the page the client asks for
    supabase.rows = {f'r{i}': {'evaluation_id': f'r{i}', 'student_id': 'S1'} for i in range(5)}
    assert len(grading_table.student_rows('S1')) == 5


def test_retries_then_fails_cleanly(supabase):
    supabase.fail_next = [503, 503]
    assert grading_table.student_rows('nobody') == []                   # recovered on the 3rd try
    supabase.fail_next = [503, 503, 503]
    with pytest.raises(pg_rest.PgRestError, match='503'):
        grading_table.student_rows('nobody')
    supabase.fail_next = [400]
    with pytest.raises(pg_rest.PgRestError, match='400'):
        grading_table.student_rows('nobody')
    assert len([r for r in supabase.requests if r.method == 'GET']) == 3 + 3 + 1  # a 400 is never retried


def test_missing_settings_are_reported(supabase, monkeypatch):
    monkeypatch.delenv('SUPABASE_SERVICE_ROLE_KEY')
    pg_rest.set_transport(httpx2.MockTransport(supabase))
    with pytest.raises(pg_rest.PgRestError, match='SUPABASE_SERVICE_ROLE_KEY are required'):
        grading_table.student_rows('S1')


def test_dynamodb_is_never_touched(supabase):
    with pytest.raises(RuntimeError, match='DynamoDB is not used when database: supabase'):
        dynamo.table('grading')


def test_upload_grades_and_writes_rows_through_rpc(supabase, grader):
    key = tg._seed()
    llm.set_client(tg.FakeLLM())
    assert grader.lambda_handler(tg.sqs_event(key), None) == {'batchItemFailures': []}
    details, summary = _details(supabase), _summary(supabase)
    assert len(details) == 3 and summary['total_marks'] == 6
    assert all(r['marks'] == 2 and r['ocr_answer'] == r['student_answer'] for r in details)
    assert 'subpart' not in details[0]                                # None values are dropped, as on DynamoDB
    rpcs = [r.url.path.rsplit('/', 1)[-1] for r in supabase.requests if r.method == 'POST']
    assert rpcs == ['grading_replace_rows', 'grading_replace_rows']   # details, then the summary LAST
    last = json.loads(supabase.requests[-1].content)
    assert [r['type'] for r in last['p_rows']] == ['summary']


def test_repeat_message_is_skipped_on_supabase_too(supabase, grader):
    key = tg._seed()
    llm.set_client(tg.FakeLLM())
    grader.lambda_handler(tg.sqs_event(key), None)
    before = json.dumps(supabase.rows, sort_keys=True)
    posts = len([r for r in supabase.requests if r.method == 'POST'])
    assert grader.lambda_handler(tg.sqs_event(key, 'm2'), None) == {'batchItemFailures': []}
    assert json.dumps(supabase.rows, sort_keys=True) == before
    assert len([r for r in supabase.requests if r.method == 'POST']) == posts


def test_supabase_outage_fails_the_message(supabase, grader):
    key = tg._seed()
    llm.set_client(tg.FakeLLM())
    supabase.fail_next = [503, 503, 503]
    assert grader.lambda_handler(tg.sqs_event(key), None) == {'batchItemFailures': [{'itemIdentifier': 'm1'}]}
    assert supabase.rows == {}


def _seed_corrected(fake, rewrite=None):
    rows = grading.detail_rows(grading.extract_qa_pairs(tg.QP_JSON, tg.ANSWERS, tg.RUBRIC), tg.AID, tg.QP, tg.SID, graded=False)
    for r in rows:
        r['status'] = 'corrected'
        if rewrite and r.get('subpart') == rewrite[0]:
            r['student_answer'] = rewrite[1]
        fake.rows[r['evaluation_id']] = grading_table.clean_none(json.loads(json.dumps(r, default=str)))
    return rows


def _payload_and_status(monkeypatch):
    """The exact "Start grading" payload and workflow's status rule. They come from the Python API's
    workflow module, which (like every API function) uses DynamoDB, so they're built with the default
    database; in the hybrid deployment that module's job is done by Node on Render."""
    monkeypatch.delenv('DATABASE_PROVIDER')
    workflow, payload = tg._workflow_payload(monkeypatch, tg.ASSIGNMENT, tg.SID)
    monkeypatch.setenv('DATABASE_PROVIDER', 'supabase')
    return payload, workflow._derived_status


def test_start_grading_through_sqs_regrades_on_supabase(supabase, grader, monkeypatch):
    """grading: sqs — "Start grading" arrives as an SQS message with the workflow payload as its body."""
    _seed_corrected(supabase, rewrite=('b', 'A completely different, much longer answer typed in by the student.'))
    llm.set_client(tg.FakeLLM(grade=lambda p: '{"marks_awarded": 10, "reasoning": "full"}'))
    payload, derived_status = _payload_and_status(monkeypatch)
    event = {'Records': [{'messageId': 'job-1', 'eventSource': 'aws:sqs', 'body': json.dumps(payload)}]}
    assert grader.lambda_handler(event, None) == {'batchItemFailures': []}

    details, summary = _details(supabase), _summary(supabase)
    assert {r['status'] for r in details} == {'graded'}
    assert {(str(r['question_number']), r.get('subpart')): r['marks'] for r in details} == {('1', None): 4, ('2', 'a'): 3, ('2', 'b'): 2.5}
    assert summary['total_marks'] == 9.5 and summary['persona'] == 'strict'
    flagged = [r for r in details if r.get('flag_reason')]
    assert [r.get('subpart') for r in flagged] == ['b'] and flagged[0]['confidence'] < 0.6
    assert derived_status({'details': details, 'summary': summary}) == 'queued_for_review'


def test_failed_regrade_job_returns_to_the_queue(supabase, grader, monkeypatch):
    _seed_corrected(supabase)
    llm.set_client(tg.FakeLLM(grade=lambda p: llm.LLMError('quota')))
    payload, _ = _payload_and_status(monkeypatch)
    event = {'Records': [{'messageId': 'job-2', 'body': json.dumps(payload)}]}
    assert grader.lambda_handler(event, None) == {'batchItemFailures': [{'itemIdentifier': 'job-2'}]}
    assert _summary(supabase) is None and {r['status'] for r in _details(supabase)} == {'corrected'}


def test_regrade_job_is_never_confused_with_an_s3_event():
    from gradeant_common.pipeline import sqs
    job = {'assignment_id': 'A', 'qp_id': 'Q', 'student_id': 'S'}
    assert sqs.regrade_job({'body': json.dumps(job)}) == job
    assert sqs.regrade_job({'body': json.dumps({'Records': [], **job})}) is None
    assert sqs.regrade_job({'body': json.dumps({'assignment_id': 'A'})}) is None
    assert sqs.regrade_job({'body': 'not json'}) is None


# ---- optional: against a real (staging) Supabase project -----------------------------

LIVE = os.environ.get('SUPABASE_TEST_URL') and os.environ.get('SUPABASE_TEST_KEY')


@pytest.mark.skipif(not LIVE, reason='set SUPABASE_TEST_URL and SUPABASE_TEST_KEY (service role) to run against a staging project')
def test_live_supabase_roundtrip(monkeypatch):
    monkeypatch.setenv('DATABASE_PROVIDER', 'supabase')
    monkeypatch.setenv('SUPABASE_URL', os.environ['SUPABASE_TEST_URL'])
    monkeypatch.setenv('SUPABASE_SERVICE_ROLE_KEY', os.environ['SUPABASE_TEST_KEY'])
    secrets.clear_cache()
    pg_rest.set_transport(None)
    sid = f'live-test-{uuid.uuid4().hex[:8]}'
    row = {'evaluation_id': f'{sid}-1', 'student_id': sid, 'assignment_id': 'LIVE', 'qp_id': 'LIVE', 'question_number': 1}
    try:
        grading_table.replace_rows([row])
        grading_table.patch_row(row['evaluation_id'], {'marks': 3}, ['nothing'])
        assert grading_table.student_rows(sid)[0]['marks'] == 3
        # The public anon key must not be able to read the table (RLS on, no grants).
        anon = os.environ.get('SUPABASE_TEST_ANON_KEY')
        if anon:
            r = httpx2.get(f"{os.environ['SUPABASE_TEST_URL']}/rest/v1/assignment_grading", params={'select': 'item'},
                           headers={'apikey': anon, 'Authorization': f'Bearer {anon}', 'Accept-Profile': 'gradeant'})
            assert r.status_code in (401, 403) or r.json() == []
    finally:
        grading_table.replace_rows([], [row['evaluation_id']])
        pg_rest.set_transport(None)
