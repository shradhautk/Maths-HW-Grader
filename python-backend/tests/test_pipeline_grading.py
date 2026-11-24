"""Grading pipeline, Group B: the grader worker (functions/grader.py + gradeant_common.pipeline.grading).

Prompts and QA pairs are compared with the ORIGINAL lambda_python_files/grader_lambda.py
(loaded with its SDKs stubbed). The model is always a FakeLLM; S3 and DynamoDB are moto.
"""
import importlib
import importlib.util
import json
import os
import sys
import threading
import types
from decimal import Decimal

import boto3
import pytest
from moto import mock_aws

ROOT = os.path.join(os.path.dirname(__file__), '..')
REPO = os.path.join(ROOT, '..')
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT]

from gradeant_common import dynamo, secrets  # noqa: E402
from gradeant_common.pipeline import PipelineInputError, grading, llm, paths, s3io  # noqa: E402

BUCKET = 'gradeant-pipeline-test'
AID, QP, SID = 'A1', 'QP1', 'stu-7'
LAYOUT = paths.Layout('Data_warehouse', AID, QP)

QP_JSON = {'questions': [
    {'question_number': 1, 'description': 'Define binding energy.', 'max_marks': 5},
    {'question_number': 2, 'subparts': [
        {'subpart': 'a', 'description': 'State the SEMF.', 'max_marks': 3},
        {'subpart': 'b', 'description': 'Compute B for Fe-56.', 'max_marks': 2.5}]},
]}
ANSWERS = {'questions': [
    {'question_number': 1, 'reply': 'Energy needed to split a nucleus.'},
    {'question_number': 2, 'subparts': [{'reply': 'B = aV A - ...'}, {'reply': '492 MeV'}]},
]}
RUBRIC = {'questions': [
    {'rubric': 'Mentions mass defect.', 'standard_answer': 'E = Δm c²'},
    {'subparts': [{'rubric': 'Five terms.', 'standard_answer': 'SEMF'}, {'rubric': '≈ 492 MeV', 'standard_answer': '492.3 MeV'}]},
]}


class FakeLLM:
    """Grade prompts get grade(prompt); feedback prompts (they start with '## Task') get `feedback`."""

    def __init__(self, grade=None, feedback='Well done.', barrier=None):
        self.grade = grade or (lambda prompt: '{"marks_awarded": 2, "reasoning": "partly right"}')
        self.feedback = feedback
        self.barrier = barrier
        self.prompts = []
        self._lock = threading.Lock()

    def generate(self, prompt, *, temperature, max_output_tokens=None):
        with self._lock:
            self.prompts.append(prompt)
        if prompt.startswith('## Task'):
            if isinstance(self.feedback, Exception):
                raise self.feedback
            return self.feedback
        if self.barrier:
            self.barrier.wait(timeout=5)
        out = self.grade(prompt)
        if isinstance(out, Exception):
            raise out
        return out

    def grade_prompts(self):
        return [p for p in self.prompts if not p.startswith('## Task')]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {'AWS_DEFAULT_REGION': 'eu-north-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x',
                 'PIPELINE_BUCKET': BUCKET, 'JWT_SECRET': 'test-secret-that-is-at-least-32-bytes-long'}.items():
        monkeypatch.setenv(k, v)
    for k in ('GEMINI_API_KEY', 'GEMINI_API_KEY_ARN', 'PIPELINE_ROOT', 'PIPELINE_BUCKET_REGION', 'AWS_DATA_REGION',
              'GRADE_ON_UPLOAD', 'GRADER_MAX_WORKERS'):
        monkeypatch.delenv(k, raising=False)
    secrets.clear_cache()
    llm.set_client(None)
    yield
    llm.set_client(None)


@pytest.fixture()
def table():
    with mock_aws():
        boto3.client('s3', region_name='eu-north-1').create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})
        boto3.client('dynamodb', region_name='eu-north-1').create_table(
            TableName='assignment_grading', BillingMode='PAY_PER_REQUEST',
            KeySchema=[{'AttributeName': 'evaluation_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': n, 'AttributeType': 'S'} for n in ('evaluation_id', 'student_id', 'type')],
            GlobalSecondaryIndexes=[
                {'IndexName': i, 'KeySchema': [{'AttributeName': k, 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}}
                for i, k in (('student_id-index', 'student_id'), ('type-index', 'type'))])
        dynamo.reset()
        s3io.reset()
        yield boto3.resource('dynamodb', region_name='eu-north-1').Table('assignment_grading')
        dynamo.reset()
        s3io.reset()


@pytest.fixture()
def grader():
    return importlib.reload(importlib.import_module('grader'))


def _seed(with_solution=False, qp=True):
    if qp:
        s3io.write_json(BUCKET, LAYOUT.qp_schema_json(with_solution), QP_JSON)
        s3io.write_json(BUCKET, LAYOUT.rubric_json(with_solution), RUBRIC)
    if with_solution:
        s3io.write_text(BUCKET, LAYOUT.solution_pdf, '%PDF')
    key = LAYOUT.parsed_json(SID)
    s3io.write_json(BUCKET, key, ANSWERS)
    return key


def sqs_event(key, message_id='m1', bucket=BUCKET):
    body = {'Records': [{'eventSource': 'aws:s3', 's3': {'bucket': {'name': bucket}, 'object': {'key': key.replace(' ', '+')}}}]}
    return {'Records': [{'messageId': message_id, 'eventSource': 'aws:sqs', 'body': json.dumps(body)}]}


def _rows(tbl):
    return sorted(tbl.scan()['Items'], key=lambda r: (r.get('type') == 'summary', str(r.get('question_number')), str(r.get('subpart'))))


def _load_original_grader(monkeypatch):
    gg = types.ModuleType('google.generativeai')
    gg.GenerativeModel, gg.configure = object, (lambda **_: None)
    google = types.ModuleType('google')
    google.generativeai = gg
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.generativeai', gg)
    monkeypatch.setitem(sys.modules, 'dotenv', dotenv)
    spec = importlib.util.spec_from_file_location('original_grader', os.path.join(REPO, 'lambda_python_files', 'grader_lambda.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gp = object.__new__(mod.GradingPipeline)
    gp.personas = {'default': gp._default_persona()}
    gp.instructions = {'default': gp._default_instructions()}
    gp._feedback_prompt = gp._build_feedback_prompt()
    gp._feedback_prompt_std = gp._build_feedback_prompt_with_std_solution()
    return gp


# --- B1: parity with the original -------------------------------------------

def test_prompt_matches_original(monkeypatch):
    original = _load_original_grader(monkeypatch)
    plain = {'question': 'Q?', 'student_answer': 'A.', 'rubric': 'R', 'max_marks': 5, 'question_number': 1, 'subpart': None}
    for pair in (plain, dict(plain, standard_answer='S'), dict(plain, rubric=None)):
        assert grading.build_eval_prompt(pair) == original._build_evaluation_prompt(pair)
        assert grading.build_feedback_prompt(pair, 3, 'why') == original._build_feedback_call(pair, 3, 'why')
    strict = grading.build_eval_prompt(plain, 'strict', partial_credit=False)
    assert 'No partial credit' in strict and 'Strictness' in strict


def test_extract_qa_pairs_matches_original(monkeypatch):
    original = _load_original_grader(monkeypatch)
    for include in (False, True):
        ours = grading.extract_qa_pairs(QP_JSON, ANSWERS, RUBRIC, include_std_answer=include)
        assert ours == original.extract_qa_pairs(QP_JSON, ANSWERS, RUBRIC, include_std_answer=include)
        assert len(ours) == 3
    with pytest.raises(PipelineInputError):
        grading.extract_qa_pairs({'nope': []}, ANSWERS)


def test_parse_marks():
    assert grading.parse_marks('```json\n{"marks_awarded": 4, "reasoning": "ok"}\n```', 5) == (4, 'ok')
    assert grading.parse_marks('{{"marks_awarded": "2.5", "reasoning": "x"}}', 5) == (2.5, 'x')
    assert grading.parse_marks('{"marks_awarded": 99, "reasoning": "x"}', Decimal('5')) == (5, 'x')  # clamped
    assert grading.parse_marks('{"marks_awarded": -1, "reasoning": "x"}', 5) == (0, 'x')
    for bad in ('ERROR: Gemini failed', '{"reasoning": "no marks"}', '{"marks_awarded": "lots"}', '{"marks_awarded": true}'):
        with pytest.raises(ValueError):
            grading.parse_marks(bad, 5)


def test_row_ids_are_stable_across_number_types():
    assert grading.row_id('A', 'Q', 'S', 1, None) == grading.row_id('A', 'Q', 'S', '1', '') == grading.row_id('A', 'Q', 'S', Decimal('1'), None)
    assert grading.row_id('A', 'Q', 'S', 2, 'a') != grading.row_id('A', 'Q', 'S', 2, 'b')


def test_questions_graded_concurrently(monkeypatch):
    monkeypatch.setenv('GRADER_MAX_WORKERS', '4')
    fake = FakeLLM(barrier=threading.Barrier(2))  # breaks (→ error) unless 2 grade calls are in flight together
    pairs = grading.extract_qa_pairs(QP_JSON, ANSWERS)[:2]
    results = grading.grade_pairs(pairs, fake)
    assert [r['question'] for r in results] == [p['question'] for p in pairs]


# --- B2: SQS path -------------------------------------------------------------

def test_sqs_grades_and_writes_rows(table, grader):
    key = _seed()
    fake = FakeLLM()
    llm.set_client(fake)
    assert grader.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    rows = _rows(table)
    details, summary = rows[:-1], rows[-1]
    assert len(details) == 3 and summary['type'] == 'summary'
    original_fields = {'evaluation_id', 'assignment_id', 'qp_id', 'student_id', 'question', 'student_answer', 'rubric',
                       'max_marks', 'question_number', 'answer_with', 'marks', 'reasoning', 'feedback', 'timestamp'}
    for r in details:
        assert original_fields <= set(r), original_fields - set(r)
        assert r['student_id'] == SID and r['marks'] == 2 and r['feedback'] == 'Well done.' and 'graded_at' in r
        assert 'standard_answer' not in r  # no solution PDF → rubric-only mode
    assert {r.get('subpart') for r in details} == {None, 'a', 'b'}
    assert summary['total_marks'] == 6 and summary['answer_with'] == 'without_standard_answer'
    assert {r['evaluation_id'] for r in details} == {grading.row_id(AID, QP, SID, q, s) for q, s in ((1, None), (2, 'a'), (2, 'b'))}
    assert summary['evaluation_id'] == grading.summary_id(AID, QP, SID)
    assert len(fake.prompts) == 6  # grade + feedback per question


def test_standard_answer_mode(table, grader):
    key = _seed(with_solution=True)
    fake = FakeLLM()
    llm.set_client(fake)
    assert grader.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    rows = _rows(table)
    assert rows[-1]['answer_with'] == 'standard_answer' and rows[0]['standard_answer'] == 'E = Δm c²'
    assert any('#Standard answer:\nE = Δm c²' in p for p in fake.grade_prompts())


def test_sqs_rerun_is_idempotent(table, grader):
    key = _seed()
    table.put_item(Item={'evaluation_id': 'legacy-uuid4', 'assignment_id': AID, 'qp_id': QP, 'student_id': SID, 'marks': 1})
    table.put_item(Item={'evaluation_id': 'other-student', 'assignment_id': AID, 'qp_id': QP, 'student_id': 'stu-8', 'marks': 1})
    llm.set_client(FakeLLM())
    grader.lambda_handler(sqs_event(key), None)
    first = {r['evaluation_id'] for r in table.scan()['Items']}
    grader.lambda_handler(sqs_event(key, 'm2'), None)
    second = {r['evaluation_id'] for r in table.scan()['Items']}
    assert first == second and len(second) == 5  # 3 details + summary + the other student's row
    assert 'legacy-uuid4' not in second and 'other-student' in second


def test_llm_failure_reports_batch_item_failure_and_writes_nothing(table, grader):
    key = _seed()
    llm.set_client(FakeLLM(grade=lambda p: llm.LLMError('quota')))
    assert grader.lambda_handler(sqs_event(key), None) == {'batchItemFailures': [{'itemIdentifier': 'm1'}]}
    assert table.scan()['Items'] == []


def test_unparseable_marks_retry_then_fail(table, grader):
    key = _seed()
    fake = FakeLLM(grade=lambda p: 'I think about 3?')
    llm.set_client(fake)
    assert grader.lambda_handler(sqs_event(key), None)['batchItemFailures'] == [{'itemIdentifier': 'm1'}]
    assert table.scan()['Items'] == []
    assert len(fake.grade_prompts()) >= 3


def test_feedback_failure_is_not_fatal(table, grader):
    key = _seed()
    llm.set_client(FakeLLM(feedback=llm.LLMError('down')))
    assert grader.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    assert {r.get('feedback') for r in _rows(table)[:-1]} == {'Feedback generation encountered an error.'}


def test_missing_qp_fails_message(table, grader):
    key = _seed(qp=False)
    llm.set_client(FakeLLM())
    assert grader.lambda_handler(sqs_event(key), None)['batchItemFailures'] == [{'itemIdentifier': 'm1'}]


def test_other_bucket_rejected_and_skips_acked(table, grader):
    key = _seed()
    llm.set_client(FakeLLM())
    assert grader.lambda_handler(sqs_event(key, bucket='evil'), None)['batchItemFailures'] == [{'itemIdentifier': 'm1'}]
    for skip in (LAYOUT.parsed_text(SID), LAYOUT.qp_pdf, LAYOUT.original_pdf(SID)):
        assert grader.lambda_handler(sqs_event(skip), None) == {'batchItemFailures': []}
    assert table.scan()['Items'] == []


def test_bad_student_name_in_json_folder_fails_loudly(table, grader):
    llm.set_client(FakeLLM())
    key = LAYOUT.parsed_json_prefix + 'Riya sheet (1).json'
    assert grader.lambda_handler(sqs_event(key), None)['batchItemFailures'] == [{'itemIdentifier': 'm1'}]


def test_s3_test_event_and_malformed_body(table, grader):
    event = {'Records': [{'messageId': 't', 'body': json.dumps({'Service': 'Amazon S3', 'Event': 's3:TestEvent'})},
                         {'messageId': 'bad', 'body': 'not json'}]}
    assert grader.lambda_handler(event, None) == {'batchItemFailures': [{'itemIdentifier': 'bad'}]}
    assert grader.lambda_handler({'Records': []}, None) == {'batchItemFailures': []}


def test_grade_on_upload_false_writes_ungraded_rows(table, grader, monkeypatch):
    monkeypatch.setenv('GRADE_ON_UPLOAD', 'false')
    key = _seed()
    fake = FakeLLM()
    llm.set_client(fake)
    assert grader.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    rows = table.scan()['Items']
    assert len(rows) == 3 and not any(r.get('type') == 'summary' for r in rows)
    assert not any('marks' in r or 'feedback' in r for r in rows) and fake.prompts == []
    workflow = importlib.import_module('workflow')
    assert workflow._derived_status({'details': rows, 'summary': None}) == 'ocr_done'


# --- B2: workflow "Start grading" invoke --------------------------------------

ASSIGNMENT = {
    'assignment_id': AID, 'qp_id': QP, 'method': 'standard', 'persona': 'strict', 'partial_credit': False,
    'rubric': [
        {'question_number': Decimal('1'), 'subpart': None, 'title': 'BE', 'max_marks': Decimal('4'), 'criteria': 'CRIT-ONE'},
        {'question_number': '2', 'subpart': 'a', 'title': 'SEMF', 'max_marks': Decimal('3'), 'criteria': 'CRIT-2A'},
    ],
}


def _workflow_payload(monkeypatch, assignment, student_id):
    """The exact payload gradeant-workflow sends (its _trigger_grading, with the Lambda client faked)."""
    monkeypatch.setenv('GRADING_FUNCTION_NAME', 'gradeant-grader')
    workflow = importlib.reload(importlib.import_module('workflow'))
    calls = []

    class FakeLambda:
        def invoke(self, **kw):
            calls.append(kw)
            return {'StatusCode': 202}
    real = workflow.boto3.client
    monkeypatch.setattr(workflow.boto3, 'client', lambda service, **kw: FakeLambda() if service == 'lambda' else real(service, **kw))
    workflow._trigger_grading(assignment, student_id)
    assert calls[0]['FunctionName'] == 'gradeant-grader' and calls[0]['InvocationType'] == 'Event'
    return workflow, json.loads(calls[0]['Payload'])


def _seed_corrected_rows(tbl):
    rows = grading.detail_rows(grading.extract_qa_pairs(QP_JSON, ANSWERS, RUBRIC), AID, QP, SID, graded=False)
    for r in rows:
        r['status'] = 'corrected'
        tbl.put_item(Item=dynamo.to_dynamo(r))
    return rows


def test_workflow_payload_is_accepted(table, grader, monkeypatch):
    rows = _seed_corrected_rows(table)
    fake = FakeLLM(grade=lambda p: '{"marks_awarded": 10, "reasoning": "full"}')
    llm.set_client(fake)
    workflow, payload = _workflow_payload(monkeypatch, ASSIGNMENT, SID)
    out = grader.lambda_handler(payload, None)

    stored = _rows(table)
    details, summary = stored[:-1], stored[-1]
    assert {r['evaluation_id'] for r in details} == {r['evaluation_id'] for r in rows}  # updated in place
    assert all(r['status'] == 'graded' for r in details)
    marks = {(str(r['question_number']), r.get('subpart')): r['marks'] for r in details}
    assert marks == {('1', None): 4, ('2', 'a'): 3, ('2', 'b'): Decimal('2.5')}  # clamped to payload / row max
    assert out == {'ok': True, 'student_id': SID, 'total_marks': 9.5}
    assert summary['method'] == 'standard' and summary['persona'] == 'strict' and summary['total_marks'] == Decimal('9.5')
    prompts_ = fake.grade_prompts()
    assert any('CRIT-ONE' in p for p in prompts_) and any('CRIT-2A' in p for p in prompts_)
    assert all('No partial credit' in p and 'Strictness' in p for p in prompts_)
    assert workflow._derived_status({'details': details, 'summary': summary}) == 'ai_graded'


def test_regrade_failure_leaves_rows_corrected(table, grader, monkeypatch):
    _seed_corrected_rows(table)
    llm.set_client(FakeLLM(grade=lambda p: llm.LLMError('down')))
    _, payload = _workflow_payload(monkeypatch, ASSIGNMENT, SID)
    with pytest.raises(grading.GradingError):
        grader.lambda_handler(payload, None)
    rows = table.scan()['Items']
    assert len(rows) == 3 and all(r['status'] == 'corrected' and 'marks' not in r for r in rows)


def test_regrade_validates_ids_and_needs_rows(table, grader):
    llm.set_client(FakeLLM())
    with pytest.raises(PipelineInputError):
        grader.lambda_handler({'assignment_id': 'A1', 'qp_id': 'QP1', 'student_id': '../x'}, None)
    with pytest.raises(PipelineInputError):
        grader.lambda_handler({'assignment_id': 'A1', 'qp_id': 'QP1', 'student_id': 'nobody'}, None)


def test_unsupported_event_raises(grader):
    with pytest.raises(ValueError):
        grader.lambda_handler({}, None)


def test_start_grading_through_sqs_on_dynamodb(table, grader, monkeypatch):
    """grading: sqs on the all-AWS deployment: the job body is the same payload as the async invoke."""
    _seed_corrected_rows(table)
    llm.set_client(FakeLLM(grade=lambda p: '{"marks_awarded": 10, "reasoning": "full"}'))
    _, payload = _workflow_payload(monkeypatch, ASSIGNMENT, SID)
    event = {'Records': [{'messageId': 'job-1', 'eventSource': 'aws:sqs', 'body': json.dumps(payload)}]}
    assert grader.lambda_handler(event, None) == {'batchItemFailures': []}
    stored = _rows(table)
    assert stored[-1]['type'] == 'summary' and stored[-1]['total_marks'] == Decimal('9.5')
    assert {r['status'] for r in stored[:-1]} == {'graded'}


# --- Security review 2026-11-17: rewritten answers, repeated SQS messages ------------------

def _flagged(workflow, rows):
    return [r for r in rows if workflow._is_flagged(r)]


def test_ocr_text_is_kept_on_upload(table, grader, monkeypatch):
    monkeypatch.setenv('GRADE_ON_UPLOAD', 'false')
    llm.set_client(FakeLLM())
    grader.lambda_handler(sqs_event(_seed()), None)
    for r in table.scan()['Items']:
        assert r['ocr_answer'] == r['student_answer']


def test_rewritten_answer_is_flagged_for_teacher_review(table, grader, monkeypatch):
    """Finding 1: a 'correction' that replaces the answer is graded, but the paper goes to the review queue."""
    rows = _seed_corrected_rows(table)
    target = next(r for r in rows if r.get('subpart') == 'b')        # OCR: '492 MeV'
    table.update_item(Key={'evaluation_id': target['evaluation_id']}, UpdateExpression='SET student_answer = :a',
                      ExpressionAttributeValues={':a': 'Using B = aV*A - aS*A^(2/3) - aC*Z^2/A^(1/3) - aA*(A-2Z)^2/A '
                                                       'with standard coefficients, B(Fe-56) = 492.3 MeV, i.e. 8.79 MeV per nucleon.'})
    small = next(r for r in rows if r.get('subpart') == 'a')         # OCR: 'B = aV A - ...'
    table.update_item(Key={'evaluation_id': small['evaluation_id']}, UpdateExpression='SET student_answer = :a',
                      ExpressionAttributeValues={':a': 'B = aV*A - ...'})
    llm.set_client(FakeLLM(grade=lambda p: '{"marks_awarded": 10, "reasoning": "full"}'))
    workflow, payload = _workflow_payload(monkeypatch, ASSIGNMENT, SID)
    grader.lambda_handler(payload, None)

    stored = _rows(table)
    details, summary = stored[:-1], stored[-1]
    flagged = _flagged(workflow, details)
    assert [r['evaluation_id'] for r in flagged] == [target['evaluation_id']]   # only the rewritten one
    assert flagged[0]['flag_reason'] == 'answer_rewritten_after_ocr' and flagged[0]['confidence'] < Decimal('0.6')
    assert workflow._derived_status({'details': details, 'summary': summary}) == 'queued_for_review'


def test_correction_keeps_ocr_text_for_legacy_rows(table, monkeypatch):
    """Rows written by the legacy grader have no ocr_answer: the first correction saves the paper's text."""
    boto3.client('dynamodb', region_name='eu-north-1').create_table(
        TableName='GradeAnt_Submissions', BillingMode='PAY_PER_REQUEST',
        KeySchema=[{'AttributeName': 'assignment_id', 'KeyType': 'HASH'}, {'AttributeName': 'student_id', 'KeyType': 'RANGE'}],
        AttributeDefinitions=[{'AttributeName': n, 'AttributeType': 'S'} for n in ('assignment_id', 'student_id')])
    table.put_item(Item={'evaluation_id': 'legacy-1', 'assignment_id': AID, 'qp_id': QP, 'student_id': SID,
                         'question_number': 1, 'student_answer': 'paper text'})
    corrections = importlib.reload(importlib.import_module('corrections'))
    caller = {'role': 'student', 'user_id': SID}
    for text in ('first fix', 'second fix'):
        corrections.submit_answer_corrections(
            {'corrections': [{'evaluation_id': 'legacy-1', 'question_number': 1, 'corrected_answer': text}]}, caller)
    row = table.get_item(Key={'evaluation_id': 'legacy-1'})['Item']
    assert row['student_answer'] == 'second fix' and row['ocr_answer'] == 'paper text'  # set once, never overwritten


def test_repeated_message_after_correction_changes_nothing(table, grader, monkeypatch):
    """Bug: a repeated SQS message used to reset corrected rows to the OCR text."""
    monkeypatch.setenv('GRADE_ON_UPLOAD', 'false')
    key = _seed()
    fake = FakeLLM()
    llm.set_client(fake)
    grader.lambda_handler(sqs_event(key), None)
    row = table.scan()['Items'][0]
    table.update_item(Key={'evaluation_id': row['evaluation_id']},
                      UpdateExpression='SET student_answer = :a, #s = :s, corrected_at = :t',
                      ExpressionAttributeNames={'#s': 'status'},
                      ExpressionAttributeValues={':a': 'my corrected text', ':s': 'corrected', ':t': '2026-11-17T10:00:00'})
    before = sorted(table.scan()['Items'], key=lambda r: r['evaluation_id'])
    assert grader.lambda_handler(sqs_event(key, 'm2'), None) == {'batchItemFailures': []}
    assert sorted(table.scan()['Items'], key=lambda r: r['evaluation_id']) == before


def test_repeated_message_after_grading_keeps_grade_and_teacher_edits(table, grader):
    """Bug: a repeated SQS message used to delete the summary (ungraded mode) or overwrite teacher mark edits."""
    key = _seed()
    llm.set_client(FakeLLM())
    grader.lambda_handler(sqs_event(key), None)
    row = next(r for r in table.scan()['Items'] if r.get('type') != 'summary')
    table.update_item(Key={'evaluation_id': row['evaluation_id']}, UpdateExpression='SET marks = :m, updated_at = :u',
                      ExpressionAttributeValues={':m': 5, ':u': '2026-11-17T11:00:00'})
    before = sorted(table.scan()['Items'], key=lambda r: r['evaluation_id'])
    fake = FakeLLM(grade=lambda p: '{"marks_awarded": 0, "reasoning": "regraded"}')
    llm.set_client(fake)
    for mode in ('true', 'false'):
        os.environ['GRADE_ON_UPLOAD'] = mode
        try:
            assert grader.lambda_handler(sqs_event(key, 'm-' + mode), None) == {'batchItemFailures': []}
        finally:
            os.environ.pop('GRADE_ON_UPLOAD', None)
    assert sorted(table.scan()['Items'], key=lambda r: r['evaluation_id']) == before and fake.prompts == []


def test_correction_similarity():
    assert grading.correction_similarity('492 MeV', '492  MeV') == 1.0           # whitespace only
    assert grading.correction_similarity(None, 'x') is None                      # nothing to compare
    assert grading.needs_review('B = aV A - ...', 'B = aV*A - ...')[0] is False  # OCR fix
    assert grading.needs_review('492 MeV', 'A completely different and much longer answer text')[0] is True
