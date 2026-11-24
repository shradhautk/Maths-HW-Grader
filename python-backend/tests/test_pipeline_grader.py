"""Grading pipeline: foundations (Group A) — secrets, S3 layout + I/O, LLM client.

The layout test compares against string literals copied from the ORIGINAL code
(lambda_python_files/grader_lambda.py L1870-1886, jsonifier.py L92-102), so a drift from
the keys the deployed Lambdas use fails here.
"""
import os
import sys
import types

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT]

from gradeant_common import secrets  # noqa: E402
from gradeant_common.pipeline import PipelineInputError, llm, paths, s3io  # noqa: E402

BUCKET = 'gradeant-pipeline-test'


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    for k, v in {'AWS_DEFAULT_REGION': 'eu-north-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x',
                 'PIPELINE_BUCKET': BUCKET}.items():
        monkeypatch.setenv(k, v)
    for k in ('GEMINI_API_KEY', 'GEMINI_API_KEY_ARN', 'PIPELINE_ROOT', 'PIPELINE_BUCKET_REGION', 'AWS_DATA_REGION'):
        monkeypatch.delenv(k, raising=False)
    secrets.clear_cache()
    s3io.reset()
    llm.set_client(None)
    yield
    secrets.clear_cache()
    s3io.reset()
    llm.set_client(None)


# --- A1 ----------------------------------------------------------------------

def test_gemini_key_from_secrets_manager(monkeypatch):
    assert secrets.gemini_api_key() == ''
    with mock_aws():
        sm = boto3.client('secretsmanager', region_name='eu-north-1')
        arn = sm.create_secret(Name='gemini', SecretString='{"GEMINI_API_KEY": "g-key-123"}')['ARN']
        monkeypatch.setenv('GEMINI_API_KEY_ARN', arn)
        assert secrets.gemini_api_key() == 'g-key-123'
    secrets.clear_cache()
    monkeypatch.setenv('GEMINI_API_KEY', 'plain-env')
    assert secrets.gemini_api_key() == 'plain-env'


# --- A2 ----------------------------------------------------------------------

def test_paths_match_original_layout():
    main_path, assignment_id, qp_id = 'Data_warehouse', 'A1', 'QP1'
    folder = f'{assignment_id}__{qp_id}'
    lay = paths.Layout(main_path, assignment_id, qp_id)

    # grader_lambda.py _validate_paths
    assert lay.student_base == f'{main_path}/students/assignments/{folder}'
    assert lay.solution_pdf == (f'{main_path}/teachers/QPRubricWithAnswers/{folder}'
                                f'/solution_pdfs/{assignment_id}_{qp_id}_answer.pdf')
    for has_solution, base in ((True, f'{main_path}/teachers/QPRubricWithAnswers/{folder}'),
                               (False, f'{main_path}/teachers/QP_Rubric/{folder}')):
        assert lay.qp_schema_json(has_solution) == f'{base}/QP_schemas/QP_schemas_json/{qp_id}_schema.json'
        assert lay.rubric_json(has_solution) == f'{base}/rubric_jsons/{qp_id}_rubric.json'
    assert lay.parsed_json_prefix == f'{lay.student_base}/Parsed_pdfs/JSON/'

    # jsonifier.py setup_s3_paths + excute_jsonifier.py output keys
    assert lay.parsed_text('s1') == f'{main_path}/students/assignments/{folder}/Parsed_pdfs/text/s1.txt'
    assert lay.parsed_seg('s1') == f'{main_path}/students/assignments/{folder}/Parsed_pdfs/JSON_Seg/s1.txt'
    assert lay.parsed_json_text('s1') == f'{main_path}/students/assignments/{folder}/Parsed_pdfs/JSON_Text/s1.txt'
    assert lay.parsed_json('s1') == f'{main_path}/students/assignments/{folder}/Parsed_pdfs/JSON/s1.json'
    assert lay.qp_pdf == f'{main_path}/teachers/QP_Rubric/{assignment_id}__{qp_id}/QP_pdfs/{qp_id}.pdf'
    assert lay.qp_schema_answer_txt == (f'{main_path}/teachers/QP_Rubric/{assignment_id}__{qp_id}'
                                        f'/QP_schemas/QP_schemas_answers/{qp_id}_schema_answer.txt')


def test_parse_student_key():
    lay, kind, stem = paths.parse_student_key('Data_warehouse/students/assignments/A1__QP1/original_pdf/stu-9.pdf')
    assert (lay, kind, stem) == (paths.Layout('Data_warehouse', 'A1', 'QP1'), 'original_pdf', 'stu-9')
    lay, kind, stem = paths.parse_student_key('Data_warehouse/students/assignments/A1__QP__2/Parsed_pdfs/JSON/s.json')
    assert (lay.assignment_id, lay.qp_id, kind, stem) == ('A1', 'QP__2', 'parsed_json', 's')
    for bad in ('Data_warehouse/teachers/QP_Rubric/A1__QP1/QP_pdfs/QP1.pdf',
                'Data_warehouse/students/assignments/A1QP1/original_pdf/s.pdf',
                'Data_warehouse/students/assignments/A1__QP1/Parsed_pdfs/text/s.txt',
                'Data_warehouse/students/assignments/A1__QP1/original_pdf/s.json',
                'Data_warehouse/students/assignments/A1__QP1/Parsed_pdfs/JSON/bad name;rm.json'):
        with pytest.raises(PipelineInputError):
            paths.parse_student_key(bad)
    # Uploaded PDFs keep the original's safe_file_name rule (+/space → _), other characters → _ too.
    _, _, stem = paths.parse_student_key('Data_warehouse/students/assignments/A1__QP1/original_pdf/John Doe+HW;1.pdf')
    assert stem == 'John_Doe_HW_1'


def test_bucket_is_required_and_checked(monkeypatch):
    s3io.check_bucket(BUCKET)
    with pytest.raises(PipelineInputError):
        s3io.check_bucket('someone-elses-bucket')
    monkeypatch.delenv('PIPELINE_BUCKET')
    with pytest.raises(PipelineInputError):
        paths.pipeline_bucket()


def test_exists_handles_string_codes(monkeypatch):
    with mock_aws():
        boto3.client('s3', region_name='eu-north-1').create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})
        assert s3io.exists(BUCKET, 'nope.json') is False
        s3io.write_json(BUCKET, 'a/b.json', {'x': 1})
        assert s3io.exists(BUCKET, 'a/b.json') is True
        assert s3io.read_json(BUCKET, 'a/b.json') == {'x': 1}
        s3io.write_text(BUCKET, 'a/folder/', '')
        assert s3io.list_keys(BUCKET, 'a/') == ['a/b.json']

    class Slow:
        def head_object(self, **_):
            raise ClientError({'Error': {'Code': 'SlowDown', 'Message': 'slow'}}, 'HeadObject')
    monkeypatch.setattr(s3io, 's3', lambda: Slow())
    with pytest.raises(ClientError):
        s3io.exists(BUCKET, 'k')


# --- A3 ----------------------------------------------------------------------

class _ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _fake_genai(monkeypatch, outcomes):
    """Insert a fake google.genai whose generate_content plays `outcomes` (exceptions or text) in order."""
    calls = []

    class Models:
        def generate_content(self, *, model, contents, config):
            calls.append({'model': model, 'contents': contents, 'config': config})
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return types.SimpleNamespace(text=outcome)

    class Client:
        def __init__(self, api_key):
            assert api_key == 'k'
            self.models = Models()

    genai_types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kw: kw,
        Part=types.SimpleNamespace(from_bytes=lambda data, mime_type: ('part', mime_type, len(data))))
    genai = types.ModuleType('google.genai')
    genai.Client = Client
    genai.types = genai_types
    google = types.ModuleType('google')
    google.genai = genai
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.genai', genai)
    monkeypatch.setitem(sys.modules, 'google.genai.types', genai_types)
    monkeypatch.setattr(llm.time, 'sleep', lambda s: None)
    return calls


def test_llm_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    calls = _fake_genai(monkeypatch, [_ApiError(429, 'RESOURCE_EXHAUSTED'), _ApiError(503, 'UNAVAILABLE'), '{"ok": 1}'])
    client = llm.get_client('grade')
    assert client.model == llm.DEFAULT_MODEL == 'gemini-3.5-flash-preview'
    assert client.generate('hi', temperature=0.6) == '{"ok": 1}'
    assert len(calls) == 3 and calls[0]['config'] == {'temperature': 0.6, 'max_output_tokens': None}


def test_llm_retries_then_raises(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    _fake_genai(monkeypatch, [_ApiError(429, 'quota')] * 3)
    with pytest.raises(llm.LLMError):
        llm.GeminiClient('m').generate('hi', temperature=0.1)


def test_llm_non_retryable_and_empty(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    calls = _fake_genai(monkeypatch, [_ApiError(400, 'INVALID_ARGUMENT')])
    with pytest.raises(llm.LLMError):
        llm.GeminiClient('m').generate('hi', temperature=0.1)
    assert len(calls) == 1
    _fake_genai(monkeypatch, ['  ', '', ' '])
    with pytest.raises(llm.LLMError, match='Empty'):
        llm.GeminiClient('m').generate('hi', temperature=0.1)


def test_llm_needs_key_and_transcribes_inline(monkeypatch):
    _fake_genai(monkeypatch, ['text'])
    with pytest.raises(llm.LLMError, match='not configured'):
        llm.GeminiClient('m').generate('hi', temperature=0.1)
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    calls = _fake_genai(monkeypatch, ['page text'])
    out = llm.get_client('parse').transcribe_pdf(b'%PDF-1', 'read it', temperature=0.2, max_output_tokens=100)
    assert out == 'page text'
    assert calls[0]['model'] == 'gemini-3.5-flash-preview'  # one model for every purpose
    assert calls[0]['contents'] == [('part', 'application/pdf', 6), 'read it']


def test_retry_delay_parsing():
    assert llm.retry_delay_from('retry_delay { seconds: 17 }') == 17
    assert llm.retry_delay_from("'retryDelay': '23s'") == 23
    assert llm.retry_delay_from('nothing here') is None
