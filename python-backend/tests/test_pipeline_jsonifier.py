"""Grading pipeline, Group C: the jsonifier worker (functions/jsonifier.py + pipeline.jsonify/pdf).

Prompts are compared with the ORIGINAL lambda_python_files/{jsonifier,excute_jsonifier}.py.
PDFs are real (made with PyMuPDF); the model is a fake; S3 is moto.
"""
import ast
import importlib
import importlib.util
import json
import os
import sys
import types

import boto3
import pytest
from moto import mock_aws

fitz = pytest.importorskip('fitz')

ROOT = os.path.join(os.path.dirname(__file__), '..')
REPO = os.path.join(ROOT, '..')
ORIG = os.path.join(REPO, 'lambda_python_files')
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT]

from gradeant_common import secrets  # noqa: E402
from gradeant_common.pipeline import PipelineInputError, jsonify, llm, paths, pdf, s3io  # noqa: E402

BUCKET = 'gradeant-pipeline-test'
AID, QP, SID = 'A1', 'QP1', 'stu-7'
LAYOUT = paths.Layout('Data_warehouse', AID, QP)
SCHEMA = '{"questions": [{"question_number": 1, "reply": ""}]}'
TRANSCRIPT = 'Answer 1: Binding energy is the energy needed to split a nucleus into its nucleons.'
FILLED = {'questions': [{'question_number': 1, 'reply': 'Binding energy is the energy needed to split a nucleus.'}]}


def make_pdf(text):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class FakeParseLLM:
    def __init__(self, transcript=TRANSCRIPT, json_reply=None, improve=None):
        self.transcript = transcript
        self.json_reply = json_reply if json_reply is not None else '```json\n' + json.dumps(FILLED) + '\n```'
        self.improve = improve
        self.calls = []

    def transcribe_pdf(self, pdf_bytes, prompt, *, temperature, max_output_tokens):
        self.calls.append(('transcribe', prompt, max_output_tokens))
        if isinstance(self.transcript, Exception):
            raise self.transcript
        return self.transcript

    def generate(self, prompt, *, temperature, max_output_tokens=None):
        self.calls.append(('generate', prompt, max_output_tokens))
        if prompt.startswith('You are a physics expert'):
            return '{"1": "Binding energy is the energy needed to split a nucleus."}'
        if prompt.startswith('Please fill the given Text'):
            return self.json_reply
        if isinstance(self.improve, Exception):
            raise self.improve
        return self.improve or 'cleaned: ' + prompt[-80:]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {'AWS_DEFAULT_REGION': 'eu-north-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x',
                 'PIPELINE_BUCKET': BUCKET}.items():
        monkeypatch.setenv(k, v)
    for k in ('GEMINI_API_KEY', 'GEMINI_API_KEY_ARN', 'PIPELINE_ROOT', 'PIPELINE_BUCKET_REGION', 'AWS_DATA_REGION'):
        monkeypatch.delenv(k, raising=False)
    secrets.clear_cache()
    llm.set_client(None)
    with mock_aws():
        boto3.client('s3', region_name='eu-north-1').create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={'LocationConstraint': 'eu-north-1'})
        s3io.reset()
        yield
        s3io.reset()
    llm.set_client(None)


@pytest.fixture()
def jsonifier():
    return importlib.reload(importlib.import_module('jsonifier'))


def _seed(layout=LAYOUT, schema=True, student_name=SID):
    s3io.s3().put_object(Bucket=BUCKET, Key=layout.qp_pdf, Body=make_pdf('Q1. Define binding energy. [5 marks]'))
    if schema:
        s3io.write_text(BUCKET, layout.qp_schema_answer_txt, SCHEMA)
    key = f'{layout.original_pdf_prefix}{student_name}.pdf'
    s3io.s3().put_object(Bucket=BUCKET, Key=key, Body=make_pdf('1) Binding energy = energy to split a nucleus'))
    return key


def sqs_event(key, message_id='m1', bucket=BUCKET):
    body = {'Records': [{'s3': {'bucket': {'name': bucket}, 'object': {'key': key.replace(' ', '+')}}}]}
    return {'Records': [{'messageId': message_id, 'body': json.dumps(body)}]}


def _keys():
    return sorted(s3io.list_keys(BUCKET, ''))


# --- parity with the original -------------------------------------------------

def _string_assigned(path, name, func=None):
    tree = ast.parse(open(path, encoding='utf-8').read())
    scope = tree
    if func:
        scope = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not found')


def _original_processor(monkeypatch):
    gg = types.ModuleType('google.generativeai')
    google = types.ModuleType('google')
    google.generativeai = gg
    dotenv = types.ModuleType('dotenv')
    dotenv.load_dotenv = lambda *a, **k: None
    tqdm = types.ModuleType('tqdm')
    tqdm.tqdm = lambda x, **k: x
    for name, mod in (('google', google), ('google.generativeai', gg), ('dotenv', dotenv), ('tqdm', tqdm)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location('original_jsonifier', os.path.join(ORIG, 'jsonifier.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    proc = object.__new__(mod.JsonifierProcessor)
    proc.qp, proc.schema = 'QP TEXT', SCHEMA
    return proc


def test_prompts_match_original(monkeypatch):
    exe = os.path.join(ORIG, 'excute_jsonifier.py')
    assert jsonify.TRANSCRIBE_PROMPT == _string_assigned(exe, 'PROMPT')
    assert jsonify.TRANSCRIBE_CONFIG == _string_assigned(exe, 'GENERATION_CONFIG')
    assert jsonify.IMPROVE_PROMPT == _string_assigned(exe, 'improvement_prompt', 'parse_pdf_with_gemini_text_method')
    proc = _original_processor(monkeypatch)
    assert jsonify.segregation_prompt('QP TEXT') == proc.get_prompt_segregation()
    assert jsonify.schema_prompt(SCHEMA) == proc.get_prompt_schema()
    for reply in ('x {"a": 1} y', 'no json', '{bad', '[1, 2]'):
        ours, theirs = jsonify.output_to_json(reply), proc.output_to_json(reply)
        assert (ours[0] is None) == (theirs[0] is None) or reply == '[1, 2]'


def test_pdf_text_helpers():
    data = make_pdf('Hello nucleus')
    assert pdf.text_with_pages(data).startswith('--- Page 1 ---') and 'Hello nucleus' in pdf.text_with_pages(data)
    assert 'Hello nucleus' in pdf.plain_text(data)
    assert pdf.text_with_pages(b'not a pdf') == '' and pdf.plain_text(b'not a pdf') == ''


# --- the worker ---------------------------------------------------------------

def test_pdf_to_json_single_write(jsonifier, monkeypatch):
    key = _seed()
    fake = FakeParseLLM()
    llm.set_client(fake)
    writes = []
    real = s3io.write_text
    monkeypatch.setattr(s3io, 'write_text', lambda b, k, *a, **kw: (writes.append(k), real(b, k, *a, **kw))[1])

    assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}

    out = LAYOUT.parsed_json(SID)
    assert writes == [LAYOUT.parsed_text(SID), LAYOUT.parsed_seg(SID), LAYOUT.parsed_json_text(SID), out]
    assert writes.count(out) == 1  # the original wrote it twice → two grader runs
    assert s3io.read_json(BUCKET, out) == FILLED
    assert s3io.read_text(BUCKET, LAYOUT.parsed_text(SID)) == TRANSCRIPT
    seg_prompt = next(p for kind, p, _ in fake.calls if p.startswith('You are a physics expert'))
    assert 'Define binding energy' in seg_prompt and seg_prompt.endswith(TRANSCRIPT)
    assert [c[2] for c in fake.calls] == [5120, 8192, 8192]


def test_existing_json_skips(jsonifier):
    key = _seed()
    s3io.write_json(BUCKET, LAYOUT.parsed_json(SID), {'done': True})
    fake = FakeParseLLM()
    llm.set_client(fake)
    assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    assert fake.calls == []


def test_teacher_pdf_and_other_files_ignored(jsonifier):
    _seed()
    fake = FakeParseLLM()
    llm.set_client(fake)
    before = _keys()
    for key in (LAYOUT.qp_pdf, LAYOUT.solution_pdf, LAYOUT.parsed_json(SID), LAYOUT.parsed_text(SID)):
        assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    assert fake.calls == [] and _keys() == before


def test_bad_json_three_attempts_then_failure(jsonifier):
    key = _seed()
    fake = FakeParseLLM(json_reply='Sorry, I cannot do that.')
    llm.set_client(fake)
    assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': [{'itemIdentifier': 'm1'}]}
    assert sum(1 for _, p, _ in fake.calls if p.startswith('Please fill')) == 3
    assert LAYOUT.parsed_json(SID) not in _keys()


def test_missing_schema_fails_before_any_model_call(jsonifier):
    key = _seed(schema=False)
    fake = FakeParseLLM()
    llm.set_client(fake)
    assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': [{'itemIdentifier': 'm1'}]}
    assert fake.calls == []


def test_model_failure_in_segregation_fails_message(jsonifier, monkeypatch):
    key = _seed()
    fake = FakeParseLLM()
    real = fake.generate

    def generate(prompt, **kw):
        if prompt.startswith('You are a physics expert'):
            raise llm.LLMError('quota')
        return real(prompt, **kw)
    fake.generate = generate
    llm.set_client(fake)
    assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': [{'itemIdentifier': 'm1'}]}
    assert LAYOUT.parsed_json(SID) not in _keys()


def test_root_from_key_not_hardcoded(jsonifier):
    other = paths.Layout('Other_root', AID, QP)
    key = _seed(layout=other)
    llm.set_client(FakeParseLLM())
    assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    assert other.parsed_json(SID) in _keys() and not any(k.startswith('Data_warehouse/') for k in _keys())


def test_legacy_file_name_is_made_safe(jsonifier):
    key = _seed(student_name='John Doe HW1')
    llm.set_client(FakeParseLLM())
    assert jsonifier.lambda_handler(sqs_event(key), None) == {'batchItemFailures': []}
    assert LAYOUT.parsed_json('John_Doe_HW1') in _keys()


def test_transcribe_fallbacks():
    data = make_pdf('1) Binding energy = energy to split a nucleus')
    # The PDF route fails → PDF text cleaned by the model (method 1)
    fake = FakeParseLLM(transcript=llm.LLMError('down'), improve='Cleaned transcript of the answer sheet, long enough to keep.')
    assert jsonify.transcribe(data, fake) == 'Cleaned transcript of the answer sheet, long enough to keep.'
    improve_prompt = next(p for kind, p, _ in fake.calls if kind == 'generate')
    assert improve_prompt.startswith(jsonify.IMPROVE_PROMPT + '\n\nContent to process:\n--- Page 1 ---')
    # Both model routes fail → raw PDF text
    fake = FakeParseLLM(transcript='', improve=llm.LLMError('down'))
    assert jsonify.transcribe(data, fake).startswith('--- Page 1 ---')
    # Nothing readable at all → input error (message fails)
    with pytest.raises(PipelineInputError):
        jsonify.transcribe(b'not a pdf', FakeParseLLM(transcript=''))
