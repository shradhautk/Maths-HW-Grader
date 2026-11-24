"""The grading-pipeline Lambda (not in this repo) must keep receiving exactly what the
deployed workflow Lambda sends it today. Runs the ORIGINAL lambda_python_files/workflow.py
and python-backend's workflow side by side and compares the invoke calls byte for byte.
"""
import importlib
import importlib.util
import json
import os
import sys
from decimal import Decimal

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')                       # python-backend/
REPO = os.path.join(ROOT, '..')
sys.path[:0] = [os.path.join(ROOT, 'functions'), ROOT]

ASSIGNMENT = {
    'assignment_id': 'A-thermo',
    'qp_id': 'QP-thermo',
    'method': 'standard',
    'persona': 'strict',
    'partial_credit': False,
    'rubric': [
        {'question_number': Decimal('1'), 'subpart': None, 'title': 'Internal energy', 'max_marks': Decimal('5'), 'criteria': 'Defines U; unit J.'},
        {'question_number': '2', 'subpart': 'a', 'title': 'First law', 'max_marks': Decimal('2.5'), 'criteria': 'dU = Q - W'},
    ],
}


class FakeLambda:
    def __init__(self):
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return {'StatusCode': 202}


def _load_original():
    spec = importlib.util.spec_from_file_location('original_workflow', os.path.join(REPO, 'lambda_python_files', 'workflow.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def env(monkeypatch):
    for k, v in {'AWS_DEFAULT_REGION': 'eu-north-1', 'AWS_ACCESS_KEY_ID': 'x', 'AWS_SECRET_ACCESS_KEY': 'x',
                 'JWT_SECRET': 'test-secret-that-is-at-least-32-bytes-long', 'GRADING_FUNCTION_NAME': 'gradeant-grader'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('AWS_DATA_REGION', raising=False)


def _capture(mod, monkeypatch, assignment, student_id):
    fake = FakeLambda()
    monkeypatch.setattr(mod.boto3, 'client', lambda service, **_: fake if service == 'lambda' else None)
    mod._trigger_grading(assignment, student_id)
    return fake.calls


def test_grader_receives_the_same_request(env, monkeypatch):
    original = _load_original()
    new = importlib.reload(importlib.import_module('workflow'))
    old_calls = _capture(original, monkeypatch, ASSIGNMENT, 'S7')
    new_calls = _capture(new, monkeypatch, ASSIGNMENT, 'S7')

    assert len(old_calls) == len(new_calls) == 1
    old, now = old_calls[0], new_calls[0]
    assert now['FunctionName'] == old['FunctionName'] == 'gradeant-grader'
    assert now['InvocationType'] == old['InvocationType'] == 'Event'  # asynchronous, as before
    assert now['Payload'] == old['Payload']                           # byte for byte
    payload = json.loads(now['Payload'])
    assert payload == {
        'assignment_id': 'A-thermo', 'qp_id': 'QP-thermo', 'student_id': 'S7', 'method': 'standard',
        'persona': 'strict', 'partial_credit': False,
        'rubric': [
            {'question_number': 1, 'subpart': None, 'title': 'Internal energy', 'max_marks': 5, 'criteria': 'Defines U; unit J.'},
            {'question_number': '2', 'subpart': 'a', 'title': 'First law', 'max_marks': 2.5, 'criteria': 'dU = Q - W'},
        ],
    }


def test_defaults_match_when_fields_are_missing(env, monkeypatch):
    bare = {'assignment_id': 'A-1', 'qp_id': 'QP-1'}
    original = _load_original()
    new = importlib.reload(importlib.import_module('workflow'))
    assert _capture(new, monkeypatch, bare, 'S1')[0]['Payload'] == _capture(original, monkeypatch, bare, 'S1')[0]['Payload']


def test_no_grader_configured_sends_nothing(env, monkeypatch):
    monkeypatch.delenv('GRADING_FUNCTION_NAME')
    new = importlib.reload(importlib.import_module('workflow'))
    assert _capture(new, monkeypatch, ASSIGNMENT, 'S7') == []
