"""gradeant_common.ai: Gemini for the assistant and rubric drafting, with the original
_ask_claude contract (text, or None on a decline). The model is the single GEMINI_MODEL."""
import os
import sys
import types

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [ROOT]

from gradeant_common import ai, secrets  # noqa: E402
from gradeant_common.pipeline import llm  # noqa: E402


class FakeModels:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({'model': model, 'contents': contents, 'config': config})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def reply(text='ok', finish='STOP', block=None):
    return types.SimpleNamespace(
        text=text,
        prompt_feedback=types.SimpleNamespace(block_reason=block) if block else None,
        candidates=[types.SimpleNamespace(finish_reason=types.SimpleNamespace(value=finish))],
    )


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@pytest.fixture()
def fake(monkeypatch):
    monkeypatch.delenv('GEMINI_MODEL', raising=False)
    monkeypatch.setattr(ai.time, 'sleep', lambda s: None)

    def install(*answers):
        models = FakeModels(answers)
        ai.set_client(types.SimpleNamespace(models=models))
        return models
    yield install
    ai.set_client(None)


def test_chat_maps_to_gemini_with_the_one_model(fake):
    models = fake(reply('Q2 was hardest.'))
    out = ai.ask('You are GradeFlow.', [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'},
                                        {'role': 'user', 'content': 'Which was hardest?'}], max_tokens=900)
    assert out == 'Q2 was hardest.'
    call = models.calls[0]
    assert call['model'] == llm.DEFAULT_MODEL == 'gemini-3.5-flash-preview'
    assert [(c['role'], c['parts'][0]['text']) for c in call['contents']] == [('user', 'Hi'), ('model', 'Hello'), ('user', 'Which was hardest?')]
    assert call['config'] == {'system_instruction': 'You are GradeFlow.', 'max_output_tokens': 900}


def test_gemini_model_setting_is_used(fake, monkeypatch):
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-other')
    models = fake(reply())
    ai.ask('S', [{'role': 'user', 'content': 'Q'}])
    assert models.calls[0]['model'] == 'gemini-other'


def test_rubric_json_schema_and_400_fallback(fake):
    schema = {'type': 'object'}
    models = fake(reply('{"items":[]}'))
    assert ai.ask('S', [{'role': 'user', 'content': 'Q'}], output_format={'type': 'json_schema', 'schema': schema}) == '{"items":[]}'
    assert models.calls[0]['config']['response_json_schema'] == schema
    models = fake(ApiError(400, 'INVALID_ARGUMENT: schema'), reply('{}'))
    assert ai.ask('S', [{'role': 'user', 'content': 'Q'}], output_format={'type': 'json_schema', 'schema': schema}) == '{}'
    assert 'response_json_schema' not in models.calls[1]['config']
    assert 'Respond with only a JSON object matching this JSON Schema' in models.calls[1]['config']['system_instruction']


def test_declines_are_none_and_transient_errors_retry(fake):
    fake(reply(text=None, block='SAFETY'))
    assert ai.ask('S', [{'role': 'user', 'content': 'Q'}]) is None
    fake(reply(text='', finish='PROHIBITED_CONTENT'))
    assert ai.ask('S', [{'role': 'user', 'content': 'Q'}]) is None
    models = fake(ApiError(429, 'RESOURCE_EXHAUSTED'), reply('fine'))
    assert ai.ask('S', [{'role': 'user', 'content': 'Q'}]) == 'fine' and len(models.calls) == 2
    fake(ApiError(400, 'INVALID_ARGUMENT'))
    with pytest.raises(ApiError):
        ai.ask('S', [{'role': 'user', 'content': 'Q'}])


def test_missing_key_is_reported(monkeypatch):
    ai.set_client(None)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY_ARN', raising=False)
    secrets.clear_cache()
    with pytest.raises(llm.LLMError, match='Gemini API key not configured'):
        ai.ask('S', [{'role': 'user', 'content': 'Q'}])
