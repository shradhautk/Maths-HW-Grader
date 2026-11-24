"""Gemini for the GradeFlow assistant and rubric drafting (functions/workflow.py).

Same contract as the original lambda_python_files/workflow.py _ask_claude, with the same
prompts (the callers pass them unchanged):

    ask(system, messages, max_tokens=16000, output_format=None) → reply text, or None when
    Gemini declines (blocked prompt / safety finish), as a Claude refusal returned None.

`output_format` ({'type': 'json_schema', 'schema': …}) asks for JSON through Gemini's
response_json_schema; if Gemini refuses the schema (400) the schema goes into the prompt
instead, as the Claude path did. The model is the single GEMINI_MODEL used everywhere
(pipeline.llm.model_name()); the key comes from GEMINI_API_KEY or GEMINI_API_KEY_ARN.
google-genai is imported on first use.
"""
import json
import threading
import time

from . import secrets
from .config import env_int
from .pipeline.llm import LLMError, _retryable, model_name, retry_delay_from

DECLINED = {'SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'RECITATION', 'IMAGE_SAFETY'}

_lock = threading.Lock()
_client = None


def set_client(client):
    """Tests only: a fake genai.Client ({models.generate_content}); None restores the real one."""
    global _client
    with _lock:
        _client = client


def _sdk():
    global _client
    with _lock:
        if _client is None:
            key = secrets.gemini_api_key()
            if not key:
                raise LLMError('Gemini API key not configured (GEMINI_API_KEY or GEMINI_API_KEY_ARN)')
            from google import genai
            _client = genai.Client(api_key=key)
        return _client


def _text(content):
    if isinstance(content, str):
        return content
    return ''.join(b.get('text', '') for b in content or [] if isinstance(b, dict) and b.get('type') == 'text')


def _generate(client, contents, config):
    retries = max(1, env_int('PIPELINE_LLM_RETRIES', 3))
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=model_name(), contents=contents, config=config)
        except Exception as exc:  # noqa: BLE001 — SDK/network errors: retry the transient ones
            if not _retryable(exc) or attempt == retries - 1:
                raise
            time.sleep(retry_delay_from(str(exc)) or min(60, 2 * 2 ** attempt))


def _status(exc):
    return getattr(exc, 'code', None) or getattr(exc, 'status_code', None)


def ask(system, messages, *, max_tokens=16000, output_format=None):
    client = _sdk()
    contents = [{'role': 'model' if m.get('role') == 'assistant' else 'user', 'parts': [{'text': _text(m.get('content'))}]}
                for m in messages]
    config = {'system_instruction': system, 'max_output_tokens': max_tokens}
    if output_format:
        config.update(response_mime_type='application/json', response_json_schema=output_format['schema'])
    try:
        response = _generate(client, contents, config)
    except Exception as exc:  # noqa: BLE001
        if not output_format or _status(exc) != 400:
            raise
        # A schema the model can't take: ask for the JSON in the prompt (callers parse leniently).
        config = {'system_instruction': f"{system}\n\nRespond with only a JSON object matching this JSON Schema, "
                                        f"and no other text:\n{json.dumps(output_format['schema'])}",
                  'max_output_tokens': max_tokens}
        response = _generate(client, contents, config)
    feedback = getattr(response, 'prompt_feedback', None)
    if feedback is not None and getattr(feedback, 'block_reason', None):
        return None
    candidates = getattr(response, 'candidates', None) or []
    reason = getattr(candidates[0], 'finish_reason', None) if candidates else None
    if str(getattr(reason, 'value', reason) or '').upper() in DECLINED:
        return None
    return getattr(response, 'text', None) or None
