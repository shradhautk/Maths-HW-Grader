"""LLM client interface for the pipeline (google-genai SDK, Gemini).

    get_client('grade') / get_client('parse') → GEMINI_MODEL (default DEFAULT_MODEL)
One model for every AI call in GradeAnt: grading, reading answer sheets, rubric drafting and the
teacher assistant (gradeant_common/ai.py, backend/src/ai/index.js use the same setting).
PIPELINE_LLM=gemini is the only provider.

Failures never turn into answer text (the original graded "ERROR: ..." strings as 0 marks):
after PIPELINE_LLM_RETRIES attempts an LLMError is raised. The key comes from
secrets.gemini_api_key() and is checked on first use, not at import.
"""
import re
import threading
import time
from typing import Optional, Protocol

from .. import secrets
from ..config import env, env_int

DEFAULT_MODEL = 'gemini-3.5-flash-preview'
PURPOSES = ('grade', 'parse')


def model_name():
    """GEMINI_MODEL, or the default. The single place the Python side names the model."""
    return env('GEMINI_MODEL', DEFAULT_MODEL)

_RETRYABLE = ('429', 'resource_exhausted', 'quota', 'rate', '500', '502', '503', '504', 'unavailable', 'internal', 'deadline')


class LLMError(Exception):
    """The model could not produce an answer (after retries), or isn't configured."""


class LLMClient(Protocol):
    def generate(self, prompt: str, *, temperature: float, max_output_tokens: Optional[int] = None) -> str: ...

    def transcribe_pdf(self, pdf_bytes: bytes, prompt: str, *, temperature: float, max_output_tokens: int) -> str: ...


def retry_delay_from(message: str) -> Optional[int]:
    """The server's suggested wait, if the error text carries one (port of grader_lambda._extract_retry_delay)."""
    for pattern in (r'retry_delay\s*\{\s*seconds:\s*(\d+)', r'retry.*?(\d+)\s*seconds?', r'retryDelay["\']?\s*:\s*["\']?(\d+)s'):
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _retryable(exc):
    code = str(getattr(exc, 'code', '') or getattr(exc, 'status', '') or '')
    text = f'{code} {exc}'.lower()
    return any(token in text for token in _RETRYABLE)


class GeminiClient:
    def __init__(self, model: str, retries: Optional[int] = None):
        self.model = model
        self.retries = max(1, retries if retries is not None else env_int('PIPELINE_LLM_RETRIES', 3))
        self._client = None
        self._lock = threading.Lock()

    def _sdk(self):
        with self._lock:
            if self._client is None:
                key = secrets.gemini_api_key()
                if not key:
                    raise LLMError('Gemini API key not configured')
                from google import genai  # lazy: only the worker layer ships it
                self._client = genai.Client(api_key=key)
            return self._client

    def _call(self, contents, temperature, max_output_tokens):
        from google.genai import types
        client = self._sdk()
        config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        last = None
        for attempt in range(self.retries):
            try:
                response = client.models.generate_content(model=self.model, contents=contents, config=config)
                text = getattr(response, 'text', None)
                if not text or not text.strip():
                    raise LLMError('Empty response from the model')
                return text
            except LLMError as exc:
                last = exc
                delay = 2
            except Exception as exc:  # SDK/network errors
                last = exc
                if not _retryable(exc):
                    raise LLMError(f'Gemini call failed: {exc}') from exc
                delay = retry_delay_from(str(exc)) or min(60, 2 * 2 ** attempt)
            if attempt < self.retries - 1:
                time.sleep(delay)
        raise LLMError(f'Gemini failed after {self.retries} attempts: {last}') from last

    def generate(self, prompt, *, temperature, max_output_tokens=None):
        return self._call(prompt, temperature, max_output_tokens)

    def transcribe_pdf(self, pdf_bytes, prompt, *, temperature, max_output_tokens):
        # Inline bytes (no Files API upload to clean up); uploads are capped well under the 20 MB inline limit.
        from google.genai import types
        part = types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf')
        return self._call([part, prompt], temperature, max_output_tokens)


_override = None
_clients = {}
_clients_lock = threading.Lock()


def get_client(purpose: str) -> LLMClient:
    if _override is not None:
        return _override
    if purpose not in PURPOSES:
        raise ValueError(f'Unknown LLM purpose {purpose!r}')
    provider = env('PIPELINE_LLM', 'gemini')
    if provider != 'gemini':
        raise LLMError(f'Unsupported PIPELINE_LLM {provider!r}')
    model = model_name()
    with _clients_lock:
        if (purpose, model) not in _clients:
            _clients[(purpose, model)] = GeminiClient(model)
        return _clients[(purpose, model)]


def set_client(client) -> None:
    """Tests only: force every get_client() to return this client (None restores the default)."""
    global _override
    _override = client
    with _clients_lock:
        _clients.clear()
