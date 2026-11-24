"""Supabase PostgREST client for the grading table (database: supabase), httpx2 only.

The grader Lambda talks to Supabase over HTTPS with the service-role key; no Postgres
driver or native wheel is needed. Everything lives in schema `gradeant`
(backend/sql/001_schema.sql), exposed to PostgREST for the service role only
(backend/sql/003_supabase_access.sql):

    select(table, filters)   GET  /rest/v1/<table>?select=item&<col>=eq.<value>   (paged)
    rpc(name, args)          POST /rest/v1/rpc/<name>                              (one transaction)

Retries 429 / 5xx / connection errors with backoff, then raises PgRestError, so the SQS
message fails and is retried. Settings: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (or _ARN).

httpx2 (an httpx fork with the same API; google-genai 3.x already puts it in the pipeline layer) is
imported only when a request is made: this module is part of the shared layer, and
the API functions (DynamoDB only) don't need it.
"""
import json
import threading
import time
from decimal import Decimal

from . import secrets
from .config import env, env_int

SCHEMA = 'gradeant'
PAGE = 1000          # rows asked for per request; PostgREST may return fewer (max_rows): paging stops on an empty page
RETRIES = 3

_lock = threading.Lock()
_client = None
_transport = None    # tests: httpx2.MockTransport


class PgRestError(Exception):
    """PostgREST refused the request or stayed unavailable after retries."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def set_transport(transport):
    """Tests only: route every request through `transport` (an httpx2.MockTransport; None restores the network)."""
    global _transport, _client
    with _lock:
        _transport, _client = transport, None


def _json_default(value):
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    raise TypeError(f'{type(value).__name__} is not JSON serialisable')


def _http():
    global _client
    with _lock:
        if _client is None:
            url = env('SUPABASE_URL').rstrip('/')
            key = secrets.supabase_service_role_key()
            if not url or not key:
                raise PgRestError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when database: supabase')
            import httpx2 as httpx  # the layer ships httpx2 (google-genai 3.x dependency), same API as httpx
            _client = httpx.Client(
                base_url=f'{url}/rest/v1',
                headers={
                    'apikey': key,
                    'Authorization': f'Bearer {key}',
                    'Accept-Profile': SCHEMA,
                    'Content-Profile': SCHEMA,
                },
                timeout=env_int('SUPABASE_TIMEOUT_SECONDS', 15),
                transport=_transport,
            )
        return _client


def _request(method, path, *, params=None, body=None):
    import httpx2 as httpx
    last = None
    for attempt in range(RETRIES):
        try:
            response = _http().request(
                method, path, params=params,
                content=None if body is None else json.dumps(body, default=_json_default),
                headers={'Content-Type': 'application/json'} if body is not None else None,
            )
        except httpx.TransportError as exc:  # connection refused, timeout, DNS…
            last = PgRestError(f'Supabase unreachable: {type(exc).__name__}')
        else:
            if response.status_code < 300:
                return response.json() if response.content else None
            detail = response.text[:300]
            last = PgRestError(f'Supabase {method} {path} → {response.status_code}: {detail}', response.status_code)
            if response.status_code != 429 and response.status_code < 500:
                raise last  # 4xx: retrying won't help (bad filter, missing grant, unknown function…)
        if attempt < RETRIES - 1:
            time.sleep(0.5 * 2 ** attempt)
    raise last


def select(table, filters, *, columns='item', order='evaluation_id'):
    """Every row of `table` matching all `filters` (column → value, equality), following pages."""
    rows, offset = [], 0
    while True:
        params = {'select': columns, 'order': order, 'limit': str(PAGE), 'offset': str(offset)}
        params.update({col: f'eq.{value}' for col, value in filters.items()})
        page = _request('GET', f'/{table}', params=params) or []
        if not page:  # a short page isn't proof of the end: the server may cap rows below PAGE
            return rows
        rows.extend(page)
        offset += len(page)


def rpc(name, args):
    """Calls gradeant.<name>(args) — one transaction on the database side."""
    return _request('POST', f'/rpc/{name}', body=args)
