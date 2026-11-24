"""Request handling shared by every API function (backend/src/middleware + controllers/respond.js).

`handle(event, fn, ...)` wraps a Lambda proxy request:
  - OPTIONS → 204 with CORS headers;
  - CORS: the allowlist from config (app.yaml urls.<mode>.corsOrigins or CORS_ALLOWED_ORIGINS);
  - ApiError → its status with {<key>: message} (or the error's own body);
  - any other exception → logged with the request id, answered with a generic 500
    (no exception text leaves the function in production).
Each API keeps the error key its original Lambda used: "error" for /getgrades,
/correctanswer, /updatedb, /workflow; "message" for /auth/* and /admin.
The operation-dispatch APIs add authentication on top: see dispatch.py.
"""
import base64
import datetime
import json
from decimal import Decimal

from . import config


class ApiError(Exception):
    """An error with an HTTP status. `body`, when given, is sent instead of {<key>: message}."""

    def __init__(self, status, message, body=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.body = body


def json_default(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj, key=str)
    raise TypeError(f'Not JSON serialisable: {type(obj).__name__}')


def create_response(status_code, body, headers=None):
    """A Lambda proxy response. CORS headers are added by handle()."""
    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json', **(headers or {})},
        'body': json.dumps(body, default=json_default),
    }


def header(event, name):
    """A request header, case-insensitively (API Gateway passes them as sent)."""
    headers = event.get('headers') or {}
    lower = name.lower()
    for k, v in headers.items():
        if k.lower() == lower:
            return v
    return None


def method(event):
    return (event.get('httpMethod') or (event.get('requestContext') or {}).get('http', {}).get('method') or 'POST').upper()


def parse_body(event, invalid_message='Invalid JSON in request body'):
    """The JSON request body as a dict ({} when absent or not an object). Malformed JSON → 400."""
    raw = event.get('body')
    if raw is None or raw == '':
        return {}
    if isinstance(raw, dict):
        return raw
    if event.get('isBase64Encoded'):
        try:
            raw = base64.b64decode(raw).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, invalid_message)
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        raise ApiError(400, invalid_message)
    return parsed if isinstance(parsed, dict) else {}


def cors_headers(event):
    origins = config.cors_origins()
    out = {
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, OPTIONS',
        'Access-Control-Max-Age': '600',
    }
    if '*' in origins or (not origins and not config.is_production()):
        out['Access-Control-Allow-Origin'] = '*'
    else:
        out['Vary'] = 'Origin'
        origin = header(event, 'origin')
        if origin and origin in origins:
            out['Access-Control-Allow-Origin'] = origin
    return out


def request_id(event, context=None):
    return getattr(context, 'aws_request_id', None) or (event.get('requestContext') or {}).get('requestId')


def log(level, msg, **fields):
    print(json.dumps({'level': level, 'msg': msg, **fields}, default=str))


def handle(event, context, fn, *, key='error', internal='Internal server error'):
    """Runs fn(event) → response dict; turns errors into responses; adds CORS to everything."""
    event = event or {}
    if method(event) == 'OPTIONS':
        res = {'statusCode': 204, 'headers': {}, 'body': ''}
    else:
        try:
            res = fn(event)
            if res is None:
                raise RuntimeError('handler returned no response')
        except ApiError as e:
            res = create_response(e.status, e.body if e.body is not None else {key: e.message})
        except Exception as e:  # noqa: BLE001 — the last line of defence for every API
            log('error', 'unhandled error', requestId=request_id(event, context), path=event.get('path'),
                error=type(e).__name__, detail=str(e))
            message = internal if config.is_production() else f'{internal}: {e}'
            res = create_response(500, {key: message})
    res.setdefault('headers', {}).update(cors_headers(event))
    return res
