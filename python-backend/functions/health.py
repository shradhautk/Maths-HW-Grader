"""GET /health, /health/ready — Lambda gradeant-health (handler health.lambda_handler).

/health        liveness, no I/O
/health/ready  also checks DynamoDB and the JWT secret, so a deployment can be
               verified end to end (503 + "degraded" when a check fails)
Same shape as backend/src/controllers/health.controller.js.
"""
import time

from gradeant_common import VERSION, config, dynamo, secrets
from gradeant_common.http import create_response, handle

_started = time.time()


def _base():
    return {
        'service': 'gradeant-api',
        'version': VERSION,
        'runtime': 'python',
        'mode': config.mode(),
        'uptimeSeconds': round(time.time() - _started),
    }


def _check(fn):
    t = time.time()
    try:
        fn()
        return {'ok': True, 'ms': round((time.time() - t) * 1000)}
    except Exception as e:  # noqa: BLE001 — reported, never raised
        return {'ok': False, 'ms': round((time.time() - t) * 1000), 'error': type(e).__name__}


def _jwt_configured():
    if not secrets.jwt_secret():
        raise LookupError('NotConfigured')


def _route(event):
    paths = config.api_paths()
    path = event.get('path')
    if (event.get('httpMethod') or 'GET') != 'GET':
        return create_response(404, {'message': 'Not Found'})
    if path == paths.get('health', '/health'):
        return create_response(200, {'status': 'ok', **_base()})
    if path == paths.get('healthReady', '/health/ready'):
        checks = {
            'dynamodb': _check(lambda: dynamo.table('users').get_item(Key={'email': '__healthcheck__'})),
            'jwtSecret': _check(_jwt_configured),
        }
        ok = all(c['ok'] for c in checks.values())
        return create_response(200 if ok else 503, {'status': 'ok' if ok else 'degraded', **_base(), 'checks': checks})
    return create_response(404, {'message': 'Not Found'})


def lambda_handler(event, context):
    return handle(event, context, _route, key='message')
