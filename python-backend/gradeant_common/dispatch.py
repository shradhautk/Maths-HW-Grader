"""The operation-dispatch APIs (/getgrades, /correctanswer, /updatedb, /workflow):
POST {"operation": "...", ...payload} with a bearer token
(backend/src/controllers/operations.controller.js).

Kept apart from http.py so functions without logins (contact, health) don't depend
on authentication code.
"""
from .auth import authenticate
from .http import ApiError, handle, parse_body
from .permissions import require_permission


def operations(event, context, ops, *, area, invalid_json='Invalid JSON format', internal='Internal server error',
               invalid_operation='Invalid operation specified'):
    """Order as in Node: body → authentication → operation lookup → app.yaml permission → operation.

    `ops` maps operation name → fn(body, caller) returning a response dict.
    """

    def run(evt):
        body = parse_body(evt, invalid_json)
        caller = authenticate(evt)
        op = body.get('operation')
        fn = ops.get(op) if isinstance(op, str) else None
        if fn is None:
            raise ApiError(400, invalid_operation)
        require_permission(area, op, caller['role'])
        return fn(body, caller)

    return handle(event, context, run, key='error', internal=internal)
