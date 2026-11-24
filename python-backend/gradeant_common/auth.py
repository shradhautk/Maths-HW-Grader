"""Bearer-token authentication (backend/src/middleware/auth.js).

An HS256 JWT from /auth/login whose account must still exist and not be suspended.
The role is read fresh from the Users table on every request, so a demoted or
deleted user loses access at once.
"""
import time

import jwt

from . import dynamo, secrets
from .http import ApiError, header

TOKEN_EXPIRY_SECONDS_DEFAULT = 86400


def bearer_token(event):
    value = header(event, 'authorization') or ''
    return value[len('Bearer '):].strip() if value.startswith('Bearer ') else None


def decode_login_token(token):
    """→ (payload, error) where error is None, 'expired' or 'invalid'."""
    try:
        return jwt.decode(token, secrets.jwt_secret(), algorithms=['HS256']), None
    except jwt.ExpiredSignatureError:
        return None, 'expired'
    except jwt.InvalidTokenError:
        return None, 'invalid'


def issue_login_token(user_id, email, expiry_seconds=TOKEN_EXPIRY_SECONDS_DEFAULT):
    now = int(time.time())
    return jwt.encode({'user_id': user_id, 'email': email, 'exp': now + expiry_seconds, 'iat': now},
                      secrets.jwt_secret(), algorithm='HS256')


def authenticate(event):
    """→ {'user_id', 'email', 'role', 'user'} or raises ApiError(401)."""
    if not secrets.jwt_secret():
        raise ApiError(500, 'Server misconfigured: JWT_SECRET is not set.')
    token = bearer_token(event)
    if token is None:
        raise ApiError(401, 'Missing Authorization: Bearer <token> header.')
    payload, error = decode_login_token(token)
    if error == 'expired':
        raise ApiError(401, 'Session expired — please log in again.')
    if error:
        raise ApiError(401, 'Invalid session token.')
    email, user_id = payload.get('email'), payload.get('user_id')
    if not email or not user_id:
        raise ApiError(401, 'Malformed session token.')
    user = dynamo.table('users').get_item(Key={'email': email}).get('Item')
    if not user or user.get('user_id') != user_id:
        raise ApiError(401, 'Account no longer exists.')
    if user.get('status') == 'suspended':
        raise ApiError(401, 'This account is suspended.')
    return {'user_id': user_id, 'email': email, 'role': user.get('userRole') or 'student', 'user': user}


def require_role(caller, roles, message):
    if caller['role'] not in roles:
        raise ApiError(403, message)
