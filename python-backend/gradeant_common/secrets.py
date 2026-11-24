"""Signing keys: Secrets Manager in production, environment variables locally.

    JWT_SECRET_ARN / SESSION_TOKEN_SECRET_ARN  → read once per execution environment
    JWT_SECRET / SESSION_TOKEN_SECRET          → used as-is (local runs, tests)
    GEMINI_API_KEY_ARN / GEMINI_API_KEY        → same rules (grading pipeline workers only)

A secret may be a plain string or JSON holding {"JWT_SECRET": "..."} / {"SESSION_TOKEN_SECRET": "..."}
/ {"GEMINI_API_KEY": "..."}.
Same rules as backend/src/aws/secrets.js. No secret ever has a default value.
"""
import json
import threading

import boto3

from .config import data_region, env

_cache = {}
_lock = threading.Lock()


def _fetch(arn, key):
    raw = boto3.client('secretsmanager', region_name=data_region()).get_secret_value(SecretId=arn).get('SecretString') or ''
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get(key), str):
            return parsed[key]
    except ValueError:
        pass
    return raw


def _resolve(key):
    value = env(key)
    if value:
        return value
    arn = env(f'{key}_ARN')
    if not arn:
        return ''
    with _lock:
        if key not in _cache:
            _cache[key] = _fetch(arn, key)  # a failure raises and is retried on the next request
        return _cache[key]


def jwt_secret():
    """The login-token (JWT, HS256) key shared with every API function. '' when not configured."""
    return _resolve('JWT_SECRET')


def session_secret():
    """The admin first-login session-token key. '' when not configured."""
    return _resolve('SESSION_TOKEN_SECRET')


def gemini_api_key():
    """The Gemini key for the grading pipeline (GEMINI_API_KEY / GEMINI_API_KEY_ARN). '' when not configured."""
    return _resolve('GEMINI_API_KEY')


def supabase_service_role_key():
    """The Supabase service-role key the grader uses for PostgREST (SUPABASE_SERVICE_ROLE_KEY / _ARN). '' when not set."""
    return _resolve('SUPABASE_SERVICE_ROLE_KEY')


def clear_cache():
    """Tests only: forget fetched secrets."""
    with _lock:
        _cache.clear()
