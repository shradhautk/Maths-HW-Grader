"""Role checks driven by config/app.yaml (backend/src/lib/permissions.js).

These sit in front of each operation; the operations keep their own built-in
checks and data rules too (app.yaml can narrow access, never widen it).
"""
from .config import app_config
from .http import ApiError


def _own(obj, key):
    return obj.get(key) if isinstance(obj, dict) and isinstance(key, str) and key in obj else None


def _list(value):
    return value if isinstance(value, list) else []


def is_allowed(area, operation, role):
    """True when app.yaml permissions.<area>.<operation> lists this role."""
    return role in _list(_own(_own(app_config().get('permissions'), area), operation))


def require_permission(area, operation, role, message="Your role can't do that."):
    if not is_allowed(area, operation, role):
        raise ApiError(403, message)


def can_sign_up(role, how):
    """app.yaml roles.<role>.signup includes 'self' / 'invite'."""
    return how in _list((_own(app_config().get('roles'), role) or {}).get('signup'))


def role_label(role):
    return (_own(app_config().get('roles'), role) or {}).get('label') or role


def self_signup_roles():
    """Roles that can sign themselves up, in app.yaml order."""
    return [r for r in (app_config().get('roles') or {}) if can_sign_up(r, 'self')]
