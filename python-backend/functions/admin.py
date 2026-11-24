"""
GradeAnt — Admin Console: POST /admin — Lambda gradeant-admin (handler admin.lambda_handler)
=============================================================================================

python-backend version: settings, table names, secrets and CORS come from
gradeant_common (same names as the Node backend); app.yaml permissions.admin can
turn individual console operations off. The deploy notes below describe the
original single-file Lambda; see python-backend/README.md for this version.

Single Lambda handler backing every admin/*.html page. Matches the
project's existing "operation dispatch" convention (see the getgrades /
updatedb endpoints the grading pages already call) instead of one Lambda
per REST route: the frontend POSTs `{"operation": "<name>", ...}` to one
API Gateway resource, and this file routes on that field.

Deploy
------
1. Runtime: Python 3.12+ (boto3 is preinstalled by the Lambda runtime —
   no `pip install` / layer needed, this file has zero third-party deps).
2. API Gateway: one POST route (e.g. `/admin`) with **Lambda proxy
   integration** pointed at this function, plus CORS enabled (or let this
   file's own `_cors_headers` handle it — see `lambda_handler`).
3. Environment variables (all optional — sensible defaults shown):
     USERS_TABLE          (default "GradeAnt_Users")
     RESET_REQUESTS_TABLE  (default "GradeAnt_ResetRequests")
     CONTENT_TABLE         (default "GradeAnt_Content")
     TICKETS_TABLE         (default "GradeAnt_Tickets")
     AUDIT_LOGS_TABLE      (default "GradeAnt_AuditLogs")
     SETTINGS_TABLE        (default "GradeAnt_Settings")
     INVITES_TABLE         (default "GradeAnt_Invites")
     ANALYTICS_TABLE       (default "GradeAnt_Analytics")
     SESSION_TOKEN_SECRET  (REQUIRED in production — HMAC signing key for
                            session tokens; falls back to an insecure
                            dev default with a loud warning if unset)
     JWT_SECRET            (REQUIRED) — same value as the /auth/login
                            Lambda (authetication.py), so admins can use the
                            token they get from the normal login page
     SES_SENDER_EMAIL      (default "no-reply@gradeant.io") — From
                            address for invite emails via SES
     APP_BASE_URL           (default "https://app.gradeant.io") — used to
                            build the invite link mailed to new users
4. IAM: this function's execution role needs dynamodb:GetItem/PutItem/
   UpdateItem/DeleteItem/Query/Scan on the tables above, and ses:SendEmail
   if invite emails are enabled.
5. DynamoDB tables (all on-demand billing is fine for admin-console
   traffic volumes):
     GradeAnt_Users          PK: user_id (S)
     GradeAnt_ResetRequests  PK: request_id (S)
     GradeAnt_Content        PK: post_id (S)
     GradeAnt_Tickets        PK: ticket_id (S)
     GradeAnt_AuditLogs      PK: log_id (S)
     GradeAnt_Settings       PK: setting_key (S)  — single item, key="platform"
     GradeAnt_Invites        PK: token (S)
     GradeAnt_Analytics      PK: metric_key (S)  — populated by a separate
                              scheduled job (EventBridge + Lambda/Glue);
                              this function only *reads* it. See
                              `get_analytics` for the fallback shape if
                              that table is empty.

Auth
----
Every operation except `getInviteDetails` / `completeFirstLogin` (the
unauthenticated first-login-setup flow, reached from an emailed invite
link rather than a login) requires `Authorization: Bearer <token>` and
the token's `role` claim to be "admin". Tokens are minimal HMAC-signed
JSON (see `_make_token` / `_verify_token`) so this file has no external
JWT dependency — if the existing `/auth/login` endpoint issues tokens in
a different format, swap `_verify_token` for whatever verifies those.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from gradeant_common import config, dynamo
from gradeant_common import secrets as secrets_store
from gradeant_common.http import ApiError as CommonApiError, create_response, handle, parse_body
from gradeant_common.permissions import require_permission

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

# Table names: gradeant_common.config (ADMIN_USERS_TABLE, RESET_REQUESTS_TABLE, ... as in backend/.env.example).
ADMIN_USERS_TABLE = config.table_name("admin_users")
RESET_REQUESTS_TABLE = config.table_name("reset_requests")
CONTENT_TABLE = config.table_name("content")
TICKETS_TABLE = config.table_name("tickets")
AUDIT_LOGS_TABLE = config.table_name("audit_logs")
SETTINGS_TABLE = config.table_name("settings")
INVITES_TABLE = config.table_name("invites")
# The table /auth/login (authetication.py) actually reads, keyed by email. Reset
# requests filed from the login page (source="login") must update *this* table
# or the temp password never works.
LOGIN_USERS_TABLE = config.table_name("users")
ANALYTICS_TABLE = config.table_name("analytics")

# Invites: INVITE_SENDER_EMAIL, else SES_SENDER_EMAIL (as the Node backend).
SES_SENDER_EMAIL = config.env("INVITE_SENDER_EMAIL") or config.env("SES_SENDER_EMAIL", "no-reply@gradeant.io")

_DEV_FALLBACK_SECRET = "dev-only-insecure-secret-CHANGE-ME"
SESSION_TOKEN_TTL_SECONDS = config.env_int("SESSION_TOKEN_TTL_SECONDS", 24 * 60 * 60)
INVITE_TOKEN_TTL_SECONDS = config.env_int("INVITE_TOKEN_TTL_SECONDS", 7 * 24 * 60 * 60)


def _session_key() -> str:
    """Admin session-token key (Secrets Manager / env). The dev fallback is never used in production."""
    key = secrets_store.session_secret()
    if key:
        return key
    if config.is_production():
        raise ApiError(500, "Server misconfigured: SESSION_TOKEN_SECRET is not set.")
    return _DEV_FALLBACK_SECRET


def _jwt_key() -> str:
    """The /auth/login JWT key, so tokens from the normal login page are accepted."""
    return secrets_store.jwt_secret()

_ses = None  # lazily created — most operations never touch SES


def _table(name):
    return dynamo.table_named(name)


def _ses_client():
    global _ses
    if _ses is None:
        _ses = boto3.client("ses", region_name=config.data_region())
    return _ses


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------

class ApiError(CommonApiError):
    """Raised by any handler to produce a specific HTTP status + message ({"message": ...})."""
    def __init__(self, status_code, message):
        super().__init__(status_code, message)
        self.status_code = status_code


class AuthError(ApiError):
    def __init__(self, message="Authentication required."):
        super().__init__(401, message)


class ForbiddenError(ApiError):
    def __init__(self, message="You don't have permission to do that."):
        super().__init__(403, message)


class NotFoundError(ApiError):
    def __init__(self, message="Not found."):
        super().__init__(404, message)


class ValidationError(ApiError):
    def __init__(self, message="Invalid request."):
        super().__init__(400, message)


class ConflictError(ApiError):
    def __init__(self, message="Conflict."):
        super().__init__(409, message)


# ----------------------------------------------------------------------
# Minimal signed session tokens (no external JWT dependency)
#
# Format: base64url(json(payload)) + "." + hex(hmac_sha256(payload_b64))
# This is deliberately simple (HMAC, not full JWT/JWS) — swap this out
# for whatever the real `/auth/login` Lambda already issues if it's
# different, so admin sessions and regular logins interoperate.
# ----------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _make_token(claims: dict, ttl_seconds: int) -> str:
    payload = dict(claims)
    payload["exp"] = int(time.time()) + ttl_seconds
    payload_b64 = _b64url_encode(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(_session_key().encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"


def _verify_token(token: str) -> dict:
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError:
        raise AuthError("Malformed session token.")

    expected_signature = hmac.new(_session_key().encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_signature, signature):
        raise AuthError("Invalid session token.")

    try:
        claims = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        raise AuthError("Malformed session token.")

    if claims.get("exp", 0) < time.time():
        raise AuthError("Session expired.")

    return claims


def _hash_password(password: str, salt_hex: str = None) -> str:
    """PBKDF2-HMAC-SHA256, stdlib only. Returns "salt_hex:hash_hex"."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}:{digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    if not stored or ":" not in stored:
        return False
    salt_hex, _ = stored.split(":", 1)
    return hmac.compare_digest(_hash_password(password, salt_hex), stored)


def _hash_password_for_login(password: str) -> str:
    """authetication.py's storage format: base64(32-byte salt + PBKDF2-SHA256/100k)."""
    salt = secrets.token_bytes(32)
    return base64.b64encode(salt + hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)).decode("utf-8")


def _password_problem(password: str):
    """authetication.py's password_problem: 8+ chars, an uppercase letter, a number."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return "Password must include an uppercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must include a number."
    return None


LOGIN_TOKEN_EXPIRY = config.env_int("TOKEN_EXPIRY_SECONDS", 86400)


def _issue_login_jwt(user_id: str, email: str) -> str:
    """The HS256 JWT /auth/login issues (payload: user_id, email, exp, iat)."""
    jwt_key = _jwt_key()
    if not jwt_key:
        raise ApiError(500, "Server misconfigured: JWT_SECRET is not set.")
    iat = int(time.time())
    header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    payload = {"user_id": user_id, "email": email, "exp": iat + LOGIN_TOKEN_EXPIRY, "iat": iat}
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(jwt_key.encode("utf-8"), f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"


def _generate_temp_password() -> str:
    # Human-typeable temp password: no ambiguous chars (0/O, 1/l/I).
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"
    return "".join(secrets.choice(alphabet) for _ in range(12))


# ----------------------------------------------------------------------
# Request helpers
# ----------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _parse_body(event) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValidationError("Request body must be valid JSON.")


def _verify_login_jwt(token: str) -> dict:
    """HS256 JWT issued by /auth/login (authetication.py's generate_token).

    It carries only user_id/email, so the role comes from the login Users
    table on every request — same as get_grades.py and the other Lambdas.
    """
    jwt_key = _jwt_key()
    if not jwt_key:
        raise AuthError("Server misconfigured: JWT_SECRET is not set.")
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        raise AuthError("Malformed session token.")
    if header.get("alg") != "HS256":
        raise AuthError("Invalid session token.")
    expected = hmac.new(jwt_key.encode("utf-8"), f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_encode(expected), signature_b64):
        raise AuthError("Invalid session token.")
    if payload.get("exp", 0) < time.time():
        raise AuthError("Session expired.")
    email, user_id = payload.get("email"), payload.get("user_id")
    if not email or not user_id:
        raise AuthError("Malformed session token.")
    user = _table(LOGIN_USERS_TABLE).get_item(Key={"email": email}).get("Item")
    if not user or user.get("user_id") != user_id:
        raise AuthError("Account no longer exists.")
    if user.get("status") == "suspended":
        raise AuthError("This account is suspended.")
    return {"sub": user_id, "role": user.get("userRole", "student"), "name": user.get("name", "Admin"), "email": email}


def _get_bearer_claims(event) -> dict:
    headers = event.get("headers") or {}
    auth_header = headers.get("Authorization") or headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        raise AuthError("Missing Authorization: Bearer <token> header.")
    token = auth_header[7:].strip()
    # Three dot-separated parts = a JWT from /auth/login; two = this file's
    # own session token (issued by completeFirstLogin).
    if token.count(".") == 2:
        return _verify_login_jwt(token)
    # Session tokens carry a role claim, but it's re-read from the account on every
    # request so a deleted, suspended or demoted admin loses access at once.
    claims = _verify_token(token)
    sub = claims.get("sub")
    account = _table(ADMIN_USERS_TABLE).get_item(Key={"user_id": str(sub)}).get("Item") if sub else None
    if not account or account.get("status") == "suspended":
        raise AuthError("Account no longer active.")
    return dict(claims, role=account.get("role", "student"), name=account.get("name", claims.get("name")))


def _require_admin(event) -> dict:
    claims = _get_bearer_claims(event)
    if claims.get("role") != "admin":
        raise ForbiddenError("This action is admin-only.")
    return claims


def _write_audit_log(actor: str, action: str, target: str):
    try:
        _table(AUDIT_LOGS_TABLE).put_item(Item={
            "log_id": str(uuid.uuid4()),
            "timestamp": Decimal(str(time.time())),
            "actor": actor,
            "action": action,
            "target": target,
        })
    except Exception as exc:  # noqa: BLE001 — audit logging must never break the caller's action
        print(f"WARNING: failed to write audit log ({action} / {target}): {exc}")


# ----------------------------------------------------------------------
# Users (admin_users.html)
# ----------------------------------------------------------------------

def list_users(payload, claims):
    search = (payload.get("search") or "").strip().lower()
    role = payload.get("role") or ""
    status = payload.get("status") or ""

    filter_expr = None
    if role:
        filter_expr = Attr("role").eq(role)
    if status:
        status_filter = Attr("status").eq(status)
        filter_expr = status_filter if filter_expr is None else (filter_expr & status_filter)

    scan_kwargs = {}
    if filter_expr is not None:
        scan_kwargs["FilterExpression"] = filter_expr

    items = _table(ADMIN_USERS_TABLE).scan(**scan_kwargs).get("Items", [])

    if search:
        items = [
            u for u in items
            if search in (u.get("name", "").lower()) or search in (u.get("email", "").lower())
        ]

    users = [{
        "id": u["user_id"],
        "name": u.get("name", ""),
        "email": u.get("email", ""),
        "role": u.get("role", "student"),
        "school": u.get("school", ""),
        "status": u.get("status", "active"),
    } for u in items]

    return {"users": users, "total": len(users)}


def _set_login_status(email: str, status: str):
    """Set the app login account's status ('active' | 'suspended'). A no-op when
    there's no login account for that email (e.g. an invite never completed)."""
    try:
        _table(LOGIN_USERS_TABLE).update_item(
            Key={"email": email},
            UpdateExpression="SET #s = :s",
            ConditionExpression="attribute_exists(email)",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": status},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise


def update_user_status(payload, claims):
    user_id = payload.get("userId")
    status = payload.get("status")
    if not user_id or status not in ("active", "suspended", "invited"):
        raise ValidationError("userId and a valid status are required.")

    account = _table(ADMIN_USERS_TABLE).get_item(Key={"user_id": user_id}).get("Item") or {}
    _table(ADMIN_USERS_TABLE).update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET #s = :status",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":status": status},
    )
    # The login account lives in the Users table (keyed by email): mirror a suspension
    # there too, or the user could keep signing in to the app.
    if account.get("email") and status != "invited":
        _set_login_status(account["email"], status)
    _write_audit_log(claims["sub"], f"Set user status to {status}", user_id)
    return {"success": True}


def delete_user(payload, claims):
    user_id = payload.get("userId")
    if not user_id:
        raise ValidationError("userId is required.")
    account = _table(ADMIN_USERS_TABLE).get_item(Key={"user_id": user_id}).get("Item") or {}
    # Block the login account rather than deleting it, so grades stay attributable.
    if account.get("email"):
        _set_login_status(account["email"], "suspended")
    _table(ADMIN_USERS_TABLE).delete_item(Key={"user_id": user_id})
    _write_audit_log(claims["sub"], "Deleted user", user_id)
    return {"success": True}


# ----------------------------------------------------------------------
# Onboarding — Invite / Import → Assign Roles (Website Wireframes.html §17)
# ----------------------------------------------------------------------

def _send_invite_email(email: str, name: str, invite_link: str):
    try:
        _ses_client().send_email(
            Source=SES_SENDER_EMAIL,
            Destination={"ToAddresses": [email]},
            Message={
                "Subject": {"Data": "You're invited to GradeAnt"},
                "Body": {"Html": {"Data": (
                    f"<p>Hi {name},</p>"
                    f"<p>You've been invited to GradeAnt. Finish setting up your account:</p>"
                    f"<p><a href=\"{invite_link}\">{invite_link}</a></p>"
                )}},
            },
        )
    except Exception as exc:  # noqa: BLE001 — invite record should still succeed even if SES isn't set up yet
        print(f"WARNING: failed to send invite email to {email}: {exc}")


def invite_users(payload, claims):
    invitees = payload.get("invitees") or []
    if not invitees:
        raise ValidationError("At least one invitee is required.")

    invited_count = 0
    for person in invitees:
        email = (person.get("email") or "").strip()
        if not email:
            continue
        name = (person.get("name") or email).strip()
        role = person.get("role") or "student"
        class_name = person.get("className") or ""

        user_id = str(uuid.uuid4())
        _table(ADMIN_USERS_TABLE).put_item(Item={
            "user_id": user_id,
            "name": name,
            "email": email,
            "role": role,
            "school": class_name,
            "status": "invited",
        })

        token = secrets.token_urlsafe(24)
        _table(INVITES_TABLE).put_item(Item={
            "token": token,
            "user_id": user_id,
            "full_name": name,
            "role": role,
            "class_name": class_name,
            "expires_at": Decimal(str(time.time() + INVITE_TOKEN_TTL_SECONDS)),
            "consumed": False,
        })

        # The React app's first-login page (web/src/App.tsx route "/first-login").
        invite_link = f"{config.app_base_url()}{_first_login_path()}?token={token}"
        _send_invite_email(email, name, invite_link)
        invited_count += 1

    _write_audit_log(claims["sub"], f"Invited {invited_count} user(s)", ", ".join(p.get("email", "") for p in invitees))
    return {"invited": invited_count}


def get_invite_details(payload, claims):
    """Unauthenticated — this is what first_login_setup.html loads."""
    token = payload.get("token")
    if not token:
        raise ValidationError("token is required.")

    invite = _table(INVITES_TABLE).get_item(Key={"token": token}).get("Item")
    if not invite:
        raise NotFoundError("This invite link is invalid or has already been used.")
    if invite.get("consumed"):
        raise ValidationError("This invite has already been used.")
    if float(invite.get("expires_at", 0)) < time.time():
        raise ValidationError("This invite link has expired — ask your admin to resend it.")

    return {
        "fullName": invite["full_name"],
        "role": invite["role"],
        "className": invite.get("class_name", ""),
    }


def complete_first_login(payload, claims):
    """Unauthenticated — sets the user's real password and logs them in."""
    token = payload.get("token")
    password = payload.get("password")
    if not token or not password or len(password) < 8:
        raise ValidationError("token and a password of at least 8 characters are required.")

    invites_table = _table(INVITES_TABLE)
    invite = invites_table.get_item(Key={"token": token}).get("Item")
    if not invite or invite.get("consumed"):
        raise ValidationError("This invite link is invalid or has already been used.")
    if float(invite.get("expires_at", 0)) < time.time():
        raise ValidationError("This invite link has expired — ask your admin to resend it.")

    problem = _password_problem(str(password))
    if problem:
        raise ValidationError(problem)

    user_id = invite["user_id"]
    account = _table(ADMIN_USERS_TABLE).get_item(Key={"user_id": user_id}).get("Item") or {}
    email = str(account.get("email") or "").strip()
    if not email:
        raise ValidationError("This invite has no email address — ask your admin to resend it.")

    # Invites are the only way to get a teacher or admin account, so this creates
    # the app login account (Users, keyed by email) that /auth/login and every app
    # API read. It never overwrites an existing account (and so never changes its role).
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    item = {
        "user_id": user_id,
        "email": email,
        "name": invite.get("full_name", ""),
        "password": _hash_password_for_login(str(password)),
        "userRole": invite["role"],
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }
    # The class the admin invited a student into: students only see their own classes' work.
    if invite["role"] == "student" and invite.get("class_name"):
        item["classes"] = [invite["class_name"]]
    try:
        _table(LOGIN_USERS_TABLE).put_item(Item=item, ConditionExpression="attribute_not_exists(email)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise ConflictError("An account with this email already exists. Log in with it, or ask your admin for help.")
        raise

    _table(ADMIN_USERS_TABLE).update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET #s = :active, password_hash = :ph",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":active": "active", ":ph": _hash_password(password)},
    )
    # "consumed" is a DynamoDB reserved word: it must go through an alias, or this
    # update fails (500) after the account was created and the invite never closes.
    invites_table.update_item(
        Key={"token": token},
        UpdateExpression="SET #c = :true",
        ExpressionAttributeNames={"#c": "consumed"},
        ExpressionAttributeValues={":true": True},
    )

    return {
        # A normal login token, accepted by the app APIs and the admin console alike.
        "token": _issue_login_jwt(user_id, email),
        "userId": user_id,
        "role": invite["role"],
        "fullName": invite["full_name"],
    }


# ----------------------------------------------------------------------
# Password resets (admin_reset_requests.html) — "teachers/students cannot
# self-reset" business rule from Website Wireframes.html §07.
# ----------------------------------------------------------------------

def list_reset_requests(payload, claims):
    status = payload.get("status") or "pending"
    items = _table(RESET_REQUESTS_TABLE).scan(
        FilterExpression=Attr("status").eq(status)
    ).get("Items", [])

    requests = [{
        "id": r["request_id"],
        "userName": r.get("user_name", ""),
        "userEmail": r.get("user_email", ""),
        "role": r.get("role", ""),
        "requestedAt": float(r.get("requested_at", 0)),
        "status": r.get("status", "pending"),
    } for r in items]
    requests.sort(key=lambda r: r["requestedAt"], reverse=True)

    pending_count = None
    if status == "pending":
        pending_count = len(requests)

    return {"requests": requests, "pendingCount": pending_count}


def approve_reset_request(payload, claims):
    request_id = payload.get("requestId")
    if not request_id:
        raise ValidationError("requestId is required.")
    _table(RESET_REQUESTS_TABLE).update_item(
        Key={"request_id": request_id},
        UpdateExpression="SET #s = :approved, approved_by = :admin",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":approved": "approved", ":admin": claims["sub"]},
    )
    _write_audit_log(claims["sub"], "Approved password reset request", request_id)
    return {"success": True}


def reject_reset_request(payload, claims):
    request_id = payload.get("requestId")
    if not request_id:
        raise ValidationError("requestId is required.")
    _table(RESET_REQUESTS_TABLE).update_item(
        Key={"request_id": request_id},
        UpdateExpression="SET #s = :rejected",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":rejected": "rejected"},
    )
    _write_audit_log(claims["sub"], "Rejected password reset request", request_id)
    return {"success": True}


def generate_temp_password(payload, claims):
    request_id = payload.get("requestId")
    if not request_id:
        raise ValidationError("requestId is required.")

    requests_table = _table(RESET_REQUESTS_TABLE)
    reset_request = requests_table.get_item(Key={"request_id": request_id}).get("Item")
    if not reset_request:
        raise NotFoundError("Reset request not found.")

    temp_password = _generate_temp_password()
    if reset_request.get("source") == "login":
        # Condition guards against creating a stray item if the account was deleted.
        try:
            _table(LOGIN_USERS_TABLE).update_item(
                Key={"email": reset_request["user_email"]},
                UpdateExpression="SET password = :p, must_change_password = :true",
                ConditionExpression=Attr("user_id").eq(reset_request["user_id"]),
                ExpressionAttributeValues={":p": _hash_password_for_login(temp_password), ":true": True},
            )
        except ClientError as exc:
            if not dynamo.is_conditional_check_failed(exc):
                raise
            raise NotFoundError("That account no longer exists.") from None
    else:
        _table(ADMIN_USERS_TABLE).update_item(
            Key={"user_id": reset_request["user_id"]},
            UpdateExpression="SET password_hash = :ph, must_change_password = :true",
            ExpressionAttributeValues={":ph": _hash_password(temp_password), ":true": True},
        )
    requests_table.update_item(
        Key={"request_id": request_id},
        UpdateExpression="SET #s = :completed",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":completed": "completed"},
    )
    _write_audit_log(claims["sub"], "Generated temp password", reset_request.get("user_email", reset_request["user_id"]))
    return {"tempPassword": temp_password}


# Fixed governance rules from Website Wireframes.html §07's Permission
# Matrix. These are product-level access rules, not per-tenant config, so
# they're returned as a constant rather than read from a table — move
# this into SETTINGS_TABLE if it ever needs to become admin-editable.
_PERMISSION_MATRIX = [
    {"label": "Change own password", "admin": "yes", "teacher": "yes", "student": "yes"},
    {"label": "Reset own pw without current", "admin": "no", "teacher": "no", "student": "no"},
    {"label": "Reset another user's pw", "admin": "yes", "teacher": "no", "student": "no"},
    {"label": "Create teacher / student", "admin": "yes", "teacher": "no", "student": "no"},
    {"label": "Edit username", "admin": "yes", "teacher": "no", "student": "no"},
    {"label": "Edit own details (limited)", "admin": "yes", "teacher": "self", "student": "self"},
    {"label": "Delete / activate users", "admin": "yes", "teacher": "no", "student": "no"},
    {"label": "View audit logs", "admin": "yes", "teacher": "no", "student": "no"},
]


def get_permission_matrix(payload, claims):
    return {"features": _PERMISSION_MATRIX}


# ----------------------------------------------------------------------
# Content & Blog (admin_content.html)
# ----------------------------------------------------------------------

def list_content(payload, claims):
    content_filter = payload.get("filter") or "all"
    scan_kwargs = {}
    if content_filter != "all":
        scan_kwargs["FilterExpression"] = Attr("status").eq(content_filter)

    items = _table(CONTENT_TABLE).scan(**scan_kwargs).get("Items", [])
    posts = [{
        "id": p["post_id"],
        "title": p.get("title", "(untitled)"),
        "author": p.get("author", ""),
        "dateLabel": p.get("date_label", ""),
        "status": p.get("status", "draft"),
    } for p in items]
    posts.sort(key=lambda p: p.get("dateLabel", ""), reverse=True)
    return {"posts": posts}


def update_post_status(payload, claims):
    post_id = payload.get("postId")
    status = payload.get("status")
    if not post_id or status not in ("draft", "pending", "published"):
        raise ValidationError("postId and a valid status are required.")
    _table(CONTENT_TABLE).update_item(
        Key={"post_id": post_id},
        UpdateExpression="SET #s = :status",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":status": status},
    )
    _write_audit_log(claims["sub"], f"Set post status to {status}", post_id)
    return {"success": True}


# ----------------------------------------------------------------------
# Analytics (admin_dashboard.html + admin_analytics.html)
# ----------------------------------------------------------------------

# Populated by a separate scheduled job (EventBridge-triggered Lambda or
# a Glue/Athena rollup) that writes pre-aggregated metrics into
# GradeAnt_Analytics — computing these live from the Users/Content
# tables on every dashboard load would not scale. This fallback is only
# what a brand-new deployment sees before that job has ever run.
_ANALYTICS_FALLBACK = {
    "activeUsers": {"value": "0", "trend": "no data yet", "direction": "flat"},
    "papersGraded": {"value": "0", "trend": "no data yet", "direction": "flat"},
    "schools": {"value": "0", "trend": "no data yet", "direction": "flat"},
    "uptime": {"value": "—", "trend": "no data yet", "direction": "flat"},
    "weeklyPapers": [],
    "topSchools": [],
}


def get_analytics(payload, claims):
    metric_range = payload.get("range") or "30d"
    item = _table(ANALYTICS_TABLE).get_item(Key={"metric_key": metric_range}).get("Item")
    if not item:
        return _ANALYTICS_FALLBACK
    return json.loads(item["data_json"])


# ----------------------------------------------------------------------
# Support tickets (admin_tickets.html)
# ----------------------------------------------------------------------

def list_tickets(payload, claims):
    ticket_filter = payload.get("filter") or "open"

    if ticket_filter == "resolved":
        filter_expr = Attr("status").eq("resolved")
    elif ticket_filter == "mine":
        filter_expr = Attr("assignee_id").eq(claims["sub"])
    else:
        filter_expr = Attr("status").eq("open")

    items = _table(TICKETS_TABLE).scan(FilterExpression=filter_expr).get("Items", [])
    tickets = [{
        "id": t["ticket_id"],
        "priority": t.get("priority", "med"),
        "subject": t.get("subject", ""),
        "ageLabel": t.get("age_label", ""),
        "requester": t.get("requester_name", ""),
        "assignee": t.get("assignee_name", ""),
    } for t in items]

    open_count = None
    if ticket_filter == "open":
        open_count = len(tickets)

    return {"tickets": tickets, "openCount": open_count}


def assign_ticket(payload, claims):
    ticket_id = payload.get("ticketId")
    if not ticket_id:
        raise ValidationError("ticketId is required.")
    _table(TICKETS_TABLE).update_item(
        Key={"ticket_id": ticket_id},
        UpdateExpression="SET assignee_id = :aid, assignee_name = :aname",
        ExpressionAttributeValues={":aid": claims["sub"], ":aname": claims.get("name", "Admin")},
    )
    _write_audit_log(claims["sub"], "Self-assigned ticket", ticket_id)
    return {"success": True}


def resolve_ticket(payload, claims):
    ticket_id = payload.get("ticketId")
    if not ticket_id:
        raise ValidationError("ticketId is required.")
    _table(TICKETS_TABLE).update_item(
        Key={"ticket_id": ticket_id},
        UpdateExpression="SET #s = :resolved",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":resolved": "resolved"},
    )
    _write_audit_log(claims["sub"], "Resolved ticket", ticket_id)
    return {"success": True}


# ----------------------------------------------------------------------
# Audit logs (admin_audit_logs.html) — read-only view of _write_audit_log
# ----------------------------------------------------------------------

def list_audit_logs(payload, claims):
    search = (payload.get("search") or "").strip().lower()
    category = payload.get("category") or ""

    items = _table(AUDIT_LOGS_TABLE).scan().get("Items", [])
    entries = [{
        "timestamp": float(e.get("timestamp", 0)),
        "actor": e.get("actor", ""),
        "action": e.get("action", ""),
        "target": e.get("target", ""),
    } for e in items]

    if category:
        entries = [e for e in entries if category.lower() in e["action"].lower()]
    if search:
        entries = [
            e for e in entries
            if search in e["actor"].lower() or search in e["action"].lower() or search in e["target"].lower()
        ]

    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return {"entries": entries[:200]}


# ----------------------------------------------------------------------
# System settings (admin_settings.html)
# ----------------------------------------------------------------------

_DEFAULT_SETTINGS = {
    "platformName": "GradeAnt",
    "supportEmail": "support@gradeant.io",
    "sessionTimeoutMinutes": 30,
    "maintenanceMode": False,
    "allowSelfSignup": False,
}


def get_settings(payload, claims):
    item = _table(SETTINGS_TABLE).get_item(Key={"setting_key": "platform"}).get("Item")
    if not item:
        return _DEFAULT_SETTINGS
    return {
        "platformName": item.get("platform_name", _DEFAULT_SETTINGS["platformName"]),
        "supportEmail": item.get("support_email", _DEFAULT_SETTINGS["supportEmail"]),
        "sessionTimeoutMinutes": int(item.get("session_timeout_minutes", _DEFAULT_SETTINGS["sessionTimeoutMinutes"])),
        "maintenanceMode": bool(item.get("maintenance_mode", False)),
        "allowSelfSignup": bool(item.get("allow_self_signup", False)),
    }


def update_settings(payload, claims):
    _table(SETTINGS_TABLE).put_item(Item={
        "setting_key": "platform",
        "platform_name": payload.get("platformName", _DEFAULT_SETTINGS["platformName"]),
        "support_email": payload.get("supportEmail", _DEFAULT_SETTINGS["supportEmail"]),
        "session_timeout_minutes": Decimal(str(payload.get("sessionTimeoutMinutes", 30))),
        "maintenance_mode": bool(payload.get("maintenanceMode", False)),
        "allow_self_signup": bool(payload.get("allowSelfSignup", False)),
    })
    _write_audit_log(claims["sub"], "Updated system settings", "platform")
    return {"success": True}


# ----------------------------------------------------------------------
# Admin's own profile (admin_profile.html) — admins edit any field
# (including username) and change their own password directly, unlike
# teachers/students who must go through the reset-request workflow.
# ----------------------------------------------------------------------

def get_profile(payload, claims):
    user = _table(ADMIN_USERS_TABLE).get_item(Key={"user_id": claims["sub"]}).get("Item")
    if not user:
        raise NotFoundError("Admin profile not found.")
    return {
        "fullName": user.get("name", ""),
        "username": user.get("username", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone", ""),
        "adminLevel": user.get("admin_level", "admin"),
    }


def update_profile(payload, claims):
    _table(ADMIN_USERS_TABLE).update_item(
        Key={"user_id": claims["sub"]},
        UpdateExpression="SET #n = :name, username = :username, email = :email, phone = :phone",
        ExpressionAttributeNames={"#n": "name"},
        ExpressionAttributeValues={
            ":name": payload.get("fullName", ""),
            ":username": payload.get("username", ""),
            ":email": payload.get("email", ""),
            ":phone": payload.get("phone", ""),
        },
    )
    _write_audit_log(claims["sub"], "Updated own profile", claims["sub"])
    return {"success": True}


def change_own_password(payload, claims):
    current_password = payload.get("currentPassword")
    new_password = payload.get("newPassword")
    if not current_password or not new_password or len(new_password) < 8:
        raise ValidationError("currentPassword and a newPassword of at least 8 characters are required.")

    users_table = _table(ADMIN_USERS_TABLE)
    user = users_table.get_item(Key={"user_id": claims["sub"]}).get("Item")
    if not user or not _verify_password(current_password, user.get("password_hash", "")):
        raise ValidationError("Current password is incorrect.")

    users_table.update_item(
        Key={"user_id": claims["sub"]},
        UpdateExpression="SET password_hash = :ph, must_change_password = :false",
        ExpressionAttributeValues={":ph": _hash_password(new_password), ":false": False},
    )
    _write_audit_log(claims["sub"], "Changed own password", claims["sub"])
    return {"success": True}


# ----------------------------------------------------------------------
# Dispatch table
# ----------------------------------------------------------------------
# `requires_auth=False` marks the two operations first_login_setup.html
# calls before the user has a session at all.

OPERATIONS = {
    "listUsers": (list_users, True),
    "updateUserStatus": (update_user_status, True),
    "deleteUser": (delete_user, True),

    "inviteUsers": (invite_users, True),
    "getInviteDetails": (get_invite_details, False),
    "completeFirstLogin": (complete_first_login, False),

    "listResetRequests": (list_reset_requests, True),
    "approveResetRequest": (approve_reset_request, True),
    "rejectResetRequest": (reject_reset_request, True),
    "generateTempPassword": (generate_temp_password, True),
    "getPermissionMatrix": (get_permission_matrix, True),

    "listContent": (list_content, True),
    "updatePostStatus": (update_post_status, True),

    "getAnalytics": (get_analytics, True),

    "listTickets": (list_tickets, True),
    "assignTicket": (assign_ticket, True),
    "resolveTicket": (resolve_ticket, True),

    "listAuditLogs": (list_audit_logs, True),

    "getSettings": (get_settings, True),
    "updateSettings": (update_settings, True),

    "getProfile": (get_profile, True),
    "updateProfile": (update_profile, True),
    "changeOwnPassword": (change_own_password, True),
}


def _first_login_path():
    """app.yaml pages.firstLogin (where invite links land)."""
    return (config.app_config().get("pages") or {}).get("firstLogin", "/first-login")


def lambda_handler(event, context):
    """OPTIONS, CORS allowlist and the generic 500 come from gradeant_common.http.handle."""
    return handle(event, context, _dispatch, key="message", internal="Internal server error.")


def _dispatch(event):
    body = parse_body(event, "Request body must be valid JSON.")
    operation = body.get("operation")
    if not operation or not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValidationError(f"Unknown or missing operation: {operation!r}")
    handler, requires_auth = OPERATIONS[operation]
    claims = _require_admin(event) if requires_auth else None
    # config/app.yaml permissions.admin can turn individual console operations off.
    if requires_auth:
        require_permission("admin", operation, claims.get("role"), "This action is admin-only.")
    return create_response(200, handler(body, claims))
