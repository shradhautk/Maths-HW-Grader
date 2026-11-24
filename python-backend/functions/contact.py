"""POST /contact — Lambda gradeant-contact (handler contact.lambda_handler).

The website's contact and request-a-pilot forms (public, no token). Each message is
checked, rate-limited per client and emailed to the team inbox (CONTACT_INBOX) with
the visitor as Reply-To. Errors are {"message": ...}.
Same behaviour as backend/src/services/contact.service.js.
"""
import re
import time

from gradeant_common import config
from gradeant_common.http import ApiError, create_response, handle, header, log, parse_body
from gradeant_common.mail import send_email

EMAIL = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
KINDS = {
    'message': {'label': 'Website message', 'fields': ['subject', 'message'], 'required': ['message']},
    'pilot': {'label': 'Pilot request', 'fields': ['school', 'subject', 'students', 'message'], 'required': ['school', 'subject', 'students']},
}
LIMITS = {'name': 120, 'email': 200, 'subject': 120, 'school': 200, 'students': 20, 'message': 5000}

# Per-execution-environment limiter: stops bursts, not a determined sender.
WINDOW_SECONDS = 10 * 60
MAX_PER_WINDOW = 5
_recent = {}


def _rate_limit(client):
    now = time.time()
    hits = [t for t in _recent.get(client, []) if now - t < WINDOW_SECONDS]
    if len(hits) >= MAX_PER_WINDOW:
        raise ApiError(429, 'Too many messages. Please try again in a few minutes, or email us directly.')
    hits.append(now)
    _recent[client] = hits
    if len(_recent) > 5000:
        _recent.clear()


def _clean(value, limit):
    text = '' if value is None else str(value) if not isinstance(value, bool) else str(value).lower()
    return text.replace('\r\n', '\n').replace('\r', '\n').strip()[:limit]


def _one_line(value):
    return re.sub(r'\s+', ' ', value)


def contact_inbox():
    return config.env('CONTACT_INBOX', 'contact@gradeant.com')


def contact_sender():
    return config.env('CONTACT_SENDER_EMAIL') or config.env('SES_SENDER_EMAIL') or config.env('INVITE_SENDER_EMAIL') or 'no-reply@gradeant.io'


def submit(body, client='unknown'):
    """Validates and sends one form submission → {'message': ...}."""
    # Honeypot: a hidden field people never fill in. Pretend success so bots move on.
    if _clean(body.get('website'), 200):
        return {'message': 'Thanks, your message has been sent.'}

    kind = body.get('kind') if body.get('kind') in KINDS else 'message'
    spec = KINDS[kind]
    f = {k: _clean(body.get(k), limit) for k, limit in LIMITS.items()}

    if not f['name']:
        raise ApiError(400, 'Please enter your name.')
    if not EMAIL.match(f['email']):
        raise ApiError(400, 'Please enter a valid email address.')
    for k in spec['required']:
        if not f[k]:
            raise ApiError(400, f"Please fill in {'the number of students' if k == 'students' else f'the {k}'}.")
    if kind == 'pilot' and not re.fullmatch(r'\d{1,6}', f['students']):
        raise ApiError(400, 'Number of students must be a whole number.')

    _rate_limit(client)

    name = _one_line(f['name'])
    topic = f['school'] if kind == 'pilot' else (f['subject'] or 'General question')
    subject = f"[{spec['label']}] {_one_line(topic)} — {name}"[:200]
    lines = [
        f"{spec['label']} from the GradeAnt website",
        '',
        f'Name: {name}',
        f"Email: {f['email']}",
        *[f'{k[0].upper()}{k[1:]}: {_one_line(f[k])}' for k in spec['fields'] if k != 'message' and f[k]],
        '',
        f['message'] or '(no message)',
    ]
    try:
        send_email(sender=contact_sender(), to=contact_inbox(), reply_to=f['email'], subject=subject, text='\n'.join(lines))
    except Exception as e:  # noqa: BLE001
        log('error', 'contact form email failed', kind=kind, error=type(e).__name__, detail=str(e))
        raise ApiError(502, f"We couldn't send your message. Please email us at {contact_inbox()}.") from None
    if kind == 'pilot':
        return {'message': "Thanks, we've got your pilot request and will be in touch within two working days."}
    return {'message': 'Thanks, your message has been sent.'}


def _client_ip(event):
    forwarded = header(event, 'x-forwarded-for')
    if forwarded:
        return forwarded.split(',')[0].strip()
    ctx = event.get('requestContext') or {}
    return (ctx.get('identity') or {}).get('sourceIp') or (ctx.get('http') or {}).get('sourceIp') or 'unknown'


def _route(event):
    if event.get('path') != config.api_paths().get('contact', '/contact') or (event.get('httpMethod') or 'POST') != 'POST':
        return create_response(404, {'message': 'Not Found'})
    return create_response(200, submit(parse_body(event), _client_ip(event)))


def lambda_handler(event, context):
    return handle(event, context, _route, key='message', internal='Could not send your message.')

