"""Settings: environment variables plus config/app.yaml (URLs, API paths, roles, permissions).

Same names and defaults as the Node backend (backend/.env.example), so one set of
Lambda environment variables works for either implementation.

app.yaml lookup (first that exists): APP_CONFIG_FILE, <layer>/config/app.yaml
(packaged copy), <repo>/config/app.yaml (source checkout).
"""
import functools
import os

import yaml

ROLES = ('student', 'teacher', 'admin')
SIGNUP_MODES = ('self', 'invite')

_S, _T, _A = 'student', 'teacher', 'admin'
_TEACHERS = [_T, _A]
_EVERYONE = [_S, _T, _A]

# The most each operation's code supports: app.yaml may narrow these, never widen them
# (same table as backend/src/config/appConfig.js SUPPORTED).
SUPPORTED = {
    'grades': {'getSingleStudentGrade': _EVERYONE, 'getAllStudentGrades': _TEACHERS, 'updateStudentGrade': _TEACHERS},
    'corrections': {
        'getAnswersForCorrection': _EVERYONE,
        'submitAnswerCorrections': [_S],
        'getQuestionsForCorrection': _TEACHERS,
        'submitQuestionCorrections': _TEACHERS,
    },
    'marks': {'updateMarks': _TEACHERS, 'updateStudentGrade': _TEACHERS},
    'upload': {'student': _EVERYONE, 'teacher': _TEACHERS, 'answer': _TEACHERS},
    'workflow': {
        **{op: _TEACHERS for op in (
            'listAssignments', 'getAssignment', 'createAssignment', 'getParsedQuestions', 'confirmQuestions',
            'setGradingMethod', 'getRubric', 'approveRubric', 'startGrading', 'getGradingRun', 'listSubmissions',
            'listFlagged', 'acceptFlagged', 'publishGrades', 'remindStudent', 'askAssistant', 'getGradingQueue',
            'listStudents', 'getTeacherAnalytics', 'getTeacherSettings', 'updateTeacherSettings',
            'postAnnouncement', 'deleteAnnouncement')},
        'listStudentAssignments': [_S],
        'requestRecheck': [_S],
        'getProgress': [_S],
        'listAnnouncements': _EVERYONE,
    },
    'admin': {op: [_A] for op in (
        'listUsers', 'updateUserStatus', 'deleteUser', 'inviteUsers', 'listResetRequests', 'approveResetRequest',
        'rejectResetRequest', 'generateTempPassword', 'getPermissionMatrix', 'listContent', 'updatePostStatus',
        'getAnalytics', 'listTickets', 'assignTicket', 'resolveTicket', 'listAuditLogs', 'getSettings',
        'updateSettings', 'getProfile', 'updateProfile', 'changeOwnPassword')},
}


def env(name, default=''):
    value = (os.environ.get(name) or '').strip()
    return value or default


def env_int(name, default):
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def env_float(name, default):
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default


def mode():
    """production on Lambda (or APP_ENV=production); development otherwise."""
    return env('APP_ENV') or ('production' if os.environ.get('AWS_LAMBDA_FUNCTION_NAME') else 'development')


def is_production():
    return mode() == 'production'


def data_region():
    """Region of the tables; AWS_REGION on Lambda is the function's own Region."""
    return env('AWS_DATA_REGION') or env('AWS_REGION') or env('AWS_DEFAULT_REGION') or 'us-east-1'


# DynamoDB table names (env var → default), as in backend/.env.example.
TABLE_ENV = {
    'users': ('USERS_TABLE', 'Users'),
    'grading': ('GRADING_TABLE', 'assignment_grading'),
    'assignments': ('ASSIGNMENTS_TABLE', 'GradeAnt_Assignments'),
    'submissions': ('SUBMISSIONS_TABLE', 'GradeAnt_Submissions'),
    'announcements': ('ANNOUNCEMENTS_TABLE', 'GradeAnt_Announcements'),
    'admin_users': ('ADMIN_USERS_TABLE', 'GradeAnt_Users'),
    'reset_requests': ('RESET_REQUESTS_TABLE', 'GradeAnt_ResetRequests'),
    'content': ('CONTENT_TABLE', 'GradeAnt_Content'),
    'tickets': ('TICKETS_TABLE', 'GradeAnt_Tickets'),
    'audit_logs': ('AUDIT_LOGS_TABLE', 'GradeAnt_AuditLogs'),
    'settings': ('SETTINGS_TABLE', 'GradeAnt_Settings'),
    'invites': ('INVITES_TABLE', 'GradeAnt_Invites'),
    'analytics': ('ANALYTICS_TABLE', 'GradeAnt_Analytics'),
}


def table_name(key):
    name, default = TABLE_ENV[key]
    return env(name, default)


def grading_assignment_index():
    return env('GRADING_ASSIGNMENT_INDEX', 'assignment_id-index')


# ---- config/app.yaml --------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))


def _config_file():
    candidates = [
        env('APP_CONFIG_FILE'),
        os.path.join(_HERE, 'config', 'app.yaml'),                       # packaged in the layer
        os.path.join(_HERE, '..', '..', 'config', 'app.yaml'),           # repo: python-backend/../config
    ]
    for f in candidates:
        if f and os.path.isfile(f):
            return f
    raise RuntimeError(f"config/app.yaml not found (looked in: {', '.join(c for c in candidates if c)}). Set APP_CONFIG_FILE.")


def _validate(doc, current_mode):
    """Same checks as validateAppConfig in backend/src/config/appConfig.js (roles and permissions)."""
    if not isinstance(doc, dict):
        raise RuntimeError('config/app.yaml: the file is empty or not a YAML mapping')
    problems = []
    if not isinstance((doc.get('urls') or {}).get(current_mode), dict):
        problems.append(f'urls.{current_mode} is missing')

    roles = doc.get('roles') if isinstance(doc.get('roles'), dict) else {}
    for name in roles:
        if name not in ROLES:
            problems.append(f'roles.{name}: unknown role (the roles are {", ".join(ROLES)})')
    for name in ROLES:
        role = roles.get(name)
        if not isinstance(role, dict):
            problems.append(f'roles.{name} is missing')
            continue
        if not isinstance(role.get('label'), str) or not role['label'].strip():
            problems.append(f'roles.{name}.label is required')
        signup = role.get('signup')
        if not isinstance(signup, list) or any(m not in SIGNUP_MODES for m in signup):
            problems.append(f'roles.{name}.signup must be a list of: {", ".join(SIGNUP_MODES)}')
        elif name == 'admin' and 'self' in signup:
            problems.append('roles.admin.signup can\'t include "self": anyone could make themselves an admin')

    perms = doc.get('permissions') if isinstance(doc.get('permissions'), dict) else {}
    for area in perms:
        if area not in SUPPORTED:
            problems.append(f'permissions.{area}: unknown section (expected {", ".join(SUPPORTED)})')
    for area, ops in SUPPORTED.items():
        given = perms.get(area) if isinstance(perms.get(area), dict) else {}
        for op in given:
            if op not in ops:
                problems.append(f'permissions.{area}.{op}: unknown operation')
        for op, supported in ops.items():
            listed = given.get(op)
            if listed is None:
                problems.append(f'permissions.{area}.{op} is missing (use [] to turn it off)')
            elif not isinstance(listed, list) or any(r not in ROLES for r in listed):
                problems.append(f'permissions.{area}.{op} must be a list of roles ({", ".join(ROLES)})')
            else:
                extra = [r for r in listed if r not in supported]
                if extra:
                    problems.append(f'permissions.{area}.{op}: {", ".join(extra)} can\'t be given this (supported: {", ".join(supported) or "none"})')

    _validate_deployments(doc, problems)

    if problems:
        raise RuntimeError(f'Invalid config/app.yaml ({current_mode}):\n  - ' + '\n  - '.join(problems))
    return doc


# ---- deployment switch (same rules as backend/src/config/appConfig.js) -------------

PROVIDERS = {
    'database': ('dynamodb', 'supabase'),
    'ai': ('gemini', 'off'),
    'email': ('ses', 'log'),
    'secrets': ('aws', 'env'),
    'grading': ('lambda', 'sqs'),
}
DEFAULT_DEPLOYMENT = 'aws'
AWS_PROVIDERS = {'database': 'dynamodb', 'ai': 'gemini', 'email': 'ses', 'secrets': 'aws', 'grading': 'lambda'}


def _validate_deployments(doc, problems):
    """`deployment` / `deployments`: every target is checked, not only the selected one."""
    name = doc.get('deployment', DEFAULT_DEPLOYMENT)
    targets = doc.get('deployments', {})
    if not isinstance(targets, dict):
        problems.append('deployments must be a mapping of target name → settings')
        targets = {}
    for key, target in targets.items():
        if not isinstance(target, dict):
            problems.append(f'deployments.{key} must be a mapping')
            continue
        given = target.get('providers', {})
        if not isinstance(given, dict):
            problems.append(f'deployments.{key}.providers must be a mapping')
            continue
        for provider, value in given.items():
            if provider not in PROVIDERS:
                problems.append(f'deployments.{key}.providers.{provider}: unknown provider (expected {", ".join(PROVIDERS)})')
            elif value not in PROVIDERS[provider]:
                problems.append(f'deployments.{key}.providers.{provider} must be one of {", ".join(PROVIDERS[provider])} (got "{value}")')
    if not isinstance(name, str) or not name:
        problems.append('deployment must be the name of one of the targets under deployments')
    elif name not in targets and not (name == DEFAULT_DEPLOYMENT and not targets):
        problems.append(f'deployment: "{name}" is not defined under deployments (defined: {", ".join(targets) or "none"})')


def providers():
    """The selected deployment target's providers (a file without the section means aws)."""
    doc = app_config()
    target = (doc.get('deployments') or {}).get(doc.get('deployment', DEFAULT_DEPLOYMENT)) or {}
    return {**AWS_PROVIDERS, **(target.get('providers') or {})}


def database():
    """'dynamodb' or 'supabase' for the grading table. DATABASE_PROVIDER wins (set it on the grader
    Lambda to switch without rebuilding the layer); otherwise app.yaml in production; else dynamodb."""
    override = env('DATABASE_PROVIDER')
    if override:
        if override not in PROVIDERS['database']:
            raise RuntimeError(f'DATABASE_PROVIDER must be one of {", ".join(PROVIDERS["database"])} (got "{override}")')
        return override
    return providers()['database'] if is_production() else 'dynamodb'


@functools.lru_cache(maxsize=None)
def app_config():
    """The parsed, validated app.yaml (cached per execution environment)."""
    path = _config_file()
    with open(path, encoding='utf-8') as fh:
        return _validate(yaml.safe_load(fh), mode())


def cors_origins():
    """CORS_ALLOWED_ORIGINS ("*" or a comma list) overrides app.yaml urls.<mode>.corsOrigins."""
    override = [o.strip() for o in env('CORS_ALLOWED_ORIGINS').split(',') if o.strip()]
    if override:
        return override
    return list(((app_config().get('urls') or {}).get(mode()) or {}).get('corsOrigins') or [])


def app_base_url():
    """The web app's URL for links in emails: APP_BASE_URL or app.yaml urls.<mode>.web."""
    return env('APP_BASE_URL', ((app_config().get('urls') or {}).get(mode()) or {}).get('web', '')).rstrip('/')


def api_paths():
    return app_config().get('api') or {}
