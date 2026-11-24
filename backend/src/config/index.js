// Single source of configuration. Picks the environment, loads the matching
// .env file, and exposes a typed config object. Nothing else in the
// app reads process.env directly.
//
// Environment selection (first match wins):
//   1. --mode=<name> on the command line   (npm run dev / npm run start:prod)
//   2. APP_ENV
//   3. AWS_LAMBDA_HTTP_ENDPOINT is set      → production (Lambda Web Functions)
//   4. development
//
// Files: .env.<mode> (committed, no secrets) then .env.<mode>.local
// (gitignored, for personal overrides and local secrets). A variable already in
// process.env always wins, so Lambda revision variables override the files.
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseEnv } from 'node:util'
import { loadAppConfig } from './appConfig.js'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')

function pickMode() {
  const arg = process.argv.find((a) => a.startsWith('--mode='))
  if (arg) return arg.slice('--mode='.length)
  if (process.env.APP_ENV) return process.env.APP_ENV
  if (process.env.AWS_LAMBDA_HTTP_ENDPOINT) return 'production'
  return 'development'
}

function loadEnvFiles(mode) {
  const loaded = []
  // Precedence: real environment variables › .env.<mode>.local › .env.<mode>.
  // Only variables that were set before any file was read count as "real", so
  // the .local file can override the committed one.
  const fromEnvironment = new Set(Object.keys(process.env))
  for (const name of [`.env.${mode}`, `.env.${mode}.local`]) {
    const file = path.join(ROOT, name)
    if (!existsSync(file)) continue
    const values = parseEnv(readFileSync(file, 'utf8'))
    for (const [key, value] of Object.entries(values)) {
      if (!fromEnvironment.has(key)) process.env[key] = value
    }
    loaded.push(name)
  }
  return loaded
}

const mode = pickMode()
const envFiles = loadEnvFiles(mode)
const env = process.env
// config/app.yaml: URLs, API paths, roles and permissions (see appConfig.js).
const app = loadAppConfig(ROOT, mode, env)

const str = (name, fallback = '') => (env[name] ?? '').trim() || fallback
const int = (name, fallback) => {
  const n = Number.parseInt(env[name] ?? '', 10)
  return Number.isFinite(n) ? n : fallback
}
const float = (name, fallback) => {
  const n = Number.parseFloat(env[name] ?? '')
  return Number.isFinite(n) ? n : fallback
}
const bool = (name, fallback = false) => {
  const v = (env[name] ?? '').trim().toLowerCase()
  if (!v) return fallback
  return ['1', 'true', 'yes', 'on'].includes(v)
}
const list = (name) =>
  str(name)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
const oneOf = (name, allowed, fallback) => {
  const v = str(name, fallback)
  if (!allowed.includes(v)) throw new Error(`${name} must be one of ${allowed.join(', ')} (got "${v}")`)
  return v
}

// Driver choice, first match wins:
//   1. an explicit <NAME>_DRIVER environment variable (tests, local overrides)
//   2. in production: config/app.yaml deployments.<deployment>.providers
//   3. the development default
// So switching AWS ↔ Supabase is the one `deployment:` line in app.yaml.
const isProductionMode = mode === 'production'
const providers = app.deployment.providers
const PROVIDER_DRIVERS = {
  db: { database: { dynamodb: 'dynamodb', supabase: 'postgres' } },
  ai: { ai: { gemini: 'gemini', off: 'canned' } },
  mail: { email: { ses: 'ses', log: 'log' } },
  grading: { grading: { lambda: 'lambda', sqs: 'sqs' } },
}
function driver(name, kind, allowed, devDefault) {
  const [[provider, mapping]] = Object.entries(PROVIDER_DRIVERS[kind] ?? { _: {} })
  const fromYaml = isProductionMode && mapping ? mapping[providers[provider]] : undefined
  return oneOf(name, allowed, fromYaml ?? devDefault)
}
// Drivers that the config accepts but whose code arrives in a later step of
// backend/FREE_TIER_HYBRID_PLAN.md. Selecting one fails at startup, never half-way through a request.
const NOT_YET = {} // every provider value is implemented (kept for future providers)

// Lambda sets AWS_LAMBDA_HTTP_ENDPOINT ("host:port"); locally HOST/PORT apply.
function listenAddress() {
  const endpoint = env.AWS_LAMBDA_HTTP_ENDPOINT
  if (endpoint) {
    const i = endpoint.lastIndexOf(':')
    return { host: endpoint.slice(0, i), port: Number(endpoint.slice(i + 1)) }
  }
  return { host: str('HOST', '127.0.0.1'), port: int('PORT', 8787) }
}

const isProduction = mode === 'production'
// AWS_REGION is reserved on Lambda Web Functions and is always the function's
// own Region. The data can live elsewhere (e.g. tables in eu-north-1 while Web
// Functions run in eu-west-1), so the data Region is configured separately.
const dataRegion = str('AWS_DATA_REGION') || str('AWS_REGION') || str('AWS_DEFAULT_REGION') || 'us-east-1'

const config = {
  mode,
  isProduction,
  envFiles,
  root: ROOT,

  server: {
    ...listenAddress(),
    // Mount the API under a prefix (e.g. "/dev") when a front door adds one.
    basePath: str('API_BASE_PATH', '/').replace(/\/+$/, '') || '',
    // Used to build links the browser follows back to this server (local storage URLs).
    publicUrl: str('PUBLIC_BASE_URL'),
    bodyLimit: str('BODY_LIMIT', '1mb'),
    logRequests: bool('LOG_REQUESTS', true),
  },

  cors: {
    // app.yaml urls.<mode>.corsOrigins; CORS_ALLOWED_ORIGINS ("*" or a comma-separated list) overrides it.
    origins: list('CORS_ALLOWED_ORIGINS').length ? list('CORS_ALLOWED_ORIGINS') : app.urls.corsOrigins,
  },

  // From config/app.yaml: API paths, web pages, roles (how each can join) and
  // permissions (which roles may run each operation; see lib/permissions.js).
  app: {
    file: app.file,
    paths: app.api,
    pages: app.pages,
    roles: app.roles,
    permissions: app.permissions,
  },

  // The selected config/app.yaml deployment target and its providers (see driver() above).
  deployment: { name: app.deployment.name, providers: { ...providers }, warnings: app.deployment.warnings },

  drivers: {
    // memory (dynalite) and pglite (in-process Postgres) are for local runs and tests only.
    db: driver('DB_DRIVER', 'db', ['memory', 'dynamodb', 'postgres', 'pglite'], 'memory'),
    storage: oneOf('STORAGE_DRIVER', ['local', 's3'], isProduction ? 's3' : 'local'),
    mail: driver('MAIL_DRIVER', 'mail', ['log', 'ses'], 'log'),
    notify: oneOf('NOTIFY_DRIVER', ['log', 'sns'], isProduction ? 'sns' : 'log'),
    jobs: oneOf('JOBS_DRIVER', ['inline', 'sqs'], 'inline'),
    ai: driver('AI_DRIVER', 'ai', ['canned', 'gemini'], 'canned'),
    grading: driver('GRADING_DRIVER', 'grading', ['log', 'lambda', 'sqs'], 'log'),
    // aws: an empty JWT_SECRET falls back to JWT_SECRET_ARN (Secrets Manager); env: environment only.
    secrets: isProduction ? providers.secrets : 'aws',
  },

  aws: {
    region: dataRegion,
    // Point DynamoDB at DynamoDB Local (e.g. http://localhost:8000) when DB_DRIVER=dynamodb.
    dynamodbEndpoint: str('DYNAMODB_ENDPOINT'),
    // Seed demo data into the database on startup (in-memory DB always seeds).
    seedDemoData: bool('SEED_DEMO_DATA', false),
  },

  // DB_DRIVER=postgres (deployment providers.database: supabase).
  postgres: {
    // Supabase → Connect → Session pooler URI (IPv4, port 5432), with the database password.
    url: str('SUPABASE_DB_URL') || str('DATABASE_URL'),
    poolMax: int('PG_POOL_MAX', 5),
    // Supabase's CA certificate (PEM text) to verify the server; without it TLS is unverified.
    sslCa: str('PG_SSL_CA').replace(/\\n/g, '\n'),
  },

  tables: {
    users: str('USERS_TABLE', 'Users'),
    grading: str('GRADING_TABLE', 'assignment_grading'),
    gradingAssignmentIndex: str('GRADING_ASSIGNMENT_INDEX', 'assignment_id-index'),
    assignments: str('ASSIGNMENTS_TABLE', 'GradeAnt_Assignments'),
    submissions: str('SUBMISSIONS_TABLE', 'GradeAnt_Submissions'),
    announcements: str('ANNOUNCEMENTS_TABLE', 'GradeAnt_Announcements'),
    adminUsers: str('ADMIN_USERS_TABLE', 'GradeAnt_Users'),
    resetRequests: str('RESET_REQUESTS_TABLE', 'GradeAnt_ResetRequests'),
    content: str('CONTENT_TABLE', 'GradeAnt_Content'),
    tickets: str('TICKETS_TABLE', 'GradeAnt_Tickets'),
    auditLogs: str('AUDIT_LOGS_TABLE', 'GradeAnt_AuditLogs'),
    settings: str('SETTINGS_TABLE', 'GradeAnt_Settings'),
    invites: str('INVITES_TABLE', 'GradeAnt_Invites'),
    analytics: str('ANALYTICS_TABLE', 'GradeAnt_Analytics'),
  },

  auth: {
    // Either the value itself (local only) or a Secrets Manager secret ARN/name.
    jwtSecret: str('JWT_SECRET'),
    jwtSecretArn: str('JWT_SECRET_ARN'),
    tokenExpirySeconds: int('TOKEN_EXPIRY_SECONDS', 86400),
    // Admin first-login session tokens (lambda_admin.py's own HMAC format).
    sessionTokenSecret: str('SESSION_TOKEN_SECRET'),
    sessionTokenSecretArn: str('SESSION_TOKEN_SECRET_ARN'),
    sessionTokenTtlSeconds: int('SESSION_TOKEN_TTL_SECONDS', 24 * 60 * 60),
    inviteTokenTtlSeconds: int('INVITE_TOKEN_TTL_SECONDS', 7 * 24 * 60 * 60),
  },

  storage: {
    profilePhotoBucket: str('PROFILE_PHOTO_BUCKET'),
    // Bucket the /uploadpdf presigner writes to. Empty disables /uploadpdf in production.
    uploadBucket: str('UPLOAD_BUCKET'),
    // {uploadType} {assignmentId} {qpId} {studentId} {courseId} {fileName} {uuid}
    uploadKeyTemplate: str('UPLOAD_KEY_TEMPLATE', 'uploads/{uploadType}/{assignmentId}/{fileName}'),
    // template: keys from UPLOAD_KEY_TEMPLATE (as before); pipeline: the grading pipeline's own
    // layout and server-chosen names (python-backend/functions/upload.py), UPLOAD_BUCKET = PIPELINE_BUCKET.
    uploadLayout: str('UPLOAD_LAYOUT', 'template'),
    pipelineBucket: str('PIPELINE_BUCKET'),
    pipelineRoot: str('PIPELINE_ROOT', 'Data_warehouse'),
    uploadMaxBytes: int('UPLOAD_MAX_BYTES', 10 * 1024 * 1024),
    uploadUrlTtlSeconds: int('UPLOAD_URL_TTL_SECONDS', 900),
    localDir: path.resolve(ROOT, str('LOCAL_STORAGE_DIR', '.data/storage')),
  },

  mail: {
    sesSenderEmail: str('SES_SENDER_EMAIL'),
    // lambda_admin.py always had a sender default for invites.
    inviteSenderEmail: str('INVITE_SENDER_EMAIL') || str('SES_SENDER_EMAIL') || 'no-reply@gradeant.io',
    // The web app's URL, for links in emails: app.yaml urls.<mode>.web, or APP_BASE_URL.
    appBaseUrl: str('APP_BASE_URL', app.urls.web).replace(/\/+$/, ''),
    // Where the public contact and request-a-pilot forms are delivered.
    contactInbox: str('CONTACT_INBOX', 'contact@gradeant.com'),
    // From address for those messages (SES-verified); the visitor goes in Reply-To.
    contactSenderEmail: str('CONTACT_SENDER_EMAIL') || str('SES_SENDER_EMAIL') || str('INVITE_SENDER_EMAIL') || 'no-reply@gradeant.io',
  },

  notify: {
    snsTopicArn: str('SNS_TOPIC_ARN'),
  },

  jobs: {
    queueUrl: str('JOBS_QUEUE_URL'),
  },

  grading: {
    functionName: str('GRADING_FUNCTION_NAME'),
    // GRADING_DRIVER=sqs: the grader's queue (it also receives the S3 events; one DLQ for both).
    queueUrl: str('GRADING_QUEUE_URL'),
    flagConfidence: float('FLAG_CONFIDENCE', 0.75),
  },

  // AI_DRIVER=gemini: the one model for every AI call in GradeAnt (the grading pipeline uses
  // the same GEMINI_MODEL). The key: GEMINI_API_KEY, or GEMINI_API_KEY_ARN (Secrets Manager).
  gemini: {
    apiKey: str('GEMINI_API_KEY'),
    apiKeyArn: str('GEMINI_API_KEY_ARN'),
    model: str('GEMINI_MODEL', 'gemini-3.5-flash-preview'),
  },
}

function validate(c) {
  const problems = []
  for (const [kind, pending] of Object.entries(NOT_YET)) {
    const step = pending[c.drivers[kind]]
    if (step) problems.push(`${kind} driver "${c.drivers[kind]}" is not available yet (arrives in ${step} of backend/FREE_TIER_HYBRID_PLAN.md)`)
  }
  if (c.isProduction) {
    if (c.drivers.secrets === 'env') {
      if (!c.auth.jwtSecret) problems.push('JWT_SECRET is required (deployment providers.secrets: env reads secrets from the environment only)')
    } else if (!c.auth.jwtSecret && !c.auth.jwtSecretArn) problems.push('JWT_SECRET_ARN (or JWT_SECRET) is required')
    if (c.cors.origins.length === 0) problems.push('CORS origins are required (app.yaml urls.production.corsOrigins)')
    if (c.drivers.db === 'memory' || c.drivers.db === 'pglite') problems.push(`DB_DRIVER=${c.drivers.db} is for local development only`)
    if (c.drivers.db === 'postgres' && !c.postgres.url) problems.push('SUPABASE_DB_URL is required for the Supabase database (Supabase → Connect → Session pooler)')
    if (c.drivers.jobs === 'sqs' && !c.jobs.queueUrl) problems.push('JOBS_DRIVER=sqs needs JOBS_QUEUE_URL')
    if (c.drivers.ai === 'gemini' && !c.gemini.apiKey && !(c.drivers.secrets === 'aws' && c.gemini.apiKeyArn)) {
      problems.push(`GEMINI_API_KEY${c.drivers.secrets === 'aws' ? ' (or GEMINI_API_KEY_ARN)' : ''} is required for the Gemini AI driver (or set deployment providers.ai: off)`)
    }
    if (c.drivers.grading === 'sqs' && !c.grading.queueUrl) problems.push('GRADING_QUEUE_URL is required for grading: sqs (the grader\'s SQS queue URL)')
  }
  if (problems.length) throw new Error(`Invalid ${c.mode} configuration:\n  - ${problems.join('\n  - ')}`)
}

validate(config)

export default config
