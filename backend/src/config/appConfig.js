// Loads and checks config/app.yaml, the main config file for URLs, API paths,
// roles and permissions (see that file for what each section means).
//
// File lookup (first that exists):
//   1. APP_CONFIG_FILE
//   2. <backend>/config/app.yaml     (inside the deployment package; scripts/package.mjs copies it)
//   3. <repo>/config/app.yaml        (the source checkout)
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { parse } from 'yaml'

export const ROLES = ['student', 'teacher', 'admin']
export const SIGNUP_MODES = ['self', 'invite']

const S = 'student'
const T = 'teacher'
const A = 'admin'
const TEACHERS = [T, A]
const EVERYONE = [S, T, A]

/**
 * The most each operation's code supports. Its data rules (own grades only,
 * own classes only, ...) are written for exactly these roles, so app.yaml may
 * narrow these lists but never widen them.
 */
export const SUPPORTED = {
  grades: { getSingleStudentGrade: EVERYONE, getAllStudentGrades: TEACHERS, updateStudentGrade: TEACHERS },
  corrections: {
    getAnswersForCorrection: EVERYONE,
    submitAnswerCorrections: [S],
    getQuestionsForCorrection: TEACHERS,
    submitQuestionCorrections: TEACHERS,
  },
  marks: { updateMarks: TEACHERS, updateStudentGrade: TEACHERS },
  upload: { student: EVERYONE, teacher: TEACHERS, answer: TEACHERS },
  workflow: {
    ...Object.fromEntries(
      [
        'listAssignments', 'getAssignment', 'createAssignment', 'getParsedQuestions', 'confirmQuestions',
        'setGradingMethod', 'getRubric', 'approveRubric', 'startGrading', 'getGradingRun', 'listSubmissions',
        'listFlagged', 'acceptFlagged', 'publishGrades', 'remindStudent', 'askAssistant', 'getGradingQueue',
        'listStudents', 'getTeacherAnalytics', 'getTeacherSettings', 'updateTeacherSettings',
        'postAnnouncement', 'deleteAnnouncement',
      ].map((op) => [op, TEACHERS]),
    ),
    listStudentAssignments: [S],
    requestRecheck: [S],
    getProgress: [S],
    listAnnouncements: EVERYONE,
  },
  admin: Object.fromEntries(
    [
      'listUsers', 'updateUserStatus', 'deleteUser', 'inviteUsers', 'listResetRequests', 'approveResetRequest',
      'rejectResetRequest', 'generateTempPassword', 'getPermissionMatrix', 'listContent', 'updatePostStatus',
      'getAnalytics', 'listTickets', 'assignTicket', 'resolveTicket', 'listAuditLogs', 'getSettings',
      'updateSettings', 'getProfile', 'updateProfile', 'changeOwnPassword',
    ].map((op) => [op, [A]]),
  ),
}

/**
 * Deployment targets (`deployment:` + `deployments:` in app.yaml). Each provider picks
 * the backing service; the values the code supports are listed here. A file without a
 * `deployment` line means `aws` with these providers, which is how GradeAnt ran before
 * the switch existed, so older app.yaml files behave exactly as before.
 */
export const PROVIDERS = {
  database: ['dynamodb', 'supabase'],
  ai: ['gemini', 'off'],
  email: ['ses', 'log'],
  secrets: ['aws', 'env'],
  grading: ['lambda', 'sqs'],
}
export const DEFAULT_DEPLOYMENT = 'aws'
export const AWS_PROVIDERS = { database: 'dynamodb', ai: 'gemini', email: 'ses', secrets: 'aws', grading: 'lambda' }
const URL_OVERRIDES = ['web', 'api', 'upload', 'corsOrigins']

const API_PATHS = ['health', 'healthReady', 'upload', 'grades', 'corrections', 'marks', 'workflow', 'admin', 'contact']
const AUTH_PATHS = ['base', 'signup', 'login', 'logout', 'forgotPassword', 'changePassword', 'me', 'profilePhoto']

export function findAppConfigFile(backendRoot, env = process.env) {
  const candidates = [
    env.APP_CONFIG_FILE && path.resolve(env.APP_CONFIG_FILE),
    path.join(backendRoot, 'config', 'app.yaml'),
    path.join(backendRoot, '..', 'config', 'app.yaml'),
  ].filter(Boolean)
  const file = candidates.find((f) => existsSync(f))
  if (!file) throw new Error(`config/app.yaml not found (looked in: ${candidates.join(', ')}). Set APP_CONFIG_FILE to its path.`)
  return file
}

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v)
const isPath = (v) => typeof v === 'string' && /^\/[\w\-/]*$/.test(v)
const isUrl = (v) => typeof v === 'string' && /^https?:\/\/[^\s]+$/.test(v)

/**
 * `deployment` / `deployments` → { name, providers, urls (overrides), warnings }.
 * Pushes every problem into `problems`.
 */
function readDeployment(doc, problems) {
  const name = doc.deployment ?? DEFAULT_DEPLOYMENT
  const all = doc.deployments
  if (all !== undefined && !isObject(all)) problems.push('deployments must be a mapping of target name → settings')
  const targets = isObject(all) ? all : {}
  // Every target is checked, not only the selected one: a typo in the other target
  // should stop startup today, not on the day someone switches to it.
  const parsed = Object.fromEntries(Object.entries(targets).map(([key, target]) => [key, readTarget(key, target, problems)]))
  if (typeof name !== 'string' || !name) {
    problems.push('deployment must be the name of one of the targets under deployments')
    return { name: DEFAULT_DEPLOYMENT, providers: { ...AWS_PROVIDERS }, urls: {}, warnings: [] }
  }
  if (!parsed[name] && !(name === DEFAULT_DEPLOYMENT && !Object.keys(targets).length)) {
    problems.push(`deployment: "${name}" is not defined under deployments (defined: ${Object.keys(targets).join(', ') || 'none'})`)
  }
  return { name, ...(parsed[name] ?? { providers: { ...AWS_PROVIDERS }, urls: {}, warnings: [] }) }
}

function readTarget(name, target, problems) {
  if (!isObject(target)) problems.push(`deployments.${name} must be a mapping`)
  const settings = isObject(target) ? target : {}

  const providers = { ...AWS_PROVIDERS }
  const given = settings.providers === undefined ? {} : settings.providers
  if (!isObject(given)) problems.push(`deployments.${name}.providers must be a mapping`)
  for (const [key, value] of Object.entries(isObject(given) ? given : {})) {
    if (!PROVIDERS[key]) {
      problems.push(`deployments.${name}.providers.${key}: unknown provider (expected ${Object.keys(PROVIDERS).join(', ')})`)
    } else if (!PROVIDERS[key].includes(value)) {
      problems.push(`deployments.${name}.providers.${key} must be one of ${PROVIDERS[key].join(', ')} (got "${value}")`)
    } else {
      providers[key] = value
    }
  }

  const urls = settings.urls === undefined ? {} : settings.urls
  if (!isObject(urls)) problems.push(`deployments.${name}.urls must be a mapping`)
  for (const key of Object.keys(isObject(urls) ? urls : {})) {
    if (!URL_OVERRIDES.includes(key)) problems.push(`deployments.${name}.urls.${key}: only ${URL_OVERRIDES.join(', ')} can be set here`)
  }

  const warnings = []
  if (providers.database === 'supabase' && providers.secrets === 'aws') {
    warnings.push(`deployments.${name}: database supabase with secrets aws reads secrets from AWS Secrets Manager; use secrets: env off AWS`)
  }
  return { providers, urls: isObject(urls) ? urls : {}, warnings }
}

/** Checks a parsed app.yaml for `mode`; returns the normalised config or throws listing every problem. */
export function validateAppConfig(raw, mode) {
  const problems = []
  const doc = isObject(raw) ? raw : {}
  if (!isObject(raw)) problems.push('the file is empty or not a YAML mapping')

  const deployment = readDeployment(doc, problems)

  // urls.<mode>; in production the selected deployment's `urls` are merged over them.
  const baseUrls = isObject(doc.urls) ? doc.urls[mode] : undefined
  const urls = isObject(baseUrls) && mode === 'production' ? { ...baseUrls, ...deployment.urls } : baseUrls
  if (!isObject(urls)) {
    problems.push(`urls.${mode} is missing`)
  } else {
    for (const key of ['web', 'api']) if (!isUrl(urls[key])) problems.push(`urls.${mode}.${key} must be an http(s) URL`)
    if (urls.upload !== undefined && urls.upload !== '' && urls.upload !== null && !isUrl(urls.upload)) {
      problems.push(`urls.${mode}.upload must be an http(s) URL or ""`)
    }
    if (!Array.isArray(urls.corsOrigins) || !urls.corsOrigins.length || !urls.corsOrigins.every((o) => o === '*' || isUrl(o))) {
      problems.push(`urls.${mode}.corsOrigins must be a list of origins (or ["*"])`)
    }
  }

  // api paths
  const api = isObject(doc.api) ? doc.api : {}
  for (const key of API_PATHS) if (!isPath(api[key])) problems.push(`api.${key} must be a path starting with "/"`)
  const auth = isObject(api.auth) ? api.auth : {}
  for (const key of AUTH_PATHS) if (!isPath(auth[key])) problems.push(`api.auth.${key} must be a path starting with "/"`)
  const mounts = API_PATHS.filter((k) => k !== 'healthReady').map((k) => api[k]).concat(auth.base)
  const duplicate = mounts.find((p, i) => p && mounts.indexOf(p) !== i)
  if (duplicate) problems.push(`api: "${duplicate}" is used for two different APIs`)

  // pages
  const pages = isObject(doc.pages) ? doc.pages : {}
  if (!isPath(pages.firstLogin)) problems.push('pages.firstLogin must be a path starting with "/"')

  // roles
  const roles = isObject(doc.roles) ? doc.roles : {}
  for (const name of Object.keys(roles)) {
    if (!ROLES.includes(name)) problems.push(`roles.${name}: unknown role (the roles are ${ROLES.join(', ')})`)
  }
  for (const name of ROLES) {
    const role = roles[name]
    if (!isObject(role)) {
      problems.push(`roles.${name} is missing`)
      continue
    }
    if (typeof role.label !== 'string' || !role.label.trim()) problems.push(`roles.${name}.label is required`)
    const signup = role.signup
    if (!Array.isArray(signup) || !signup.every((m) => SIGNUP_MODES.includes(m))) {
      problems.push(`roles.${name}.signup must be a list of: ${SIGNUP_MODES.join(', ')}`)
    } else if (name === 'admin' && signup.includes('self')) {
      problems.push('roles.admin.signup can\'t include "self": anyone could make themselves an admin')
    }
  }

  // permissions
  const permissions = isObject(doc.permissions) ? doc.permissions : {}
  for (const area of Object.keys(permissions)) {
    if (!SUPPORTED[area]) problems.push(`permissions.${area}: unknown section (expected ${Object.keys(SUPPORTED).join(', ')})`)
  }
  for (const [area, ops] of Object.entries(SUPPORTED)) {
    const given = isObject(permissions[area]) ? permissions[area] : {}
    for (const op of Object.keys(given)) {
      if (!ops[op]) problems.push(`permissions.${area}.${op}: unknown operation`)
    }
    for (const [op, supported] of Object.entries(ops)) {
      const list = given[op]
      if (list === undefined) {
        problems.push(`permissions.${area}.${op} is missing (use [] to turn it off)`)
      } else if (!Array.isArray(list) || !list.every((r) => ROLES.includes(r))) {
        problems.push(`permissions.${area}.${op} must be a list of roles (${ROLES.join(', ')})`)
      } else {
        const extra = list.filter((r) => !supported.includes(r))
        if (extra.length) {
          problems.push(`permissions.${area}.${op}: ${extra.join(', ')} can't be given this (supported: ${supported.join(', ') || 'none'})`)
        }
      }
    }
  }

  if (problems.length) throw new Error(`Invalid config/app.yaml (${mode}):\n  - ${problems.join('\n  - ')}`)

  return {
    urls: { web: urls.web.replace(/\/+$/, ''), api: urls.api, upload: urls.upload || '', corsOrigins: urls.corsOrigins },
    api,
    pages,
    roles,
    permissions,
    deployment: { name: deployment.name, providers: deployment.providers, warnings: deployment.warnings },
  }
}

export function loadAppConfig(backendRoot, mode, env = process.env) {
  const file = findAppConfigFile(backendRoot, env)
  let raw
  try {
    raw = parse(readFileSync(file, 'utf8'))
  } catch (err) {
    throw new Error(`config/app.yaml is not valid YAML (${file}): ${err.message}`)
  }
  return { file, ...validateAppConfig(raw, mode) }
}
