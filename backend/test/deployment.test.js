// The config/app.yaml `deployment:` switch, end to end through src/config/index.js:
// production takes its drivers from the selected target, explicit *_DRIVER variables
// still win, development keeps local drivers, and options whose code isn't built yet
// stop startup instead of failing half-way through a request.
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { describe, test } from 'node:test'
import { fileURLToPath } from 'node:url'

const BACKEND = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const SHIPPED = path.join(BACKEND, '..', 'config', 'app.yaml')

function yamlWith(deployment) {
  const dir = mkdtempSync(path.join(tmpdir(), 'gradeant-cfg-'))
  const file = path.join(dir, 'app.yaml')
  writeFileSync(file, readFileSync(SHIPPED, 'utf8').replace(/^deployment: [a-z]+\b/m,`deployment: ${deployment}`))
  return file
}

/** Loads src/config/index.js in a fresh process; returns {drivers, deployment} or {error}. */
function loadConfig({ mode = 'production', deployment = 'aws', env = {} } = {}) {
  const script = `import('./src/config/index.js').then(
    (m) => console.log(JSON.stringify({ drivers: m.default.drivers, deployment: m.default.deployment })),
    (e) => console.log(JSON.stringify({ error: e.message })))`
  const clean = Object.fromEntries(Object.entries(process.env).filter(([k]) => !/_DRIVER$|^APP_ENV$|^JWT_SECRET|^AWS_LAMBDA|^GEMINI_/.test(k)))
  const r = spawnSync(process.execPath, ['--input-type=module', '-e', script], {
    cwd: BACKEND,
    encoding: 'utf8',
    env: { ...clean, APP_ENV: mode, APP_CONFIG_FILE: yamlWith(deployment), JWT_SECRET: 'test-secret-at-least-32-characters-long', ...env },
  })
  const line = r.stdout.trim().split('\n').pop()
  assert.ok(line, `no output: ${r.stderr}`)
  return JSON.parse(line)
}

describe('deployment switch → drivers', () => {
  test('aws in production: DynamoDB, SES, Lambda grading, and Gemini for AI (key from Secrets Manager)', () => {
    const c = loadConfig({ env: { GEMINI_API_KEY_ARN: 'arn:aws:secretsmanager:eu-north-1:1:secret:gemini' } })
    assert.equal(c.error, undefined)
    assert.deepEqual(
      { db: c.drivers.db, ai: c.drivers.ai, mail: c.drivers.mail, grading: c.drivers.grading, storage: c.drivers.storage, notify: c.drivers.notify, jobs: c.drivers.jobs },
      { db: 'dynamodb', ai: 'gemini', mail: 'ses', grading: 'lambda', storage: 's3', notify: 'sns', jobs: 'inline' },
    )
    assert.equal(c.drivers.secrets, 'aws')
    assert.equal(c.deployment.name, 'aws')
  })

  test('Gemini is the only AI: without a key production refuses to start', () => {
    assert.match(loadConfig().error, /GEMINI_API_KEY \(or GEMINI_API_KEY_ARN\) is required/)
    assert.match(loadConfig({ env: { AI_DRIVER: 'bedrock' } }).error, /AI_DRIVER must be one of canned, gemini/)
  })

  test('development ignores the providers and keeps local drivers', () => {
    const c = loadConfig({ mode: 'development', deployment: 'hybrid' })
    assert.deepEqual([c.drivers.db, c.drivers.ai, c.drivers.mail, c.drivers.grading], ['memory', 'canned', 'log', 'log'])
  })

  test('an explicit *_DRIVER variable wins over app.yaml', () => {
    const c = loadConfig({ env: { MAIL_DRIVER: 'log', AI_DRIVER: 'canned' } })
    assert.equal(c.drivers.mail, 'log')
    assert.equal(c.drivers.ai, 'canned')
    assert.equal(c.drivers.db, 'dynamodb')
  })

  test('hybrid names every setting it is missing', () => {
    const c = loadConfig({ deployment: 'hybrid' })
    assert.match(c.error, /SUPABASE_DB_URL is required/)
    assert.match(c.error, /GEMINI_API_KEY is required/)
    assert.match(c.error, /GRADING_QUEUE_URL is required/)
    assert.doesNotMatch(c.error, /not available yet/)
  })

  test('a fully configured hybrid starts with Supabase, Gemini, SQS grading and log-only email', () => {
    const c = loadConfig({
      deployment: 'hybrid',
      env: {
        SUPABASE_DB_URL: 'postgresql://u:p@db.example:5432/postgres',
        GEMINI_API_KEY: 'g-key',
        GRADING_QUEUE_URL: 'https://sqs.eu-north-1.amazonaws.com/123456789012/gradeant-grader',
      },
    })
    assert.equal(c.error, undefined)
    assert.deepEqual(
      { db: c.drivers.db, ai: c.drivers.ai, mail: c.drivers.mail, grading: c.drivers.grading, secrets: c.drivers.secrets },
      { db: 'postgres', ai: 'gemini', mail: 'log', grading: 'sqs', secrets: 'env' },
    )
  })

  test('hybrid uses the postgres driver and needs SUPABASE_DB_URL', () => {
    const env = { AI_DRIVER: 'canned', GRADING_DRIVER: 'log' }
    assert.match(loadConfig({ deployment: 'hybrid', env }).error, /SUPABASE_DB_URL is required/)
    const c = loadConfig({ deployment: 'hybrid', env: { ...env, SUPABASE_DB_URL: 'postgresql://u:p@db.example:5432/postgres' } })
    assert.equal(c.drivers.db, 'postgres')
    assert.equal(c.drivers.secrets, 'env')
  })

  test('the in-process databases are refused in production', () => {
    assert.match(loadConfig({ env: { DB_DRIVER: 'pglite' } }).error, /DB_DRIVER=pglite is for local development only/)
    assert.match(loadConfig({ env: { DB_DRIVER: 'memory' } }).error, /DB_DRIVER=memory is for local development only/)
  })

  test('secrets: env needs the value itself, not a Secrets Manager ARN', () => {
    const c = loadConfig({
      deployment: 'hybrid',
      env: { JWT_SECRET: '', JWT_SECRET_ARN: 'arn:aws:secretsmanager:eu-north-1:1:secret:x', DB_DRIVER: 'dynamodb', AI_DRIVER: 'canned', GRADING_DRIVER: 'log' },
    })
    assert.match(c.error, /JWT_SECRET is required \(deployment providers\.secrets: env/)
  })
})
