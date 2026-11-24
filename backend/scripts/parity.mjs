// Parity check: runs the same scripted requests against the original Python
// Lambdas (scripts/parity/run_python.py, on moto) and this Node API (on the
// in-process DynamoDB emulator), seeded from the same fixture, and diffs every
// response after masking values that are random by design (ids, tokens,
// timestamps).
//
//   PYTHON=/path/to/venv/bin/python npm run check:parity
//
// Needs Python 3 with lambda_python_files/requirements-dev.txt installed.
// Exits non-zero on any difference not listed in KNOWN_DIFFERENCES.
import { execFileSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { isDeepStrictEqual } from 'node:util'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const SECRET = 'parity-jwt-secret-parity-jwt-secret-0123'
const SESSION_SECRET = 'parity-session-secret-0123456789abcdef'

// Intentional behaviour changes, each documented in backend/README.md.
const PY_RESERVED_WORD_BUG =
  'Python bug fixed: lambda_admin.py completeFirstLogin writes "SET consumed", a DynamoDB reserved word, so it always fails with a 500'
const INVALID_JSON = 'Malformed JSON is a 400 with the endpoint\'s usual error shape (Python returned a 500 with the parser\'s exception text)'
const KNOWN_DIFFERENCES = {
  'grades.invalidJson': INVALID_JSON,
  'correct.invalidJson': 'Same 400; the message no longer echoes the JSON parser\'s exception text',
  'grades.update.noSummary': 'Python returned nothing (API Gateway answers 502); Node answers 404 with a message',
  'updatedb.detailsOnly': 'Python crashed (500) when only per-question marks were sent; Node applies them',
  'admin.firstLogin.ok': PY_RESERVED_WORD_BUG,
  'admin.firstLogin.again': PY_RESERVED_WORD_BUG,
  'admin.session.profile': `${PY_RESERVED_WORD_BUG} (so Python never issued the session token used here)`,
  'admin.session.tampered': `${PY_RESERVED_WORD_BUG} (so Python never issued the session token used here)`,
}

Object.assign(process.env, {
  APP_ENV: 'test',
  JWT_SECRET: SECRET,
  SESSION_TOKEN_SECRET: SESSION_SECRET,
  PROFILE_PHOTO_BUCKET: '',
  SES_SENDER_EMAIL: '',
  SNS_TOPIC_ARN: '',
  GRADING_FUNCTION_NAME: '',
  LOG_REQUESTS: 'false',
})

const fixture = JSON.parse(readFileSync(path.join(HERE, 'parity/fixture.json'), 'utf8'))
const cases = JSON.parse(readFileSync(path.join(HERE, 'parity/cases.json'), 'utf8'))

// ---- Python ------------------------------------------------------------------------
const python = process.env.PYTHON || 'python3'
console.log(`Running ${cases.length} cases against the Python Lambdas (${python})…`)
const pyResults = JSON.parse(
  execFileSync(python, [path.join(HERE, 'parity/run_python.py')], { maxBuffer: 64 * 1024 * 1024, stdio: ['ignore', 'pipe', 'inherit'] }),
)

// ---- Node --------------------------------------------------------------------------
console.log('Running the same cases against the Node API…')
const { post, startServer, stopServer, table } = await import('../test/helpers.js')
const { default: jwt } = await import('jsonwebtoken')
const { setAIOverride: setClaudeOverride } = await import('../src/ai/index.js')
const { hashAdminPassword, hashLoginPassword } = await import('../src/lib/passwords.js')

await startServer({ assignmentIndex: true, seed: false })

async function resolveValue(v) {
  if (v && typeof v === 'object' && '$loginHash' in v) return hashLoginPassword(v.$loginHash)
  if (v && typeof v === 'object' && '$adminHash' in v) return hashAdminPassword(v.$adminHash)
  if (Array.isArray(v)) return Promise.all(v.map(resolveValue))
  if (v && typeof v === 'object') {
    return Object.fromEntries(await Promise.all(Object.entries(v).map(async ([k, x]) => [k, await resolveValue(x)])))
  }
  return v
}
for (const [name, items] of Object.entries(fixture.tables)) {
  for (const item of items) await table(name).put(await resolveValue(item))
}

const claudeCalls = []
setClaudeOverride(async ({ system, messages, effort, outputFormat }) => {
  claudeCalls.push({ system, messages, effort, format: outputFormat ?? null })
  return outputFormat ? null : 'CANNED'
})

const users = Object.fromEntries(fixture.tables.Users.map((u) => [u.email, u]))
function tokenFor(spec) {
  if (spec === '@bad') return 'not-a-token'
  const sign = (payload) => jwt.sign(payload, SECRET, { algorithm: 'HS256', noTimestamp: true })
  const now = Math.floor(Date.now() / 1000)
  if (spec === '@ghost') return sign({ email: 'ghost@x.edu', user_id: 'G1', exp: now + 600 })
  const expired = spec.startsWith('@expired:')
  const u = users[expired ? spec.slice('@expired:'.length) : spec]
  return sign({ email: u.email, user_id: u.user_id, exp: now + (expired ? -10 : 600) })
}

const nodeResults = {}
const ref = (expr) => {
  const parts = expr.split('.')
  for (let i = parts.length; i > 0; i--) {
    const name = parts.slice(0, i).join('.')
    if (nodeResults[name]) return parts.slice(i).reduce((v, k) => v[k], nodeResults[name].body)
  }
  throw new Error(`unresolved ref ${expr}`)
}
const subst = (v) => {
  if (typeof v === 'string') return v.replace(/\$\{ref:([^}]+)\}/g, (_, e) => String(ref(e)))
  if (Array.isArray(v)) return v.map(subst)
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, subst(x)]))
  return v
}
for (const c of cases) {
  const token = c.as ? tokenFor(c.as) : c.token ? subst(c.token) : undefined
  const [status, body] = await post(c.path, subst(c.body ?? {}), { token, raw: c.raw })
  nodeResults[c.name] = { status, body }
}
nodeResults.__claude_calls__ = { status: 0, body: claudeCalls }
setClaudeOverride(undefined)
await stopServer()

// ---- Compare -----------------------------------------------------------------------
const today = new Date().toISOString().slice(0, 10)
const RUNTIME_EPOCH = 1790000300 // fixture timestamps are below this
function mask(v, key) {
  if (Array.isArray(v)) return v.map((x) => mask(x))
  if (v && typeof v === 'object') return Object.fromEntries(Object.keys(v).sort().map((k) => [k, mask(v[k], k)]))
  if (['token', 'tempPassword'].includes(key) && typeof v === 'string') return `<${key}>`
  if (['timestamp', 'requestedAt'].includes(key) && typeof v === 'number' && v > RUNTIME_EPOCH) return `<${key}>`
  if (typeof v !== 'string') return v
  return v
    .replace(/eyJ[\w-]+\.[\w-]+\.[\w-]+/g, '<jwt>')
    .replace(/[\w-]+\.[0-9a-f]{64}\b/g, '<session>')
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g, '<uuid>')
    .replace(/\b(A|QP|N)-[0-9a-f]{10}\b/g, '$1-<id>')
    .replace(new RegExp(`${today}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?(Z|[+-]\\d{2}:\\d{2})?`, 'g'), '<now>')
}
const canonical = (v) => {
  if (Array.isArray(v)) return v.map(canonical).sort((a, b) => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1))
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, canonical(x)]))
  return v
}

// Claude prompts: assignments are listed in DynamoDB query order (unordered), and a
// stored 4.0 prints as "4.0" from Python's Decimal but "4" from a JS number.
function promptKey(calls) {
  return (calls?.body ?? []).map((c) => ({
    ...c,
    system: String(c.system)
      .replace(/(\d+)\.0 marks/g, '$1 marks')
      .split(/\n+(?=Assignment: )/)
      .map((block) => block.trim())
      .sort(),
    messages: JSON.parse(JSON.stringify(c.messages).replace(/(\d+)\.0 marks/g, '$1 marks')),
  }))
}

const report = []
let same = 0
let orderOnly = 0
let known = 0
let different = 0
for (const name of [...cases.map((c) => c.name), '__claude_calls__']) {
  const py = mask(pyResults[name])
  const nd = mask(nodeResults[name])
  if (isDeepStrictEqual(py, nd)) {
    same++
    continue
  }
  if (name === '__claude_calls__' && isDeepStrictEqual(promptKey(py), promptKey(nd))) {
    orderOnly++
    report.push({ name, result: 'same prompts apart from assignment order and 4.0-vs-4 number formatting' })
    continue
  }
  if (isDeepStrictEqual(canonical(py), canonical(nd))) {
    orderOnly++
    report.push({ name, result: 'order-only (unordered DynamoDB scan/query results)' })
    continue
  }
  if (KNOWN_DIFFERENCES[name]) {
    known++
    report.push({ name, result: `known: ${KNOWN_DIFFERENCES[name]}`, python: py, node: nd })
    continue
  }
  different++
  report.push({ name, result: 'DIFFERENT', python: py, node: nd })
}

mkdirSync(path.join(HERE, '..', '.data'), { recursive: true })
const out = path.join(HERE, '..', '.data', 'parity-report.json')
writeFileSync(out, JSON.stringify(report, null, 2))
for (const r of report.filter((x) => x.result === 'DIFFERENT')) {
  console.log(`\n✗ ${r.name}\n  python: ${JSON.stringify(r.python)}\n  node:   ${JSON.stringify(r.node)}`)
}
console.log(
  `\n${cases.length + 1} comparisons: ${same} identical, ${orderOnly} identical apart from ordering, ` +
    `${known} known/intentional differences, ${different} unexplained differences. Report: ${path.relative(process.cwd(), out)}`,
)
process.exit(different ? 1 : 0)
