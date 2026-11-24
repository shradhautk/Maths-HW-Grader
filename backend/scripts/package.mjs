// Builds the Lambda deployment package (nothing is uploaded):
//   dist/package/      index.js + run.sh at the root, src/, config/app.yaml, production node_modules
//   dist/function.zip  the same, zipped, with its SHA-256 printed
//
// The one package runs both ways (see deploy/DEPLOY.md):
//   - Lambda + AWS Lambda Web Adapter (handler run.sh)   ← the recommended setup, in eu-north-1
//   - Lambda Web Functions (index.js, AWS_LAMBDA_HTTP_ENDPOINT)
//
//   npm run package              build
//   npm run package -- --verify  build, check the zip (run.sh is LF + mode 755), extract it
//                                into a clean folder and start it both ways to check /health
//
// The zip is written here rather than by tar/zip so it is identical on Windows,
// Linux and CI: every entry carries Unix permissions (run.sh 755, the rest 644),
// which Lambda needs to execute run.sh, and timestamps are fixed so the same
// sources always give the same SHA-256.
import { execSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { listFiles, readEntries, writeZip } from './zip.mjs'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const DIST = path.join(ROOT, 'dist')
const STAGE = path.join(DIST, 'package')
const ZIP = path.join(DIST, 'function.zip')
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
const EXECUTABLES = new Set(['run.sh'])

// ---- Stage the files ---------------------------------------------------------

rmSync(DIST, { recursive: true, force: true })
mkdirSync(STAGE, { recursive: true })
for (const entry of ['index.js', 'worker.js', 'package.json', 'package-lock.json', '.env.production', 'src']) {
  cpSync(path.join(ROOT, entry), path.join(STAGE, entry), { recursive: true })
}
// run.sh must have LF line endings on Lambda, whatever git did on checkout.
writeFileSync(path.join(STAGE, 'run.sh'), readFileSync(path.join(ROOT, 'run.sh'), 'utf8').replace(/\r\n/g, '\n'))
// The main config (URLs, API paths, roles, permissions) lives at the repo root;
// the package carries a copy at config/app.yaml (found by src/config/appConfig.js).
mkdirSync(path.join(STAGE, 'config'), { recursive: true })
cpSync(path.join(ROOT, '..', 'config', 'app.yaml'), path.join(STAGE, 'config', 'app.yaml'))
// Development-only code (in-memory DynamoDB, demo seed) is never loaded in production.
rmSync(path.join(STAGE, 'src', 'dev', 'localDynamo.js'), { force: true })
console.log('Installing production dependencies…')
// One command string (npm is npm.cmd on Windows, which needs a shell; passing args with shell: true is deprecated).
execSync(`${npm} ci --omit=dev --ignore-scripts --no-audit --no-fund`, { cwd: STAGE, stdio: 'inherit' })

// ---- Collect files -----------------------------------------------------------

const files = listFiles(STAGE)

// Lambda runs Arm64 here: native (.node) modules built on x86 would not load.
const native = files.filter((f) => f.endsWith('.node'))
if (native.length) console.warn(`WARNING: native modules found (rebuild them for linux/arm64):\n  ${native.join('\n  ')}`)

// ---- Write the zip (scripts/zip.mjs) -------------------------------------------

writeZip(ZIP, STAGE, files, { executables: EXECUTABLES })
const sha256 = createHash('sha256').update(readFileSync(ZIP)).digest('hex')
console.log(`\nPackage: ${path.relative(process.cwd(), ZIP)} (${files.length} files, ${(statSync(ZIP).size / 1048576).toFixed(1)} MB)\nSHA-256: ${sha256}`)

if (!process.argv.includes('--verify')) process.exit(0)

// ---- Verify ------------------------------------------------------------------

const entries = readEntries(readFileSync(ZIP))
if (entries.size !== files.length) throw new Error(`zip holds ${entries.size} entries, expected ${files.length}`)
for (const required of ['index.js', 'run.sh', 'config/app.yaml', '.env.production']) {
  if (!entries.has(required)) throw new Error(`${required} is missing from the root of the package`)
}
const runSh = entries.get('run.sh')
if ((runSh.mode & 0o777) !== 0o755) throw new Error(`run.sh has mode ${(runSh.mode & 0o777).toString(8)}, Lambda needs 755`)
if (runSh.data.includes(0x0d)) throw new Error('run.sh has CRLF line endings, Lambda needs LF')
console.log(`Zip checked: ${entries.size} entries, every CRC valid; run.sh is mode 755 with LF endings.`)

// Extract from the zip itself (not dist/package), so the test runs exactly what Lambda would receive.
const dir = mkdtempSync(path.join(tmpdir(), 'gradeant-package-'))
for (const [name, { data }] of entries) {
  const target = path.join(dir, ...name.split('/'))
  mkdirSync(path.dirname(target), { recursive: true })
  writeFileSync(target, data)
}
if (!existsSync(path.join(dir, 'index.js'))) throw new Error('index.js is not at the root of the extracted package')

/** Starts the extracted package with `env` and waits up to 10 s for GET /health → 200. */
async function startAndCheck(label, env, port) {
  const child = spawn(process.execPath, ['index.js'], {
    cwd: dir,
    // Production mode, with placeholder values (/health makes no AWS calls).
    env: { PATH: process.env.PATH, JWT_SECRET: 'package-verification-only', GEMINI_API_KEY: 'package-verification-only', CORS_ALLOWED_ORIGINS: 'https://example.invalid', LOG_REQUESTS: 'false', ...env },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let output = ''
  child.stdout.on('data', (d) => (output += d))
  child.stderr.on('data', (d) => (output += d))
  const started = Date.now()
  let res
  while (!res && Date.now() - started < 10_000) {
    res = await fetch(`http://127.0.0.1:${port}/health`).catch(() => null)
    if (!res) await new Promise((r) => setTimeout(r, 200))
  }
  // Wait for the server to exit: on Windows its files stay locked until then.
  const exited = new Promise((resolve) => child.once('exit', resolve))
  child.kill('SIGTERM')
  await exited
  if (!res || res.status !== 200) {
    rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 })
    console.error(`\nPackage verification FAILED (${label}):\n${output}`)
    process.exit(1)
  }
  const body = await res.json()
  console.log(`Package verified (${label}): /health → ${body.status}, mode ${body.mode}, ready in ${Date.now() - started} ms`)
}

const base = 18000 + Math.floor(Math.random() * 1000)
// As behind the Lambda Web Adapter: run.sh → node index.js on 127.0.0.1:$PORT.
await startAndCheck('Lambda + Web Adapter', { APP_ENV: 'production', HOST: '127.0.0.1', PORT: String(base) }, base)
// As on Lambda Web Functions.
await startAndCheck('Lambda Web Functions', { AWS_LAMBDA_HTTP_ENDPOINT: `127.0.0.1:${base + 1}` }, base + 1)
rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 })
