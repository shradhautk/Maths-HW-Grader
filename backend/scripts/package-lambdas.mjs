// Builds the one-Lambda-per-API-path packages (nothing is uploaded):
//
//   dist/lambdas/layer-gradeant-deps.zip  Layer: production node_modules        → /opt/nodejs/node_modules
//   dist/lambdas/layer-gradeant-app.zip   Layer: the backend code               → /opt/nodejs/gradeant
//   dist/lambdas/gradeant-<name>.zip      Function code: index.mjs (handler index.handler), one per API path
//   dist/lambdas/gradeant-jobs.zip        Function code: the SQS worker (AI rubric drafting)
//   dist/lambdas/manifest.json            What each zip is: handler, layers, paths, AWS services, sizes, SHA-256
//
//   npm run package:lambdas              build
//   npm run package:lambdas -- --verify  build, then unpack the layers into a stand-in /opt, unpack
//                                        each function, and invoke it with an API Gateway event
//
// Functions and their AWS needs: lambdas/functions.js. Guide: lambdas/LAMBDAS.md.
import { execSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { FUNCTIONS, WORKER } from '../lambdas/functions.js'
import { listFiles, readEntries, writeZip } from './zip.mjs'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const OUT = path.join(ROOT, 'dist', 'lambdas')
const STAGE = path.join(OUT, 'stage')
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
const LAYER_APP_DIR = '/opt/nodejs/gradeant' // where the app layer lands on Lambda
const MAX_UNZIPPED = 250 * 1024 * 1024 // Lambda: function + all layers, unzipped
const mb = (n) => `${(n / 1048576).toFixed(1)} MB`
const sha = (file) => createHash('sha256').update(readFileSync(file)).digest('hex')

rmSync(OUT, { recursive: true, force: true })
mkdirSync(OUT, { recursive: true })
const manifest = { builtAt: new Date().toISOString(), runtime: 'nodejs24.x', architecture: 'arm64', layers: {}, functions: {} }

// ---- Layer 1: dependencies -----------------------------------------------------
const depsDir = path.join(STAGE, 'deps')
mkdirSync(depsDir, { recursive: true })
for (const f of ['package.json', 'package-lock.json']) cpSync(path.join(ROOT, f), path.join(depsDir, f))
console.log('Installing production dependencies for the deps layer…')
execSync(`${npm} ci --omit=dev --ignore-scripts --no-audit --no-fund`, { cwd: depsDir, stdio: 'inherit' })
const depFiles = listFiles(depsDir).filter((f) => f.startsWith('node_modules/'))
const native = depFiles.filter((f) => f.endsWith('.node'))
if (native.length) console.warn(`WARNING: native modules found (rebuild them for linux/arm64):\n  ${native.join('\n  ')}`)
const depsZip = path.join(OUT, 'layer-gradeant-deps.zip')
const deps = writeZip(depsZip, depsDir, depFiles, { prefix: 'nodejs/' })
manifest.layers['gradeant-deps'] = { zip: path.basename(depsZip), mountsAt: '/opt/nodejs/node_modules', files: deps.files, unzippedBytes: deps.bytes, sha256: sha(depsZip) }

// ---- Layer 2: the backend code ---------------------------------------------------
const appDir = path.join(STAGE, 'app')
mkdirSync(appDir, { recursive: true })
for (const entry of ['package.json', '.env.production', 'worker.js', 'src']) {
  cpSync(path.join(ROOT, entry), path.join(appDir, entry), { recursive: true })
}
for (const f of ['adapter.js', 'functions.js']) {
  mkdirSync(path.join(appDir, 'lambdas'), { recursive: true })
  cpSync(path.join(ROOT, 'lambdas', f), path.join(appDir, 'lambdas', f))
}
mkdirSync(path.join(appDir, 'config'), { recursive: true })
cpSync(path.join(ROOT, '..', 'config', 'app.yaml'), path.join(appDir, 'config', 'app.yaml'))
// Development-only code (in-memory DynamoDB) is never loaded in production.
rmSync(path.join(appDir, 'src', 'dev', 'localDynamo.js'), { force: true })
const appZip = path.join(OUT, 'layer-gradeant-app.zip')
const app = writeZip(appZip, appDir, listFiles(appDir), { prefix: 'nodejs/gradeant/' })
manifest.layers['gradeant-app'] = { zip: path.basename(appZip), mountsAt: LAYER_APP_DIR, files: app.files, unzippedBytes: app.bytes, sha256: sha(appZip) }

// ---- Functions -----------------------------------------------------------------
const { default: config } = await import('../src/config/index.js')
const fnStage = path.join(STAGE, 'functions')

function writeFunction(name, source, meta) {
  const dir = path.join(fnStage, name)
  mkdirSync(dir, { recursive: true })
  writeFileSync(path.join(dir, 'index.mjs'), source)
  const zip = path.join(OUT, `gradeant-${name}.zip`)
  const f = writeZip(zip, dir, ['index.mjs'])
  const total = f.bytes + deps.bytes + app.bytes
  if (total > MAX_UNZIPPED) throw new Error(`gradeant-${name}: ${mb(total)} unzipped with its layers, over Lambda's 250 MB limit`)
  manifest.functions[`gradeant-${name}`] = {
    zip: path.basename(zip),
    handler: 'index.handler',
    layers: ['gradeant-deps', 'gradeant-app'],
    ...meta,
    unzippedWithLayers: mb(total),
    sha256: sha(zip),
  }
}

for (const [name, spec] of Object.entries(FUNCTIONS)) {
  const paths = spec.paths(config.app.paths)
  writeFunction(
    name,
    [
      `// gradeant-${name}: ${spec.description}`,
      `// Code lives in the gradeant-app layer (${LAYER_APP_DIR}); npm packages in gradeant-deps.`,
      `import { createHandler } from '${LAYER_APP_DIR}/lambdas/adapter.js'`,
      '',
      `export const handler = createHandler('${name}')`,
      '',
    ].join('\n'),
    { routes: paths.map((p) => `${name === 'health' ? 'GET' : 'POST'} ${p}${name === 'auth' ? '/*' : ''}`), aws: spec.aws, tables: spec.tables, memoryMB: spec.memory, timeoutSeconds: spec.timeout },
  )
}
writeFunction(
  WORKER.name,
  [
    `// gradeant-${WORKER.name}: ${WORKER.description}`,
    `// Triggered by the SQS jobs queue (JOBS_DRIVER=sqs), not by API Gateway.`,
    `export { handler } from '${LAYER_APP_DIR}/worker.js'`,
    '',
  ].join('\n'),
  { trigger: 'SQS (jobs queue, ReportBatchItemFailures)', aws: WORKER.aws, tables: WORKER.tables, memoryMB: WORKER.memory, timeoutSeconds: WORKER.timeout },
)

writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
rmSync(STAGE, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 })

console.log(`\nLayers:`)
for (const [n, l] of Object.entries(manifest.layers)) console.log(`  ${l.zip.padEnd(26)} ${String(l.files).padStart(6)} files  ${mb(statSync(path.join(OUT, l.zip)).size).padStart(8)} zipped  ${mb(l.unzippedBytes).padStart(8)} unzipped  → ${l.mountsAt}`)
console.log(`Functions (handler index.handler, layers gradeant-deps + gradeant-app):`)
for (const [n, f] of Object.entries(manifest.functions)) console.log(`  ${f.zip.padEnd(26)} ${(f.routes ?? [f.trigger]).join(', ')}`)
console.log(`\nWritten to ${path.relative(process.cwd(), OUT)}`)

if (!process.argv.includes('--verify')) process.exit(0)

// ---- Verify: unpack like Lambda does and invoke every function ------------------
const tmp = mkdtempSync(path.join(tmpdir(), 'gradeant-lambdas-'))
const opt = path.join(tmp, 'opt')
const unpack = (zip, into) => {
  for (const [name, { data, mode }] of readEntries(readFileSync(zip))) {
    const target = path.join(into, ...name.split('/'))
    mkdirSync(path.dirname(target), { recursive: true })
    writeFileSync(target, data)
    if ((mode & 0o777) !== 0o644) throw new Error(`${path.basename(zip)}: ${name} has mode ${(mode & 0o777).toString(8)}, expected 644`)
  }
}
unpack(depsZip, opt)
unpack(appZip, opt)

// The same events API Gateway (REST, payload 1.0) sends, answered without calling AWS.
const preflight = (p) => ({ httpMethod: 'OPTIONS', path: p, headers: { Origin: 'https://example.invalid', 'Access-Control-Request-Method': 'POST' }, multiValueHeaders: { Origin: ['https://example.invalid'], 'Access-Control-Request-Method': ['POST'] }, requestContext: { stage: 'dev' }, body: null })
const checks = Object.entries(FUNCTIONS).map(([name, spec]) => {
  const p = spec.paths(config.app.paths)[0]
  return name === 'health'
    ? { name, event: { httpMethod: 'GET', path: p, headers: {}, requestContext: { stage: 'dev' }, body: null }, expect: 200 }
    : { name, event: preflight(p), expect: 204 }
})

let failed = 0
for (const c of [...checks, { name: WORKER.name }]) {
  const task = path.join(tmp, `task-${c.name}`)
  unpack(path.join(OUT, `gradeant-${c.name}.zip`), task)
  // /opt/... only exists on Lambda: point the import at the stand-in folder.
  const index = path.join(task, 'index.mjs')
  writeFileSync(index, readFileSync(index, 'utf8').replaceAll(`'/opt/`, `'${pathToFileURL(opt).href}/`))
  const script = c.event
    ? `const { handler } = await import(${JSON.stringify(pathToFileURL(index).href)});
       const r = await handler(${JSON.stringify(c.event)}, { awsRequestId: 'verify' });
       console.log(JSON.stringify({ status: r.statusCode }));
       process.exit(0)`
    : `const m = await import(${JSON.stringify(pathToFileURL(index).href)});
       console.log(JSON.stringify({ status: typeof m.handler === 'function' ? 'handler' : 'missing' }));
       process.exit(0)`
  const r = spawnSync(process.execPath, ['--input-type=module', '-e', script], {
    cwd: task,
    env: { PATH: process.env.PATH, APP_ENV: 'production', JWT_SECRET: 'package-verification-only', GEMINI_API_KEY: 'package-verification-only', CORS_ALLOWED_ORIGINS: 'https://example.invalid', LOG_REQUESTS: 'false', AWS_REGION: 'eu-north-1' },
    encoding: 'utf8',
    timeout: 30_000,
  })
  const line = (r.stdout || '').trim().split('\n').pop()
  let got
  try {
    got = JSON.parse(line).status
  } catch {
    got = undefined
  }
  const want = c.event ? c.expect : 'handler'
  const ok = got === want
  if (!ok) failed++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} gradeant-${c.name.padEnd(12)} ${c.event ? `${c.event.httpMethod} ${c.event.path} → ${got}` : `worker.handler ${got}`}${ok ? '' : `\n${r.stdout}\n${r.stderr}`}`)
}
rmSync(tmp, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 })
if (failed) {
  console.error(`\n${failed} function(s) failed verification`)
  process.exit(1)
}
console.log('\nAll functions start from their zips + layers and answer correctly.')
