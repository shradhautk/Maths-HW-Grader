// The entry point as Lambda Web Functions runs it: index.js must listen on the
// host:port in AWS_LAMBDA_HTTP_ENDPOINT, quickly, and production config must be
// validated before the server starts.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// These tests are about the aws (Lambda) target, whatever config/app.yaml currently selects.
const AWS_CONFIG = path.join(mkdtempSync(path.join(tmpdir(), 'gradeant-rt-')), 'app.yaml')
writeFileSync(AWS_CONFIG, readFileSync(path.join(ROOT, '..', 'config', 'app.yaml'), 'utf8').replace(/^deployment: [a-z]+\b/m, 'deployment: aws'))

const freePort = () =>
  new Promise((resolve) => {
    const s = createServer().listen(0, '127.0.0.1', () => {
      const { port } = s.address()
      s.close(() => resolve(port))
    })
  })

function run(env, args = []) {
  const child = spawn(process.execPath, ['index.js', ...args], {
    cwd: ROOT,
    env: { PATH: process.env.PATH, APP_CONFIG_FILE: AWS_CONFIG, ...env },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let output = ''
  child.stdout.on('data', (d) => (output += d))
  child.stderr.on('data', (d) => (output += d))
  const exited = new Promise((resolve) => child.on('exit', (code) => resolve(code)))
  return { child, exited, output: () => output }
}

test('listens on AWS_LAMBDA_HTTP_ENDPOINT within the 10 s startup limit', async () => {
  const port = await freePort()
  const started = Date.now()
  const proc = run({ AWS_LAMBDA_HTTP_ENDPOINT: `127.0.0.1:${port}`, APP_ENV: 'test' })
  try {
    let res
    while (!res && Date.now() - started < 10_000) {
      res = await fetch(`http://127.0.0.1:${port}/health`).catch(() => null)
      if (!res) await new Promise((r) => setTimeout(r, 100))
    }
    assert.ok(res, `server did not listen: ${proc.output()}`)
    assert.equal(res.status, 200)
    assert.ok(Date.now() - started < 10_000)
  } finally {
    proc.child.kill('SIGTERM')
    assert.equal(await proc.exited, 0) // graceful SIGTERM shutdown
  }
})

test('production refuses to start without its required configuration', async () => {
  const proc = run({ APP_ENV: 'production', CORS_ALLOWED_ORIGINS: '', JWT_SECRET_ARN: '' })
  const code = await proc.exited
  assert.notEqual(code, 0)
  assert.match(proc.output(), /JWT_SECRET_ARN \(or JWT_SECRET\) is required/)
})

test('Lambda (AWS_LAMBDA_HTTP_ENDPOINT set, no APP_ENV) means production mode', async () => {
  const proc = run({ AWS_LAMBDA_HTTP_ENDPOINT: '127.0.0.1:1', CORS_ALLOWED_ORIGINS: '', JWT_SECRET_ARN: '' })
  await proc.exited
  assert.match(proc.output(), /Invalid production configuration/)
})
