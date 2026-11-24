// GET /health: liveness (no I/O). GET /health/ready: also checks the database (DynamoDB or Postgres)
// (and the secret store) so a deployment can be verified end to end.
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { pingDatabase } from '../data/index.js'
import { getJwtSecret } from '../aws/secrets.js'
import config from '../config/index.js'

const { version } = JSON.parse(readFileSync(path.join(config.root, 'package.json'), 'utf8'))
const started = Date.now()

const base = () => ({
  service: 'gradeant-api',
  version,
  mode: config.mode,
  uptimeSeconds: Math.round((Date.now() - started) / 1000),
})

export function health(req, res) {
  res.json({ status: 'ok', ...base() })
}

export async function ready(req, res) {
  const checks = {}
  const run = async (name, fn) => {
    const t = Date.now()
    try {
      await fn()
      checks[name] = { ok: true, ms: Date.now() - t }
    } catch (err) {
      checks[name] = { ok: false, ms: Date.now() - t, error: err?.name || 'Error' }
    }
  }
  // "dynamodb" on AWS (as before, so existing monitors keep working); "database" on Supabase/Postgres.
  const usesPostgres = config.drivers.db === 'postgres' || config.drivers.db === 'pglite'
  await Promise.all([
    run(usesPostgres ? 'database' : 'dynamodb', pingDatabase),
    run('jwtSecret', async () => {
      if (!(await getJwtSecret())) throw Object.assign(new Error(), { name: 'NotConfigured' })
    }),
  ])
  const ok = Object.values(checks).every((c) => c.ok)
  res.status(ok ? 200 : 503).json({ status: ok ? 'ok' : 'degraded', ...base(), drivers: config.drivers, checks })
}
