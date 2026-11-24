// npm run test:pg — the whole test suite again, on the Postgres driver (in-process PGlite)
// instead of the DynamoDB emulator. Same tests, same expectations: the Supabase database
// must behave exactly like DynamoDB for every service.
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const r = spawnSync(
  process.execPath,
  ['--test', '--test-concurrency=1', '--test-reporter=spec', ...(process.argv.slice(2).length ? process.argv.slice(2) : ['test/*.test.js'])],
  { cwd: root, stdio: 'inherit', env: { ...process.env, DB_DRIVER: 'pglite' } },
)
process.exit(r.status ?? 1)
