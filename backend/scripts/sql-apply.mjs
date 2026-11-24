// Creates / updates the GradeAnt schema on Supabase Postgres. Safe to re-run: each file in
// backend/sql runs once and is recorded in gradeant.schema_migrations.
//
//   SUPABASE_DB_URL=postgresql://... node scripts/sql-apply.mjs            apply pending files
//   SUPABASE_DB_URL=postgresql://... node scripts/sql-apply.mjs --dry-run  only list them
//
// SUPABASE_DB_URL: Supabase → Connect → Session pooler URI, with the database password.
// The connection string is read from the environment only; never pass it on the command line.
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import pg from 'pg'

const SQL_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'sql')
const dryRun = process.argv.includes('--dry-run')
const url = process.env.SUPABASE_DB_URL || process.env.DATABASE_URL
if (!url) {
  console.error('Set SUPABASE_DB_URL (Supabase → Connect → Session pooler URI).')
  process.exit(2)
}

const ca = (process.env.PG_SSL_CA ?? '').replace(/\\n/g, '\n')
const client = new pg.Client({ connectionString: url, ssl: ca ? { ca, rejectUnauthorized: true } : { rejectUnauthorized: false } })
await client.connect()
try {
  await client.query('create schema if not exists gradeant')
  await client.query('create table if not exists gradeant.schema_migrations (file text primary key, applied_at timestamptz not null default now())')
  const done = new Set((await client.query('select file from gradeant.schema_migrations')).rows.map((r) => r.file))
  const files = readdirSync(SQL_DIR).filter((f) => /^\d+_.+\.sql$/.test(f)).sort()
  const pending = files.filter((f) => !done.has(f))
  if (!pending.length) console.log('Schema is up to date.')
  for (const file of pending) {
    if (dryRun) {
      console.log(`would apply ${file}`)
      continue
    }
    await client.query('begin')
    try {
      await client.query(readFileSync(path.join(SQL_DIR, file), 'utf8'))
      await client.query('insert into gradeant.schema_migrations (file) values ($1)', [file])
      await client.query('commit')
      console.log(`applied ${file}`)
    } catch (err) {
      await client.query('rollback')
      console.error(`FAILED ${file}: ${err.message}`)
      process.exitCode = 1
      break
    }
  }
} finally {
  await client.end()
}
