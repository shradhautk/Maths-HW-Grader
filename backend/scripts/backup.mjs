// Backups of the Supabase database (schema gradeant), with a restore test every time.
//
//   node scripts/backup.mjs backup  --out FILE.jsonl.gz      dump every gradeant table (SUPABASE_DB_URL)
//   node scripts/backup.mjs verify  FILE.jsonl.gz            restore test: load into a fresh in-process
//                                                            Postgres built from backend/sql, compare
//                                                            every table's row count and checksum
//   node scripts/backup.mjs restore FILE.jsonl.gz --yes      disaster recovery: upsert every row into
//                                                            SUPABASE_DB_URL (run npm run sql:apply first)
//
// File: gzipped JSON lines — a header, one line per row {table, key, item}, then a manifest
// {table: {count, sha256}}. The checksum covers each row's key and item as canonical JSON
// (sorted keys), in key order, so a backup that loads with the same checksums is complete and intact.
// Generated index columns aren't stored: Postgres recomputes them from `item` on load.
// Connection strings come from the environment only.
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { gunzipSync, gzipSync } from 'node:zlib'

const SQL_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'sql')
export const SCHEMA_FILES = ['001_schema.sql', '002_pipeline_api.sql'] // 003 needs Supabase's roles
const SKIP = new Set(['schema_migrations']) // bookkeeping of sql-apply.mjs, recreated by it
export const FORMAT = 'gradeant-backup'

/** JSON with object keys sorted, so equal data always gives equal text. */
export function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

/** The gradeant tables and their primary-key columns, read from the catalog (no hard-coded list). */
export async function listTables(query) {
  const rows = await query(
    `select c.relname as table, array_agg(a.attname::text order by array_position(i.indkey::int2[], a.attnum)) as keys
       from pg_index i
       join pg_class c on c.oid = i.indrelid
       join pg_namespace n on n.oid = c.relnamespace
       join pg_attribute a on a.attrelid = c.oid and a.attnum = any(i.indkey)
      where n.nspname = 'gradeant' and i.indisprimary
      group by c.relname
      order by c.relname`,
  )
  return rows.filter((r) => !SKIP.has(r.table)).map((r) => ({ table: r.table, keys: toArray(r.keys) }))
}

// pg returns text[] as an array; PGlite may return the Postgres text form "{a,b}".
const toArray = (v) => (Array.isArray(v) ? v : String(v).replace(/^\{|\}$/g, '').split(',').filter(Boolean))
const ident = (name) => `"${String(name).replace(/"/g, '""')}"`

/** Every row of every table, in key order: [{table, key, item}], plus the manifest. */
export async function dump(query) {
  const rows = []
  const manifest = {}
  for (const { table, keys } of await listTables(query)) {
    const order = keys.map(ident).join(', ')
    const result = await query(`select ${order}, item from gradeant.${ident(table)} order by ${order}`)
    const hash = createHash('sha256')
    for (const r of result) {
      const key = Object.fromEntries(keys.map((k) => [k, r[k]]))
      rows.push({ table, key, item: r.item })
      hash.update(canonical({ key, item: r.item })).update('\n')
    }
    manifest[table] = { count: result.length, sha256: hash.digest('hex') }
  }
  return { rows, manifest }
}

export function encode({ rows, manifest }, createdAt = new Date().toISOString()) {
  const lines = [JSON.stringify({ format: FORMAT, version: 1, created_at: createdAt })]
  for (const r of rows) lines.push(JSON.stringify(r))
  lines.push(JSON.stringify({ manifest }))
  return gzipSync(lines.join('\n') + '\n')
}

export function decode(buffer) {
  const lines = gunzipSync(buffer).toString('utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
  const [header, ...rest] = lines
  if (header?.format !== FORMAT) throw new Error('Not a GradeAnt backup file')
  const last = rest.pop()
  if (!last?.manifest) throw new Error('The backup has no manifest (truncated file?)')
  return { header, rows: rest, manifest: last.manifest }
}

/** Upserts rows into the gradeant tables (key columns + item; generated columns recompute). */
export async function load(query, rows) {
  for (const { table, key, item } of rows) {
    const cols = Object.keys(key)
    const params = [...cols.map((c) => key[c]), JSON.stringify(item)]
    await query(
      `insert into gradeant.${ident(table)} (${[...cols.map(ident), 'item'].join(', ')}) values (${params.map((_, i) => `$${i + 1}`).join(', ')}) ` +
        `on conflict (${cols.map(ident).join(', ')}) do update set item = excluded.item`,
      params,
    )
  }
}

/** Compares two manifests; returns a list of differences (empty = identical). */
export function compare(expected, actual) {
  const problems = []
  for (const table of new Set([...Object.keys(expected), ...Object.keys(actual)])) {
    const e = expected[table]
    const a = actual[table] ?? { count: 0, sha256: createHash('sha256').digest('hex') }
    if (!e) problems.push(`${table}: in the database but not in the backup`)
    else if (e.count !== a.count) problems.push(`${table}: ${e.count} rows in the backup, ${a.count} after restore`)
    else if (e.sha256 !== a.sha256) problems.push(`${table}: same row count but different contents (checksum mismatch)`)
  }
  return problems
}

/** Restore test: a fresh in-process Postgres from backend/sql, load the backup, compare. */
export async function verify(buffer) {
  const { rows, manifest } = decode(buffer)
  const fileProblems = []
  const counts = {}
  for (const r of rows) counts[r.table] = (counts[r.table] ?? 0) + 1
  for (const [table, m] of Object.entries(manifest)) {
    if ((counts[table] ?? 0) !== m.count) fileProblems.push(`${table}: manifest says ${m.count} rows, the file holds ${counts[table] ?? 0}`)
  }
  const { PGlite } = await import('@electric-sql/pglite')
  const db = new PGlite()
  try {
    for (const file of SCHEMA_FILES) await db.exec(readFileSync(path.join(SQL_DIR, file), 'utf8'))
    const query = async (sql, params) => (await db.query(sql, params)).rows
    await load(query, rows)
    const restored = await dump(query)
    return { problems: [...fileProblems, ...compare(manifest, restored.manifest)], manifest, restored: restored.manifest }
  } finally {
    await db.close()
  }
}

async function connect() {
  const url = process.env.SUPABASE_DB_URL || process.env.DATABASE_URL
  if (!url) throw new Error('Set SUPABASE_DB_URL (Supabase → Connect → Session pooler URI).')
  const { default: pg } = await import('pg')
  const ca = (process.env.PG_SSL_CA ?? '').replace(/\\n/g, '\n')
  const client = new pg.Client({ connectionString: url, ssl: ca ? { ca, rejectUnauthorized: true } : { rejectUnauthorized: false } })
  await client.connect()
  return { query: async (sql, params) => (await client.query(sql, params)).rows, close: () => client.end() }
}

const summary = (manifest) => Object.entries(manifest).map(([t, m]) => `${t}=${m.count}`).join(' ')

async function main([command, ...args]) {
  const flag = (name) => args.includes(name)
  const value = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined)
  const out = value('--out')
  const file = args.find((a, i) => !a.startsWith('--') && !(out !== undefined && args[i - 1] === '--out'))
  if (command === 'backup') {
    if (!out) throw new Error('Usage: backup --out FILE.jsonl.gz')
    const db = await connect()
    try {
      const dumped = await dump(db.query)
      writeFileSync(out, encode(dumped))
      console.log(`backup written: ${out} (${dumped.rows.length} rows: ${summary(dumped.manifest)})`)
    } finally {
      await db.close()
    }
  } else if (command === 'verify') {
    if (!file) throw new Error('Usage: verify FILE.jsonl.gz')
    const { problems, manifest } = await verify(readFileSync(file))
    if (problems.length) {
      console.error(`RESTORE TEST FAILED for ${file}:\n  - ${problems.join('\n  - ')}`)
      process.exitCode = 1
    } else {
      console.log(`restore test passed: every table restored with identical counts and checksums (${summary(manifest)})`)
    }
  } else if (command === 'restore') {
    if (!file) throw new Error('Usage: restore FILE.jsonl.gz --yes')
    if (!flag('--yes')) throw new Error('restore writes into SUPABASE_DB_URL: add --yes to confirm (after npm run sql:apply)')
    const { rows } = decode(readFileSync(file))
    const db = await connect()
    try {
      await load(db.query, rows)
      console.log(`restored ${rows.length} rows`)
    } finally {
      await db.close()
    }
  } else {
    throw new Error('Commands: backup --out FILE | verify FILE | restore FILE --yes')
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(err.message)
    process.exitCode = 2
  })
}
