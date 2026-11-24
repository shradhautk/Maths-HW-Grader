// Supabase backups (scripts/backup.mjs): a dump restores into a fresh database built from
// backend/sql with identical counts and checksums, and any damage is detected.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { gunzipSync, gzipSync } from 'node:zlib'
import { after, before, describe, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { PGlite } from '@electric-sql/pglite'

const { SCHEMA_FILES, canonical, decode, dump, encode, listTables, load, verify } = await import('../scripts/backup.mjs')

const SQL = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'sql')
let db
const query = async (sql, params) => (await db.query(sql, params)).rows

const ROWS = [
  { table: 'users', key: { email: 'a@x.edu' }, item: { email: 'a@x.edu', user_id: 'S1', classes: ['PHY'], name: 'Ä Okafor', meta: { n: 1.5, empty: '', nul: null } } },
  { table: 'users', key: { email: 't@x.edu' }, item: { email: 't@x.edu', user_id: 'T1', userRole: 'teacher' } },
  { table: 'assignment_grading', key: { evaluation_id: 'E1' }, item: { evaluation_id: 'E1', student_id: 'S1', assignment_id: 'A1', qp_id: 'Q1', marks: 4 } },
  { table: 'assignment_grading', key: { evaluation_id: 'E2' }, item: { evaluation_id: 'E2', student_id: 'S1', assignment_id: 'A1', qp_id: 'Q1', type: 'summary', total_marks: 4 } },
  { table: 'submissions', key: { assignment_id: 'A1', student_id: 'S2' }, item: { assignment_id: 'A1', student_id: 'S2', status: 'corrected' } },
  { table: 'submissions', key: { assignment_id: 'A1', student_id: 'S1' }, item: { assignment_id: 'A1', student_id: 'S1', status: 'ai_graded' } },
]

before(async () => {
  db = new PGlite()
  for (const f of SCHEMA_FILES) await db.exec(readFileSync(path.join(SQL, f), 'utf8'))
  await load(query, ROWS)
})
after(() => db.close())

describe('backup → restore test', () => {
  test('every gradeant table is found with its key columns (composite keys in order)', async () => {
    const tables = Object.fromEntries((await listTables(query)).map((t) => [t.table, t.keys]))
    assert.equal(Object.keys(tables).length, 13)
    assert.deepEqual(tables.submissions, ['assignment_id', 'student_id'])
    assert.deepEqual(tables.invites, ['token'])
  })

  test('a backup restores with identical counts and checksums', async () => {
    const dumped = await dump(query)
    assert.equal(dumped.manifest.users.count, 2)
    assert.equal(dumped.manifest.submissions.count, 2)
    assert.equal(dumped.manifest.content.count, 0) // empty tables are in the manifest too
    assert.deepEqual(dumped.rows.filter((r) => r.table === 'submissions').map((r) => r.key.student_id), ['S1', 'S2']) // key order
    const { problems } = await verify(encode(dumped))
    assert.deepEqual(problems, [])
    // Generated index columns come back from `item` after a restore.
    assert.deepEqual(await query("select student_id, \"type\" from gradeant.assignment_grading where evaluation_id = 'E2'"), [{ student_id: 'S1', type: 'summary' }])
  })

  test('damage is detected: a changed row, a missing row, a truncated file', async () => {
    const dumped = await dump(query)
    const lines = gunzipSync(encode(dumped)).toString('utf8').split('\n').filter(Boolean)

    const changed = lines.map((l) => l.replace('"marks":4', '"marks":5'))
    assert.match((await verify(gzipSync(changed.join('\n')))).problems.join(), /assignment_grading: same row count but different contents/)

    const missing = lines.filter((l) => !l.includes('"t@x.edu"'))
    assert.match((await verify(gzipSync(missing.join('\n')))).problems.join(), /users: manifest says 2 rows, the file holds 1/)

    await assert.rejects(verify(gzipSync(lines.slice(0, -1).join('\n'))), /no manifest/)
    await assert.rejects(verify(gzipSync('{"something":"else"}\n')), /Not a GradeAnt backup file/)
  })

  test('canonical JSON ignores key order, keeps everything else', () => {
    assert.equal(canonical({ b: 1, a: [{ d: null, c: 'x' }] }), '{"a":[{"c":"x","d":null}],"b":1}')
    assert.notEqual(canonical({ a: 1 }), canonical({ a: '1' }))
  })

  test('the file is self-describing', async () => {
    const { header, rows, manifest } = decode(encode(await dump(query), '2026-11-18T03:00:00.000Z'))
    assert.deepEqual(header, { format: 'gradeant-backup', version: 1, created_at: '2026-11-18T03:00:00.000Z' })
    assert.equal(rows.length, 6)
    assert.equal(Object.keys(manifest).length, 13)
  })
})
