// The Postgres driver (Supabase) must behave exactly like DynamoDB. Every operation the
// repositories use runs against BOTH the DynamoDB emulator and in-process Postgres, and the
// results (items, ordering where DynamoDB guarantees one, ReturnValues, errors) must match.
import assert from 'node:assert/strict'
import { after, before, describe, test } from 'node:test'

process.env.APP_ENV = 'test'
process.env.DB_DRIVER = 'memory' // this file runs BOTH drivers itself (emulator + PGlite), also under npm run test:pg
const { default: config } = await import('../src/config/index.js')
const dynamo = await import('../src/data/dynamodb.js')
const postgres = await import('../src/data/postgres.js')
const { UnsupportedExpression, compileCondition, compileUpdate } = await import('../src/data/expressions.js')

const T = config.tables
const drivers = { dynamodb: dynamo, postgres }

before(async () => {
  await dynamo.initDatabase({ seed: false, assignmentIndex: true })
  await postgres.initDatabase({ seed: false, local: true })
})
after(async () => {
  await dynamo.closeDatabase()
  await postgres.closeDatabase()
})

/** Runs fn(table) on both drivers; returns {dynamodb, postgres} results ({ok} or {error: name}). */
async function both(tableName, fn) {
  const out = {}
  for (const [name, d] of Object.entries(drivers)) {
    try {
      const result = await fn(d.table(tableName))
      // The SDK's response envelope ($metadata) is never read by the services.
      out[name] = { ok: result && typeof result === 'object' && '$metadata' in result ? (({ $metadata, ...rest }) => rest)(result) : result }
    } catch (err) {
      out[name] = { error: err.name }
    }
  }
  return out
}

async function same(tableName, fn, { sortBy } = {}) {
  const r = await both(tableName, fn)
  const norm = (x) => (sortBy && Array.isArray(x?.ok) ? { ok: [...x.ok].sort((a, b) => String(a[sortBy]).localeCompare(String(b[sortBy]))) } : x)
  assert.deepEqual(norm(r.postgres), norm(r.dynamodb))
  return r.dynamodb
}

const ids = (items) => items.map((i) => i.evaluation_id).sort()

describe('Postgres driver = DynamoDB, operation by operation', () => {
  test('put, get, overwrite, delete', async () => {
    const item = { email: 'p@x.edu', user_id: 'U1', name: 'P', classes: ['PHY', 'CHEM'], meta: { n: 1, empty: '', nul: null }, skip: undefined }
    await same(T.users, (t) => t.put(item).then(() => 'ok'))
    const got = await same(T.users, (t) => t.get({ email: 'p@x.edu' }))
    assert.deepEqual(got.ok, { email: 'p@x.edu', user_id: 'U1', name: 'P', classes: ['PHY', 'CHEM'], meta: { n: 1, empty: '', nul: null } })
    await same(T.users, (t) => t.put({ email: 'p@x.edu', user_id: 'U1b' }).then(() => 'ok'))
    assert.deepEqual((await same(T.users, (t) => t.get({ email: 'p@x.edu' }))).ok, { email: 'p@x.edu', user_id: 'U1b' })
    await same(T.users, (t) => t.delete({ email: 'p@x.edu' }).then(() => 'ok'))
    assert.equal((await same(T.users, (t) => t.get({ email: 'p@x.edu' }))).ok, undefined)
  })

  test('conditional put (createUser): taken email is a ConditionalCheckFailed', async () => {
    const create = (t) => t.put({ email: 'new@x.edu', user_id: 'N1' }, { ConditionExpression: 'attribute_not_exists(email)' }).then(() => 'ok')
    assert.deepEqual(await same(T.users, create), { ok: 'ok' })
    assert.deepEqual(await same(T.users, create), { error: 'ConditionalCheckFailedException' })
  })

  test('update: SET creates a missing item (upsert), REMOVE, if_not_exists null vs absent, ALL_NEW', async () => {
    await same(T.users, (t) => t.update({ Key: { email: 'ghost@x.edu' }, UpdateExpression: 'set last_login = :t', ExpressionAttributeValues: { ':t': '2026-01-01' } }))
    assert.deepEqual((await same(T.users, (t) => t.get({ email: 'ghost@x.edu' }))).ok, { email: 'ghost@x.edu', last_login: '2026-01-01' })

    await same(T.grading, (t) => t.put({ evaluation_id: 'E0', student_id: 'S', teacher_id: null, rubric: [1, 2], flag: 'x' }).then(() => 'ok'))
    const r = await same(T.grading, (t) =>
      t.update({
        Key: { evaluation_id: 'E0' },
        UpdateExpression: 'SET marks = :m, teacher_id = if_not_exists(teacher_id, :tid), owner_id = if_not_exists(owner_id, :tid) REMOVE flag, nothing_here',
        ExpressionAttributeValues: { ':m': 4.5, ':tid': 'T9' },
        ReturnValues: 'ALL_NEW',
      }),
    )
    assert.deepEqual(r.ok.Attributes, { evaluation_id: 'E0', student_id: 'S', teacher_id: null, rubric: [1, 2], marks: 4.5, owner_id: 'T9' })
  })

  test('conditional updates: #qn = :qn, #uid = :uid, attribute_exists on a missing item', async () => {
    await same(T.grading, (t) => t.put({ evaluation_id: 'E1', question_number: 1, student_answer: 'a' }).then(() => 'ok'))
    const correct = (qn) => (t) =>
      t.update({
        Key: { evaluation_id: 'E1' },
        UpdateExpression: 'SET student_answer = :sa',
        ConditionExpression: '#qn = :qn',
        ExpressionAttributeNames: { '#qn': 'question_number' },
        ExpressionAttributeValues: { ':sa': 'b', ':qn': qn },
      }).then(() => 'ok')
    assert.deepEqual(await same(T.grading, correct('1')), { error: 'ConditionalCheckFailedException' }) // "1" is not 1
    assert.deepEqual(await same(T.grading, correct(1)), { ok: 'ok' })
    assert.deepEqual(
      await same(T.users, (t) =>
        t.update({ Key: { email: 'nobody@x.edu' }, UpdateExpression: 'SET #s = :s', ConditionExpression: 'attribute_exists(email)', ExpressionAttributeNames: { '#s': 'status' }, ExpressionAttributeValues: { ':s': 'suspended' } }),
      ),
      { error: 'ConditionalCheckFailedException' },
    )
    assert.equal((await same(T.users, (t) => t.get({ email: 'nobody@x.edu' }))).ok, undefined) // a failed condition writes nothing
  })

  test('queries on indexes with = / <> on missing attributes / IN filters; unknown index', async () => {
    const rows = [
      { evaluation_id: 'G1', student_id: 'S1', assignment_id: 'A', qp_id: 'Q', question_number: 1 },
      { evaluation_id: 'G2', student_id: 'S1', assignment_id: 'A', qp_id: 'Q', question_number: 2 },
      { evaluation_id: 'G3', student_id: 'S1', assignment_id: 'A', qp_id: 'Q', type: 'summary', total_marks: 7 },
      { evaluation_id: 'G4', student_id: 'S1', assignment_id: 'B', qp_id: 'Q' },
      { evaluation_id: 'G5', student_id: 'S2', assignment_id: 'A', qp_id: 'Q' },
      { evaluation_id: 'G6', assignment_id: 'A', qp_id: 'Q' }, // no student_id: not in student_id-index
    ]
    for (const row of rows) await same(T.grading, (t) => t.put(row).then(() => 'ok'))
    // An index key attribute must be a string: DynamoDB refuses null or a number.
    for (const bad of [{ type: null }, { student_id: 5 }]) {
      assert.deepEqual(await same(T.grading, (t) => t.put({ evaluation_id: 'BAD', ...bad }).then(() => 'ok')), { error: 'ValidationException' })
    }
    const q = (FilterExpression, extraValues = {}) => (t) =>
      t.queryAll({
        IndexName: 'student_id-index',
        KeyConditionExpression: 'student_id = :sid',
        ...(FilterExpression ? { FilterExpression } : {}),
        ...(FilterExpression
          ? { ExpressionAttributeNames: Object.fromEntries([['#type', 'type'], ['#aid', 'assignment_id']].filter(([k]) => FilterExpression.includes(k))) }
          : {}),
        ExpressionAttributeValues: { ':sid': 'S1', ...extraValues },
      })
    assert.deepEqual(ids((await same(T.grading, q(), { sortBy: 'evaluation_id' })).ok), ['G1', 'G2', 'G3', 'G4'])
    const notSummary = await same(T.grading, q('#type <> :summary AND #aid = :aid', { ':summary': 'summary', ':aid': 'A' }), { sortBy: 'evaluation_id' })
    assert.deepEqual(ids(notSummary.ok), ['G1', 'G2']) // detail rows have no `type`: <> is true for them
    assert.deepEqual(ids((await same(T.grading, q('#aid IN (:a, :b)', { ':a': 'B', ':b': 'Z' }), { sortBy: 'evaluation_id' })).ok), ['G4'])
    const summaries = await same(
      T.grading,
      (t) => t.queryAll({ IndexName: 'type-index', KeyConditionExpression: '#type = :summary', FilterExpression: '#aid = :aid AND #qp = :qp', ExpressionAttributeNames: { '#type': 'type', '#aid': 'assignment_id', '#qp': 'qp_id' }, ExpressionAttributeValues: { ':summary': 'summary', ':aid': 'A', ':qp': 'Q' } }),
      { sortBy: 'evaluation_id' },
    )
    assert.deepEqual(ids(summaries.ok), ['G3'])
    const byAssignment = await same(
      T.grading,
      (t) => t.queryAll({ IndexName: T.gradingAssignmentIndex, KeyConditionExpression: '#aid = :aid', FilterExpression: '#qp = :qp', ExpressionAttributeNames: { '#aid': 'assignment_id', '#qp': 'qp_id' }, ExpressionAttributeValues: { ':aid': 'A', ':qp': 'Q' } }),
      { sortBy: 'evaluation_id' },
    )
    assert.deepEqual(ids(byAssignment.ok), ['G1', 'G2', 'G3', 'G5', 'G6'])
    const missing = await same(T.grading, (t) => t.queryAll({ IndexName: 'no-such-index', KeyConditionExpression: 'x = :x', ExpressionAttributeValues: { ':x': 'y' } }))
    assert.equal(missing.error, 'ValidationException')
    assert.ok(postgres.isMissingIndexError({ name: missing.error }))
  })

  test('submissions: base-table query is sorted by student_id (the range key)', async () => {
    for (const sid of ['s3', 's1', 's2']) {
      await same(T.submissions, (t) => t.update({ Key: { assignment_id: 'A9', student_id: sid }, UpdateExpression: 'SET #s = :s', ExpressionAttributeNames: { '#s': 'status' }, ExpressionAttributeValues: { ':s': 'answer_uploaded' } }))
    }
    const r = await same(T.submissions, (t) => t.queryAll({ KeyConditionExpression: 'assignment_id = :a', ExpressionAttributeValues: { ':a': 'A9' } }))
    assert.deepEqual(r.ok.map((s) => s.student_id), ['s1', 's2', 's3'])
    assert.deepEqual((await same(T.submissions, (t) => t.get({ assignment_id: 'A9', student_id: 's2' }))).ok, { assignment_id: 'A9', student_id: 's2', status: 'answer_uploaded' })
  })

  test('scans with filters and projections', async () => {
    for (const a of [
      { assignment_id: 'X1', teacher_id: 'T1', content_status: 'published', class_name: 'PHY' },
      { assignment_id: 'X2', teacher_id: 'T2', content_status: 'draft', class_name: 'CHEM' },
      { assignment_id: 'X3', teacher_id: 'T1', content_status: 'published', class_name: 'BIO' },
    ]) {
      await same(T.assignments, (t) => t.put(a).then(() => 'ok'))
    }
    const published = await same(T.assignments, (t) => t.scanAll({ FilterExpression: 'content_status = :p', ExpressionAttributeValues: { ':p': 'published' } }), { sortBy: 'assignment_id' })
    assert.deepEqual(published.ok.map((a) => a.assignment_id).sort(), ['X1', 'X3'])
    const byTeacher = await same(T.assignments, (t) => t.queryAll({ IndexName: 'teacher_id-index', KeyConditionExpression: 'teacher_id = :t', ExpressionAttributeValues: { ':t': 'T1' } }), { sortBy: 'assignment_id' })
    assert.deepEqual(byTeacher.ok.map((a) => a.assignment_id).sort(), ['X1', 'X3']) // index order isn't guaranteed
    await same(T.users, (t) => t.put({ email: 'q@x.edu', user_id: 'Q1', name: 'Q', userRole: 'student', password: 'secret' }).then(() => 'ok'))
    const directory = await same(T.users, (t) => t.scanAll({ ProjectionExpression: 'user_id, email, #n', ExpressionAttributeNames: { '#n': 'name' } }), { sortBy: 'email' })
    assert.ok(directory.ok.every((u) => !('password' in u)))
  })

  test('unused ExpressionAttributeNames / Values are refused, as DynamoDB does', async () => {
    const extraName = await same(T.users, (t) => t.scanAll({ FilterExpression: 'email = :e', ExpressionAttributeNames: { '#n': 'name' }, ExpressionAttributeValues: { ':e': 'x' } }))
    assert.equal(extraName.error, 'ValidationException')
    const extraValue = await same(T.users, (t) => t.update({ Key: { email: 'q@x.edu' }, UpdateExpression: 'SET a = :a', ExpressionAttributeValues: { ':a': 1, ':b': 2 } }))
    assert.equal(extraValue.error, 'ValidationException')
  })

  test('changing a key attribute is refused', async () => {
    const r = await same(T.grading, (t) => t.update({ Key: { evaluation_id: 'E0' }, UpdateExpression: 'SET evaluation_id = :x', ExpressionAttributeValues: { ':x': 'E-other' } }))
    assert.equal(r.error, 'ValidationException')
  })
})

describe('expression parser', () => {
  test('constructs the services never use fail loudly', () => {
    const bad = [
      () => compileCondition('a = :v OR b = :v', {}, { ':v': 1 }),
      () => compileCondition('begins_with(a, :v)', {}, { ':v': 1 }),
      () => compileCondition('a > :v', {}, { ':v': 1 }),
      () => compileCondition('a.b = :v', {}, { ':v': 1 }),
      () => compileCondition('#missing = :v', {}, { ':v': 1 }),
      () => compileUpdate('ADD counter :one', {}, { ':one': 1 }),
      () => compileUpdate('SET n = n + :one', {}, { ':one': 1 }),
    ]
    for (const fn of bad) assert.throws(fn, UnsupportedExpression)
  })

  test('the Postgres driver rejects request parameters it does not implement', async () => {
    await assert.rejects(postgres.table(T.users).scanAll({ Limit: 5 }), UnsupportedExpression)
  })
})
