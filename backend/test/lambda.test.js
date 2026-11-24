// The per-path Lambda functions (lambdas/): each handler, fed real API Gateway
// events, must answer exactly like the Express app does for the same request.
import assert from 'node:assert/strict'
import { after, before, describe, test } from 'node:test'
import { post, putUsers, startServer, stopServer } from './helpers.js'

const { hashLoginPassword } = await import('../src/lib/passwords.js')
const { createHandler, toRequest } = await import('../lambdas/adapter.js')
const { FUNCTIONS } = await import('../lambdas/functions.js')

const PASSWORD = 'Lambda-Test-Pass1!'
const ctx = { awsRequestId: 'req-test-1' }
const handlers = Object.fromEntries(Object.keys(FUNCTIONS).map((n) => [n, createHandler(n)]))

/** API Gateway REST (payload 1.0) proxy event, as the existing /dev API sends. */
const restEvent = (method, path, body, headers = {}) => ({
  resource: '/{proxy+}',
  path,
  httpMethod: method,
  headers: { 'Content-Type': 'application/json', ...headers },
  multiValueHeaders: Object.fromEntries(Object.entries({ 'Content-Type': 'application/json', ...headers }).map(([k, v]) => [k, [v]])),
  queryStringParameters: null,
  multiValueQueryStringParameters: null,
  requestContext: { stage: 'dev', identity: { sourceIp: '203.0.113.9' } },
  body: body === undefined ? null : typeof body === 'string' ? body : JSON.stringify(body),
  isBase64Encoded: false,
})

/** HTTP API / Function URL (payload 2.0) event. */
const httpEvent = (method, rawPath, body, { stage = '$default', headers = {} } = {}) => ({
  version: '2.0',
  rawPath,
  rawQueryString: '',
  headers: { 'content-type': 'application/json', ...headers },
  requestContext: { stage, http: { method, sourceIp: '203.0.113.9' } },
  body: body === undefined ? undefined : Buffer.from(JSON.stringify(body)).toString('base64'),
  isBase64Encoded: true,
})

const parse = (r) => (r.body ? JSON.parse(r.body) : null)

let token
before(async () => {
  await startServer()
  const password = await hashLoginPassword(PASSWORD)
  await putUsers([
    { email: 't@x.edu', user_id: 'T1', name: 'Priya', userRole: 'teacher', password },
    { email: 's@x.edu', user_id: 'S1', name: 'Okafor', userRole: 'student', password },
  ])
})
after(async () => {
  await Promise.all(Object.values(handlers).map((h) => h.close()))
  await stopServer()
})

describe('lambda functions', () => {
  test('every API path in config/app.yaml belongs to exactly one function', async () => {
    const { default: config } = await import('../src/config/index.js')
    const p = config.app.paths
    const owned = Object.values(FUNCTIONS).flatMap((f) => f.paths(p))
    const expected = [p.health, p.healthReady, p.auth.base, p.upload, p.grades, p.corrections, p.marks, p.workflow, p.admin, p.contact]
    assert.deepEqual([...owned].sort(), [...expected].sort())
  })

  test('health answers GET /health', async () => {
    const r = await handlers.health(restEvent('GET', '/health'), ctx)
    assert.equal(r.statusCode, 200)
    assert.equal(parse(r).status, 'ok')
  })

  test('auth logs in exactly like the Express app', async () => {
    const body = { email: 't@x.edu', password: PASSWORD, userType: 'teacher' }
    const r = await handlers.auth(restEvent('POST', '/auth/login', body), ctx)
    const [status, direct] = await post('/auth/login', body)
    assert.equal(r.statusCode, status)
    assert.equal(r.statusCode, 200)
    token = parse(r).token
    assert.ok(token, 'login returns a token')
    assert.deepEqual(Object.keys(parse(r)).sort(), Object.keys(direct).sort())
  })

  test('a wrong password is refused the same way', async () => {
    const body = { email: 't@x.edu', password: 'nope', userType: 'teacher' }
    const r = await handlers.auth(restEvent('POST', '/auth/login', body), ctx)
    const [status, direct] = await post('/auth/login', body)
    assert.deepEqual([r.statusCode, parse(r)], [status, direct])
  })

  test('grades: same status and body as the Express app (REST event)', async () => {
    const body = { operation: 'getAllStudentGrades', qp_id: 'QP-1', assignment_id: 'A-1' }
    const r = await handlers.grades(restEvent('POST', '/getgrades', body, { Authorization: `Bearer ${token}` }), ctx)
    const [status, direct] = await post('/getgrades', body, { token })
    assert.deepEqual([r.statusCode, parse(r)], [status, direct])
  })

  test('grades: HTTP API event with a named stage and a base64 body', async () => {
    const body = { operation: 'getAllStudentGrades', qp_id: 'QP-1', assignment_id: 'A-1' }
    const r = await handlers.grades(httpEvent('POST', '/dev/getgrades', body, { stage: 'dev', headers: { authorization: `Bearer ${token}` } }), ctx)
    const [status, direct] = await post('/getgrades', body, { token })
    assert.deepEqual([r.statusCode, parse(r)], [status, direct])
    assert.ok(r.headers, 'payload 2.0 response uses headers')
  })

  test('no token is a 401, as before', async () => {
    const r = await handlers.marks(restEvent('POST', '/updatedb', { operation: 'updateMarks' }), ctx)
    const [status, direct] = await post('/updatedb', { operation: 'updateMarks' })
    assert.deepEqual([r.statusCode, parse(r)], [status, direct])
    assert.equal(r.statusCode, 401)
  })

  test('invalid JSON gets the endpoint’s 400', async () => {
    const r = await handlers.workflow(restEvent('POST', '/workflow', '{not json', { Authorization: `Bearer ${token}` }), ctx)
    const [status, direct] = await post('/workflow', null, { token, raw: '{not json' })
    assert.deepEqual([r.statusCode, parse(r)], [status, direct])
  })

  test('a function never serves another function’s path', async () => {
    const r = await handlers.grades(restEvent('POST', '/workflow', { operation: 'listAssignments' }, { Authorization: `Bearer ${token}` }), ctx)
    assert.equal(r.statusCode, 404)
    const h = await handlers.contact(restEvent('GET', '/health'), ctx)
    assert.equal(h.statusCode, 404)
  })

  test('CORS preflight is answered by the function', async () => {
    const r = await handlers.workflow(
      restEvent('OPTIONS', '/workflow', undefined, { Origin: 'https://app.gradeant.io', 'Access-Control-Request-Method': 'POST' }),
      ctx,
    )
    assert.equal(r.statusCode, 204)
    assert.ok(r.multiValueHeaders['access-control-allow-origin'], 'allow-origin header present')
  })

  test('event translation: query strings, multi-value headers, request id, client IP', () => {
    const ev = restEvent('GET', '/health', undefined, { Accept: 'application/json' })
    ev.multiValueQueryStringParameters = { a: ['1', '2'], b: ['x y'] }
    const req = toRequest(ev, 'rid-1')
    assert.equal(req.path, '/health?a=1&a=2&b=x+y')
    assert.equal(req.headers.get('lambda-web-request-id'), 'rid-1')
    assert.equal(req.headers.get('x-forwarded-for'), '203.0.113.9')
    assert.equal(req.body, undefined)
    const v2 = toRequest(httpEvent('POST', '/dev', { x: 1 }, { stage: 'dev' }))
    assert.equal(v2.path, '/')
  })
})
