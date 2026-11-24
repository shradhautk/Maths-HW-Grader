// Runs the Express API inside a plain AWS Lambda handler, one function per API
// path (functions.js). Each execution environment starts the real Express app
// once on 127.0.0.1 (random port) and forwards every invocation to it as an HTTP
// request, so routing, auth, CORS and error handling behave exactly as they do
// locally and on Lambda Web Functions. No extra dependency.
//
// Accepts API Gateway REST (payload 1.0) and HTTP API / Function URL (2.0)
// events and answers in the same format.
import { createServer } from 'node:http'
import express from 'express'
import config from '../src/config/index.js'
import { initDatabase } from '../src/aws/dynamodb.js'
import { cors } from '../src/middleware/cors.js'
import { notFoundHandler } from '../src/middleware/errors.js'
import { requestLogger } from '../src/middleware/requestLogger.js'
import routes from '../src/routes/index.js'
import { FUNCTIONS } from './functions.js'

/** The Express app for one function: the shared routes, limited to that function's paths. */
export function buildApp(name) {
  const spec = FUNCTIONS[name]
  if (!spec) throw new Error(`Unknown function "${name}" (expected one of: ${Object.keys(FUNCTIONS).join(', ')})`)
  const base = config.server.basePath || ''
  const prefixes = spec.paths(config.app.paths).map((p) => base + p)
  const app = express()
  app.disable('x-powered-by')
  app.set('env', config.isProduction ? 'production' : 'development')
  app.use(requestLogger())
  app.use(cors())
  // A request routed to the wrong function gets a 404, never another function's code.
  app.use((req, res, next) => (prefixes.some((p) => req.path === p || req.path.startsWith(`${p}/`)) ? next() : notFoundHandler(req, res)))
  app.use(base || '/', routes)
  app.use(notFoundHandler)
  return app
}

const HOP_BY_HOP = new Set(['host', 'connection', 'content-length', 'transfer-encoding', 'keep-alive', 'expect', 'upgrade'])
const TEXT_TYPES = /json|text|javascript|xml|x-www-form-urlencoded/i

/** API Gateway / Function URL event → { method, path, headers, body } for the local server. */
export function toRequest(event, requestId) {
  const v2 = event.version === '2.0'
  const method = (v2 ? event.requestContext?.http?.method : event.httpMethod) || 'GET'
  let path = (v2 ? event.rawPath : event.path) || '/'
  // HTTP APIs with a named stage put it in the path (/dev/getgrades); REST APIs don't.
  const stage = event.requestContext?.stage
  if (v2 && stage && stage !== '$default' && (path === `/${stage}` || path.startsWith(`/${stage}/`))) path = path.slice(stage.length + 1) || '/'

  let query = ''
  if (v2) query = event.rawQueryString || ''
  else {
    const multi = event.multiValueQueryStringParameters
    const single = event.queryStringParameters
    const params = new URLSearchParams()
    if (multi) for (const [k, vs] of Object.entries(multi)) for (const v of vs ?? []) params.append(k, v)
    else if (single) for (const [k, v] of Object.entries(single)) params.append(k, v ?? '')
    query = params.toString()
  }

  const headers = new Headers()
  if (!v2 && event.multiValueHeaders) {
    for (const [k, vs] of Object.entries(event.multiValueHeaders)) for (const v of vs ?? []) if (!HOP_BY_HOP.has(k.toLowerCase())) headers.append(k, v)
  } else {
    for (const [k, v] of Object.entries(event.headers ?? {})) if (v != null && !HOP_BY_HOP.has(k.toLowerCase())) headers.set(k, v)
  }
  if (v2 && Array.isArray(event.cookies) && event.cookies.length) headers.set('cookie', event.cookies.join('; '))
  if (requestId) headers.set('lambda-web-request-id', requestId) // the app's logs key on this header
  // The caller's IP for rate limiting / logs, if API Gateway didn't already add it.
  const ip = v2 ? event.requestContext?.http?.sourceIp : event.requestContext?.identity?.sourceIp
  if (ip && !headers.has('x-forwarded-for')) headers.set('x-forwarded-for', ip)

  const hasBody = event.body != null && event.body !== '' && method !== 'GET' && method !== 'HEAD'
  const body = hasBody ? Buffer.from(event.body, event.isBase64Encoded ? 'base64' : 'utf8') : undefined
  return { v2, method, path: path + (query ? `?${query}` : ''), headers, body }
}

/** The local server's reply → the response shape API Gateway expects for the event's format. */
async function toResponse(res, v2) {
  const buf = Buffer.from(await res.arrayBuffer())
  const type = res.headers.get('content-type') || ''
  const text = buf.length === 0 || TEXT_TYPES.test(type)
  const body = text ? buf.toString('utf8') : buf.toString('base64')
  const cookies = res.headers.getSetCookie()
  const single = {}
  const multi = {}
  res.headers.forEach((value, key) => {
    if (key === 'set-cookie' || HOP_BY_HOP.has(key)) return
    single[key] = value
    multi[key] = [value]
  })
  if (v2) return { statusCode: res.status, headers: single, ...(cookies.length ? { cookies } : {}), body, isBase64Encoded: !text }
  if (cookies.length) multi['set-cookie'] = cookies
  return { statusCode: res.status, multiValueHeaders: multi, body, isBase64Encoded: !text }
}

/**
 * The Lambda handler for one function. The server starts on the first
 * invocation of each execution environment and is reused afterwards.
 */
export function createHandler(name) {
  let origin
  let server
  const start = async () => {
    await initDatabase()
    server = createServer(buildApp(name))
    server.keepAliveTimeout = 60_000
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
    return `http://127.0.0.1:${server.address().port}`
  }

  async function handler(event, context) {
    origin ??= start().catch((err) => {
      origin = undefined // retry the start on the next invocation
      throw err
    })
    const base = await origin
    const req = toRequest(event ?? {}, context?.awsRequestId)
    try {
      const res = await fetch(base + req.path, { method: req.method, headers: req.headers, body: req.body, redirect: 'manual' })
      return await toResponse(res, req.v2)
    } catch (err) {
      console.error(JSON.stringify({ level: 'error', msg: 'lambda adapter failed', fn: name, requestId: context?.awsRequestId, error: err?.name, detail: err?.message }))
      const body = JSON.stringify({ error: 'Internal server error' })
      const headers = { 'content-type': 'application/json' }
      return req.v2 ? { statusCode: 502, headers, body } : { statusCode: 502, multiValueHeaders: { 'content-type': ['application/json'] }, body }
    }
  }

  /** Stops the local server (tests; Lambda simply freezes the environment). */
  handler.close = async () => {
    const s = server
    server = undefined
    origin = undefined
    if (s) await new Promise((resolve) => s.close(resolve).closeAllConnections())
  }
  return handler
}
