// Test harness: the real Express app on a random port, backed by a fresh
// in-memory DynamoDB per suite.
process.env.APP_ENV = 'test'

const { default: jwt } = await import('jsonwebtoken')
const { default: config } = await import('../src/config/index.js')
const { createApp } = await import('../src/app.js')
const dynamo = await import('../src/aws/dynamodb.js')

export { config }
export const table = dynamo.table

let server
export let baseUrl

export async function startServer({ assignmentIndex = true, seed = false } = {}) {
  await stopServer()
  await dynamo.initDatabase({ seed, assignmentIndex })
  await new Promise((resolve) => {
    server = createApp().listen(0, '127.0.0.1', resolve)
  })
  baseUrl = `http://127.0.0.1:${server.address().port}`
  config.server.publicUrl = baseUrl // local-storage URLs must point at this test server
  return baseUrl
}

export async function stopServer() {
  if (server) await new Promise((resolve) => server.close(resolve))
  server = undefined
  await dynamo.closeDatabase()
}

export function tokenFor(user, { secret = config.auth.jwtSecret, expiresIn = 600 } = {}) {
  return jwt.sign(
    { email: user.email, user_id: user.user_id, exp: Math.floor(Date.now() / 1000) + expiresIn },
    secret,
    { algorithm: 'HS256' },
  )
}

/** POST JSON; returns [status, body, headers]. */
export async function post(path, body, { token, headers = {}, raw } = {}) {
  const res = await fetch(`${baseUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}), ...headers },
    body: raw ?? JSON.stringify(body ?? {}),
  })
  const text = await res.text()
  let json = null
  try {
    json = text ? JSON.parse(text) : null
  } catch {
    json = text
  }
  return [res.status, json, res.headers]
}

export async function get(path) {
  const res = await fetch(`${baseUrl}${path}`)
  return [res.status, await res.json().catch(() => null), res.headers]
}

export async function putUsers(users) {
  for (const u of users) await table(config.tables.users).put(u)
}
