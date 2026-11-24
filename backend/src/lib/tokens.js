// Two token formats, both inherited from the Python Lambdas:
//  - HS256 JWTs issued by /auth/login (payload: user_id, email, iat, exp)
//  - the admin console's own HMAC session tokens, issued by completeFirstLogin:
//    base64url(json(claims)) + "." + hex(hmac_sha256(that base64 string))
import { createHmac, timingSafeEqual } from 'node:crypto'
import jwt from 'jsonwebtoken'
import { getJwtSecret, getSessionTokenSecret } from '../aws/secrets.js'
import config from '../config/index.js'
import { unauthorized } from './errors.js'

const nowSeconds = () => Math.floor(Date.now() / 1000)

export async function issueLoginToken(userId, email) {
  const iat = nowSeconds()
  const payload = { user_id: userId, email, exp: iat + config.auth.tokenExpirySeconds, iat }
  return jwt.sign(payload, await getJwtSecret(), { algorithm: 'HS256' })
}

/** Payload of a valid login JWT, or a reason it's invalid: 'expired' | 'invalid'. */
export async function decodeLoginToken(token) {
  try {
    return { payload: jwt.verify(token, await getJwtSecret(), { algorithms: ['HS256'] }) }
  } catch (err) {
    return { error: err?.name === 'TokenExpiredError' ? 'expired' : 'invalid' }
  }
}

const b64url = (buf) => Buffer.from(buf).toString('base64url')
const fromB64url = (s) => {
  if (!/^[A-Za-z0-9_-]*$/.test(s)) throw new Error('bad base64')
  return Buffer.from(s, 'base64url')
}
const safeEqual = (a, b) => {
  const x = Buffer.from(a)
  const y = Buffer.from(b)
  return x.length === y.length && timingSafeEqual(x, y)
}

/**
 * lambda_admin.py's _verify_login_jwt: verifies the login JWT by hand with the
 * admin console's error messages (and, like it, treats a missing exp as expired).
 */
export async function verifyLoginJwtForAdmin(token) {
  const secret = await getJwtSecret()
  if (!secret) throw unauthorized('Server misconfigured: JWT_SECRET is not set.')
  let header, payload
  const [headerB64, payloadB64, signatureB64] = token.split('.')
  try {
    header = JSON.parse(fromB64url(headerB64).toString('utf8'))
    payload = JSON.parse(fromB64url(payloadB64).toString('utf8'))
  } catch {
    throw unauthorized('Malformed session token.')
  }
  if (header?.alg !== 'HS256') throw unauthorized('Invalid session token.')
  const expected = b64url(createHmac('sha256', secret).update(`${headerB64}.${payloadB64}`).digest())
  if (!safeEqual(expected, signatureB64)) throw unauthorized('Invalid session token.')
  if (!((payload?.exp ?? 0) >= Date.now() / 1000)) throw unauthorized('Session expired.')
  return payload
}

export async function makeSessionToken(claims, ttlSeconds) {
  const payloadB64 = b64url(JSON.stringify({ ...claims, exp: nowSeconds() + ttlSeconds }))
  const signature = createHmac('sha256', await sessionSecret()).update(payloadB64).digest('hex')
  return `${payloadB64}.${signature}`
}

export async function verifySessionToken(token) {
  const dot = token.indexOf('.')
  if (dot < 0) throw unauthorized('Malformed session token.')
  const payloadB64 = token.slice(0, dot)
  const signature = token.slice(dot + 1)
  const expected = createHmac('sha256', await sessionSecret()).update(payloadB64).digest('hex')
  if (!safeEqual(expected, signature)) throw unauthorized('Invalid session token.')
  let claims
  try {
    claims = JSON.parse(fromB64url(payloadB64).toString('utf8'))
  } catch {
    throw unauthorized('Malformed session token.')
  }
  if (!((claims?.exp ?? 0) >= Date.now() / 1000)) throw unauthorized('Session expired.')
  return claims
}

async function sessionSecret() {
  const secret = await getSessionTokenSecret()
  if (secret) return secret
  // lambda_admin.py fell back to an insecure default; only allow that outside production.
  if (config.isProduction) throw unauthorized('Server misconfigured: SESSION_TOKEN_SECRET is not set.')
  return 'dev-only-insecure-secret-CHANGE-ME'
}
