// Password hashing in the two storage formats already in the tables, so
// existing accounts keep working:
//   Users.password (login)          base64(32-byte salt + PBKDF2-SHA256, 100k)
//   GradeAnt_Users.password_hash    "<salt hex>:<hash hex>", PBKDF2-SHA256, 200k
// Hashing runs on the libuv pool so it never blocks the event loop.
import { pbkdf2, randomBytes, randomInt, timingSafeEqual } from 'node:crypto'
import { promisify } from 'node:util'

const pbkdf2Async = promisify(pbkdf2)

export async function hashLoginPassword(password, salt = randomBytes(32)) {
  const hash = await pbkdf2Async(Buffer.from(password, 'utf8'), salt, 100_000, 32, 'sha256')
  return Buffer.concat([salt, hash]).toString('base64')
}

export async function verifyLoginPassword(stored, provided) {
  const raw = Buffer.from(String(stored), 'base64')
  const salt = raw.subarray(0, 32)
  const expected = raw.subarray(32)
  const actual = await pbkdf2Async(Buffer.from(String(provided), 'utf8'), salt, 100_000, 32, 'sha256')
  return expected.length === actual.length && timingSafeEqual(expected, actual)
}

export async function hashAdminPassword(password, saltHex) {
  const salt = saltHex ? Buffer.from(saltHex, 'hex') : randomBytes(16)
  const digest = await pbkdf2Async(Buffer.from(password, 'utf8'), salt, 200_000, 32, 'sha256')
  return `${salt.toString('hex')}:${digest.toString('hex')}`
}

export async function verifyAdminPassword(password, stored) {
  if (!stored || !String(stored).includes(':')) return false
  const [saltHex] = String(stored).split(':', 1)
  const candidate = Buffer.from(await hashAdminPassword(password, saltHex))
  const expected = Buffer.from(String(stored))
  return candidate.length === expected.length && timingSafeEqual(candidate, expected)
}

/** Same rules the change-password page shows: 8+ chars, an uppercase letter, a number. */
export function passwordProblem(password) {
  if (Array.from(password).length < 8) return 'Password must be at least 8 characters.'
  if (!/\p{Lu}/u.test(password)) return 'Password must include an uppercase letter.'
  if (!/\p{Nd}/u.test(password)) return 'Password must include a number.'
  return null
}

/** Human-typeable temp password: no ambiguous chars (0/O, 1/l/I). */
export function generateTempPassword() {
  const alphabet = 'ABCDEFGHJKMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789'
  let out = ''
  for (let i = 0; i < 12; i++) out += alphabet[randomInt(alphabet.length)]
  return out
}
