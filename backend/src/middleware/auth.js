// Bearer-token authentication for /getgrades, /correctanswer, /updatedb and
// /workflow (same rules as their Python Lambdas): an HS256 JWT from
// /auth/login, whose account must still exist. The role is read fresh from the
// Users table on every request, so a demoted or deleted user loses access at once.
import { getJwtSecret } from '../aws/secrets.js'
import { HttpError, unauthorized } from '../lib/errors.js'
import { decodeLoginToken } from '../lib/tokens.js'
import { getUserByEmail } from '../repositories/users.repository.js'

export function bearerToken(req) {
  const header = req.get('authorization') ?? ''
  return header.startsWith('Bearer ') ? header.slice('Bearer '.length).trim() : null
}

/** Returns { user_id, email, role, user } or throws a 401 HttpError. */
export async function authenticate(req) {
  if (!(await getJwtSecret())) throw new HttpError(500, 'Server misconfigured: JWT_SECRET is not set.')
  const token = bearerToken(req)
  if (token === null) throw unauthorized('Missing Authorization: Bearer <token> header.')
  const { payload, error } = await decodeLoginToken(token)
  if (error === 'expired') throw unauthorized('Session expired — please log in again.')
  if (error) throw unauthorized('Invalid session token.')
  const { email, user_id: userId } = payload
  if (!email || !userId) throw unauthorized('Malformed session token.')
  const user = await getUserByEmail(email)
  if (!user || user.user_id !== userId) throw unauthorized('Account no longer exists.')
  if (user.status === 'suspended') throw unauthorized('This account is suspended.')
  return { user_id: userId, email, role: user.userRole ?? 'student', user }
}

/** Express middleware: sets req.caller. */
export const requireCaller = async (req, res, next) => {
  req.caller = await authenticate(req)
  next()
}

/** Throws 403 unless the caller has one of `roles`. */
export function requireRole(caller, roles, message) {
  if (!roles.includes(caller.role)) throw new HttpError(403, message)
}
