// /auth/* — port of lambda_python_files/authetication.py. Same request and
// response bodies, status codes and messages.
import { randomUUID } from 'node:crypto'
import { storage } from '../aws/s3.js'
import config from '../config/index.js'
import { HttpError, badRequest, reply, unauthorized } from '../lib/errors.js'
import { hashLoginPassword, passwordProblem, verifyLoginPassword } from '../lib/passwords.js'
import { canSignUp, roleLabel, selfSignupRoles } from '../lib/permissions.js'
import { isoNaive } from '../lib/py.js'
import { decodeLoginToken, issueLoginToken } from '../lib/tokens.js'
import { pendingResetRequestsFor, resetRequests } from '../repositories/admin.repository.js'
import { getUserByEmail, putUser, setLastLogin, setPassword, setPhotoKey } from '../repositories/users.repository.js'

const PHOTO_URL_TTL = 3600
const PHOTO_TYPES = { 'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp' }
const PHOTO_MAX_BYTES = 5 * 1024 * 1024

/** body['field'] with Python's KeyError → 400 "Missing required field: 'field'". */
function field(obj, name) {
  if (!obj || typeof obj !== 'object' || !(name in obj)) throw badRequest(`Missing required field: '${name}'`)
  return obj[name]
}

async function photoViewUrl(user) {
  const key = user.photo_key
  const bucket = config.storage.profilePhotoBucket
  if (!key || !bucket) return null
  return storage().presignGet(bucket, key, PHOTO_URL_TTL)
}

/** The Users item for a valid bearer token, or null. */
async function authenticatedUser(token) {
  if (!token) return null
  const { payload } = await decodeLoginToken(token)
  if (!payload) return null
  const item = await getUserByEmail(payload.email)
  if (!item || item.user_id !== payload.user_id || item.status === 'suspended') return null
  return item
}

/**
 * The Users item for a typed email. Tries it as typed (trimmed) first, so
 * accounts stored with capitals keep working, then lowercased, so
 * "Teacher@School.edu" or a trailing space from autofill still finds the account.
 */
async function findUserByTypedEmail(email) {
  const typed = String(email ?? '').trim()
  if (!typed) return null
  const exact = await getUserByEmail(typed)
  if (exact) return exact
  const lower = typed.toLowerCase()
  return lower === typed ? null : getUserByEmail(lower)
}

export async function signup(body) {
  // New accounts are stored trimmed and lowercased; login matches either form.
  const email = String(field(body, 'email') ?? '').trim().toLowerCase()
  if (!email) throw badRequest('Email is required.')
  const password = field(body, 'password')
  const name = body.name ?? ''
  const userRole = body.role || 'student'
  // Which roles can sign themselves up is set in config/app.yaml (roles.<role>.signup).
  if (!canSignUp(userRole, 'self')) {
    if (canSignUp(userRole, 'invite')) {
      throw new HttpError(403, `${roleLabel(userRole)} accounts are invite-only. Ask your school administrator for an invite.`)
    }
    throw badRequest(`Role must be one of: ${selfSignupRoles().join(', ')}.`)
  }
  const problem = passwordProblem(String(password))
  if (problem) throw badRequest(problem)
  if (await getUserByEmail(email)) return reply(409, { message: 'User already exists' })

  const userId = randomUUID()
  const timestamp = isoNaive()
  await putUser({
    user_id: userId,
    email,
    name,
    password: await hashLoginPassword(String(password)),
    userRole,
    created_at: timestamp,
    updated_at: timestamp,
  })
  return reply(201, {
    message: 'User created successfully',
    user_id: userId,
    userRole,
    name,
    email,
    token: await issueLoginToken(userId, email),
  })
}

export async function login(body) {
  const password = field(body, 'password')
  const userRole = field(body, 'userType')
  const typedEmail = field(body, 'email')
  const user = await findUserByTypedEmail(typedEmail)
  // The reply is always "Invalid credentials" (no account probing); in
  // development the API log also says which check failed. Never the password.
  const reject = (reason) => {
    if (config.mode === 'development') {
      console.log(JSON.stringify({ level: 'warn', msg: `login rejected: ${reason}`, email: String(typedEmail), role: userRole }))
    }
    return unauthorized('Invalid credentials')
  }
  if (!user) throw reject('no account with this email (the local in-memory DB is reset whenever the API restarts)')
  if (userRole !== field(user, 'userRole')) throw reject(`role mismatch: account is "${user.userRole}", login page sent "${userRole}"`)
  if (!(await verifyLoginPassword(field(user, 'password'), password))) throw reject('wrong password')
  // Checked after the password so the message can't be used to probe which accounts exist.
  if (user.status === 'suspended') throw new HttpError(403, 'This account is suspended. Contact your administrator.')

  const token = await issueLoginToken(user.user_id, user.email)
  await setLastLogin(user.email, isoNaive())
  return {
    message: 'Login successful',
    user_id: user.user_id,
    userRole: user.userRole,
    name: field(user, 'name'),
    email: user.email,
    must_change_password: Boolean(user.must_change_password),
    photoUrl: await photoViewUrl(user),
    token,
  }
}

export async function logout(authorization) {
  if (!authorization || !authorization.startsWith('Bearer ')) throw unauthorized('Invalid authorization header')
  const token = authorization.split(' ')[1]
  const { payload } = await decodeLoginToken(token ?? '')
  if (!payload) throw unauthorized('Invalid or expired token')
  return { message: 'Logout successful' }
}

/**
 * Teachers/students can't reset their own password — this files a request for
 * an admin. Always answers the same way so it can't reveal which emails exist.
 */
export async function forgotPassword(body) {
  const generic = { message: 'If that account exists, your password reset request has been sent to the administrator.' }
  const email = String(body.email ?? '').trim()
  if (!email) throw badRequest('Email is required.')
  try {
    const user = await getUserByEmail(email)
    // Admins reset their own password from their profile page.
    if (!user || !['student', 'teacher'].includes(user.userRole)) return generic
    if ((await pendingResetRequestsFor(user.user_id)).length) return generic
    await resetRequests.put({
      request_id: randomUUID(),
      user_id: user.user_id,
      user_name: user.name ?? '',
      user_email: user.email,
      role: user.userRole,
      requested_at: Math.floor(Date.now() / 1000),
      status: 'pending',
      source: 'login',
    })
    return generic
  } catch (err) {
    console.error(JSON.stringify({ level: 'error', msg: 'forgot password failed', error: err?.message }))
    throw new HttpError(500, 'Could not submit your request. Please try again.')
  }
}

/** Authenticated change; also clears the must-change flag set by an admin-issued temp password. */
export async function changePassword(token, body) {
  const user = await authenticatedUser(token)
  if (!user) throw unauthorized('Invalid or expired token')
  const current = body.currentPassword || ''
  const next = body.newPassword || ''
  if (!(await verifyLoginPassword(user.password, current))) throw badRequest('Current password is incorrect.')
  const problem = passwordProblem(next)
  if (problem) throw badRequest(problem)
  if (next === current) throw badRequest('Choose a password different from your current one.')
  await setPassword(user.email, await hashLoginPassword(next), isoNaive())
  return { message: 'Password updated.' }
}

/** Fresh copy of the account fields the profile page shows. */
export async function me(token) {
  const user = await authenticatedUser(token)
  if (!user) throw unauthorized('Invalid or expired token')
  return {
    user_id: user.user_id,
    name: user.name ?? '',
    email: user.email,
    userRole: user.userRole ?? '',
    status: user.status ?? 'active',
    must_change_password: Boolean(user.must_change_password),
    photoUrl: await photoViewUrl(user),
  }
}

/** Two steps: action=upload-url returns a presigned PUT, then action=confirm records it. */
export async function profilePhoto(token, body) {
  const user = await authenticatedUser(token)
  if (!user) throw unauthorized('Invalid or expired token')
  const bucket = config.storage.profilePhotoBucket
  if (!bucket) throw new HttpError(503, 'Profile photos are not configured.')

  const isPhotoType = Object.hasOwn(PHOTO_TYPES, String(body.contentType))
  const key = `profile-photos/${user.user_id}.${isPhotoType ? PHOTO_TYPES[body.contentType] : ''}`
  const store = storage()

  if (body.action === 'upload-url') {
    if (!isPhotoType) throw badRequest('Use a JPG, PNG or WebP image.')
    const size = body.size
    if (typeof size !== 'number' || !(size > 0) || size > PHOTO_MAX_BYTES) throw badRequest('Photo must be under 5 MB.')
    // The size is part of the signature: S3 refuses an upload of any other length.
    return { uploadUrl: await store.presignPut(bucket, key, body.contentType, 300, Math.floor(size)), key }
  }

  if (body.action === 'confirm') {
    // Only accept the key we issued for this user, and only once the object exists.
    if (!isPhotoType || body.key !== key) throw badRequest('Invalid photo key.')
    const head = await store.head(bucket, key)
    if (head.contentLength > PHOTO_MAX_BYTES) {
      await store.delete(bucket, key)
      throw badRequest('Photo must be under 5 MB.')
    }
    await setPhotoKey(user.email, key)
    return { photoUrl: await photoViewUrl({ photo_key: key }) }
  }

  throw badRequest('Unknown action.')
}
