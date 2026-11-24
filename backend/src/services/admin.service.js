// /admin — port of admin/lambda_admin.py: the admin console, one resource
// dispatching on `operation`. Every operation is admin-only except
// getInviteDetails / completeFirstLogin (reached from an emailed invite link).
import { randomBytes, randomUUID } from 'node:crypto'
import { sendEmail } from '../aws/ses.js'
import config from '../config/index.js'
import { badRequest, conflict, forbidden, isConditionalCheckFailed, notFound, unauthorized } from '../lib/errors.js'
import {
  generateTempPassword as makeTempPassword,
  hashAdminPassword,
  hashLoginPassword,
  passwordProblem,
  verifyAdminPassword,
} from '../lib/passwords.js'
import { canSignUp, requirePermission } from '../lib/permissions.js'
import { isoNaive, pyRepr, sortBy, truthy } from '../lib/py.js'
import { issueLoginToken, verifyLoginJwtForAdmin, verifySessionToken } from '../lib/tokens.js'
import {
  adminUsers,
  analytics,
  auditLogs,
  content,
  equalsFilter,
  invites,
  resetRequests,
  setFields,
  settings,
  tickets,
} from '../repositories/admin.repository.js'
import { createUser, getUserByEmail, setAccountStatus, setTempPassword } from '../repositories/users.repository.js'

// ---- Auth ------------------------------------------------------------------------

/** Claims for the bearer token: a login JWT (3 parts) or an admin session token (2 parts). */
export async function bearerClaims(authorization) {
  const header = authorization ?? ''
  if (!header.toLowerCase().startsWith('bearer ')) throw unauthorized('Missing Authorization: Bearer <token> header.')
  const token = header.slice(7).trim()
  if (token.split('.').length === 3) {
    const payload = await verifyLoginJwtForAdmin(token)
    const { email, user_id: userId } = payload
    if (!email || !userId) throw unauthorized('Malformed session token.')
    const user = await getUserByEmail(email)
    if (!user || user.user_id !== userId) throw unauthorized('Account no longer exists.')
    if (user.status === 'suspended') throw unauthorized('This account is suspended.')
    return { sub: userId, role: user.userRole ?? 'student', name: user.name ?? 'Admin', email }
  }
  // Session tokens carry a role claim, but it's re-read from the account on every
  // request so a deleted, suspended or demoted admin loses access at once.
  const claims = await verifySessionToken(token)
  const account = claims?.sub ? await adminUsers.get({ user_id: String(claims.sub) }) : undefined
  if (!account || account.status === 'suspended') throw unauthorized('Account no longer active.')
  return { ...claims, role: account.role ?? 'student', name: account.name ?? claims.name }
}

async function requireAdmin(authorization) {
  const claims = await bearerClaims(authorization)
  if (claims.role !== 'admin') throw forbidden('This action is admin-only.')
  return claims
}

async function writeAuditLog(actor, action, target) {
  try {
    await auditLogs.put({ log_id: randomUUID(), timestamp: Date.now() / 1000, actor, action, target })
  } catch (err) {
    // Audit logging must never break the admin's action.
    console.warn(JSON.stringify({ level: 'warn', msg: 'failed to write audit log', action, target, error: err?.message }))
  }
}

const str = (v) => (v === undefined || v === null ? '' : String(v))

// ---- Users -----------------------------------------------------------------------

async function listUsers(p) {
  const search = str(p.search).trim().toLowerCase()
  const items = await adminUsers.scanAll(equalsFilter({ role: p.role || undefined, status: p.status || undefined }))
  const matching = search
    ? items.filter((u) => str(u.name).toLowerCase().includes(search) || str(u.email).toLowerCase().includes(search))
    : items
  const users = matching.map((u) => ({
    id: u.user_id,
    name: u.name ?? '',
    email: u.email ?? '',
    role: u.role ?? 'student',
    school: u.school ?? '',
    status: u.status ?? 'active',
  }))
  return { users, total: users.length }
}

async function updateUserStatus(p, claims) {
  if (!p.userId || !['active', 'suspended', 'invited'].includes(p.status)) throw badRequest('userId and a valid status are required.')
  const account = await adminUsers.get({ user_id: String(p.userId) })
  await adminUsers.update({ Key: { user_id: p.userId }, ...setFields({ status: p.status }) })
  // The login account lives in the Users table (keyed by email): mirror a suspension
  // there too, or the user could keep signing in to the app.
  if (account?.email && p.status !== 'invited') await setAccountStatus(account.email, p.status)
  await writeAuditLog(claims.sub, `Set user status to ${p.status}`, p.userId)
  return { success: true }
}

async function deleteUser(p, claims) {
  if (!p.userId) throw badRequest('userId is required.')
  const account = await adminUsers.get({ user_id: String(p.userId) })
  // Block the login account rather than deleting it, so grades stay attributable.
  if (account?.email) await setAccountStatus(account.email, 'suspended')
  await adminUsers.delete({ user_id: p.userId })
  await writeAuditLog(claims.sub, 'Deleted user', p.userId)
  return { success: true }
}

// ---- Onboarding: invite → first login (wireframe §17) ------------------------------

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch])

async function sendInviteEmail(email, name, link) {
  try {
    await sendEmail({
      from: config.mail.inviteSenderEmail,
      to: email,
      subject: "You're invited to GradeAnt",
      html:
        `<p>Hi ${escapeHtml(name)},</p>` +
        "<p>You've been invited to GradeAnt. Finish setting up your account:</p>" +
        `<p><a href="${escapeHtml(link)}">${escapeHtml(link)}</a></p>`,
    })
  } catch (err) {
    // The invite record should still succeed even if SES isn't set up yet.
    console.warn(JSON.stringify({ level: 'warn', msg: 'failed to send invite email', error: err?.message }))
  }
}

async function inviteUsers(p, claims) {
  const invitees = Array.isArray(p.invitees) ? p.invitees : []
  if (!invitees.length) throw badRequest('At least one invitee is required.')
  // Only roles whose config/app.yaml roles.<role>.signup includes "invite".
  for (const person of invitees) {
    const role = person?.role || 'student'
    if (str(person?.email).trim() && !canSignUp(role, 'invite')) throw badRequest(`${str(role)} accounts can't be created by invite.`)
  }
  let invited = 0
  for (const person of invitees) {
    const email = str(person?.email).trim()
    if (!email) continue
    const name = str(person.name || email).trim()
    const role = person.role || 'student'
    const className = person.className || ''
    const userId = randomUUID()
    await adminUsers.put({ user_id: userId, name, email, role, school: className, status: 'invited' })
    const token = randomBytes(24).toString('base64url')
    await invites.put({
      token,
      user_id: userId,
      full_name: name,
      role,
      class_name: className,
      expires_at: Date.now() / 1000 + config.auth.inviteTokenTtlSeconds,
      consumed: false,
    })
    // The web app's first-login page (config/app.yaml pages.firstLogin).
    await sendInviteEmail(email, name, `${config.mail.appBaseUrl}${config.app.pages.firstLogin}?token=${token}`)
    invited += 1
  }
  await writeAuditLog(claims.sub, `Invited ${invited} user(s)`, invitees.map((x) => str(x?.email)).join(', '))
  return { invited }
}

async function loadInvite(token) {
  if (!token) throw badRequest('token is required.')
  return invites.get({ token: String(token) })
}

async function getInviteDetails(p) {
  const invite = await loadInvite(p.token)
  if (!invite) throw notFound('This invite link is invalid or has already been used.')
  if (invite.consumed) throw badRequest('This invite has already been used.')
  if (Number(invite.expires_at ?? 0) < Date.now() / 1000) throw badRequest('This invite link has expired — ask your admin to resend it.')
  return { fullName: invite.full_name, role: invite.role, className: invite.class_name ?? '' }
}

async function completeFirstLogin(p) {
  const { token, password } = p
  if (!token || !password || Array.from(String(password)).length < 8) {
    throw badRequest('token and a password of at least 8 characters are required.')
  }
  const invite = await invites.get({ token: String(token) })
  if (!invite || invite.consumed) throw badRequest('This invite link is invalid or has already been used.')
  if (Number(invite.expires_at ?? 0) < Date.now() / 1000) throw badRequest('This invite link has expired — ask your admin to resend it.')
  const problem = passwordProblem(String(password))
  if (problem) throw badRequest(problem)

  const account = await adminUsers.get({ user_id: invite.user_id })
  const email = str(account?.email).trim()
  if (!email) throw badRequest('This invite has no email address — ask your admin to resend it.')

  // Invites are the only way to get a teacher or admin account, so this creates
  // the app login account (Users, keyed by email) that /auth/login and every app
  // API read. It never overwrites an existing account (and so never changes its role).
  const at = isoNaive()
  try {
    await createUser({
      user_id: invite.user_id,
      email,
      name: invite.full_name ?? '',
      password: await hashLoginPassword(String(password)),
      userRole: invite.role,
      status: 'active',
      created_at: at,
      updated_at: at,
      // The class the admin invited a student into: students only see their own classes' work.
      ...(invite.role === 'student' && truthy(invite.class_name) ? { classes: [invite.class_name] } : {}),
    })
  } catch (err) {
    if (isConditionalCheckFailed(err)) throw conflict('An account with this email already exists. Log in with it, or ask your admin for help.')
    throw err
  }
  await adminUsers.update({
    Key: { user_id: invite.user_id },
    ...setFields({ status: 'active', password_hash: await hashAdminPassword(String(password)) }),
  })
  await invites.update({ Key: { token: String(token) }, ...setFields({ consumed: true }) })
  return {
    // A normal login token, accepted by the app APIs and the admin console alike.
    token: await issueLoginToken(invite.user_id, email),
    userId: invite.user_id,
    role: invite.role,
    fullName: invite.full_name,
  }
}

// ---- Password resets (teachers/students can't self-reset, wireframe §07) ------------

async function listResetRequests(p) {
  const status = p.status || 'pending'
  const items = await resetRequests.scanAll(equalsFilter({ status }))
  const requests = sortBy(
    items.map((r) => ({
      id: r.request_id,
      userName: r.user_name ?? '',
      userEmail: r.user_email ?? '',
      role: r.role ?? '',
      requestedAt: Number(r.requested_at ?? 0),
      status: r.status ?? 'pending',
    })),
    (r) => r.requestedAt,
    { reverse: true },
  )
  return { requests, pendingCount: status === 'pending' ? requests.length : null }
}

async function approveResetRequest(p, claims) {
  if (!p.requestId) throw badRequest('requestId is required.')
  await resetRequests.update({ Key: { request_id: p.requestId }, ...setFields({ status: 'approved', approved_by: claims.sub }) })
  await writeAuditLog(claims.sub, 'Approved password reset request', p.requestId)
  return { success: true }
}

async function rejectResetRequest(p, claims) {
  if (!p.requestId) throw badRequest('requestId is required.')
  await resetRequests.update({ Key: { request_id: p.requestId }, ...setFields({ status: 'rejected' }) })
  await writeAuditLog(claims.sub, 'Rejected password reset request', p.requestId)
  return { success: true }
}

async function generateTempPassword(p, claims) {
  if (!p.requestId) throw badRequest('requestId is required.')
  const request = await resetRequests.get({ request_id: p.requestId })
  if (!request) throw notFound('Reset request not found.')

  const temp = makeTempPassword()
  if (request.source === 'login') {
    // Requests from the login page reset the login account (Users, keyed by email).
    try {
      await setTempPassword(request.user_email, request.user_id, await hashLoginPassword(temp))
    } catch (err) {
      if (isConditionalCheckFailed(err)) throw notFound('That account no longer exists.')
      throw err
    }
  } else {
    await adminUsers.update({
      Key: { user_id: request.user_id },
      ...setFields({ password_hash: await hashAdminPassword(temp), must_change_password: true }),
    })
  }
  await resetRequests.update({ Key: { request_id: p.requestId }, ...setFields({ status: 'completed' }) })
  await writeAuditLog(claims.sub, 'Generated temp password', request.user_email ?? request.user_id)
  return { tempPassword: temp }
}

// Fixed governance rules from the wireframe's Permission Matrix (§07).
const PERMISSION_MATRIX = [
  { label: 'Change own password', admin: 'yes', teacher: 'yes', student: 'yes' },
  { label: 'Reset own pw without current', admin: 'no', teacher: 'no', student: 'no' },
  { label: "Reset another user's pw", admin: 'yes', teacher: 'no', student: 'no' },
  { label: 'Create teacher / student', admin: 'yes', teacher: 'no', student: 'no' },
  { label: 'Edit username', admin: 'yes', teacher: 'no', student: 'no' },
  { label: 'Edit own details (limited)', admin: 'yes', teacher: 'self', student: 'self' },
  { label: 'Delete / activate users', admin: 'yes', teacher: 'no', student: 'no' },
  { label: 'View audit logs', admin: 'yes', teacher: 'no', student: 'no' },
]

const getPermissionMatrix = async () => ({ features: PERMISSION_MATRIX })

// ---- Content & blog --------------------------------------------------------------

async function listContent(p) {
  const filter = p.filter || 'all'
  const items = await content.scanAll(filter !== 'all' ? equalsFilter({ status: filter }) : {})
  const posts = items.map((x) => ({
    id: x.post_id,
    title: x.title ?? '(untitled)',
    author: x.author ?? '',
    dateLabel: x.date_label ?? '',
    status: x.status ?? 'draft',
  }))
  return { posts: sortBy(posts, (x) => x.dateLabel ?? '', { reverse: true }) }
}

async function updatePostStatus(p, claims) {
  if (!p.postId || !['draft', 'pending', 'published'].includes(p.status)) throw badRequest('postId and a valid status are required.')
  await content.update({ Key: { post_id: p.postId }, ...setFields({ status: p.status }) })
  await writeAuditLog(claims.sub, `Set post status to ${p.status}`, p.postId)
  return { success: true }
}

// ---- Analytics (pre-aggregated by a separate scheduled job) -------------------------

const ANALYTICS_FALLBACK = {
  activeUsers: { value: '0', trend: 'no data yet', direction: 'flat' },
  papersGraded: { value: '0', trend: 'no data yet', direction: 'flat' },
  schools: { value: '0', trend: 'no data yet', direction: 'flat' },
  uptime: { value: '—', trend: 'no data yet', direction: 'flat' },
  weeklyPapers: [],
  topSchools: [],
}

async function getAnalytics(p) {
  const item = await analytics.get({ metric_key: p.range || '30d' })
  return item ? JSON.parse(item.data_json) : ANALYTICS_FALLBACK
}

// ---- Support tickets -------------------------------------------------------------

async function listTickets(p, claims) {
  const filter = p.filter || 'open'
  const condition =
    filter === 'resolved' ? { status: 'resolved' } : filter === 'mine' ? { assignee_id: claims.sub } : { status: 'open' }
  const items = await tickets.scanAll(equalsFilter(condition))
  const list = items.map((t) => ({
    id: t.ticket_id,
    priority: t.priority ?? 'med',
    subject: t.subject ?? '',
    ageLabel: t.age_label ?? '',
    requester: t.requester_name ?? '',
    assignee: t.assignee_name ?? '',
  }))
  return { tickets: list, openCount: filter === 'open' ? list.length : null }
}

async function assignTicket(p, claims) {
  if (!p.ticketId) throw badRequest('ticketId is required.')
  await tickets.update({ Key: { ticket_id: p.ticketId }, ...setFields({ assignee_id: claims.sub, assignee_name: claims.name ?? 'Admin' }) })
  await writeAuditLog(claims.sub, 'Self-assigned ticket', p.ticketId)
  return { success: true }
}

async function resolveTicket(p, claims) {
  if (!p.ticketId) throw badRequest('ticketId is required.')
  await tickets.update({ Key: { ticket_id: p.ticketId }, ...setFields({ status: 'resolved' }) })
  await writeAuditLog(claims.sub, 'Resolved ticket', p.ticketId)
  return { success: true }
}

// ---- Audit logs --------------------------------------------------------------------

async function listAuditLogs(p) {
  const search = str(p.search).trim().toLowerCase()
  const category = p.category || ''
  let entries = (await auditLogs.scanAll()).map((e) => ({
    timestamp: Number(e.timestamp ?? 0),
    actor: e.actor ?? '',
    action: e.action ?? '',
    target: e.target ?? '',
  }))
  if (category) entries = entries.filter((e) => e.action.toLowerCase().includes(String(category).toLowerCase()))
  if (search) {
    entries = entries.filter(
      (e) => e.actor.toLowerCase().includes(search) || e.action.toLowerCase().includes(search) || e.target.toLowerCase().includes(search),
    )
  }
  return { entries: sortBy(entries, (e) => e.timestamp, { reverse: true }).slice(0, 200) }
}

// ---- System settings -----------------------------------------------------------------

const DEFAULT_SETTINGS = {
  platformName: 'GradeAnt',
  supportEmail: 'support@gradeant.io',
  sessionTimeoutMinutes: 30,
  maintenanceMode: false,
  allowSelfSignup: false,
}

async function getSettings() {
  const item = await settings.get({ setting_key: 'platform' })
  if (!item) return DEFAULT_SETTINGS
  return {
    platformName: item.platform_name ?? DEFAULT_SETTINGS.platformName,
    supportEmail: item.support_email ?? DEFAULT_SETTINGS.supportEmail,
    sessionTimeoutMinutes: Math.trunc(Number(item.session_timeout_minutes ?? DEFAULT_SETTINGS.sessionTimeoutMinutes)),
    maintenanceMode: truthy(item.maintenance_mode ?? false),
    allowSelfSignup: truthy(item.allow_self_signup ?? false),
  }
}

async function updateSettings(p, claims) {
  const timeout = Number(p.sessionTimeoutMinutes ?? 30)
  if (!Number.isFinite(timeout)) throw badRequest('sessionTimeoutMinutes must be a number.')
  await settings.put({
    setting_key: 'platform',
    platform_name: p.platformName ?? DEFAULT_SETTINGS.platformName,
    support_email: p.supportEmail ?? DEFAULT_SETTINGS.supportEmail,
    session_timeout_minutes: timeout,
    maintenance_mode: truthy(p.maintenanceMode ?? false),
    allow_self_signup: truthy(p.allowSelfSignup ?? false),
  })
  await writeAuditLog(claims.sub, 'Updated system settings', 'platform')
  return { success: true }
}

// ---- Admin's own profile ---------------------------------------------------------------

async function getProfile(p, claims) {
  const user = await adminUsers.get({ user_id: claims.sub })
  if (!user) throw notFound('Admin profile not found.')
  return {
    fullName: user.name ?? '',
    username: user.username ?? '',
    email: user.email ?? '',
    phone: user.phone ?? '',
    adminLevel: user.admin_level ?? 'admin',
  }
}

async function updateProfile(p, claims) {
  await adminUsers.update({
    Key: { user_id: claims.sub },
    ...setFields({ name: p.fullName ?? '', username: p.username ?? '', email: p.email ?? '', phone: p.phone ?? '' }),
  })
  await writeAuditLog(claims.sub, 'Updated own profile', claims.sub)
  return { success: true }
}

async function changeOwnPassword(p, claims) {
  const { currentPassword, newPassword } = p
  if (!currentPassword || !newPassword || Array.from(String(newPassword)).length < 8) {
    throw badRequest('currentPassword and a newPassword of at least 8 characters are required.')
  }
  const user = await adminUsers.get({ user_id: claims.sub })
  if (!user || !(await verifyAdminPassword(String(currentPassword), user.password_hash ?? ''))) {
    throw badRequest('Current password is incorrect.')
  }
  await adminUsers.update({
    Key: { user_id: claims.sub },
    ...setFields({ password_hash: await hashAdminPassword(String(newPassword)), must_change_password: false }),
  })
  await writeAuditLog(claims.sub, 'Changed own password', claims.sub)
  return { success: true }
}

// ---- Dispatch ------------------------------------------------------------------------

// [handler, requiresAuth]
const OPERATIONS = {
  listUsers: [listUsers, true],
  updateUserStatus: [updateUserStatus, true],
  deleteUser: [deleteUser, true],
  inviteUsers: [inviteUsers, true],
  getInviteDetails: [getInviteDetails, false],
  completeFirstLogin: [completeFirstLogin, false],
  listResetRequests: [listResetRequests, true],
  approveResetRequest: [approveResetRequest, true],
  rejectResetRequest: [rejectResetRequest, true],
  generateTempPassword: [generateTempPassword, true],
  getPermissionMatrix: [getPermissionMatrix, true],
  listContent: [listContent, true],
  updatePostStatus: [updatePostStatus, true],
  getAnalytics: [getAnalytics, true],
  listTickets: [listTickets, true],
  assignTicket: [assignTicket, true],
  resolveTicket: [resolveTicket, true],
  listAuditLogs: [listAuditLogs, true],
  getSettings: [getSettings, true],
  updateSettings: [updateSettings, true],
  getProfile: [getProfile, true],
  updateProfile: [updateProfile, true],
  changeOwnPassword: [changeOwnPassword, true],
}

export async function dispatch(body, authorization) {
  const operation = body.operation
  if (!operation || typeof operation !== 'string' || !Object.hasOwn(OPERATIONS, operation)) {
    throw badRequest(`Unknown or missing operation: ${pyRepr(operation ?? null)}`)
  }
  const [handler, requiresAuth] = OPERATIONS[operation]
  const claims = requiresAuth ? await requireAdmin(authorization) : null
  // config/app.yaml permissions.admin can turn individual console operations off.
  if (requiresAuth) requirePermission('admin', operation, claims.role, 'This action is admin-only.')
  return handler(body, claims)
}
