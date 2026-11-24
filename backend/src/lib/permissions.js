// Role checks driven by config/app.yaml (permissions and roles sections).
// These sit in front of each operation; the operations keep their own
// built-in checks and data rules too (app.yaml can narrow access, never widen it).
import config from '../config/index.js'
import { forbidden } from './errors.js'

// Own keys only: area/operation/role names can come from request bodies.
const own = (obj, key) => (obj && typeof key === 'string' && Object.hasOwn(obj, key) ? obj[key] : undefined)
const listOf = (v) => (Array.isArray(v) ? v : [])

/** True when app.yaml's permissions.<area>.<operation> lists this role. */
export const isAllowed = (area, operation, role) => listOf(own(own(config.app.permissions, area), operation)).includes(role)

/** Throws 403 unless app.yaml lets this role run the operation. */
export function requirePermission(area, operation, role, message = "Your role can't do that.") {
  if (!isAllowed(area, operation, role)) throw forbidden(message)
}

/** How an account with this role may be created: app.yaml roles.<role>.signup includes 'self' / 'invite'. */
export const canSignUp = (role, mode) => listOf(own(config.app.roles, role)?.signup).includes(mode)

/** The role's display label from app.yaml. */
export const roleLabel = (role) => own(config.app.roles, role)?.label ?? role

/** Roles that can sign themselves up, in app.yaml order. */
export const selfSignupRoles = () => Object.keys(config.app.roles).filter((r) => canSignUp(r, 'self'))
