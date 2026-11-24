// Shared controller for the "operation dispatch" APIs (/getgrades,
// /correctanswer, /updatedb, /workflow): POST {"operation": "...", ...payload}
// with a bearer token. Authentication runs before the operation is checked,
// as in the Python handlers.
import { badRequest } from '../lib/errors.js'
import { requirePermission } from '../lib/permissions.js'
import { authenticate } from '../middleware/auth.js'
import { bodyOf, handle } from './respond.js'

/** `area` is this API's section under permissions in config/app.yaml. */
export function operationsController(operations, { area, internalMessage = 'Internal server error', invalidMessage = 'Invalid operation specified' } = {}) {
  return handle(async (req) => {
    const body = bodyOf(req)
    const caller = await authenticate(req)
    const op = body.operation
    const fn = typeof op === 'string' && Object.hasOwn(operations, op) ? operations[op] : null
    if (!fn) throw badRequest(invalidMessage)
    requirePermission(area, op, caller.role)
    return fn(body, caller)
  }, internalMessage)
}
