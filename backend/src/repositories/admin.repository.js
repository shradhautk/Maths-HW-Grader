// Admin console tables (lambda_admin.py). All keyed by a single string id:
//   GradeAnt_Users (user_id) · GradeAnt_ResetRequests (request_id) ·
//   GradeAnt_Content (post_id) · GradeAnt_Tickets (ticket_id) ·
//   GradeAnt_AuditLogs (log_id) · GradeAnt_Settings (setting_key) ·
//   GradeAnt_Invites (token) · GradeAnt_Analytics (metric_key)
import { table } from '../data/index.js'
import config from '../config/index.js'

export const adminUsers = table(() => config.tables.adminUsers)
export const resetRequests = table(() => config.tables.resetRequests)
export const content = table(() => config.tables.content)
export const tickets = table(() => config.tables.tickets)
export const auditLogs = table(() => config.tables.auditLogs)
export const settings = table(() => config.tables.settings)
export const invites = table(() => config.tables.invites)
export const analytics = table(() => config.tables.analytics)

/** Build a scan filter from equality conditions, e.g. { role: 'teacher', status: 'active' }. */
export function equalsFilter(conditions) {
  const entries = Object.entries(conditions).filter(([, v]) => v !== undefined)
  if (!entries.length) return {}
  const names = {}
  const values = {}
  const parts = entries.map(([field, value], i) => {
    names[`#f${i}`] = field
    values[`:v${i}`] = value
    return `#f${i} = :v${i}`
  })
  return { FilterExpression: parts.join(' AND '), ExpressionAttributeNames: names, ExpressionAttributeValues: values }
}

/** Set one or more attributes on an item: setFields({ status: 'active' }). */
export function setFields(fields, extra = {}) {
  const names = {}
  const values = {}
  const parts = Object.entries(fields).map(([field, value], i) => {
    names[`#f${i}`] = field
    values[`:v${i}`] = value
    return `#f${i} = :v${i}`
  })
  return { UpdateExpression: `SET ${parts.join(', ')}`, ExpressionAttributeNames: names, ExpressionAttributeValues: values, ...extra }
}

export const pendingResetRequestsFor = (userId) =>
  resetRequests.scanAll(equalsFilter({ user_id: userId, status: 'pending' }))
