// Users table (login accounts), keyed by email. Written by /auth/*, read by
// every authenticated API to look up the caller's role.
import { table } from '../data/index.js'
import config from '../config/index.js'

const users = table(() => config.tables.users)

export const getUserByEmail = (email) => users.get({ email })

export const putUser = (item) => users.put(item)

/** Like putUser, but fails (ConditionalCheckFailedException) when the email is already taken. */
export const createUser = (item) => users.put(item, { ConditionExpression: 'attribute_not_exists(email)' })

export const setLastLogin = (email, at) =>
  users.update({
    Key: { email },
    UpdateExpression: 'set last_login = :login_time',
    ExpressionAttributeValues: { ':login_time': at },
  })

export const setPassword = (email, passwordHash, at) =>
  users.update({
    Key: { email },
    UpdateExpression: 'set password = :p, must_change_password = :f, updated_at = :t',
    ExpressionAttributeValues: { ':p': passwordHash, ':f': false, ':t': at },
  })

export const setPhotoKey = (email, key) =>
  users.update({
    Key: { email },
    UpdateExpression: 'set photo_key = :k',
    ExpressionAttributeValues: { ':k': key },
  })

/** Admin-issued temp password; the condition stops it creating a stray item for a deleted account. */
export const setTempPassword = (email, userId, passwordHash) =>
  users.update({
    Key: { email },
    UpdateExpression: 'SET password = :p, must_change_password = :true',
    ConditionExpression: '#uid = :uid',
    ExpressionAttributeNames: { '#uid': 'user_id' },
    ExpressionAttributeValues: { ':p': passwordHash, ':true': true, ':uid': userId },
  })

/** Account status ('active' | 'suspended'); a no-op when there's no login account for that email. */
export async function setAccountStatus(email, status) {
  try {
    await users.update({
      Key: { email },
      UpdateExpression: 'SET #s = :s',
      ConditionExpression: 'attribute_exists(email)',
      ExpressionAttributeNames: { '#s': 'status' },
      ExpressionAttributeValues: { ':s': status },
    })
  } catch (err) {
    if (err?.name !== 'ConditionalCheckFailedException') throw err
  }
}

export const setGradingSettings = (email, settings) =>
  users.update({
    Key: { email },
    UpdateExpression: 'SET gradeant_settings = :s',
    ExpressionAttributeValues: { ':s': settings },
  })

/** Everyone, with just the fields the workflow screens need. */
export const listDirectory = () =>
  users.scanAll({
    ProjectionExpression: 'user_id, email, #n, userRole, classes',
    ExpressionAttributeNames: { '#n': 'name' },
  })
