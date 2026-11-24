// Table definitions the backend expects (keys and GSIs). Used to create tables
// in the local emulator / DynamoDB Local; also the reference for production.
import { CreateTableCommand, ListTablesCommand } from '@aws-sdk/client-dynamodb'
import config from '../config/index.js'

const gsi = (IndexName, hash) => ({
  IndexName,
  KeySchema: [{ AttributeName: hash, KeyType: 'HASH' }],
  Projection: { ProjectionType: 'ALL' },
})

export function tableDefinitions({ assignmentIndex = true } = {}) {
  const t = config.tables
  const defs = [
    { name: t.users, keys: [['email', 'HASH']] },
    {
      name: t.grading,
      keys: [['evaluation_id', 'HASH']],
      gsis: [
        gsi('student_id-index', 'student_id'),
        gsi('type-index', 'type'),
        ...(assignmentIndex ? [gsi(t.gradingAssignmentIndex, 'assignment_id')] : []),
      ],
    },
    { name: t.assignments, keys: [['assignment_id', 'HASH']], gsis: [gsi('teacher_id-index', 'teacher_id')] },
    {
      name: t.submissions,
      keys: [
        ['assignment_id', 'HASH'],
        ['student_id', 'RANGE'],
      ],
      gsis: [gsi('student_id-index', 'student_id')],
    },
    { name: t.announcements, keys: [['announcement_id', 'HASH']] },
    { name: t.adminUsers, keys: [['user_id', 'HASH']] },
    { name: t.resetRequests, keys: [['request_id', 'HASH']] },
    { name: t.content, keys: [['post_id', 'HASH']] },
    { name: t.tickets, keys: [['ticket_id', 'HASH']] },
    { name: t.auditLogs, keys: [['log_id', 'HASH']] },
    { name: t.settings, keys: [['setting_key', 'HASH']] },
    { name: t.invites, keys: [['token', 'HASH']] },
    { name: t.analytics, keys: [['metric_key', 'HASH']] },
  ]
  return defs.map(({ name, keys, gsis = [] }) => {
    const attrs = new Set([...keys.map(([k]) => k), ...gsis.map((g) => g.KeySchema[0].AttributeName)])
    return {
      TableName: name,
      BillingMode: 'PAY_PER_REQUEST',
      KeySchema: keys.map(([AttributeName, KeyType]) => ({ AttributeName, KeyType })),
      AttributeDefinitions: [...attrs].sort().map((AttributeName) => ({ AttributeName, AttributeType: 'S' })),
      ...(gsis.length ? { GlobalSecondaryIndexes: gsis } : {}),
    }
  })
}

export async function createTables(client, { ifMissing = false, assignmentIndex = true } = {}) {
  const existing = ifMissing ? new Set((await client.send(new ListTablesCommand({}))).TableNames ?? []) : new Set()
  for (const def of tableDefinitions({ assignmentIndex })) {
    if (existing.has(def.TableName)) continue
    await client.send(new CreateTableCommand(def))
  }
}
