// Optional background worker for JOBS_DRIVER=sqs. Deploy the same package as a
// regular (event) Lambda function on Node.js 22+ with handler "worker.handler"
// and the jobs queue as its SQS event source. Failed messages are reported
// individually so SQS retries only those (enable ReportBatchItemFailures).
import { initDatabase } from './src/aws/dynamodb.js'
import { runJob } from './src/services/jobs.service.js'

const ready = initDatabase()

export async function handler(event) {
  await ready
  const batchItemFailures = []
  for (const record of event.Records ?? []) {
    try {
      await runJob(JSON.parse(record.body))
    } catch (err) {
      console.error(JSON.stringify({ level: 'error', msg: 'job failed', messageId: record.messageId, error: err?.message }))
      batchItemFailures.push({ itemIdentifier: record.messageId })
    }
  }
  return { batchItemFailures }
}
