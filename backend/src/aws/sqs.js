// Background jobs queue (JOBS_DRIVER=sqs). Messages are consumed by worker.js
// deployed as a regular Lambda function with this queue as its event source.
import { SendMessageCommand, SQSClient } from '@aws-sdk/client-sqs'
import config from '../config/index.js'

let sqs

export async function sendJobMessage(job) {
  sqs ??= new SQSClient({ region: config.aws.region })
  await sqs.send(new SendMessageCommand({ QueueUrl: config.jobs.queueUrl, MessageBody: JSON.stringify(job) }))
}
