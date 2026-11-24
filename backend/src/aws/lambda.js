// Hands a corrected submission to the grading pipeline ("Start grading"). The payload is
// the one workflow.py sent (python-backend/tests/test_grader_contract.py pins it):
//   GRADING_DRIVER=lambda  asynchronous "Event" invoke of GRADING_FUNCTION_NAME (all-AWS)
//   GRADING_DRIVER=sqs     a message to the grader's queue, GRADING_QUEUE_URL (free-tier hybrid:
//                          durable, retried, dead-lettered; python-backend sqs.regrade_job reads it)
//   GRADING_DRIVER=log     development: only logged
import { InvokeCommand, LambdaClient } from '@aws-sdk/client-lambda'
import { SendMessageCommand, SQSClient } from '@aws-sdk/client-sqs'
import config from '../config/index.js'

let lambda
let sqs
let sendOverride
export const gradingRequests = [] // development/test inspection only

export async function triggerGrading(payload) {
  const driver = config.drivers.grading
  const target = driver === 'sqs' ? config.grading.queueUrl : config.grading.functionName
  if (!target) return // grading hand-off not configured: nothing to send (as before)
  if (driver === 'log') {
    gradingRequests.push(payload)
    console.log(JSON.stringify({ level: 'info', msg: 'grading request (not sent: GRADING_DRIVER=log)', ...payload, rubric: undefined }))
    return
  }
  const body = JSON.stringify(payload)
  if (driver === 'sqs') {
    if (sendOverride) return sendOverride({ QueueUrl: target, MessageBody: body })
    sqs ??= new SQSClient({ region: config.aws.region })
    await sqs.send(new SendMessageCommand({ QueueUrl: target, MessageBody: body }))
    return
  }
  lambda ??= new LambdaClient({ region: config.aws.region })
  await lambda.send(new InvokeCommand({ FunctionName: target, InvocationType: 'Event', Payload: Buffer.from(body) }))
}

/** Tests: capture SQS sends ({ QueueUrl, MessageBody }); undefined restores the real client. */
export function setGradingSender(fn) {
  sendOverride = fn
}
