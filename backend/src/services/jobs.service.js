// Background jobs (currently: AI rubric drafting). workflow.py ran these by
// invoking its own Lambda asynchronously; a Web Function has no such
// self-invoke, so:
//   JOBS_DRIVER=inline  run the job before the request returns (development,
//                       and production without a queue — set the revision
//                       timeout to 60 s or more, since it waits for Claude)
//   JOBS_DRIVER=sqs     send it to JOBS_QUEUE_URL; worker.js (a regular Lambda
//                       function subscribed to the queue) runs it, with SQS retries
import { sendJobMessage } from '../aws/sqs.js'
import config from '../config/index.js'

const HANDLERS = {
  draftRubric: async (job) => {
    const { draftRubric } = await import('./workflow/assistant.js')
    await draftRubric(job.assignment_id)
  },
}

export async function enqueueJob(job) {
  if (!Object.hasOwn(HANDLERS, job.type)) throw new Error(`Unknown job type: ${job.type}`)
  if (config.drivers.jobs === 'sqs') return sendJobMessage(job)
  return runJob(job)
}

export async function runJob(job) {
  if (!job || !Object.hasOwn(HANDLERS, job.type)) throw new Error(`Unknown job type: ${job?.type}`)
  await HANDLERS[job.type](job)
}
