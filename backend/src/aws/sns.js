// "Grades published" notifications: Amazon SNS in production, the console in development.
import { PublishCommand, SNSClient } from '@aws-sdk/client-sns'
import config from '../config/index.js'

let sns
export const publishedNotifications = [] // development/test inspection only

export async function publishNotification({ subject, message }) {
  const topic = config.notify.snsTopicArn
  if (!topic) return
  if (config.drivers.notify === 'log') {
    publishedNotifications.push({ subject, message })
    console.log(JSON.stringify({ level: 'info', msg: 'notification (not sent: NOTIFY_DRIVER=log)', subject }))
    return
  }
  sns ??= new SNSClient({ region: config.aws.region })
  await sns.send(new PublishCommand({ TopicArn: topic, Subject: subject, Message: message }))
}
