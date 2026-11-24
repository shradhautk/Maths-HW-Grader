// Outgoing email: Amazon SES in production, the console in development (MAIL_DRIVER=log).
import { SendEmailCommand, SESClient } from '@aws-sdk/client-ses'
import config from '../config/index.js'

let ses
export const sentMail = [] // development/test inspection only

/** sendEmail({ from, to, subject, html?, text?, replyTo? }) */
export async function sendEmail({ from, to, subject, html, text, replyTo }) {
  if (config.drivers.mail === 'log') {
    sentMail.push({ from, to, subject, html, text, replyTo })
    console.log(JSON.stringify({ level: 'info', msg: 'email (not sent: MAIL_DRIVER=log)', to, subject }))
    return
  }
  ses ??= new SESClient({ region: config.aws.region })
  const Body = html ? { Html: { Data: html } } : { Text: { Data: text ?? '' } }
  await ses.send(
    new SendEmailCommand({
      Source: from,
      Destination: { ToAddresses: [to] },
      ...(replyTo ? { ReplyToAddresses: [replyTo] } : {}),
      Message: { Subject: { Data: subject }, Body },
    }),
  )
}
