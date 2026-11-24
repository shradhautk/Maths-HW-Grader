// Public contact and request-a-pilot forms (website /contact and /request-a-pilot).
// No login: each message is checked, rate-limited per client, and emailed to the
// team inbox (config.mail.contactInbox) with the visitor as Reply-To.
import { sendEmail } from '../aws/ses.js'
import config from '../config/index.js'
import { badRequest, HttpError } from '../lib/errors.js'

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
const KINDS = {
  message: { label: 'Website message', fields: ['subject', 'message'], required: ['message'] },
  pilot: { label: 'Pilot request', fields: ['school', 'subject', 'students', 'message'], required: ['school', 'subject', 'students'] },
}
const LIMITS = { name: 120, email: 200, subject: 120, school: 200, students: 20, message: 5000 }

// Per-instance limiter (Lambda instances are short-lived; this stops bursts, not a determined sender).
const WINDOW_MS = 10 * 60 * 1000
const MAX_PER_WINDOW = 5
const recent = new Map()

function rateLimit(client) {
  const now = Date.now()
  const hits = (recent.get(client) ?? []).filter((t) => now - t < WINDOW_MS)
  if (hits.length >= MAX_PER_WINDOW) throw new HttpError(429, 'Too many messages. Please try again in a few minutes, or email us directly.')
  hits.push(now)
  recent.set(client, hits)
  if (recent.size > 5000) recent.clear()
}

const clean = (v, max) => String(v ?? '').replace(/\r\n?/g, '\n').trim().slice(0, max)
const oneLine = (v) => v.replace(/\s+/g, ' ')

/** Validates and sends one form submission. Returns { message } on success. */
export async function submit(body, client = 'unknown') {
  // Honeypot: a hidden field people never fill in. Pretend success so bots move on.
  if (clean(body.website, 200)) return { message: 'Thanks, your message has been sent.' }

  const kind = KINDS[body.kind] ? body.kind : 'message'
  const spec = KINDS[kind]
  const f = Object.fromEntries(Object.entries(LIMITS).map(([k, max]) => [k, clean(body[k], max)]))

  if (!f.name) throw badRequest('Please enter your name.')
  if (!EMAIL.test(f.email)) throw badRequest('Please enter a valid email address.')
  for (const k of spec.required) if (!f[k]) throw badRequest(`Please fill in ${k === 'students' ? 'the number of students' : `the ${k}`}.`)
  if (kind === 'pilot' && !/^\d{1,6}$/.test(f.students)) throw badRequest('Number of students must be a whole number.')

  rateLimit(client)

  const name = oneLine(f.name)
  const subject = `[${spec.label}] ${oneLine(kind === 'pilot' ? f.school : f.subject || 'General question')} — ${name}`.slice(0, 200)
  const lines = [
    `${spec.label} from the GradeAnt website`,
    '',
    `Name: ${name}`,
    `Email: ${f.email}`,
    ...spec.fields.filter((k) => k !== 'message' && f[k]).map((k) => `${k[0].toUpperCase()}${k.slice(1)}: ${oneLine(f[k])}`),
    '',
    f.message || '(no message)',
  ]
  try {
    await sendEmail({
      from: config.mail.contactSenderEmail,
      to: config.mail.contactInbox,
      replyTo: f.email,
      subject,
      text: lines.join('\n'),
    })
  } catch (err) {
    console.error(JSON.stringify({ level: 'error', msg: 'contact form email failed', kind, error: err?.name, detail: err?.message }))
    throw new HttpError(502, `We couldn't send your message. Please email us at ${config.mail.contactInbox}.`)
  }
  return { message: kind === 'pilot' ? "Thanks, we've got your pilot request and will be in touch within two working days." : 'Thanks, your message has been sent.' }
}
