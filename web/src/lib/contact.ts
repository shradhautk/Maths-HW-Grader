import { API } from '../config'
import { CONTACT } from '../components/landing/site'
import { apiRequest, isRouteMissing } from './api'

export interface ContactMessage {
  kind: 'message' | 'pilot'
  name: string
  email: string
  subject?: string
  message?: string
  school?: string
  students?: string
  /** Honeypot: hidden from people, so only bots fill it in. */
  website?: string
}

/**
 * Sends the website contact / request-a-pilot form; the API emails it to the team.
 * Throws an Error whose message is safe to show the visitor.
 */
export async function sendContactMessage(msg: ContactMessage): Promise<string> {
  try {
    const data = await apiRequest<{ message?: string }>(API.contact, { body: msg, auth: false })
    return data?.message ?? 'Thanks, your message has been sent.'
  } catch (err) {
    // Network failures and undeployed routes surface as "Failed to fetch" etc.; don't show that.
    if (isRouteMissing(err) || !(err instanceof Error) || !err.message) {
      throw new Error(`We couldn't send your message. Please email us at ${CONTACT.email}.`, { cause: err })
    }
    throw err
  }
}
