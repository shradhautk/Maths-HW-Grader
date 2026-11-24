import { HttpError, Reply } from '../lib/errors.js'

/** Sends a service result: a Reply keeps its status, anything else is a 200. */
export function send(res, result) {
  if (result instanceof Reply) return res.status(result.status).json(result.body)
  return res.json(result)
}

/**
 * Wraps a controller. Unexpected errors become `internalMessage` (with the
 * router's error key) instead of leaking exception text, as the Python
 * handlers' outer try/except did.
 */
export const handle = (fn, internalMessage) => async (req, res) => {
  try {
    send(res, await fn(req, res))
  } catch (err) {
    if (err instanceof HttpError || !internalMessage) throw err
    console.error(JSON.stringify({ level: 'error', msg: internalMessage, requestId: req.get('lambda-web-request-id'), error: err?.name, detail: err?.message }))
    throw new HttpError(500, internalMessage)
  }
}

/** The request body as an object ({} when absent or not an object). */
export const bodyOf = (req) => (req.body && typeof req.body === 'object' && !Array.isArray(req.body) ? req.body : {})
