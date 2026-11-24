// Error handling. Each API keeps the error shape its Python Lambda used:
// {"error": "..."} for /getgrades, /correctanswer, /updatedb, /workflow and
// {"message": "..."} for /auth/* and /admin. Unexpected errors are logged with
// the request id and answered with a generic 500 (no internals leaked).
import express from 'express'
import config from '../config/index.js'
import { HttpError } from '../lib/errors.js'

/** JSON body parser for one router; malformed JSON becomes a 400 in that router's shape. */
export const jsonBody = () => express.json({ type: () => true, limit: config.server.bodyLimit })

export function errorHandler({ key, invalidJson = 'Invalid JSON in request body', internal = 'Internal server error' }) {
  // Express recognises error handlers by their four parameters.
  // eslint-disable-next-line no-unused-vars
  return (err, req, res, _next) => {
    if (err instanceof HttpError) {
      return res.status(err.status).json(err.body ?? { [key]: err.message })
    }
    if (err?.type === 'entity.parse.failed') return res.status(400).json({ [key]: invalidJson })
    if (err?.type === 'entity.too.large') return res.status(413).json({ [key]: 'Request body too large.' })
    console.error(
      JSON.stringify({
        level: 'error',
        msg: 'unhandled error',
        requestId: req.get('lambda-web-request-id'),
        path: req.path,
        error: err?.name,
        detail: err?.message,
        stack: config.isProduction ? undefined : err?.stack,
      }),
    )
    res.status(500).json({ [key]: config.isProduction ? internal : `${internal}: ${err?.message}` })
  }
}

export function notFoundHandler(req, res) {
  res.status(404).json({ message: 'Not Found' })
}
