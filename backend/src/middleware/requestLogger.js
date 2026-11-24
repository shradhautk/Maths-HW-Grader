// One structured log line per request, tagged with the lambda-web-request-id
// header Lambda adds (so a client error can be matched to its log). Never logs
// bodies or tokens.
import config from '../config/index.js'

export function requestLogger() {
  return (req, res, next) => {
    if (!config.server.logRequests) return next()
    const started = process.hrtime.bigint()
    res.on('finish', () => {
      console.log(
        JSON.stringify({
          level: 'info',
          requestId: req.get('lambda-web-request-id'),
          method: req.method,
          path: req.originalUrl.split('?')[0],
          operation: typeof req.body?.operation === 'string' ? req.body.operation : undefined,
          status: res.statusCode,
          ms: Number((process.hrtime.bigint() - started) / 1000000n),
        }),
      )
    })
    next()
  }
}
