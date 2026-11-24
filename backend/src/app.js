// Express application (no listen() here, so tests can mount it on any port).
import express from 'express'
import config from './config/index.js'
import { cors } from './middleware/cors.js'
import { notFoundHandler } from './middleware/errors.js'
import { requestLogger } from './middleware/requestLogger.js'
import routes from './routes/index.js'

export function createApp() {
  const app = express()
  app.disable('x-powered-by')
  app.set('env', config.isProduction ? 'production' : 'development')
  app.use(requestLogger())
  app.use(cors())
  app.use(config.server.basePath || '/', routes)
  app.use(notFoundHandler)
  return app
}
