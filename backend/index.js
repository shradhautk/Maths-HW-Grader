// Entry point. Lambda Web Functions load index.js at the package root and route
// HTTP traffic to the host:port in AWS_LAMBDA_HTTP_ENDPOINT; the server must be
// listening within 10 seconds. Locally it listens on HOST:PORT (default
// 127.0.0.1:8787).
//
//   npm run dev          development (.env.development: in-memory DynamoDB, local files)
//   npm run start:prod   production  (.env.production: AWS services)
import config from './src/config/index.js'
import { createApp } from './src/app.js'
import { closeDatabase, initDatabase } from './src/aws/dynamodb.js'

await initDatabase()

const app = createApp()
const { host, port } = config.server
const server = app.listen(port, host, () => {
  console.log(
    JSON.stringify({
      level: 'info',
      msg: `GradeAnt API listening on http://${host}:${port}${config.server.basePath}`,
      mode: config.mode,
      envFiles: config.envFiles,
      drivers: config.drivers,
    }),
  )
})

// Lambda sends SIGTERM before removing an environment and allows ~1 s for cleanup.
for (const signal of ['SIGTERM', 'SIGINT']) {
  process.on(signal, () => {
    server.close()
    closeDatabase().finally(() => process.exit(0))
    setTimeout(() => process.exit(0), 800).unref()
  })
}
