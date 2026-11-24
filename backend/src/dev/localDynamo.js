// In-process DynamoDB emulator for local development and tests (dynalite,
// in-memory). Development-only dependency: never imported in production.
import dynalite from 'dynalite'

export async function startLocalDynamo() {
  const server = dynalite({ createTableMs: 0, deleteTableMs: 0, updateTableMs: 0 })
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const { port } = server.address()
  return {
    endpoint: `http://127.0.0.1:${port}`,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  }
}
