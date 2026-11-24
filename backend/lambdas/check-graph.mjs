// Checks the per-path Lambda split against the code dependency graph built by
// graphify (extra/leftovers/graphify-out/graph.json; refresh it with
// `graphify update .` from the repo root after code changes).
//
// For each function it walks the graph from its entry files (lambdas/functions.js)
// through imports and calls, collects every src/aws/* module reached, and
// compares that with the function's declared `aws` list, which its IAM role in
// lambdas/template.yaml is built from. A service the code reaches but the list
// lacks would fail at runtime with AccessDenied, so the check fails.
//
//   npm run check:lambdas                    uses ../extra/leftovers/graphify-out/graph.json
//   npm run check:lambdas -- path/graph.json
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { FUNCTIONS, WORKER } from './functions.js'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const graphFile = path.resolve(process.argv[2] ?? path.join(ROOT, '..', 'extra', 'leftovers', 'graphify-out', 'graph.json'))
if (!existsSync(graphFile)) {
  console.error(`Graph not found: ${graphFile}\nBuild it from the repo root with: graphify update .`)
  process.exit(2)
}

const graph = JSON.parse(readFileSync(graphFile, 'utf8'))
const fileOf = new Map(graph.nodes.map((n) => [n.id, (n.source_file || '').replaceAll('\\', '/')]))
const FOLLOW = new Set(['imports', 'imports_from', 'calls', 'indirect_call', 'dynamic_import'])
const deps = new Map()
for (const e of graph.links) {
  if (!FOLLOW.has(e.relation)) continue
  const from = fileOf.get(e.source)
  const to = fileOf.get(e.target)
  if (!from?.startsWith('backend/') || !to?.startsWith('backend/src/') || from === to) continue
  if (!deps.has(from)) deps.set(from, new Set())
  deps.get(from).add(to)
}

// Middleware every API function runs (the adapter mounts them); auth is reached from each entry that uses it.
const SHARED = ['src/middleware/cors.js', 'src/middleware/errors.js', 'src/middleware/requestLogger.js', 'src/controllers/respond.js']

function reach(entries) {
  const seen = new Set()
  const stack = entries.map((e) => `backend/${e}`)
  while (stack.length) {
    const f = stack.pop()
    if (seen.has(f)) continue
    seen.add(f)
    for (const d of deps.get(f) ?? []) if (!d.includes('/src/dev/')) stack.push(d) // dev-only code never runs on Lambda
  }
  return seen
}

const built = graph.built_at_commit ? ` (graph built at commit ${String(graph.built_at_commit).slice(0, 8)})` : ''
console.log(`Checking Lambda permissions against ${path.relative(process.cwd(), graphFile)}${built}\n`)

let problems = 0
const rows = [
  ...Object.entries(FUNCTIONS).map(([name, spec]) => ({ name, spec, entries: [...spec.entry, ...SHARED] })),
  { name: WORKER.name, spec: WORKER, entries: WORKER.entry },
]
for (const { name, spec, entries } of rows) {
  const missingEntry = entries.filter((e) => !fileOf.size || ![...fileOf.values()].includes(`backend/${e}`))
  if (missingEntry.length) {
    console.log(`  STALE gradeant-${name}: ${missingEntry.join(', ')} not in the graph; run \`graphify update .\``)
    problems++
    continue
  }
  const files = reach(entries)
  // AWS services = the src/aws/* modules reached, plus the DynamoDB driver behind the
  // database seam (src/data/dynamodb.js; the Postgres driver there is not an AWS service).
  const awsModule = (f) => f.startsWith('backend/src/aws/') || f === 'backend/src/data/dynamodb.js'
  const reached = [...new Set([...files].filter(awsModule).map((f) => path.basename(f, '.js')))].sort()
  const unused = spec.unused ?? {}
  const missing = reached.filter((a) => !spec.aws.includes(a) && !unused[a])
  const extra = spec.aws.filter((a) => !reached.includes(a))
  const status = missing.length ? 'FAIL' : 'ok  '
  if (missing.length) problems++
  console.log(`  ${status} gradeant-${name.padEnd(12)} ${String(files.size).padStart(3)} files  reaches: ${reached.join(', ') || '—'}`)
  if (missing.length) console.log(`         missing from functions.js aws (and its IAM role): ${missing.join(', ')}`)
  for (const [a, why] of Object.entries(unused)) if (reached.includes(a)) console.log(`         ${a}: reachable but not used (${why})`)
  if (extra.length) console.log(`         declared but not reached: ${extra.join(', ')} (harmless; can be removed)`)
}

if (problems) {
  console.error(`\n${problems} function(s) need attention.`)
  process.exit(1)
}
console.log('\nEvery function’s IAM role covers the AWS services its code reaches.')
