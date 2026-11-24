// Moved: the DynamoDB code now lives in src/data/dynamodb.js, behind the database
// seam in src/data/index.js (which also offers the Supabase/Postgres driver).
// This re-export keeps existing imports (lambdas/adapter.js, scripts) working.
export { clean, closeDatabase, initDatabase, isMissingIndexError, pingDatabase, table } from '../data/index.js'
