// One AWS Lambda per API path (see lambdas/LAMBDAS.md). The split and each
// function's AWS needs come from the dependency graph in
// extra/leftovers/graphify-out (rebuilt with `graphify update`): every function
// runs the same backend code from the shared layer, and serves only its paths.
//
//   paths   API paths (config/app.yaml `api` keys) this function answers
//   aws     AWS services its code reaches (graph: route → service → src/aws/*)
//   tables  DynamoDB table groups it reads/writes (graph: → repositories/*)
//   entry   files where its requests enter the code (the graph check walks from here)
//
// `npm run check:lambdas` re-reads the graph and fails if a function's code now
// reaches an AWS service its `aws` list (and so its IAM role) doesn't have.

/** @typedef {{ entry: string[], description: string, paths: (p: any) => string[], aws: string[], tables: string[], memory: number, timeout: number }} LambdaSpec */

const WORKFLOW_TABLES = ['Users', 'assignment_grading', 'GradeAnt_Assignments', 'GradeAnt_Submissions', 'GradeAnt_Announcements']
const ADMIN_TABLES = [
  'Users',
  'GradeAnt_Users',
  'GradeAnt_ResetRequests',
  'GradeAnt_Content',
  'GradeAnt_Tickets',
  'GradeAnt_AuditLogs',
  'GradeAnt_Settings',
  'GradeAnt_Invites',
  'GradeAnt_Analytics',
]

/** @type {Record<string, LambdaSpec>} */
export const FUNCTIONS = {
  auth: {
    entry: ['src/routes/auth.routes.js'],
    description: 'POST /auth/* — login, signup, logout, forgot/change password, me, profile photo',
    paths: (p) => [p.auth.base],
    aws: ['dynamodb', 'secrets', 's3'],
    tables: ADMIN_TABLES, // auth.service also files password-reset requests
    memory: 1024, // PBKDF2 (100k rounds) on every login is CPU-bound
    timeout: 29,
  },
  grades: {
    entry: ['src/services/grades.service.js', 'src/controllers/operations.controller.js'],
    description: 'POST /getgrades — one student’s result, class roster, grade updates',
    paths: (p) => [p.grades],
    aws: ['dynamodb', 'secrets'],
    tables: WORKFLOW_TABLES,
    memory: 512,
    timeout: 29,
  },
  corrections: {
    entry: ['src/services/correction.service.js', 'src/controllers/operations.controller.js'],
    description: 'POST /correctanswer — answer and question corrections',
    paths: (p) => [p.corrections],
    aws: ['dynamodb', 'secrets'],
    tables: WORKFLOW_TABLES,
    memory: 512,
    timeout: 29,
  },
  marks: {
    entry: ['src/services/marks.service.js', 'src/controllers/operations.controller.js'],
    description: 'POST /updatedb — teacher mark edits',
    paths: (p) => [p.marks],
    aws: ['dynamodb', 'secrets'],
    tables: WORKFLOW_TABLES,
    memory: 512,
    timeout: 29,
  },
  workflow: {
    entry: ['src/services/workflow/index.js', 'src/services/jobs.service.js', 'src/controllers/operations.controller.js'],
    description: 'POST /workflow — assignments, rubric, grading runs, publishing, assistant, student screens',
    paths: (p) => [p.workflow],
    aws: ['dynamodb', 'secrets', 'lambda', 'ses', 'sns', 'sqs'],
    tables: WORKFLOW_TABLES,
    memory: 1024,
    timeout: 29, // API Gateway stops at 29 s; AI rubric drafting runs in gradeant-jobs instead
  },
  admin: {
    entry: ['src/controllers/admin.controller.js'],
    description: 'POST /admin — admin console and invite links',
    paths: (p) => [p.admin],
    aws: ['dynamodb', 'secrets', 'ses'],
    tables: ADMIN_TABLES,
    memory: 1024, // invites and password resets hash passwords
    timeout: 29,
  },
  contact: {
    entry: ['src/routes/contact.routes.js'],
    description: 'POST /contact — public contact and request-a-pilot forms',
    paths: (p) => [p.contact],
    aws: ['ses'],
    tables: [],
    memory: 256,
    timeout: 15,
  },
  upload: {
    entry: ['src/routes/upload.routes.js'],
    description: 'POST /uploadpdf — presigned S3 upload URLs (off until UPLOAD_BUCKET is set)',
    paths: (p) => [p.upload],
    aws: ['dynamodb', 'secrets', 's3'],
    tables: ['Users', 'GradeAnt_Assignments'], // a student uploads only for their own classes' assignments
    memory: 512,
    timeout: 15,
  },
  health: {
    entry: ['src/controllers/health.controller.js'],
    description: 'GET /health, /health/ready — liveness and readiness checks',
    paths: (p) => [p.health, p.healthReady],
    aws: ['dynamodb', 'secrets'],
    tables: ['Users'],
    memory: 256,
    timeout: 10,
  },
}

/** The SQS worker for background jobs (AI rubric drafting): handler worker.handler, not behind API Gateway. */
export const WORKER = {
  name: 'jobs',
  entry: ['worker.js'],
  description: 'SQS consumer for background jobs (AI rubric drafting)',
  aws: ['dynamodb', 'secrets', 'sqs'],
  tables: WORKFLOW_TABLES,
  memory: 1024,
  timeout: 300,
  // Reachable in the graph but never called by a job: the worker only uses defaultRubric from
  // workflow/assignments.js, which also imports aws/lambda.js for "Start grading".
  unused: { lambda: 'workflow/assignments.js is imported for defaultRubric only' },
}
