# GradeAnt API — Node.js + Express

The GradeAnt backend as one Express app. It replaces the six Python Lambdas, and the same code runs:

- **locally**: `npm run dev`. It uses an in-process DynamoDB, local file storage and logged emails, so no AWS account is needed.
- **on AWS**: `npm run start:prod`, or as a **Lambda Web Function**. It uses DynamoDB, S3, SES, SNS, SQS, Lambda, and Gemini (`gemini-3.5-flash-preview`) for AI.

Which environment runs is decided by configuration only. The React app in [`../web`](../web) is unchanged apart from its env files.

```
backend/
├── index.js              entry point (Lambda Web Functions loads index.js and routes to AWS_LAMBDA_HTTP_ENDPOINT)
├── worker.js             optional SQS consumer for background jobs (JOBS_DRIVER=sqs)
├── .env.example          every setting, documented
├── .env.development      npm run dev   (local drivers, demo data)
├── .env.production       npm run start:prod / Lambda (AWS drivers, no secrets)
├── .env.test             npm test
├── src/
│   ├── config/           the only place that reads process.env; picks the mode and .env file
│   ├── routes/           Express routers (URL → controller), one per API resource
│   ├── controllers/      request parsing, auth, operation dispatch, response shape
│   ├── services/         business logic (auth, grades, correction, marks, admin, upload, jobs, workflow/*)
│   ├── repositories/     DynamoDB access per table (keys, indexes, expressions)
│   ├── aws/              DynamoDB, S3, SES, SNS, SQS, Lambda, Secrets Manager (+ local drivers)
│   ├── middleware/       CORS, bearer-token auth, JSON body + error handling, request logging
│   ├── lib/              tokens, password hashing, errors, Python-compatible helpers
│   └── dev/              table schema, in-memory DynamoDB, demo seed (development only)
├── test/                 node:test suites (42 tests)
└── scripts/              package.mjs (deployment zip), parity.mjs (Python vs Node comparison)
```

## Run it locally

You need **Node.js 22+** (Lambda runs Node.js 24).

```bash
# Terminal 1 — API on http://localhost:8787
cd backend
npm install
npm run dev

# Terminal 2 — web app on http://localhost:5173 (web/.env.development already points at :8787)
cd web
npm install
npm run dev
```

Demo logins: choose the matching role on the login page. The password for every account is **`Password123!`**.

| Role | Email |
|---|---|
| Admin | `admin@gradeant.dev` |
| Teacher | `teacher@gradeant.dev` |
| Student | `riya@gradeant.dev`, `sam@gradeant.dev`, `li@gradeant.dev` |

How local development behaves:

- **Data** lives in memory and resets on restart. Set `DB_DRIVER=dynamodb` and `DYNAMODB_ENDPOINT=http://localhost:8000` to use [DynamoDB Local](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.html) instead.
- **Uploads** go to `backend/.data/storage` through signed, expiring URLs, so the browser follows the same flow as with S3.
- **Emails, notifications and grading-pipeline calls** are logged, not sent.
- **The GradeFlow assistant** gives a canned reply. To call Gemini, put `AI_DRIVER=gemini` and `GEMINI_API_KEY=...` in `.env.development.local`.

`GET /health` reports liveness. `GET /health/ready` also checks DynamoDB and the JWT secret.

## Environments

| Command | Mode | Settings file |
|---|---|---|
| `npm run dev` | development (auto-restarts on changes) | `.env.development` |
| `npm run start:prod` | production | `.env.production` |
| `npm test` | test | `.env.test` |
| Lambda Web Function | production (detected from `AWS_LAMBDA_HTTP_ENDPOINT`, or `APP_ENV`) | `.env.production` |

The mode is chosen by the first of these that applies: `--mode=…`, then `APP_ENV`, then production when running on Lambda, otherwise development.

The loading order is `.env.<mode>`, then `.env.<mode>.local`. The `.local` file is gitignored and is the place for personal overrides and local secrets. Real environment variables always win, so Lambda revision variables override the files.

Each integration has a *driver* setting, so local and AWS implementations swap without code changes:

| Setting | Development | Production |
|---|---|---|
| `DB_DRIVER` | `memory` (in-process DynamoDB) | `dynamodb` |
| `STORAGE_DRIVER` | `local` (files + signed URLs) | `s3` (presigned URLs) |
| `MAIL_DRIVER` | `log` | `ses` |
| `NOTIFY_DRIVER` | `log` | `sns` |
| `AI_DRIVER` | `canned` | `gemini` |
| `GRADING_DRIVER` | `log` | `lambda` |
| `JOBS_DRIVER` | `inline` | `inline` or `sqs` |

Every variable is documented in [`.env.example`](.env.example).

### Secrets

- **Production.** Secrets are never put in files. `JWT_SECRET_ARN` and `SESSION_TOKEN_SECRET_ARN` name AWS Secrets Manager secrets, which are read once per execution environment at first use. The AWS Web Functions guide recommends this over environment variables.
- **Shared JWT secret.** The JWT secret **must be the one the Python Lambdas use**, so sessions keep working across the switch.
  - If `JWT_SECRET` was never set on those Lambdas, they are using the default hard-coded in `authetication.py`. Rotate it: store a new value in Secrets Manager and set it on both.
- **Development.** Uses throwaway local keys from `.env.development`.
- **Browser bundle.** The React app gets only public URLs (`VITE_*`). AWS credentials never reach it.

## API: what replaced what

Request and response bodies, status codes, error messages and authorization rules are preserved. The frontend calls exactly the same paths.

| Route | Python Lambda (kept, unchanged) | Node |
|---|---|---|
| `POST /auth/{login,signup,logout,forgot-password,change-password,me,profile-photo}` | `lambda_python_files/authetication.py` | `routes/auth.routes.js` → `services/auth.service.js` |
| `POST /getgrades` | `lambda_python_files/get_grades.py` | `services/grades.service.js` |
| `POST /correctanswer` | `lambda_python_files/correct_questions_answer.py` | `services/correction.service.js` |
| `POST /updatedb` | `lambda_python_files/update_db.py` | `services/marks.service.js` |
| `POST /admin` | `admin/lambda_admin.py` | `services/admin.service.js` |
| `POST /workflow` (27 operations) | `lambda_python_files/workflow.py` | `services/workflow/*` |
| `POST /uploadpdf` | *not in this repo* | `services/upload.service.js` (see below) |
| `GET /health`, `GET /health/ready` | — | `controllers/health.controller.js` |

AWS integrations, Python → Node (AWS SDK v3):

| Service | Used for | Node |
|---|---|---|
| DynamoDB | all data (same tables, keys and indexes) | `aws/dynamodb.js` (`@aws-sdk/lib-dynamodb`) |
| S3 | profile photos, upload URLs (presigned) | `aws/s3.js` |
| Gemini (Google, not AWS) | GradeFlow assistant, AI rubric drafting | `ai/index.js` (`@google/genai`; a declined request returns null) |
| SES | invites, submission reminders | `aws/ses.js` |
| SNS | "grades published" notification | `aws/sns.js` |
| Lambda | async hand-off to the grading pipeline | `aws/lambda.js` |
| SQS | *new, optional:* background jobs (replaces `workflow.py` invoking itself) | `aws/sqs.js` + `worker.js` |
| Secrets Manager | *new:* JWT/session secrets | `aws/secrets.js` |

### What stays in Python

Nothing at runtime. All six Lambdas are request/response logic over DynamoDB plus SDK calls, which Node handles fully, so no FastAPI service was needed.

- **ML in this repo.** None. Gemini is called through the Google Gen AI SDK.
- **OCR and grading pipeline.** It sits outside this repo. The API only hands it work, through an async Lambda invoke (`GRADING_FUNCTION_NAME`), exactly as before.
- **The Python files are kept unchanged** until the Node API has run in production. They are the rollback path, and the parity check runs against them. Delete `lambda_python_files/` and `admin/lambda_admin.py` only after cutover.

### `/uploadpdf`

The original presign Lambda is not in the repository, so the S3 key layout the grading pipeline reads is unknown.

- The Node endpoint writes keys from `UPLOAD_KEY_TEMPLATE`. It also now requires a login and validates the files (PDF, size, count, and students upload only their own work).
- In production it is **off until `UPLOAD_BUCKET` is set**.
- Until then, `web/.env.production` keeps `VITE_UPLOAD_BASE_URL` pointing at the existing upload Lambda.
- To move uploads to Node, set the template to the pipeline's layout, set `UPLOAD_BUCKET`, then remove `VITE_UPLOAD_BASE_URL`.

### Intentional differences from the Python Lambdas

These are listed by the parity check below. Everything else matches.

1. **`completeFirstLogin` works.** `lambda_admin.py` writes `SET consumed = :true`, and `CONSUMED` is a DynamoDB reserved word. Real DynamoDB rejects that expression, so first-login setup currently saves the password and then fails with a 500. Node uses attribute-name placeholders for every expression.
2. **Malformed JSON is a 400** with the endpoint's usual error shape. Some Lambdas returned a 500 with the parser's text.
3. **`/getgrades` `updateStudentGrade` with no matching summary row returns 404.** Python returned nothing, which API Gateway turns into a 502.
4. **`/updatedb` `updateMarks` with only per-question marks works.** Python crashed with a 500 when `summary` was missing.
5. **Unexpected errors don't leak exception text.** They return each endpoint's generic message; details go to the logs.
6. **Queries and scans follow pagination,** so results over DynamoDB's 1 MB page are complete. `get_grades.py`, `correct_questions_answer.py` and `lambda_admin.py` read only the first page.
7. **AI rubric drafting can't get stuck.** If Gemini rejects the JSON schema (400), the request is retried with the schema in the prompt. If drafting fails anyway, the teacher gets the question-based rubric instead of "Drafting rubric…" forever.
8. **CORS uses an allowlist** (`CORS_ALLOWED_ORIGINS`) instead of `*`, or echoing any origin as `/admin` did.

### Security notes on behaviour that was kept as-is

- **Self-signup is limited to `student` and `teacher`.** `POST /auth/signup` rejects any other `role` (including `admin`) with a 400, and defaults a missing role to `student`. `authetication.py` enforces the same rule.
- **Two password stores for admins.** The admin console's "change own password" updates `GradeAnt_Users.password_hash`, but login reads `Users.password`, so it doesn't change the login password. This matches `lambda_admin.py`.

## Tests

```bash
npm test                    # 42 tests: workflow (Python suite ported, with/without the assignment_id GSI),
                            # HTTP/auth/CORS/uploads/admin, Gemini client against a fake,
                            # Lambda startup contract, Python-compatible helpers

# Python vs Node: the same 210 requests against both implementations, diffed
python -m venv .venv && .venv/bin/pip install -r ../lambda_python_files/requirements-dev.txt
PYTHON=.venv/bin/python npm run check:parity

npm run package -- --verify # build dist/function.zip, extract it, start it as Lambda would
```

The parity check seeds the same fixture into both implementations. The Python side runs on moto and the Node side on the in-process DynamoDB. It then compares every response, after masking random ids, tokens and timestamps. It also compares the prompts sent to Claude.

The current result: **197 identical, 6 identical apart from DynamoDB result order, 8 intentional differences (above), 0 unexplained.**

## Deploying to AWS Lambda Web Functions

> **One Lambda per API path, behind the existing API Gateway: [`lambdas/LAMBDAS.md`](lambdas/LAMBDAS.md)** (layers, 10 functions, per-function IAM from the dependency graph).
>
> **Recommended path: [`deploy/DEPLOY.md`](deploy/DEPLOY.md).** Lambda Web Functions are not offered in eu-north-1, where the tables are, so the API is deployed as a regular Lambda with the AWS Lambda Web Adapter and a Function URL, in eu-north-1 (SAM template, pre-flight checks and a GitHub workflow included). The steps below remain valid for a Region that has Web Functions.


Nothing in this repository deploys anything. These steps follow the [Lambda Web Functions developer guide](https://docs.aws.amazon.com/lambda/latest/dg/web-functions.html); check it for changes before running them.

**Regions.** Web Functions are available in us-east-1, us-east-2, us-west-2, ap-northeast-1 and eu-west-1. The current API and tables are in **eu-north-1**, so the nearest option is **eu-west-1**, with `AWS_DATA_REGION=eu-north-1`. Each DynamoDB call then crosses Regions, which adds latency and inter-Region data transfer cost. Moving the tables or using Global Tables avoids that.

**Runtime facts the app relies on:**

- Node.js 24 (`nodejs24.x`), Arm64.
- `index.js` at the package root, listening on `AWS_LAMBDA_HTTP_ENDPOINT` within 10 s.
- SIGTERM on scale-down.
- `AWS_REGION` is reserved.
- No WebSockets.
- The default request timeout is 30 s.

### 1. Prerequisites

- AWS CLI v2 with the `lambda-web` commands (`aws lambda-web help`).
- A paid-tier Gemini API key (Google AI Studio), as `GEMINI_API_KEY` or a Secrets Manager secret in `GEMINI_API_KEY_ARN`.
- Secrets in Secrets Manager, in the data Region: the JWT secret (the value the Python Lambdas use) and an admin session secret.

### 2. Execution role

Trust policy: `lambda.amazonaws.com` → `sts:AssumeRole`. Permissions (replace `ACCOUNT`, Regions, names):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": ["arn:aws:logs:eu-west-1:ACCOUNT:log-group:/aws/lambda/web/gradeant-api:*",
                   "arn:aws:logs:*:ACCOUNT:log-group:/aws/lambda/web/gradeant-api.local:*"] },
    { "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"],
      "Resource": ["arn:aws:dynamodb:eu-north-1:ACCOUNT:table/Users", "arn:aws:dynamodb:eu-north-1:ACCOUNT:table/assignment_grading",
                   "arn:aws:dynamodb:eu-north-1:ACCOUNT:table/assignment_grading/index/*", "arn:aws:dynamodb:eu-north-1:ACCOUNT:table/GradeAnt_*",
                   "arn:aws:dynamodb:eu-north-1:ACCOUNT:table/GradeAnt_*/index/*"] },
    { "Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"],
      "Resource": ["arn:aws:secretsmanager:eu-north-1:ACCOUNT:secret:gradeant/jwt-*", "arn:aws:secretsmanager:eu-north-1:ACCOUNT:secret:gradeant/session-*"] },
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": ["arn:aws:s3:::PROFILE_PHOTO_BUCKET/profile-photos/*", "arn:aws:s3:::UPLOAD_BUCKET/*"] },
    { "Effect": "Allow", "Action": "ses:SendEmail", "Resource": "*" },
    { "Effect": "Allow", "Action": "sns:Publish", "Resource": "arn:aws:sns:eu-north-1:ACCOUNT:TOPIC" },
    { "Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": "arn:aws:lambda:eu-north-1:ACCOUNT:function:GRADING_FUNCTION" },
    { "Effect": "Allow", "Action": "sqs:SendMessage", "Resource": "arn:aws:sqs:eu-north-1:ACCOUNT:gradeant-jobs" }
  ]
}
```

Notes on the policy:

- **Gemini.** Called over HTTPS with an API key; the only AWS permission is `secretsmanager:GetSecretValue` on the key secret, if you use one.
- **Optional statements.** Drop the SNS, Lambda, SQS and S3 statements for any feature you don't configure.
- **Deploying identity.** The identity that deploys needs `iam:PassRole` on this role, plus the S3 and IAM permissions listed in the guide's prerequisites.

### 3. Configure and package

1. Fill in `backend/.env.production`. It holds no secrets, only names, ARNs and URLs:
   - `CORS_ALLOWED_ORIGINS`: the web app's origin.
   - `JWT_SECRET_ARN` and `SESSION_TOKEN_SECRET_ARN`.
   - Buckets, `SES_SENDER_EMAIL`, `SNS_TOPIC_ARN`, `GRADING_FUNCTION_NAME`.
2. Build and check the package:

```bash
cd backend
npm run package -- --verify     # → dist/function.zip (+ dist/package/), checked like Lambda would run it
```

### 4. Create the Web Function

**Quick path** (creates the function, a revision, and an endpoint named `dev` that serves each new revision):

```bash
aws lambda-web deploy --name gradeant-api --code dist/package \
  --execution-role-arn arn:aws:iam::ACCOUNT:role/gradeant-api-role \
  --bucket-name YOUR-VERSIONED-BUCKET --create --region eu-west-1
```

- The endpoint uses the default application-managed auth. That is what this app expects, because it checks bearer tokens itself.
- Its URL looks like `https://<id>.<x>.lambda-web.eu-west-1.on.aws`.
- Revisions get the default 30 s timeout.

**Production path.** Create the function from an uploaded zip with explicit revision settings, using the commands in [Creating a Web Function](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-create.html#web-functions-create-cli-api):

```bash
aws s3 cp dist/function.zip s3://YOUR-VERSIONED-BUCKET/gradeant-api/function.zip
aws lambda-web create-web-function-revision --function-name gradeant-api \
  --build-config '{"codeConfig":{"s3Object":{"bucket":"YOUR-VERSIONED-BUCKET","key":"gradeant-api/function.zip"}},"runtimeConfig":{"runtime":"nodejs24.x"}}' \
  --service-config '{"executionRoleArn":"arn:aws:iam::ACCOUNT:role/gradeant-api-role","timeoutSeconds":90,"maxConcurrencyPerEnvironment":32}'
aws lambda-web wait web-function-revision-active --function-name gradeant-api --revision-id REV_ID
```

- **Timeout.** `timeoutSeconds: 90` leaves room for Claude calls. AI rubric drafting runs inside the request with `JOBS_DRIVER=inline`.
- **Concurrency.** `maxConcurrencyPerEnvironment: 32` keeps the CPU-bound password hashing from crowding one environment; tune it under load.
- **Revisions.** Revisions are immutable and don't copy settings, so pass the full `serviceConfig` each time.
- **Revision variables (optional).** Add `"environmentVariables": {...}` to `serviceConfig` to override `.env.production` for a revision.

### 5. Verify, then switch the web app

```bash
curl https://<endpoint>/health          # {"status":"ok",...,"mode":"production"}
curl https://<endpoint>/health/ready    # checks DynamoDB access and the JWT secret
```

1. Set `VITE_API_BASE_URL` in `web/.env.production` to the endpoint URL (or your CloudFront domain).
2. Rebuild the web app.

**Rollback** is setting it back. The Python API and the Node API share the same tables and JWT secret, so sessions carry over in both directions.

**Optional extras:**

- **CloudFront.** Put CloudFront in front for a custom domain or WAF. With an `IamAuth` endpoint and origin access control, direct calls can't bypass it.
- **SQS jobs.** For background jobs, set `JOBS_DRIVER=sqs` and `JOBS_QUEUE_URL`. Then deploy the same zip as a regular Node.js 22+ Lambda with:
  - handler `worker.handler`
  - `APP_ENV=production`
  - the queue as its event source, with ReportBatchItemFailures enabled
  - the same DynamoDB permissions (and the Gemini key secret), plus `sqs:ReceiveMessage`, `sqs:DeleteMessage` and `sqs:GetQueueAttributes`
