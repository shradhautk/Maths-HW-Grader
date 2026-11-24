# GradeAnt API as one Lambda per API path

> **Python version:** the same nine API functions in Python 3.13, with the same routes, permissions and API Gateway steps, are in [`python-backend/`](../../python-backend/README.md). Deploy one or the other: both use the `gradeant-*` function names.

This replaces the six Python Lambdas with Node functions, one per API path, behind your **existing** API Gateway (`rus9nultj9`, stage `dev`).
- **Same code as the backend.** Every function runs `backend/src` unchanged, from a shared layer.
- **The single-function deployment stays.** It is still in `backend/deploy/` for later. Nothing here changes it.
- **Nothing deploys on its own.** Every step below is one you run.

| File | What it is |
|---|---|
| `functions.js` | The functions: paths, entry files, AWS services, tables, memory, timeout |
| `adapter.js` | Turns an API Gateway event into a request to the Express routes, and the reply back |
| `template.yaml` | Layers, 10 functions, one IAM role each, jobs queue, logs, alarms (SAM) |
| `samconfig.example.toml` | Your account's settings (copy to `samconfig.toml`, gitignored) |
| `check-graph.mjs` | Checks each role against the code dependency graph (`npm run check:lambdas`) |
| `../scripts/package-lambdas.mjs` | Builds every zip into `dist/lambdas/` (`npm run package:lambdas -- --verify`) |
| `../test/lambda.test.js` | Feeds real API Gateway events to each handler; replies must match the Express app |

---

## 1. How it fits together

```
Browser ──► API Gateway REST API rus9nultj9 (stage /dev, eu-north-1), Lambda proxy integrations
              │ POST /auth/*        ──► gradeant-auth
              │ POST /getgrades     ──► gradeant-grades
              │ POST /correctanswer ──► gradeant-corrections
              │ POST /updatedb      ──► gradeant-marks
              │ POST /workflow      ──► gradeant-workflow ──(SQS gradeant-jobs)──► gradeant-jobs
              │ POST /admin         ──► gradeant-admin
              │ POST /contact       ──► gradeant-contact
              │ POST /uploadpdf     ──► gradeant-upload   (later: legacy upload Lambda until then)
              │ GET  /health[/ready]──► gradeant-health
              ▼
   every function = index.mjs (a few lines) + layer gradeant-deps (npm packages)
                                            + layer gradeant-app  (backend/src, config/app.yaml)
```

**How a request flows:**
1. API Gateway passes the request to the function.
2. The function's handler (`adapter.js`, in the app layer) starts the backend's Express routes once per container.
3. It forwards the request to them and returns the reply in API Gateway's format.

Login checks, permissions, CORS, error messages and response bodies are therefore exactly the backend's. `test/lambda.test.js` checks this against the Express app.

**Why AI drafting moves to a queue.** API Gateway stops every request at 29 s, and AI rubric drafting can take longer. So `gradeant-workflow` puts drafting jobs on the SQS queue `gradeant-jobs` (`JOBS_DRIVER=sqs`), and `gradeant-jobs` runs them with up to 5 minutes each. This uses the backend's existing `worker.js`, so no new code was needed.

---

## 2. Each function: services, permissions and settings

The split comes from the dependency graph in `extra/leftovers/graphify-out` (rebuilt today with `graphify update`). For each path, the graph was walked from its entry files to every `src/aws/*` module and repository it reaches. `npm run check:lambdas` repeats that walk and fails if a role ever misses a service.

All functions: **Node.js 24, Arm64, handler `index.handler`, layers `gradeant-deps` + `gradeant-app`, Region eu-north-1.**

| Function | Route | Memory / timeout | AWS services it is allowed to use | DynamoDB tables (read/write) |
|---|---|---|---|---|
| `gradeant-auth` | `POST /auth/*` | 1024 MB / 29 s | DynamoDB; Secrets Manager (JWT + session); S3 (`profile-photos/*` only) | `Users`, `GradeAnt_*` (password-reset requests) |
| `gradeant-grades` | `POST /getgrades` | 512 MB / 29 s | DynamoDB; Secrets Manager (JWT) | `Users`, `assignment_grading` (+indexes), `GradeAnt_Assignments`, `GradeAnt_Submissions`, `GradeAnt_Announcements` |
| `gradeant-corrections` | `POST /correctanswer` | 512 MB / 29 s | same as grades | same as grades |
| `gradeant-marks` | `POST /updatedb` | 512 MB / 29 s | same as grades | same as grades |
| `gradeant-workflow` | `POST /workflow` | 1024 MB / 29 s | DynamoDB; Secrets Manager (JWT, Gemini key*); Gemini (assistant, outside AWS); SQS send (jobs queue); SES*; SNS*; Lambda invoke of the grading pipeline* | same as grades |
| `gradeant-admin` | `POST /admin` | 1024 MB / 29 s | DynamoDB; Secrets Manager (JWT + session); SES* (invites) | `Users`, `GradeAnt_*` |
| `gradeant-contact` | `POST /contact` | 256 MB / 15 s | SES* only | none |
| `gradeant-upload` | `POST /uploadpdf` | 512 MB / 15 s | DynamoDB; Secrets Manager (JWT); S3 upload bucket* | `Users`, `GradeAnt_Assignments` (a student uploads only for their own classes) |
| `gradeant-health` | `GET /health`, `GET /health/ready` | 256 MB / 10 s | DynamoDB `GetItem` on `Users`; Secrets Manager (JWT) | `Users` (one read) |
| `gradeant-jobs` | SQS `gradeant-jobs` (not API Gateway) | 1024 MB / 300 s | DynamoDB; Secrets Manager (Gemini key*); Gemini (rubric drafting); SQS receive (added by SAM) | same as grades |

\* Only when its setting is filled in: `SesSenderEmail`, `SnsTopicArn`, `GradingFunctionName`, `UploadBucketName`. Each has its own IAM policy that exists only then. SES sending is limited to that one From address.

**Notes on these permissions:**
- **Memory.** Login, signup and admin invites hash passwords (PBKDF2, 100k rounds), which is CPU-bound, so those functions get 1024 MB. Lambda CPU scales with memory.
- **Logs.** Every role can also write its own logs (`/aws/lambda/gradeant-<name>`, kept 30 days).
- **One exception from the graph.** The graph shows `gradeant-jobs` can *reach* the Lambda-invoke module, but only through an import of `defaultRubric`; it never starts grading. So it is deliberately not given that permission, and `check:lambdas` reports it as "reachable but not used".

**Environment variables (same on every function; set by the template).** No secret values, only ARNs:

| Variable | From parameter | Meaning |
|---|---|---|
| `APP_ENV` | – | `production` |
| `AWS_DATA_REGION` | `DataRegion` | Region of the tables, SNS and grading Lambda |
| `CORS_ALLOWED_ORIGINS` | `WebOrigins` | Sites allowed to call the API |
| `JWT_SECRET_ARN` | `JwtSecretArn` | Login token key (Secrets Manager) |
| `SESSION_TOKEN_SECRET_ARN` | `SessionSecretArn` | Admin first-login key (created if empty) |
| `PROFILE_PHOTO_BUCKET` | `ProfilePhotoBucketName` | Created if empty (private, encrypted, TLS-only) |
| `UPLOAD_BUCKET`, `UPLOAD_KEY_TEMPLATE` | same names | `/uploadpdf` (off while empty) |
| `SES_SENDER_EMAIL`, `CONTACT_INBOX` | same names | Email |
| `SNS_TOPIC_ARN` | `SnsTopicArn` | "Grades published" |
| `GRADING_FUNCTION_NAME` | `GradingFunctionName` | Grading pipeline hand-off |
| `GEMINI_MODEL`, `GEMINI_API_KEY_ARN`, `GEMINI_API_KEY` | same names | Gemini (the one model for all AI) |
| `JOBS_DRIVER`, `JOBS_QUEUE_URL` | – | `sqs` and the queue the stack creates |

Table names, token lifetimes and limits come from `backend/.env.production` inside the app layer (documented in `backend/.env.example`).

**Shared resources the stack creates:**
- **Layers** `gradeant-deps` and `gradeant-app`.
- **Queue** `gradeant-jobs`: visibility 30 min, encrypted. After 3 failed tries a job moves to the dead-letter queue `gradeant-jobs-dlq`, kept 14 days.
- **Secret** `gradeant/session`, if you don't pass one.
- **Photo bucket**, if you don't pass one.
- **Permissions** for your REST API to invoke each function (when `RestApiId` is set).
- **Alarms** `gradeant-lambdas-errors` (5+ crashes or timeouts in 5 minutes, across all functions) and `gradeant-jobs-failed` (a job reached the dead-letter queue).

---

## 3. One-time AWS setup

These steps are the same as for the single-function deployment. Follow **`backend/deploy/DEPLOY.md` §3**:

1. **Tools:** AWS CLI v2, SAM CLI, Node 22/24, and a profile in the account that owns the tables (§3.1).
2. **JWT secret** (§3.2): store the key the Python Lambdas use as `gradeant/jwt`, and note its ARN. This is the step that matters most.
3. **A Gemini API key** (paid tier) as `GeminiSecretArn` or `GeminiApiKey` (DEPLOY.md §3.3).
4. **SES** domain verification and production access (§3.4), if you want email.
5. **The grading Lambda's name** (§3.5).
6. **Pre-flight check** (read-only), §3.7: `bash deploy/preflight.sh`.

---

## 4. Build and test the zips

```bash
cd backend
npm ci
npm test                                  # includes test/lambda.test.js (11 checks on the handlers)
npm run check:lambdas                     # roles vs. the dependency graph
npm run package:lambdas -- --verify       # builds dist/lambdas/ and invokes every function from its zips
```

`dist/lambdas/` then holds:

| Zip | Upload as | Lambda path | Size |
|---|---|---|---|
| `layer-gradeant-deps.zip` | Layer **gradeant-deps** (Node.js 22/24, arm64) | `/opt/nodejs/node_modules` | ~20 MB zipped, ~56 MB unzipped |
| `layer-gradeant-app.zip` | Layer **gradeant-app** (Node.js 22/24, arm64) | `/opt/nodejs/gradeant` | ~0.1 MB |
| `gradeant-<name>.zip` × 10 | Function code, handler `index.handler` | `/var/task/index.mjs` | < 1 KB each |
| `manifest.json` | – | – | Routes, services, tables, sizes, SHA-256 of every zip |

- **Size limit.** Each function plus both layers comes to about 56 MB unzipped, well under Lambda's 250 MB limit.
- **Windows builds are fine.** The zips are written with Unix permissions, so they work wherever they were built.
- **What a code change means.** Only `layer-gradeant-app.zip` changes when backend code changes, and every function picks up the new layer version on deploy. `layer-gradeant-deps.zip` only changes when `package-lock.json` does.

---

## 5. Create everything (SAM, recommended)

```bash
cd backend/lambdas
cp samconfig.example.toml samconfig.toml   # fill in JwtSecretArn, RestApiId, and any optional settings
sam validate --lint
sam deploy                                 # shows every change and asks before applying
```

This takes about 3–5 minutes. It creates the layers, the 10 functions with their roles, the queue, logs and alarms, plus permission for `rus9nultj9` to invoke each function. **It does not touch API Gateway routes.** The website keeps using the Python Lambdas until you do §7.

### Doing it by hand in the console instead

For each function in §2:
1. **Lambda → Create function.** Name `gradeant-<name>`, runtime **Node.js 24.x**, architecture **arm64**.
2. **Code → Upload from → .zip file:** `gradeant-<name>.zip`. Runtime settings → handler `index.handler`.
3. **Layers → Add a layer → Custom layers:** `gradeant-deps`, then `gradeant-app`. Create both first under *Lambda → Layers → Create layer* from the two layer zips, with compatible runtimes Node.js 22.x/24.x and architecture arm64.
4. **Configuration → General:** the memory and timeout from §2.
5. **Configuration → Environment variables:** the table in §2.
6. **Configuration → Permissions → role:** add an inline policy with exactly the services and tables in §2's row. `template.yaml` has the exact JSON for each.
7. **For `gradeant-jobs`:** add the trigger **SQS → `gradeant-jobs`**, batch size 1, *Report batch item failures* on.

SAM does all of this for you and keeps it repeatable, so prefer it.

---

## 6. Test the functions before switching any traffic

```bash
# Invoke directly (no API Gateway, no website):
aws lambda invoke --function-name gradeant-health --region eu-north-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"httpMethod":"GET","path":"/health/ready","headers":{},"requestContext":{"stage":"dev"},"body":null}' out.json
cat out.json          # statusCode 200, body {"status":"ok",…,"checks":{"dynamodb":{"ok":true…},"jwtSecret":{"ok":true…}}}

aws lambda invoke --function-name gradeant-auth --region eu-north-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"httpMethod":"POST","path":"/auth/login","headers":{"Content-Type":"application/json"},"requestContext":{"stage":"dev"},"body":"{\"email\":\"you@school.edu\",\"password\":\"…\",\"userType\":\"teacher\"}"}' out.json
cat out.json          # statusCode 200 with a token = it reads the real Users table with the shared JWT key

aws logs tail /aws/lambda/gradeant-auth --follow --region eu-north-1
```

---

## 7. Switch API Gateway to the new functions, one route at a time

Do the low-risk routes first and check the website after each: `/health`, then `/getgrades`, then `/auth/*`, then the rest.

**In the console:**
1. *API Gateway → `rus9nultj9` → Resources* → pick the method (for example `/getgrades` → `POST`).
2. *Integration request → Edit.* Integration type **Lambda function**, **Lambda proxy integration on**, function **`gradeant-grades`** (eu-north-1). Save.
3. *Deploy API* → stage **dev**. The change goes live only on deploy.
4. Check the website, and watch `aws logs tail /aws/lambda/gradeant-grades --follow`.

For `/auth`, change the proxy resource under it (for example `/auth/{proxy+}`), or each `/auth/<name>` method. `/contact` and `/health` probably don't exist on this API yet: *Create resource* → `contact` → *Create method* `POST` → Lambda proxy → `gradeant-contact`.

**The same from the CLI:**

```bash
API=rus9nultj9; R=eu-north-1
aws apigateway get-resources --rest-api-id $API --region $R \
  --query "items[].[id,path,join(',',keys(resourceMethods||{}))]" --output table      # find each resource id

FN_ARN=$(aws lambda get-function --function-name gradeant-grades --region $R --query Configuration.FunctionArn --output text)
aws apigateway put-integration --rest-api-id $API --region $R \
  --resource-id <id of /getgrades> --http-method POST \
  --type AWS_PROXY --integration-http-method POST \
  --uri "arn:aws:apigateway:$R:lambda:path/2015-03-31/functions/$FN_ARN/invocations"
aws apigateway create-deployment --rest-api-id $API --region $R --stage-name dev \
  --description "getgrades -> gradeant-grades"
```

**Notes:**
- **CORS.** If a resource has an `OPTIONS` method (console "Enable CORS"), you can leave it: its mock reply still answers preflight. Or point it at the same function, which answers preflight from the allowlist in `WebOrigins`.
- **Website.** No change needed: `config/app.yaml` keeps the same URL (`…/dev`).
- **Rollback.** Point the method's integration back at the Python Lambda and *Deploy API*. Both use the same tables and JWT key, so nobody is logged out and no data is lost.
- **Cleanup.** After 2–4 weeks with no issues, delete the Python Lambdas. Keep `lambda_python_files/` in git history.

---

## 8. Changing code later (and keeping the graph current)

```bash
# from the repo root: refresh the dependency graph (no AI calls)
graphify update .        # writes graphify-out/; the copy this setup reads is extra/leftovers/graphify-out
cd backend
npm test && npm run check:lambdas && npm run package:lambdas -- --verify
cd lambdas && sam deploy
```

If `check:lambdas` fails, your change made a function reach a new AWS service. Add it to that function's `aws` list in `functions.js` and its role in `template.yaml`, then rerun.

---

## 9. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `{"message":"Internal server error"}` from API Gateway on every call | The function couldn't start. `aws logs tail /aws/lambda/gradeant-<name>` shows the config error (for example missing `JWT_SECRET_ARN`). |
| `Cannot find package 'express'` in the logs | The `gradeant-deps` layer is missing, or the function is x86 while the layer is arm64. Both layers must be attached and the architecture must be arm64. |
| `Cannot find module '/opt/nodejs/gradeant/lambdas/adapter.js'` | The `gradeant-app` layer is missing from the function. |
| 403 / "not authorized to perform lambda:InvokeFunction" from API Gateway | `RestApiId` was empty at deploy: set it and `sam deploy`, or add the invoke permission in the console. |
| 404 `Not found` from a function | That route is wired to the wrong function: each one only serves its own paths (§2). |
| 504 after ~29 s on `/workflow` | A long AI call in the request. Drafting already runs in `gradeant-jobs`; if the **assistant** chat is slow, ask AWS for a higher REST API integration timeout (Service Quotas) or keep prompts short. |
| AI rubric never appears | Check `aws logs tail /aws/lambda/gradeant-jobs` and the `gradeant-jobs-dlq` queue (alarm `gradeant-jobs-failed`), usually the Gemini key or a retired model. |
| `AccessDeniedException` in a log | That function's role lacks the action. `npm run check:lambdas` tells you which service; add it in `template.yaml`. |
