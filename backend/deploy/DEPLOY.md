# Deploying the GradeAnt API to AWS

How the Node API in `backend/` goes live on AWS, replacing the six Python Lambdas without downtime.
Nothing in this folder deploys on its own. Every step here is one you run.

| File | What it is |
|---|---|
| `template.yaml` | Everything the API needs in AWS, as one CloudFormation (SAM) stack |
| `samconfig.example.toml` | Your account's settings for `sam deploy` (copy to `samconfig.toml`, which is gitignored) |
| `preflight.sh` | Read-only checks: tables, indexes, secret, SES, grading Lambda, adapter layer |
| `../scripts/package.mjs` | Builds `dist/function.zip` and checks it starts (`npm run package -- --verify`) |
| `../../.github/workflows/backend-deploy.yml` | The same deploy from GitHub Actions, run by hand |

---

## 1. Where it runs, and why

```
                        eu-north-1 (Stockholm): same Region as today's tables
 Browser ──HTTPS──► Lambda Function URL ──► Lambda "gradeant-api" (Node 24, Arm64)
 (app.gradeant.io,                            │  Lambda Web Adapter layer
  grade-ant.com)                              │  └─ run.sh → node index.js (Express on 127.0.0.1:8080)
                                              │
            ┌───────────────┬─────────────────┼────────────────┬──────────────┬───────────────┐
            ▼               ▼                 ▼                ▼              ▼               ▼
        DynamoDB      Secrets Manager     S3 bucket          SES            SNS        grading-pipeline
     (existing 14     JWT + session     profile photos   invites, mail   "grades     Lambda (existing,
      tables, as is)   keys              (new, private)                   published"   async invoke)
                                              Gemini (gemini-3.5-flash-preview): GradeFlow assistant, rubric drafting
```

**Recommended: a regular Lambda function with the AWS Lambda Web Adapter, behind a Lambda Function URL, in eu-north-1.**

| Option | Verdict | Why |
|---|---|---|
| **Lambda + Web Adapter + Function URL, eu-north-1** | **Use this** | Runs next to the tables, so there is no cross-Region delay or transfer cost. No code changes: the adapter forwards each request to the Express server. Requests can run up to 90 s, which AI rubric drafting needs. Pay per request; idles at about $0. |
| Lambda Web Functions | Later, maybe | Built for exactly this app, but at launch it is only in us-east-1, us-east-2, us-west-2, ap-northeast-1 and eu-west-1, not eu-north-1. Every database call would cross Regions. The same zip already runs there if the tables ever move (see §11). |
| API Gateway (HTTP API) + Lambda | No | Integrations stop at 30 s, which cuts off AI rubric drafting. It adds cost and gives nothing this app uses: the app already does its own auth and CORS. |
| ECS Fargate / EC2 | No | Pays for servers 24/7 and needs a load balancer, for a workload that is mostly idle. |

**Why the Web Adapter needs no code change.**
- Lambda starts `run.sh`, which runs `node index.js`.
- The adapter (`/opt/bootstrap` in the layer) waits for `GET /health` to answer, then turns each Function URL request into a normal HTTP request to `127.0.0.1:8080`.
- The app sees ordinary Express traffic, exactly as on your PC.

**What stays where it is:**
- The DynamoDB tables.
- The Python Lambdas and their API Gateway (`rus9nultj9`). They are the rollback path for a few weeks.
- The legacy `/uploadpdf` Lambda. Uploads keep using it until §10.
- The grading pipeline.

---

## 2. What the stack creates

Stack name `gradeant-api`, Region `eu-north-1`:

| Resource | Settings |
|---|---|
| **Lambda `gradeant-api`** | Node.js 24, Arm64 (cheaper and faster per $ than x86), 1024 MB, 90 s timeout, reserved concurrency 50 (a hard cap on simultaneous requests, which protects cost and DynamoDB), Lambda Web Adapter layer v30, handler `run.sh` |
| **Function URL** | Public (`AuthType NONE`): the app checks login tokens itself, like the Python Lambdas. Buffered responses. CORS is answered by the app's allowlist. It carries both permissions a public URL needs since Oct 2025 (`InvokeFunctionUrl` + `InvokeFunction`, the latter only through the URL). |
| **IAM role** (auto, one per stack) | Least privilege. DynamoDB `Get/Put/Update/Delete/Query/Scan` on `Users`, `assignment_grading` and `GradeAnt_*` (with their indexes) only. `GetSecretValue` on the two secrets only. S3 on `profile-photos/*` only. `GetSecretValue` on the Gemini key secret (if `GeminiSecretArn` is set). SES send only from your sender address, SNS publish to your topic, invoke of your grading Lambda, and uploads only if those settings are filled in. |
| **Secret `gradeant/session`** | A new random key for admin first-login sessions, created only if you don't pass one. |
| **S3 `gradeant-profile-photos-<account>-<region>`** | Created only if you don't pass a bucket. Private (all public access blocked), SSE-S3 encrypted, TLS-only, versioned with old versions expiring after 30 days, CORS `PUT/GET` from your web origins. Kept if the stack is deleted. |
| **Log group** `/aws/lambda/gradeant-api` | 30-day retention. |
| **Alarms** | `gradeant-api-5xx` (≥5 server errors in 5 min) and `gradeant-api-throttles` (cap reached). They notify `AlarmTopicArn` if you set one. |

Settings go into the function as environment variables and override `.env.production` inside the zip. **No secret value is ever in an environment variable or file.** Only the secrets' ARNs are, and the app reads the values from Secrets Manager.

---

## 3. One-time setup in AWS

Use an admin-capable profile in the account that owns the tables. All commands assume Region `eu-north-1`.

### 3.1 Tools on your PC

| Tool | Version | Check |
|---|---|---|
| AWS CLI v2 | any recent | `aws --version` (already installed on this PC) |
| AWS SAM CLI | 1.x | `sam --version` (install: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) |
| Node.js | 22 or 24 | `node --version` |

```bash
aws configure --profile gradeant        # access key of an admin user, Region eu-north-1
export AWS_PROFILE=gradeant             # PowerShell: $env:AWS_PROFILE = "gradeant"
aws sts get-caller-identity             # shows the account you are about to change
```

### 3.2 The JWT secret (required, and the one that matters most)

Logins are signed JWTs. **The Node API must use the same key as the Python Lambdas.** Then a user logged in on one keeps working on the other, and rollback is invisible.

1. Find the current key: Lambda console → the auth Lambda (`authetication.py`) → Configuration → Environment variables → `JWT_SECRET`.
   - **If it is set,** store that exact value.
   - **If it is not set,** the Python code is using the default written in `authetication.py`, which is public in the repo. Generate a new key, store it, **and** set it as `JWT_SECRET` on every Python Lambda that checks tokens (auth, get_grades, correct_questions_answer, update_db, workflow, admin). Everyone is logged out once.
2. Store it. Write the value to a file first, so it never lands in your shell history:

```bash
# put the key (one line, nothing else) in jwt.txt, then:
aws secretsmanager create-secret --name gradeant/jwt \
  --description "GradeAnt JWT signing key (shared by the Python and Node APIs)" \
  --secret-string file://jwt.txt --region eu-north-1
rm jwt.txt
# Note the "ARN" in the output → JwtSecretArn
```

To generate a new key: `node -e "console.log(require('crypto').randomBytes(48).toString('base64url'))"`.

### 3.3 Gemini API key (GradeFlow assistant, AI rubric drafting)

1. Create a **paid-tier** Gemini API key in Google AI Studio (the free tier may use content to improve Google's products; this is student data).
2. Store it in Secrets Manager (eu-north-1) as a plain string, and pass the secret ARN as `GeminiSecretArn`. Or pass the key itself as `GeminiApiKey` (NoEcho).
3. `GeminiModel` defaults to `gemini-3.5-flash-preview`, a preview model: Google can change or retire it, so check before each release.

Without model access the API still works. Only the assistant and AI drafting fail, and rubric drafting falls back to the question-based rubric.

### 3.4 Email (SES): invites, reminders, contact form

1. SES console (eu-north-1) → *Identities* → *Create identity* → **Domain** `gradeant.io`. Add the DKIM CNAME records it shows at your DNS provider, then wait for *Verified*.
2. New SES accounts are in the **sandbox**: mail only reaches verified addresses, at most 200 a day. *Account dashboard → Request production access* (usually approved within a day).
3. Choose the sender, for example `no-reply@gradeant.io` → `SesSenderEmail`. Contact-form messages go to `ContactInbox` (default `contact@gradeant.com`).

Leave `SesSenderEmail` empty to deploy without email. Invites then have to be shared by hand.

### 3.5 The grading pipeline Lambda

"Start grading" hands work to the existing pipeline Lambda, asynchronously, exactly like `workflow.py`.

```bash
aws lambda list-functions --region eu-north-1 --query "Functions[].FunctionName" --output table
```

Find the one `workflow.py` invokes: its name is in that Lambda's configuration or in `workflow.py`. Set it as `GradingFunctionName`.

### 3.6 Optional settings

| Setting | When | How |
|---|---|---|
| `SnsTopicArn` | Teachers/admins want a "grades published" notification | `aws sns create-topic --name gradeant-grades-published`, then subscribe emails to it |
| `AlarmTopicArn` | You want the 5xx/throttle alarms to email someone | `aws sns create-topic --name gradeant-alarms` + `aws sns subscribe --protocol email --notification-endpoint you@…` |
| `SessionSecretArn` | Keep admin first-login sessions valid across the switch | Store the admin Lambda's `SESSION_TOKEN_SECRET` value (Lambda console → `lambda_admin` → Environment variables) like §3.2. If empty, a new one is created and admins sign in again once. |
| `ProfilePhotoBucketName` | You already have a photo bucket | Pass its name. It needs CORS `PUT/GET` from your web origins. |

### 3.7 Pre-flight check (read-only)

```bash
cd backend
JWT_SECRET_ARN=arn:aws:secretsmanager:eu-north-1:ACCOUNT_ID:secret:gradeant/jwt-XXXXXX \
SES_SENDER=no-reply@gradeant.io GRADING_FUNCTION=your-grading-fn \
bash deploy/preflight.sh
```

It confirms all 14 tables and their indexes exist, plus the secret, SES verification and sandbox status, the grading Lambda and the adapter layer. Fix every `[FAIL]` before deploying. The one expected `[warn]` is the optional `assignment_id-index`: without it, class lists use a scan, as they do today.

---

## 4. Settings reference

Template parameter → environment variable the app reads → meaning.

| Parameter | Env var | Default | What it controls |
|---|---|---|---|
| `JwtSecretArn` | `JWT_SECRET_ARN` | *(required)* | Login token key (§3.2) |
| `SessionSecretArn` | `SESSION_TOKEN_SECRET_ARN` | new secret | Admin first-login sessions |
| `WebOrigins` | `CORS_ALLOWED_ORIGINS` | app.gradeant.io, grade-ant.com, www.grade-ant.com | Which sites may call the API and upload photos |
| `DataRegion` | `AWS_DATA_REGION` | eu-north-1 | Region of the tables, SNS and grading Lambda |
| `ProfilePhotoBucketName` | `PROFILE_PHOTO_BUCKET` | new bucket | Profile photos |
| `UploadBucketName` | `UPLOAD_BUCKET` | *(off)* | `/uploadpdf` in Node (§10) |
| `UploadKeyTemplate` | `UPLOAD_KEY_TEMPLATE` | `uploads/{uploadType}/{assignmentId}/{fileName}` | S3 key layout for uploads |
| `SesSenderEmail` | `SES_SENDER_EMAIL` | *(off)* | From address for all mail |
| `ContactInbox` | `CONTACT_INBOX` | contact@gradeant.com | Where contact/pilot forms go |
| `SnsTopicArn` | `SNS_TOPIC_ARN` | *(off)* | "Grades published" notification |
| `GradingFunctionName` | `GRADING_FUNCTION_NAME` | *(off)* | Grading pipeline hand-off |
| `GeminiModel` | `GEMINI_MODEL` | gemini-3.5-flash-preview | The one model for all AI |
| `GeminiSecretArn` / `GeminiApiKey` | `GEMINI_API_KEY_ARN` / `GEMINI_API_KEY` | empty | The Gemini key (one of the two) |
| `MemorySize` | – | 1024 MB | Memory and CPU (login hashing is CPU-bound) |
| `TimeoutSeconds` | – | 90 | Longest request (AI drafting) |
| `ReservedConcurrency` | – | 50 | Max simultaneous requests (0 = no cap) |
| `LambdaAdapterLayerVersion` | – | 30 | Web Adapter layer version |
| `LogRetentionDays` | – | 30 | How long logs are kept |
| `AlarmTopicArn` | – | *(none)* | Who hears about alarms |

The adapter and runtime variables (`AWS_LAMBDA_EXEC_WRAPPER`, `AWS_LWA_PORT`, `AWS_LWA_READINESS_CHECK_PATH`, `PORT`, `HOST`, `APP_ENV`) are fixed in the template. Don't change them.

Everything else (table names, token lifetimes, upload size limits, the AI flag threshold) comes from `backend/.env.production` inside the zip. It is documented in `backend/.env.example`.

---

## 5. Deploy from your PC

```bash
cd backend
npm ci
npm test                                   # 42 tests
npm run package -- --verify                # dist/function.zip, checked and started both ways

cd deploy
cp samconfig.example.toml samconfig.toml   # then fill in the ARNs/names from §3
sam validate --lint
sam deploy                                 # shows the change set, asks "Deploy this changeset? [y/N]"
```

- **The first deploy takes about 3–5 minutes.** At the end it prints the outputs. `ApiUrl` is the new API address, for example `https://abc123xyz.lambda-url.eu-north-1.on.aws/`.
- **Building on Windows is fine.** `npm run package` writes the zip itself, with Unix permissions (`run.sh` 755) and LF line endings. `--verify` checks both, and Python's zip reader reads it the same way.
- **To update later,** run the same three commands: `npm test`, `npm run package -- --verify`, `sam deploy`. Each deploy replaces the code, and in-flight requests finish on the old version.

---

## 6. Deploy from GitHub Actions (recommended once it works)

The workflow `.github/workflows/backend-deploy.yml` runs **only when started by hand** (Actions → *Deploy backend API* → *Run workflow*). It:
1. Tests and packages the API on Linux.
2. Signs in to AWS with OIDC, so there are no stored access keys.
3. Runs `sam deploy`.
4. Smoke-tests `/health` and `/health/ready`.

Tick *dry run* to only test and package.

One-time setup:

1. **Let GitHub sign in to AWS (OIDC).** IAM → *Identity providers* → *Add provider* → OpenID Connect. URL `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.
2. **Create a role `gradeant-github-deploy`** with this trust policy (replace `ACCOUNT_ID` and `OWNER/REPO`):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
        "token.actions.githubusercontent.com:sub": "repo:OWNER/REPO:environment:production"
      }
    }
  }]
}
```

3. **Give the role what `sam deploy` needs.** For a first version, attach `AWSCloudFormationFullAccess` plus this inline policy, scoped to this stack's resources:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["lambda:*"], "Resource": [
        "arn:aws:lambda:eu-north-1:ACCOUNT_ID:function:gradeant-api*",
        "arn:aws:lambda:eu-north-1:753240598075:layer:LambdaAdapterLayerArm64:*"] },
    { "Effect": "Allow", "Action": ["iam:CreateRole","iam:DeleteRole","iam:GetRole","iam:TagRole","iam:UntagRole","iam:PassRole",
        "iam:AttachRolePolicy","iam:DetachRolePolicy","iam:PutRolePolicy","iam:DeleteRolePolicy","iam:GetRolePolicy"],
      "Resource": "arn:aws:iam::ACCOUNT_ID:role/gradeant-api-*" },
    { "Effect": "Allow", "Action": ["s3:*"], "Resource": [
        "arn:aws:s3:::aws-sam-cli-managed-default-*", "arn:aws:s3:::aws-sam-cli-managed-default-*/*",
        "arn:aws:s3:::gradeant-profile-photos-*"] },
    { "Effect": "Allow", "Action": ["secretsmanager:CreateSecret","secretsmanager:DeleteSecret","secretsmanager:DescribeSecret","secretsmanager:TagResource","secretsmanager:GetRandomPassword"],
      "Resource": "*" },
    { "Effect": "Allow", "Action": ["logs:CreateLogGroup","logs:DeleteLogGroup","logs:PutRetentionPolicy","logs:DescribeLogGroups","logs:TagResource"], "Resource": "*" },
    { "Effect": "Allow", "Action": ["cloudwatch:PutMetricAlarm","cloudwatch:DeleteAlarms","cloudwatch:DescribeAlarms"], "Resource": "*" }
  ]
}
```

4. **In GitHub** → *Settings* → *Environments*, create **`production`** and add yourself as a *required reviewer*, so every deploy waits for your click.
5. **Under** *Settings → Secrets and variables → Actions*, add:
   - **Secret `AWS_DEPLOY_ROLE_ARN`:** the role from step 2.
   - **Variable `JWT_SECRET_ARN`:** required.
   - **Optional variables:** `AWS_REGION`, `WEB_ORIGINS`, `SESSION_SECRET_ARN`, `PROFILE_PHOTO_BUCKET`, `UPLOAD_BUCKET`, `SES_SENDER_EMAIL`, `SNS_TOPIC_ARN`, `GRADING_FUNCTION_NAME`, `GEMINI_SECRET_ARN`, `GEMINI_MODEL`, `ALARM_TOPIC_ARN`.

Unrelated: the existing `aws.yml` (the web app's ECS deploy) still uses long-lived access keys. Moving it to the same OIDC approach is worth doing.

---

## 7. Check the new API before anyone uses it

```bash
API=https://abc123xyz.lambda-url.eu-north-1.on.aws     # ApiUrl, no trailing slash

curl -s $API/health          # {"status":"ok",...,"mode":"production"}
curl -s $API/health/ready    # {"status":"ok",…,"checks":{"dynamodb":{"ok":true,…},"jwtSecret":{"ok":true,…}}}  (503 + "degraded" if not)

# Real login against the production Users table (use your own account):
curl -s -X POST $API/auth/login -H "Content-Type: application/json" \
  -d '{"email":"you@school.edu","password":"…","userType":"teacher"}'

# CORS: the browser's preflight from the web app's origin must be allowed…
curl -si -X OPTIONS $API/getgrades -H "Origin: https://app.gradeant.io" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control-allow-origin
# …and refused for any other site (no Access-Control-Allow-Origin header):
curl -si -X OPTIONS $API/getgrades -H "Origin: https://evil.example" \
  -H "Access-Control-Request-Method: POST" | grep -ci access-control-allow-origin

aws logs tail /aws/lambda/gradeant-api --follow --region eu-north-1   # live logs
```

The **real-browser test**:
1. Run the web app locally against the new API: in `web/.env.development.local`, set `VITE_API_BASE_URL=<ApiUrl>`.
2. Add `http://localhost:5173` to `WebOrigins` temporarily.
3. Log in as a student, a teacher and an admin, and click through.

---

## 8. Switch the website over

1. In `config/app.yaml`, under `urls.production`, set `api: <ApiUrl without trailing slash>`. Leave `upload:` pointing at the old API until §10.
2. Rebuild and redeploy the web app: push to `main` runs `.github/workflows/aws.yml`, which rebuilds the image and deploys it to ECS.
3. Watch for an hour: `aws logs tail …`, the `gradeant-api-5xx` alarm, and a quick login yourself.

**Rollback:** set `urls.production.api` back to `https://rus9nultj9.execute-api.eu-north-1.amazonaws.com/dev` and redeploy the web app.
- Both APIs use the same tables and the same JWT key, so nobody is logged out and no data is lost either way.
- Keep the Python Lambdas for 2–4 weeks after the switch, then remove them (`lambda_python_files/`, `admin/lambda_admin.py`, and their API Gateway).

---

## 9. Running it

| Task | How |
|---|---|
| Live logs | `aws logs tail /aws/lambda/gradeant-api --follow` (one JSON line per request) |
| Errors | CloudWatch → Alarms → `gradeant-api-5xx`; Lambda console → Monitor |
| Deploy a change | `npm test && npm run package -- --verify && sam deploy` (or the GitHub workflow) |
| Change a setting | Edit `samconfig.toml` (or the GitHub variable) → `sam deploy` |
| Rotate the JWT key | `aws secretsmanager put-secret-value --secret-id gradeant/jwt --secret-string file://new.txt`; update the Python Lambdas' `JWT_SECRET` too while they exist. Running functions pick it up on their next cold start: force that with a deploy. Everyone is logged out. |
| Busy day | Raise `ReservedConcurrency`; Lambda scales by itself up to that cap |
| Cost | Lambda: per request and duration, about $0 when idle. DynamoDB, S3, SES: as today. Gemini: per token, paid tier. Set an AWS Budget with an email alert. |
| Cold starts | The first request after idle takes about 1–3 s (Node start plus adapter readiness). If that matters, add provisioned concurrency of 1 later. |

---

## 10. Later steps

1. **Move uploads to Node.**
   - Find the S3 key layout the grading pipeline reads, from the pipeline's code or the legacy upload Lambda's config.
   - Set `UploadKeyTemplate` to match it and `UploadBucketName` to the bucket, then `sam deploy`.
   - Add a CORS `PUT` rule for your web origins on that bucket.
   - Set `upload: ""` in `app.yaml` and redeploy the web app.
2. **Custom domain** (`api.gradeant.io`): put CloudFront in front of the Function URL with an ACM certificate, and optionally AWS WAF with rate limiting. Then point `urls.production.api` at it.
3. **Background jobs on SQS** (if AI drafting ever needs more than 90 s): create a queue, set `JOBS_DRIVER=sqs` and `JOBS_QUEUE_URL`, and deploy the same zip as a second function with handler `worker.handler` (see `backend/README.md`).
4. **Point-in-time recovery** on the DynamoDB tables if it isn't on: `aws dynamodb update-continuous-backups --table-name Users --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true` (repeat per table).

---

## 11. If the tables ever move: Lambda Web Functions

The same `dist/function.zip` runs on Lambda Web Functions. `index.js` listens on `AWS_LAMBDA_HTTP_ENDPOINT`, and `--verify` checks this too. The commands are in `backend/README.md` ("Deploying to AWS Lambda Web Functions"). It only makes sense in a Region that has both Web Functions and your tables (today: eu-west-1, us-east-1, us-east-2, us-west-2, ap-northeast-1).

---

## 12. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `403 Forbidden` from the URL | The function URL's public permissions are missing. The stack adds both; check that `aws lambda get-policy --function-name gradeant-api` shows `lambda:InvokeFunctionUrl` **and** `lambda:InvokeFunction`. |
| `502` or a timeout on every request | The app didn't start. Run `aws logs tail /aws/lambda/gradeant-api`; usually it's a configuration error printed at startup (for example a missing `JWT_SECRET_ARN`). |
| `/health/ready` → `dynamodb: AccessDeniedException` | Wrong `DataRegion`, or a table name in `.env.production` differs from the real one. |
| `/health/ready` → `jwtSecret: ResourceNotFoundException` | Wrong `JwtSecretArn`, or the secret is in another Region. |
| Browser: "blocked by CORS policy" | Your site's exact origin (scheme + host, no slash) is missing from `WebOrigins`. |
| Login works on one API, "invalid token" on the other | The JWT keys differ (§3.2). |
| Invites/emails not arriving | SES sandbox or unverified sender (§3.4). Check `aws ses get-send-quota`. |
| Assistant says it can't reach the AI | Gemini key missing or wrong, or the model was retired (§3.3). |
| `run.sh: permission denied` / `not found` | The zip was built some other way. Always use `npm run package` (its `--verify` checks mode 755 and LF). |
