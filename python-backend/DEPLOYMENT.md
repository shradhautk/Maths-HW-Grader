# GradeAnt API on AWS Lambda — Design & Deployment Guide

| | |
|---|---|
| **Covers** | `python-backend/`: 9 Python 3.13 API Lambda functions, 2 grading-pipeline workers (`gradeant-grader`, `gradeant-jsonifier`), 3 layers |
| **Region** | `eu-north-1` (Stockholm), the same Region as the DynamoDB tables |
| **Front door** | The existing API Gateway REST API `rus9nultj9`, stage `dev` (unchanged URL) |
| **Status** | Built and tested; not yet deployed |
| **Last updated** | 2026-11-17 (grading pipeline port: B15) |
| **Related** | `README.md` (overview, tests), `deploy/template.yaml` (AWS SAM), `tools/backfill_classes.py`, `PIPELINE_PORT_PLAN.md` (pipeline port design) |

Part A records **what was built and why**. Part B is the **step-by-step deployment**, with every setting per function. Part C covers **running and changing it** afterwards.

---

# Part A — Design

## A1. Architecture

```
Browser (app.gradeant.io)
   │  HTTPS + "Authorization: Bearer <login token>"
   ▼
API Gateway REST API  rus9nultj9  /dev   (Lambda proxy integration, one route per function)
   │
   ├─ POST /auth/*          → gradeant-auth          login, signup, logout, passwords, profile photo
   ├─ POST /getgrades       → gradeant-grades        a student's result, class roster, total-mark edits
   ├─ POST /correctanswer   → gradeant-corrections   answer & question-text corrections
   ├─ POST /updatedb        → gradeant-marks         teacher mark edits
   ├─ POST /workflow        → gradeant-workflow      assignments, rubric, grading, publishing, assistant
   │                              └─ invokes itself (async) to draft rubrics with Claude
   │                              └─ invokes GradingFunctionName (async) on "Start grading"
   ├─ POST /admin           → gradeant-admin         admin console, invites
   ├─ POST /contact         → gradeant-contact       public contact / pilot form → email
   ├─ POST /uploadpdf       → gradeant-upload        presigned S3 upload links (off until a bucket is set)
   └─ GET  /health[/ready]  → gradeant-health        health checks
            │
            ▼
   DynamoDB (existing tables) · Secrets Manager (login keys) · S3 (photos, uploads)
   SES (email) · SNS (notifications) · Gemini (Google, outside AWS)

Grading pipeline (not behind API Gateway; B15):
   S3 <root>/students/assignments/<aid>__<qp>/original_pdf/<student>.pdf
      └─ S3 event → SQS → gradeant-jsonifier   PDF → transcript → answers by question → JSON (Gemini)
   S3 …/Parsed_pdfs/JSON/<student>.json
      └─ S3 event → SQS → gradeant-grader      grades each answer (Gemini) → assignment_grading rows + summary
   gradeant-workflow "Start grading" ──(async)──► gradeant-grader   regrades the student's corrected answers
```

## A2. How a function is packaged

Every function is **its own small zip plus two shared layers**:

| Part | Contents | On Lambda at |
|---|---|---|
| Function zip `gradeant-<name>.zip` | One file: `<name>.py` | `/var/task/<name>.py` |
| Layer **gradeant-python-common** | `gradeant_common/` (shared code) plus `config/app.yaml` | `/opt/python/gradeant_common/` |
| Layer **gradeant-python-deps** | PyJWT, PyYAML, boto3/botocore, built for Linux arm64 / Python 3.13 (gradeant-workflow also adds gradeant-pipeline-deps for Gemini) | `/opt/python/` |

Python on Lambda searches `/opt/python` automatically, so `from gradeant_common import ...` in each function just works. Shared code is uploaded **once**, not copied into every function.

The two pipeline workers use **`gradeant-pipeline-deps` instead of `gradeant-python-deps`** (never both): boto3, PyYAML, PyMuPDF and google-genai, pinned in `pipeline-requirements.txt`, ~112 MB unzipped. Stacking both package layers would put two separately installed copies of shared libraries (httpx, pydantic) into one `/opt/python`. That is why the template sets layers per function, not in `Globals`.

What `gradeant_common` provides:

| Module | Job |
|---|---|
| `config` | Settings from environment variables, plus `config/app.yaml` (URLs, API paths, roles, permissions) |
| `secrets` | Login keys from Secrets Manager, read once per container; no built-in fallback key |
| `http` | Request wrapper: CORS allowlist, OPTIONS, JSON errors, never leaks internal error text |
| `dispatch` | Operation APIs: body → login token → operation → `app.yaml` permission |
| `auth`, `permissions` | Token checks (account exists and isn't suspended; role read fresh) and role rules |
| `dynamo`, `grading`, `ownership` | Table access (all pages), and teacher-to-teacher isolation |
| `mail` | Email through SES |

## A3. Design decisions

| Decision | Why |
|---|---|
| **One Lambda per API path** | Each function gets only the permissions its code needs, and a problem stays inside one function. It also matches how the original Lambdas were split. |
| **Shared code in a layer** | One copy of auth, permissions, CORS and database code. A fix to the shared code is one layer upload, not nine copies. |
| **Python 3.13, arm64** | 3.13 is the newest supported runtime (the code also runs on 3.10+). Arm64 (Graviton) costs about 20% less per second than x86. |
| **Same API Gateway, same tables, same login key** | No website change, no data migration. Each route can be switched separately and switched back. |
| **New function names (`gradeant-*`)** | The old Lambdas keep running as the rollback until each route is switched. |
| **Behaviour identical to the Node backend** | Proven by the parity check: 211 requests, 0 differences. |
| **Rubric drafting by self-invocation** | Claude can take longer than API Gateway's 29 s limit. Workflow invokes itself asynchronously; its 300 s timeout covers that background run. |
| **Grader contract unchanged** | A test proves the grading-pipeline Lambda receives byte-for-byte the same request as before, and another proves `gradeant-grader` accepts it. (The original grader rejected it with a 400.) |
| **Secrets only as ARNs** | Environment variables hold Secrets Manager ARNs, never key values. |
| **Pipeline ported, prompts verbatim** | `gradeant-jsonifier` / `gradeant-grader` replace `excute_jsonifier.py` + `jsonifier.py` / `grader_lambda.py`. Tests pin every prompt to the original text, so grading behaviour doesn't drift. |
| **Gemini stays, behind an interface** | Same model family as today, via the supported `google-genai` SDK. The key lives in Secrets Manager. `PIPELINE_LLM` can later select another provider without touching grading code. |
| **Pipeline failures are retried, never graded** | A model error fails the SQS message (retried, then dead-lettered) instead of becoming 0 marks. Rows have deterministic ids, so a retry overwrites instead of duplicating. |

## A4. Security rules the code enforces

- **Who can sign up** is set in `config/app.yaml` `roles.<role>.signup`. Students and teachers can register themselves or join from an admin invite (`[self, invite]`). Admins join by invite only. To make teachers invite-only again, set `roles.teacher.signup: [invite]`; a self-registered teacher can create an assignment for any class name and reach that class's students.
- **Teachers see only their own assignments** and grading data; data with no owner is refused.
- **Students see only their own classes** (`Users.classes`). A student with no classes sees nothing. An invited student gets the invite's class at first login.
- **Students change only their own answers**, and not after grading has started.
- **Reminders reach only students on that assignment's roster.**
- **Uploads:** students upload only for published assignments in their own classes. Question papers and solutions only by the teacher who owns the assignment (or an admin). In the pipeline layout the server picks every file name. The file size is signed into each upload link: PDFs up to `UPLOAD_MAX_BYTES`, photos up to 5 MB.
- **Pipeline workers** accept events only from `PIPELINE_BUCKET`, and only file names that are plain ids. They never log answer text.
- **CORS allowlist on every function**, and no internal error text in replies.
- **Each function's IAM role** allows only its own tables and services. `check_permissions.py` verifies this against the code graph.

---

# Part B — Step-by-step deployment

Do the steps in order. Steps B1–B10 don't affect the live website: it keeps using the old Lambdas until step B11.

## B1. What you need

- **Console access:** an AWS account with admin console access, Region **eu-north-1**.
- **On your PC:** Python 3.10+ (the build), and the AWS CLI (only for the test and backfill commands).
- **The project:** this repository, with the branch that contains `python-backend/`.
- **Four values**, collected in B2–B5:
  - the JWT secret ARN;
  - the session secret ARN;
  - the email sender address;
  - the grading Lambda's name.

## B2. Login key secret (JWT) — required

The new functions must sign and check login tokens with **the same key as the current Lambdas**. That way nobody is logged out, and you can switch back at any time.

1. **Find the current key.** In the Lambda console, open the current auth Lambda (the one running `authetication.py`). Go to **Configuration**, then **Environment variables**, and copy the value of `JWT_SECRET`.
   - **If there is no `JWT_SECRET`**, the old code is using a default key that is visible in the repository. Make a new key (`python -c "import secrets; print(secrets.token_urlsafe(48))"`), use it below, **and** set it as `JWT_SECRET` on every old Lambda too. Everyone is logged out once.
2. **Store it.** Open **Secrets Manager**, then **Store a new secret**:
   - Secret type: **Other type of secret**, then **Plaintext**. Paste only the key, no quotes.
   - Secret name: `gradeant/jwt`. Leave rotation off.
3. **Note the ARN.** Open the secret and copy its **ARN**: this is **`JWT_SECRET_ARN`**.

## B3. Admin session secret

1. **Find the current value.** In the current admin Lambda (`lambda_admin.py`), copy the `SESSION_TOKEN_SECRET` environment variable.
   - If it isn't set, make a new value as in B2. Admins with an open first-login link will need a new link.
2. **Store it** in Secrets Manager as `gradeant/session`, the same way as B2.
3. **Note the ARN:** this is **`SESSION_TOKEN_SECRET_ARN`**.

## B4. Email (SES) — needed for invites, reminders and the contact form

1. Open **Amazon SES** (eu-north-1), then **Identities**, then **Create identity**. Choose **Domain**, enter `gradeant.io`, and add the DKIM records it shows at your DNS provider. Wait for **Verified**.
2. **Account dashboard → Request production access.** Until approved, SES only sends to verified addresses, at most 200 a day.
3. Choose the sender address, for example `no-reply@gradeant.io`: this is **`SES_SENDER_EMAIL`**.

You can deploy without email; leave `SES_SENDER_EMAIL` empty and no email is sent.

## B5. The Gemini key and the grading Lambda

1. **Gemini**: a paid-tier API key (Google AI Studio), stored in Secrets Manager as a plain string (`GeminiSecretArn`) or passed as `GeminiApiKey` (NoEcho). The one model for all AI is `GeminiModel` (`gemini-3.5-flash-preview`, a preview: Google may change or retire it).
2. **Grading Lambda:** note the name of your existing grading-pipeline function. It's the one the current workflow Lambda invokes; check its `GRADING_FUNCTION_NAME` variable. This is **`GRADING_FUNCTION_NAME`** until B15 switches it to `gradeant-grader`.

## B6. Build the zips

From the project folder:

```bash
python python-backend/build.py --verify
```

This creates `python-backend/dist/` with 14 zips: 3 layers, the 9 API functions and the 2 pipeline workers. Upload **only these zips**: never `.py` files, and never a folder you zipped yourself. B7–B14 cover the API functions; the workers and `gradeant-pipeline-deps` are B15.

## B7. Upload the two layers (once)

Lambda console (eu-north-1): **Layers**, then **Create layer**. Do this twice:

| Field | Layer 1 | Layer 2 |
|---|---|---|
| Name | `gradeant-python-deps` | `gradeant-python-common` |
| Upload a .zip file | `layer-gradeant-python-deps.zip` (~23 MB) | `layer-gradeant-python-common.zip` |
| Compatible architectures | **arm64** | **arm64** |
| Compatible runtimes | **Python 3.13** | **Python 3.13** |

## B8. Create the 9 functions

For **each** card below, in the Lambda console:

1. **Functions → Create function → Author from scratch.** Set the **name**, runtime **Python 3.13**, architecture **arm64**, then **Create function**.
2. **Code** tab: **Upload from → .zip file**, choose the card's **zip**, then **Save**.
3. **Code** tab: **Runtime settings → Edit**, set the card's **handler**, then **Save**.
4. **Code** tab, bottom: **Layers → Add a layer → Custom layers**. Add `gradeant-python-deps`, then add `gradeant-python-common`. Every function needs both.
5. **Configuration → General configuration → Edit:** set the card's **memory** and **timeout**.
6. **Configuration → Environment variables → Edit:** add the card's **variables**.
7. **Configuration → Permissions:** click the role name, then **Add permissions → Create inline policy → JSON**. Paste the card's **policy** with your values filled in (see B9), and name it `gradeant`.

**Variables every function gets** ("Common" in the cards):

| Key | Value |
|---|---|
| `APP_ENV` | `production` |
| `AWS_DATA_REGION` | `eu-north-1` |
| `CORS_ALLOWED_ORIGINS` | `https://app.gradeant.io,https://grade-ant.com,https://www.grade-ant.com` |

---

### Card 1 — gradeant-auth

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-auth` / `gradeant-auth.zip` / `auth.lambda_handler` |
| Memory / timeout | 1024 MB / 29 s (password hashing is CPU-heavy) |
| Variables | Common + `JWT_SECRET_ARN`, `PROFILE_PHOTO_BUCKET` (your photo bucket name) |
| Route | `POST /auth/*` |

Policy: B9 **Policy AUTH**.

### Card 2 — gradeant-grades

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-grades` / `gradeant-grades.zip` / `grades.lambda_handler` |
| Memory / timeout | 512 MB / 29 s |
| Variables | Common + `JWT_SECRET_ARN` |
| Route | `POST /getgrades` |

Policy: B9 **Policy GRADING**.

### Card 3 — gradeant-corrections

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-corrections` / `gradeant-corrections.zip` / `corrections.lambda_handler` |
| Memory / timeout | 512 MB / 29 s |
| Variables | Common + `JWT_SECRET_ARN` |
| Route | `POST /correctanswer` |

Policy: B9 **Policy GRADING**.

### Card 4 — gradeant-marks

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-marks` / `gradeant-marks.zip` / `marks.lambda_handler` |
| Memory / timeout | 512 MB / 29 s |
| Variables | Common + `JWT_SECRET_ARN` |
| Route | `POST /updatedb` |

Policy: B9 **Policy GRADING**.

### Card 5 — gradeant-workflow

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-workflow` / `gradeant-workflow.zip` / `workflow.lambda_handler` |
| Memory / timeout | 1024 MB / **300 s** (background rubric drafting; browser requests are still cut at 29 s by API Gateway) |
| Variables | Common + `JWT_SECRET_ARN`, `GRADING_FUNCTION_NAME`, `SES_SENDER_EMAIL`, `GEMINI_API_KEY_ARN` (or `GEMINI_API_KEY`), `GEMINI_MODEL`, optional `SNS_TOPIC_ARN` |
| Route | `POST /workflow` |

Policy: B9 **Policy WORKFLOW**.

### Card 6 — gradeant-admin

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-admin` / `gradeant-admin.zip` / `admin.lambda_handler` |
| Memory / timeout | 1024 MB / 29 s |
| Variables | Common + `JWT_SECRET_ARN`, `SESSION_TOKEN_SECRET_ARN`, `SES_SENDER_EMAIL` |
| Route | `POST /admin` |

Policy: B9 **Policy ADMIN**.

### Card 7 — gradeant-contact

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-contact` / `gradeant-contact.zip` / `contact.lambda_handler` |
| Memory / timeout | 256 MB / 15 s |
| Variables | Common + `SES_SENDER_EMAIL`, `CONTACT_INBOX` (`contact@gradeant.com`) |
| Route | `POST /contact` (new on this API) |

Policy: B9 **Policy CONTACT**.

### Card 8 — gradeant-upload

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-upload` / `gradeant-upload.zip` / `upload.lambda_handler` |
| Memory / timeout | 512 MB / 15 s |
| Variables | Common + `JWT_SECRET_ARN`. Leave `UPLOAD_BUCKET` **unset** for now (see C4). |
| Route | `POST /uploadpdf`, but **don't switch it yet**: the current upload Lambda keeps serving uploads |

Policy: B9 **Policy UPLOAD**.

### Card 9 — gradeant-health

| Setting | Value |
|---|---|
| Name / zip / handler | `gradeant-health` / `gradeant-health.zip` / `health.lambda_handler` |
| Memory / timeout | 256 MB / 10 s |
| Variables | Common + `JWT_SECRET_ARN` |
| Route | `GET /health`, `GET /health/ready` (new on this API) |

Policy: B9 **Policy HEALTH**.

---

## B9. IAM policies (paste one per function)

Replace these placeholders with your values:
- `ACCOUNT_ID`: your 12-digit account ID.
- `JWT_SECRET_ARN`, `SESSION_TOKEN_SECRET_ARN`: from B2 and B3.
- `SES_SENDER_EMAIL`: from B4.
- `GRADING_FUNCTION_NAME`: from B5.
- `PROFILE_PHOTO_BUCKET`: your photo bucket name.

Leave out any statement for a feature you don't use. These are exactly the permissions `deploy/template.yaml` creates.

**Policy GRADING** (grades, corrections, marks):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"],
      "Resource": [
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/assignment_grading",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/assignment_grading/index/*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Assignments",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Assignments/index/*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Submissions",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Submissions/index/*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Announcements" ] },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": "JWT_SECRET_ARN" }
  ]
}
```

**Policy WORKFLOW:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"],
      "Resource": [
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/assignment_grading",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/assignment_grading/index/*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Assignments",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Assignments/index/*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Submissions",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Submissions/index/*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Announcements" ] },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": "JWT_SECRET_ARN" },
    { "Effect": "Allow", "Action": "lambda:InvokeFunction",
      "Resource": [ "arn:aws:lambda:eu-north-1:ACCOUNT_ID:function:gradeant-workflow",
                    "arn:aws:lambda:eu-north-1:ACCOUNT_ID:function:GRADING_FUNCTION_NAME" ] },
    { "Effect": "Allow", "Action": "ses:SendEmail", "Resource": "*",
      "Condition": { "StringEquals": { "ses:FromAddress": "SES_SENDER_EMAIL" } } },
    { "Effect": "Allow", "Action": "sns:Publish", "Resource": "SNS_TOPIC_ARN" }
  ]
}
```

**Policy AUTH:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"],
      "Resource": [
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_*/index/*" ] },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": "JWT_SECRET_ARN" },
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::PROFILE_PHOTO_BUCKET/profile-photos/*" }
  ]
}
```

**Policy ADMIN:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"],
      "Resource": [
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_*",
        "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_*/index/*" ] },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue",
      "Resource": [ "JWT_SECRET_ARN", "SESSION_TOKEN_SECRET_ARN" ] },
    { "Effect": "Allow", "Action": "ses:SendEmail", "Resource": "*",
      "Condition": { "StringEquals": { "ses:FromAddress": "SES_SENDER_EMAIL" } } }
  ]
}
```

**Policy CONTACT:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "ses:SendEmail", "Resource": "*",
      "Condition": { "StringEquals": { "ses:FromAddress": "SES_SENDER_EMAIL" } } }
  ]
}
```

**Policy UPLOAD:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:Query"],
      "Resource": [ "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users",
                    "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Assignments" ] },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": "JWT_SECRET_ARN" }
  ]
}
```

(When you turn uploads on in C4, add `{ "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": "arn:aws:s3:::UPLOAD_BUCKET/*" }`.)

**Policy HEALTH:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "dynamodb:GetItem", "Resource": "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users" },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": "JWT_SECRET_ARN" }
  ]
}
```

## B10. Test every function before switching traffic

**1. Health first.** Open `gradeant-health`, go to the **Test** tab, create an event with this JSON, then **Test**:

```json
{"httpMethod": "GET", "path": "/health/ready", "headers": {}}
```

Expected: `"statusCode": 200`, with `"dynamodb": {"ok": true …}` and `"jwtSecret": {"ok": true …}`.

**2. A start-up check for each other function.** Use its own path, for example `/getgrades`, `/workflow`, `/auth/login`:

```json
{"httpMethod": "OPTIONS", "path": "/getgrades", "headers": {}}
```

Expected: `"statusCode": 204`.

**3. A real login** on `gradeant-auth`, with your own account:

```json
{"httpMethod": "POST", "path": "/auth/login", "headers": {"Content-Type": "application/json"},
 "body": "{\"email\": \"you@school.edu\", \"password\": \"…\", \"userType\": \"teacher\"}"}
```

Expected: `"statusCode": 200` and a `token`. This proves the function reads the real `Users` table with the shared key.

If a test fails, see C6.

## B11. Give existing students their class (before switching any route)

Students now see only their own classes' work, and accounts created before this change have no `classes`. From the project folder, using AWS CLI credentials for this account:

```bash
python python-backend/tools/backfill_classes.py --region eu-north-1           # dry run: lists what would change
python python-backend/tools/backfill_classes.py --region eu-north-1 --apply   # writes Users.classes
```

Students who signed themselves up have no invite class. The script lists them; place each one in a class by setting `classes` (a list of class names) on their `Users` item in DynamoDB.

## B12. Switch API Gateway, one route at a time

For each route, in this order: **`/health`** (new) → **`/getgrades`** → **`/correctanswer`** → **`/updatedb`** → **`/auth`** → **`/workflow`** → **`/admin`** → **`/contact`** (new).

1. **API Gateway → `rus9nultj9` → Resources** → select the method (for example `/getgrades` → `POST`).
2. **Integration request → Edit:** integration type **Lambda function**, **Lambda proxy integration ON**, Region **eu-north-1**, function `gradeant-<name>`. **Save**, and accept "Grant API Gateway permission to invoke".
3. **Deploy API → stage `dev`.** Nothing changes for users until you deploy.
4. **Check the website** for that feature, and watch the logs: `aws logs tail /aws/lambda/gradeant-<name> --follow --region eu-north-1`.

**Notes:**
- **New routes:** `/health`, `/health/ready` and `/contact` don't exist on this API yet. Use **Create resource** (`health`, then `ready` under it; `contact`), then **Create method** (`GET` or `POST`) → Lambda proxy → the function.
- **`/auth`:** if it uses one proxy resource (`/auth/{proxy+}`), switch that. If it has one resource per action (`/auth/login`, `/auth/signup`, …), switch each of them to `gradeant-auth`.
- **`OPTIONS` methods** (from "Enable CORS") can stay as they are, or point to the same function.
- **The website URL doesn't change**, because `config/app.yaml` still points at `…/dev`.

## B13. Deploy the web app

The web app now sends the profile-photo size. Deploy the updated web build (push to `main` runs `.github/workflows/aws.yml`) **together with** switching `/auth`. Otherwise photo uploads from the old web app are refused.

## B14. Rollback (any time)

Point a route's integration back at its old Lambda, then **Deploy API** to `dev`. Both versions use the same tables and login key, so nobody is logged out and no data is lost. Keep the old Lambdas for 2–4 weeks after the last route is switched, then delete them.

## B15. Grading pipeline workers (cut-over)

`gradeant-jsonifier` and `gradeant-grader` replace the legacy pipeline Lambdas (`excute_jsonifier.py`, `grader_lambda.py`). Deploying them changes nothing until step 5: their SQS mappings are created **disabled**, and `GradingFunctionName` still points at the old grader.

**Template parameters** (`samconfig.toml`):

| Parameter | Value |
|---|---|
| `PipelineBucketName` | The pipeline's bucket (today `assignmentgrader`). Empty = workers get no S3/secret permissions. |
| `PipelineRoot` | `Data_warehouse` (top folder of the layout) |
| `GeminiSecretArn` | The secret from step 1 |
| `GeminiModel` | `gemini-3.5-flash-preview`: the one model for grading, reading answer sheets, rubric drafting and the assistant (change here, no code change) |
| `GradeOnUpload` | `true` = grade as soon as a parsed sheet lands (as today). `false` = store answers ungraded and wait for the teacher's "Start grading". |
| `JsonifierQueueArn`, `GraderQueueArn` | The SQS queues that receive the bucket's S3 events (step 2) |
| `PipelineQueuesEnabled` | `false` until step 5 |
| `UploadLayout` | `template` (keep) until step 8 |

**Runbook:**

1. **Gemini key secret.** Copy the key the legacy Lambdas use (`GEMINI_API_KEY`) into a new Secrets Manager secret `gradeant/gemini`, either as a plain string or as `{"GEMINI_API_KEY": "..."}`. Never paste it into chat or the repo. Note its ARN.
2. **Find the bucket's Region and its event wiring** (read-only):
   ```bash
   aws s3api get-bucket-location --bucket assignmentgrader
   aws s3api get-bucket-notification-configuration --bucket assignmentgrader
   ```
   Note the queue ARNs and their prefix/suffix filters. The jsonifier's queue gets `…/original_pdf/*.pdf`, the grader's gets `…/Parsed_pdfs/JSON/*.json`. If the bucket is not in eu-north-1, add `PIPELINE_BUCKET_REGION` to both workers.
3. **Deploy** with `PipelineQueuesEnabled=false`, `GradeOnUpload=true`, and `GradingFunctionName` still set to the old grader.
4. **Smoke test.** Invoke `gradeant-grader` with a hand-made SQS event that points at an existing parsed JSON of a test assignment (`{"Records":[{"messageId":"t1","body":"{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/students/assignments/<aid>__<qp>/Parsed_pdfs/JSON/<student>.json\"}}}]}"}]}`). Expect `{"batchItemFailures": []}`, and compare the new rows with the legacy ones.
5. **Switch the queues.** First **disable the legacy Lambdas' SQS triggers** (Lambda → the old function → Configuration → Triggers → disable), then redeploy with `PipelineQueuesEnabled=true`. **Never run both on one queue**: they would compete for the messages.
6. **Queue settings.** Set each queue's visibility timeout to at least 5400 s (6 × the 900 s function timeout). Give each queue a dead-letter queue (redrive, `maxReceiveCount` 3) if it has none. Failed messages end up there, not graded as 0.
7. **"Start grading".** Set `GradingFunctionName=gradeant-grader` and redeploy. Failed asynchronous runs go to the `gradeant-grader-failures` queue (stack output `GraderFailureQueueUrl`).
8. **Later, as a separate decision: uploads straight into the pipeline layout.** Set `UploadLayout=pipeline` and `UploadBucketName` = `PipelineBucketName`, add a CORS `PUT` rule for your web origins on that bucket, then switch `/uploadpdf` (B12). `GradeOnUpload=false` belongs with the new teacher/student workflow screens.

**Rollback:** set `PipelineQueuesEnabled=false` and re-enable the legacy SQS triggers. Set `GradingFunctionName` back to the old grader (or empty), and keep `UploadLayout=template`. Rows written by the new grader have the same fields, so every screen keeps reading them.

**Before deploying, still open:** the `strict` / `lenient` persona wording (placeholders marked `TODO(user, plan O5)` in `gradeant_common/pipeline/prompts.py`), and which process produces the question-paper schema JSON, rubric JSON and schema-answer text (the workers only read them).

## B16. Free-tier hybrid: pipeline-only stack on Supabase

For the hybrid deployment (web on Netlify, Node API on Render, Supabase database; `backend/FREE_TIER_HYBRID_PLAN.md`), AWS runs **only** the grading pipeline. Deploy `deploy/pipeline-template.yaml` **instead of** `template.yaml`, because they share function names.

1. **Supabase:** run `npm run sql:apply` in `backend/` with `SUPABASE_DB_URL` set. Then in the dashboard go to **Project Settings → API → Exposed schemas** and add `gradeant`.
2. **Parameters**, in addition to B15's:

   | Parameter | Value |
   |---|---|
   | `DatabaseProvider` | `supabase` (or leave empty and set `deployment: hybrid` in `config/app.yaml`) |
   | `SupabaseUrl` | `https://<project>.supabase.co` |
   | `SupabaseServiceRoleKey` | the service-role key (NoEcho; stored as an encrypted Lambda environment variable) |
   | `GeminiApiKey` | the Gemini key (NoEcho), instead of `GeminiSecretArn` (no Secrets Manager cost) |

3. **"Start grading":** the Node API sends the regrade payload to the **grader's queue** (`grading: sqs`). Give the Render IAM user `sqs:SendMessage` on that queue only. No Lambda invoke permission is needed.
4. **Check:** after an upload, rows appear in `gradeant.assignment_grading`. The grader logs show no `message failed` lines, and the `gradeant-pipeline-errors` alarm stays quiet.

Rollback: set `DatabaseProvider=dynamodb` (or redeploy `template.yaml`). The grader then writes to DynamoDB again. The two stores aren't synchronised: the hybrid starts with an empty Supabase database (no migration, decided 2026-11-18).

## B-alt. One-command alternative (AWS SAM)

`deploy/template.yaml` creates all of B7–B9 (layers, functions, variables, policies, log groups, alarm) in one go:

```bash
cd python-backend
python build.py --verify
cd deploy
cp samconfig.example.toml samconfig.toml   # fill in JwtSecretArn, RestApiId=rus9nultj9, …
sam validate --lint
sam deploy
```

You still do B2–B5 first, then B10–B15 afterwards. The pipeline workers, their layer, failure queue and (disabled) SQS mappings are in the same template.

---

# Part C — Running and changing it

## C1. Changing code later

| What changed | What to do |
|---|---|
| One function, for example `functions/grades.py` | `python python-backend/build.py`, then upload only `gradeant-grades.zip` (**Code → Upload from → .zip**) |
| `gradeant_common/` or `config/app.yaml` | Build. Then **Layers → gradeant-python-common → Create version** and upload the new zip. Then, **in every function:** **Layers → Edit**, and select the new version. |
| `requirements.txt` | The same, for `gradeant-python-deps` |
| `pipeline-requirements.txt` | The same, for `gradeant-pipeline-deps` (the two workers only) |

Layer versions never change once uploaded, and each function stays on the version it was given. That's why a shared-code change needs the second step. `sam deploy` does it automatically.

**Before uploading any change**, run:
- `pytest python-backend/tests`
- the parity check (README §3)
- `python python-backend/check_permissions.py`, after `graphify update .`

## C2. Logs and monitoring

- **Logs:** each function logs to `/aws/lambda/gradeant-<name>`. Set retention to 30 days: **CloudWatch → Log groups → Actions → Edit retention**.
- **Errors:** **Lambda → function → Monitor** shows errors, duration and throttles. Optionally, create a CloudWatch alarm on the `Errors` metric.

## C3. Cost

- **Lambda:** billed per request and per GB-second, and close to $0 at pilot scale (the free tier covers 1 million requests and 400,000 GB-s a month).
- **The real costs:**
  - Gemini for all AI;
  - DynamoDB reads, especially scans as tables grow;
  - API Gateway: REST API is about $3.50 per million calls;
  - CloudWatch logs.

## C4. Turning on uploads in the new function (later)

1. **Pipeline layout (recommended):** set `UPLOAD_LAYOUT=pipeline` and `UPLOAD_BUCKET` = `PIPELINE_BUCKET` (B15 step 8). Keys and file names then match what the workers read. One PDF per request, and the assignment must exist.
2. **Or template layout:** set `UPLOAD_BUCKET` and `UPLOAD_KEY_TEMPLATE` to the layout you need.
3. Either way: add the S3 statement to its policy (B9) and a CORS `PUT` rule for your web origins on that bucket. Then switch the `/uploadpdf` route (B12), and set `urls.production.upload: ""` in `config/app.yaml`.

## C5. Settings reference (optional variables)

Defaults match `backend/.env.example`. Change them only if your setup differs.

| Variable | Default | Meaning |
|---|---|---|
| `USERS_TABLE`, `GRADING_TABLE`, `ASSIGNMENTS_TABLE`, `SUBMISSIONS_TABLE`, `ANNOUNCEMENTS_TABLE`, `ADMIN_USERS_TABLE`, `RESET_REQUESTS_TABLE`, `CONTENT_TABLE`, `TICKETS_TABLE`, `AUDIT_LOGS_TABLE`, `SETTINGS_TABLE`, `INVITES_TABLE`, `ANALYTICS_TABLE` | `Users`, `assignment_grading`, `GradeAnt_…` | DynamoDB table names |
| `GRADING_ASSIGNMENT_INDEX` | `assignment_id-index` | Optional index on `assignment_grading` |
| `TOKEN_EXPIRY_SECONDS` | `86400` | Login token lifetime |
| `SESSION_TOKEN_TTL_SECONDS`, `INVITE_TOKEN_TTL_SECONDS` | 1 day, 7 days | Admin session and invite-link lifetimes |
| `FLAG_CONFIDENCE` | `0.75` | Answers below this AI confidence are flagged for review |
| `UPLOAD_MAX_BYTES`, `UPLOAD_URL_TTL_SECONDS` | 10 MB, 900 s | Upload limits |
| `INVITE_SENDER_EMAIL`, `CONTACT_SENDER_EMAIL` | `SES_SENDER_EMAIL` | Separate From addresses, if wanted |
| `APP_BASE_URL` | `app.yaml` `urls.production.web` | Web address used in invite links |
| `PIPELINE_BUCKET`, `PIPELINE_ROOT`, `PIPELINE_BUCKET_REGION` | –, `Data_warehouse`, `AWS_DATA_REGION` | Pipeline bucket (required by the workers; no default), top folder, bucket Region |
| `GEMINI_API_KEY_ARN` (`GEMINI_API_KEY` locally) | – | Gemini key for the workers |
| `GEMINI_MODEL`, `PIPELINE_LLM` | `gemini-3.5-flash-preview`, `gemini` | The one model, and the provider |
| `GRADE_ON_UPLOAD` | `true` | Grade when a parsed sheet lands, or store it ungraded for "Start grading" |
| `GRADER_MAX_WORKERS`, `GRADER_TEMPERATURE`, `PIPELINE_LLM_RETRIES` | 4, 0.6, 3 | Questions graded in parallel, model temperature, attempts per model call |
| `UPLOAD_LAYOUT` | `template` | `/uploadpdf` keys: `template` (`UPLOAD_KEY_TEMPLATE`) or `pipeline` |
| `CORRECTION_SIMILARITY_MIN` (grader) | `0.6` | A corrected answer keeping less of the paper's text (`ocr_answer`) than this is flagged for teacher review |

## C6. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `No module named 'gradeant_common'` | The `gradeant-python-common` layer isn't attached (B8 step 4) |
| `No module named 'jwt'` / `'yaml'` | The `gradeant-python-deps` layer isn't attached, or the function isn't **arm64** |
| `Handler 'lambda_handler' missing` / `Unable to import module` | Wrong handler: it must be `<file>.lambda_handler`, for example `grades.lambda_handler` |
| `"jwtSecret": {"ok": false}` or `Server misconfigured: JWT_SECRET is not set.` | `JWT_SECRET_ARN` is wrong or missing, or the policy lacks `secretsmanager:GetSecretValue` |
| `AccessDeniedException` in the logs | The policy is missing a table or service. Compare with B9. |
| Browser: "blocked by CORS policy" | The site's exact origin is missing from `CORS_ALLOWED_ORIGINS` |
| Everyone logged out after switching | `gradeant/jwt` isn't the same key as the old Lambdas' `JWT_SECRET` (B2) |
| A student sees no assignments | No `classes` on their `Users` item. Run B11, or set it by hand. |
| `{"message":"Internal server error"}` from API Gateway | The function failed to start: read `aws logs tail /aws/lambda/gradeant-<name>` |
| AI rubric stuck on "drafting" | Workflow logs: usually the Gemini key or a retired model (B5), or the self-invoke permission (Policy WORKFLOW) |
| Invites or reminders not emailed | SES sender not verified, SES still in sandbox (B4), or `SES_SENDER_EMAIL` missing |
| Worker log `Event from unexpected bucket` | The S3 notification comes from another bucket than `PipelineBucketName` |
| Worker log `Gemini API key not configured` | `GeminiSecretArn` empty, or the `PipelinePolicy` wasn't created (needs both bucket and secret set) |
| Worker log `Question-paper schema not found` / `…answer schema not found` | The teacher-side files for that assignment aren't in the bucket yet. The message is retried, then dead-lettered. |
| Messages piling up in a dead-letter queue | Read the worker logs (`message failed` lines show the reason). Fix it, then redrive the DLQ. |
| A student's sheet graded twice | Both the legacy and the new worker are on one queue (B15 step 5) |
