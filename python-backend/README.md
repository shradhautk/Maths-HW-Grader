# GradeAnt API — Python 3.13 Lambdas

The GradeAnt API as **nine Python Lambda functions**, one per API path. They sit behind the existing API Gateway (`rus9nultj9`, stage `dev`) and share two layers. **Two more functions, `gradeant-jsonifier` and `gradeant-grader`, are the grading pipeline** (answer-sheet PDF → JSON → grades), ported from `lambda_python_files/`. They are fed by SQS, not API Gateway (§6).

- **Same behaviour as the Node backend.** Every one of the 211 parity requests gets the same reply as `backend/` (205 identical, 6 identical apart from ordering, 0 differences).
- **Same data.** Same DynamoDB tables, same `config/app.yaml`, same login tokens.
- **Code runs on Python 3.10+.** It is deployed on `python3.13`, arm64.
- **The live Lambdas are untouched.** `lambda_python_files/` and `admin/` are what's deployed today; they stay unchanged as the rollback.
- **The Node backend stays.** `backend/` remains for future deployment. Nothing here deploys on its own.

| Path | What it is |
|---|---|
| `functions/<name>.py` | One Lambda each; handler `<name>.lambda_handler` |
| `gradeant_common/` | Shared layer: config, secrets, request handling, auth, permissions, DynamoDB, mail |
| `lambdas.py` | The functions: routes, AWS services, tables, memory, timeout. Source for the build, template and check |
| `build.py` | Builds `dist/` (layers + function zips); `--verify` starts each one from its zips |
| `deploy/generate_template.py` → `deploy/template.yaml` | AWS SAM template generated from `lambdas.py` |
| `deploy/samconfig.example.toml` | Your account's deploy settings (copy to `samconfig.toml`, gitignored) |
| `check_permissions.py` | Checks each function's IAM services against the graphify code graph |
| `gradeant_common/pipeline/` | Grading pipeline code: S3 layout, LLM client (Gemini), PDF → JSON, grading (§6) |
| `pipeline-requirements.txt` | Packages of the pipeline workers' own layer (`gradeant-pipeline-deps`) |
| `tests/` | 97 pytest tests (moto, fake model); plus the parity check in `backend/` |
| `DEPLOYMENT.md` | **Design & step-by-step AWS deployment guide**: architecture, decisions, every setting and IAM policy per function; pipeline cut-over in B15 |
| `PIPELINE_PORT_PLAN.md` | Design record of the pipeline port: findings about the originals, decisions, test plan |

---

## 1. How it fits together

```
Browser ──► API Gateway REST API rus9nultj9 (/dev, eu-north-1), Lambda proxy integrations
              POST /auth/*          ──► gradeant-auth          auth.py
              POST /getgrades       ──► gradeant-grades        grades.py
              POST /correctanswer   ──► gradeant-corrections   corrections.py
              POST /updatedb        ──► gradeant-marks         marks.py
              POST /workflow        ──► gradeant-workflow      workflow.py ──(async self-invoke)──► rubric drafting
              POST /admin           ──► gradeant-admin         admin.py
              POST /contact         ──► gradeant-contact       contact.py
              POST /uploadpdf       ──► gradeant-upload        upload.py   (off until UPLOAD_BUCKET is set)
              GET  /health[/ready]  ──► gradeant-health        health.py

every function = its own <name>.py
               + layer gradeant-python-deps    /opt/python                  PyJWT, PyYAML, boto3 (workflow also: pipeline layer, Gemini)
               + layer gradeant-python-common  /opt/python/gradeant_common  shared code + config/app.yaml

Grading pipeline (§6), fed by SQS (S3 events), not API Gateway:
   …/original_pdf/<student>.pdf    ──► gradeant-jsonifier  jsonifier.py  ──► …/Parsed_pdfs/JSON/<student>.json
   …/Parsed_pdfs/JSON/<student>.json ──► gradeant-grader   grader.py     ──► assignment_grading rows + summary
   gradeant-workflow "Start grading" ──(async)──► gradeant-grader (regrade the student's corrected answers)
   workers = <name>.py + gradeant-pipeline-deps (boto3, PyYAML, PyMuPDF, google-genai) + gradeant-python-common
```

**What `gradeant_common` does for every function** (mirrors `backend/src`):

| Module | Job |
|---|---|
| `config` | Settings from environment variables, using the same names and defaults as `backend/.env.example`. Also loads and validates `config/app.yaml`, which sets URLs, API paths, roles and who-may-do-what. |
| `secrets` | `JWT_SECRET_ARN` / `SESSION_TOKEN_SECRET_ARN`, read from Secrets Manager once per container. Plain `JWT_SECRET` / `SESSION_TOKEN_SECRET` are accepted locally. There is **no built-in fallback key**. |
| `http` | `handle()` answers OPTIONS and applies the CORS allowlist (app.yaml or `CORS_ALLOWED_ORIGINS`). It turns errors into `{"error"}` / `{"message"}` replies as each API always did, and never sends exception text in production. |
| `dispatch` | `operations()` runs the operation APIs in this order: body → login token → operation → `app.yaml` permission. |
| `auth`, `permissions` | Login-token checks (account must exist and not be suspended; role read fresh), and role checks from `app.yaml`. |
| `dynamo`, `grading`, `ownership` | Tables, every query reading all pages, and teacher-to-teacher isolation. |
| `mail` | SES in production, logged locally. |

**Behaviour changes from the deployed Lambdas.** All of them match the Node backend.

*Matching Node's rules:*
- **Roles:** who may run each operation comes from `config/app.yaml`, and so do the roles allowed to sign themselves up.
- **Graded answers locked:** once a paper is graded, or its submission has reached `grading_triggered`, a student can no longer change the answer text.

*Better error replies:*
- **Malformed JSON:** a 400, without the parser's text.
- **Grade update with no summary row:** a 404 (it used to be a 502).
- **Marks with only per-question entries:** applied (it used to be a 500).
- **Unexpected errors:** never leak details to the browser.

*Security:*
- **CORS allowlist on every function.** The admin Lambda used to echo back any website with credentials.
- **No fallback keys.** `auth.py` had a hard-coded fallback JWT key; the admin session key's development fallback is now refused in production.

*Security review fixes (2026-11-16, same in the Node backend):*
- **Teacher sign-up** is set in `config/app.yaml` `roles.teacher.signup`. It is now `[self, invite]`: teachers can register themselves or be invited. Set `[invite]` for invite-only, because a self-registered "teacher" could create an assignment for any class name, list that class's students' names and emails, post announcements to them, and send them GradeAnt reminder emails.
- **Reminders reach only students on the assignment's roster**: enrolled in its class, or with a submission or answers for it.
- **Students see only their own classes' assignments and announcements.** A student with no `classes` used to see every school's published work; now they see none.
- **Invites record the class.** An invited student's account gets `classes: [<invite class>]` at first login.
- **Answer uploads only for the student's own classes.** The assignment must be published and in one of their classes.
- **Profile photos:** the browser sends the file size; it must be at most 5 MB and is signed into the upload URL, so a larger file is refused by S3.
- **Auth error replies never include exception text.**

*Plumbing:*
- **Table names:** the same variables as Node (`USERS_TABLE` = `Users`, `ADMIN_USERS_TABLE` = `GradeAnt_Users`). The old admin Lambda used `USERS_TABLE` for the admin table.
- **Pagination:** queries follow DynamoDB pagination.
- **New functions:** `/contact`, `/uploadpdf` and `/health` are new in Python.

---

## 2. Each function: services, permissions, settings

All functions: **Python 3.13, arm64**. API functions use layers `gradeant-python-deps` and `gradeant-python-common`; the pipeline workers (‡) use `gradeant-pipeline-deps` and `gradeant-python-common`. Every role can also write its own logs (`/aws/lambda/gradeant-<name>`, kept 30 days).

| Function | Route | Memory / timeout | AWS services it may use | DynamoDB tables |
|---|---|---|---|---|
| `gradeant-auth` | `POST /auth/*` | 1024 MB / 29 s | DynamoDB; Secrets Manager (JWT); S3 `profile-photos/*` | `Users`, `GradeAnt_*` |
| `gradeant-grades` | `POST /getgrades` | 512 MB / 29 s | DynamoDB; Secrets Manager (JWT) | `Users`, `assignment_grading`, `GradeAnt_Assignments`, `GradeAnt_Submissions`, `GradeAnt_Announcements` (+ indexes) |
| `gradeant-corrections` | `POST /correctanswer` | 512 MB / 29 s | same as grades | same as grades |
| `gradeant-marks` | `POST /updatedb` | 512 MB / 29 s | same as grades | same as grades |
| `gradeant-workflow` | `POST /workflow` | 1024 MB / 300 s † | DynamoDB; Secrets Manager (JWT, Gemini key*); Gemini; invoke **itself** (rubric drafting); SES*; SNS*; grading Lambda* | same as grades |
| `gradeant-admin` | `POST /admin` | 1024 MB / 29 s | DynamoDB; Secrets Manager (JWT + session); SES* | `Users`, `GradeAnt_*` |
| `gradeant-contact` | `POST /contact` | 256 MB / 15 s | SES* only | none |
| `gradeant-upload` | `POST /uploadpdf` | 512 MB / 15 s | DynamoDB; Secrets Manager (JWT); S3 upload bucket* | `Users`, `GradeAnt_Assignments` (a student uploads only for their own classes) |
| `gradeant-health` | `GET /health`, `/health/ready` | 256 MB / 10 s | DynamoDB `GetItem` on `Users`; Secrets Manager (JWT) | `Users` (one read) |
| `gradeant-grader` ‡ | SQS (`GraderQueueArn`) + async invoke from workflow | 1024 MB / 900 s | DynamoDB incl. `BatchWriteItem`; S3 pipeline bucket*; Gemini secret*; SQS poll*; failure queue | `assignment_grading` (+ indexes) |
| `gradeant-jsonifier` ‡ | SQS (`JsonifierQueueArn`) | 2048 MB / 900 s | S3 pipeline bucket*; Gemini secret*; SQS poll* | none |

\* Only when the setting is filled in: `SesSenderEmail`, `SnsTopicArn`, `GradingFunctionName`, `UploadBucketName`, `PipelineBucketName` + `GeminiSecretArn`, the queue ARNs. Each has its own policy that exists only then, and SES sending is limited to that one From address.

‡ Layers `gradeant-pipeline-deps` + `gradeant-python-common` (not `gradeant-python-deps`); the template sets layers per function for this reason.

† **Why workflow has 300 s.** API Gateway still stops browser requests at 29 s. The longer limit is for the background run that drafts a rubric with Claude: `workflow.py` invokes itself asynchronously (`InvocationType=Event`), as the deployed Lambda already does.

**Environment variables (set on every function by the template).** These hold only ARNs, never secret values:

| Variable | Meaning |
|---|---|
| `APP_ENV=production` | Production behaviour (no error details, no dev fallbacks) |
| `AWS_DATA_REGION` | Region of the tables, SNS and grading Lambda |
| `CORS_ALLOWED_ORIGINS` | Sites allowed to call the API |
| `JWT_SECRET_ARN`, `SESSION_TOKEN_SECRET_ARN` | Keys in Secrets Manager |
| `PROFILE_PHOTO_BUCKET` | Profile photos (created if you don't pass one) |
| `UPLOAD_BUCKET`, `UPLOAD_KEY_TEMPLATE` | `/uploadpdf` (off while empty) |
| `SES_SENDER_EMAIL`, `CONTACT_INBOX` | Email |
| `SNS_TOPIC_ARN`, `GRADING_FUNCTION_NAME` | Notifications, grading hand-off |
| `GEMINI_MODEL` (default `gemini-3.5-flash-preview`) | The one model for every AI call |
| `UPLOAD_LAYOUT` (upload only) | `template` (keys from `UPLOAD_KEY_TEMPLATE`) or `pipeline` (the pipeline's layout, §6) |
| `PIPELINE_BUCKET`, `PIPELINE_ROOT` (workers + upload) | The pipeline's bucket and top folder (`Data_warehouse`) |
| `GEMINI_API_KEY_ARN` / `GEMINI_API_KEY` (workers and workflow), `GRADE_ON_UPLOAD`, `GRADER_MAX_WORKERS` (workers) | Gemini key, grade-on-upload switch, questions graded in parallel |

Optional overrides use the same names as `backend/.env.example`: table names (`USERS_TABLE`, `GRADING_TABLE`, …), `TOKEN_EXPIRY_SECONDS`, `FLAG_CONFIDENCE`, `UPLOAD_MAX_BYTES` and so on.

---

## 3. Build and test

```bash
cd python-backend
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest tests                    # 97 tests (PyMuPDF from requirements-dev.txt)

# Same replies as the Node backend (211 requests), from backend/:
cd ../backend && PARITY_PYTHON=python-backend PYTHON=../python-backend/.venv/Scripts/python.exe npm run check:parity

# Each IAM role vs. the code graph (refresh the graph first after code changes):
cd .. && graphify update .            # copy kept at extra/leftovers/graphify-out
cd python-backend && python check_permissions.py
#   graphify skips git-ignored folders. If python-backend/ is git-ignored, build a graph of a copy
#   (functions/, gradeant_common/, lambdas.py under <dir>/python-backend), run `graphify update .`
#   in <dir>, then: python check_permissions.py <dir>/graphify-out/graph.json

# The zips (dependencies are fetched for Linux arm64 / Python 3.13, so this works on Windows too):
python build.py --verify
```

**Windows note.** moto has deeply nested files. If `pip install` fails with a long-path error, create the virtual environment at a short path (for example `%TEMP%\g313`), or enable Windows long paths.

`dist/` then holds:

| Zip | Upload as | Lands at | Size |
|---|---|---|---|
| `layer-gradeant-python-deps.zip` | Layer **gradeant-python-deps** (Python 3.13, arm64) | `/opt/python` | ~23 MB zipped, ~42 MB unzipped |
| `layer-gradeant-python-common.zip` | Layer **gradeant-python-common** (Python 3.13, arm64) | `/opt/python/gradeant_common` | < 0.1 MB |
| `layer-gradeant-pipeline-deps.zip` | Layer **gradeant-pipeline-deps** (Python 3.13, arm64), pipeline workers only | `/opt/python` | ~50 MB zipped, ~112 MB unzipped |
| `gradeant-<name>.zip` × 11 | Function code, handler `<name>.lambda_handler` | `/var/task/<name>.py` | a few KB each |
| `manifest.json` | – | – | Routes, services, tables, layers and SHA-256 of every zip |

An API function plus its two layers is about 42 MB unzipped; a pipeline worker plus its two layers about 112 MB. Both are under Lambda's 250 MB limit, which `build.py` enforces. `--verify` starts each function with only its own layers, as Lambda does. If you change `lambdas.py`, regenerate the template with `python deploy/generate_template.py`.

---

## 4. Deploy

### 4.1 One-time AWS setup

These steps are the same for every version of the backend. Follow **`backend/deploy/DEPLOY.md` §3**:

1. **JWT secret.** Store the key the deployed Lambdas use as `gradeant/jwt` in Secrets Manager, and note its ARN.
2. **Session key (optional).** To keep admin first-login links working across the switch, store the admin Lambda's `SESSION_TOKEN_SECRET` too, and pass it as `SessionSecretArn`.
3. **A Gemini API key** (paid tier) as `GeminiSecretArn` or `GeminiApiKey`.
4. **SES:** domain verification and production access, if you want email.
5. **The grading Lambda's name**, for `GradingFunctionName`.
6. **Pre-flight check** (read-only): `bash backend/deploy/preflight.sh`.
7. **Give existing students their class, before switching any route.** Students now see only their own classes' work, and accounts created before this fix have no `classes`. Copy each invited student's class from their invite record:

   ```bash
   python tools/backfill_classes.py --region eu-north-1           # dry run: lists what would change
   python tools/backfill_classes.py --region eu-north-1 --apply   # writes Users.classes
   ```

   Students who signed themselves up have no invite class. The script lists them, and an admin has to place them in a class: set `classes` (a list of class names) on their `Users` item. Until then they see no assignments.

### 4.2 Create the functions (AWS SAM)

```bash
cd python-backend
python build.py --verify
cd deploy
cp samconfig.example.toml samconfig.toml      # JwtSecretArn, RestApiId=rus9nultj9, optional settings
sam validate --lint
sam deploy                                    # shows every change and asks before applying
```

This creates the three layers, the nine API functions and the two pipeline workers with their roles, and the log groups and alarm. It also lets `rus9nultj9` invoke each API function. **It does not change API Gateway routes**: the website keeps using the current Lambdas until §4.4. The workers stay idle until their cut-over (DEPLOYMENT.md B15): their SQS mappings are created disabled, and "Start grading" keeps invoking `GradingFunctionName`.

Only one version of these functions can be deployed: the Python stack and the Node `backend/lambdas` stack use the same function names (`gradeant-*`).

### 4.2b Or: upload everything by hand in the AWS console

This does the same as `sam deploy`, by clicking. It takes longer (about 80 clicks), but you can see every step.

#### How the shared code (`gradeant_common`) gets to every Lambda

You don't copy `gradeant_common` into each function. It is uploaded **once, as a Lambda layer**, and every function is linked to that layer.

- **Function zip** (one per function): contains only that function's own file. For example, `gradeant-auth.zip` contains just `auth.py`.
- **Layer `gradeant-python-common`** (shared by all nine functions): `gradeant_common` and `config/app.yaml`.
- **Layer `gradeant-python-deps`** (shared by all nine functions): the libraries PyJWT, PyYAML and boto3. `gradeant-workflow` also gets `gradeant-pipeline-deps` for Gemini (google-genai).

When a function runs, AWS unpacks its own zip into `/var/task` and each attached layer into `/opt`. Python on Lambda automatically looks in `/opt/python`, so `from gradeant_common import config` in `auth.py` works:

```
What a Lambda sees at runtime:

/var/task/auth.py                          ← gradeant-auth.zip (only this function's file)
/opt/python/gradeant_common/__init__.py    ← layer gradeant-python-common
/opt/python/gradeant_common/http.py
/opt/python/gradeant_common/...
/opt/python/gradeant_common/config/app.yaml
/opt/python/jwt/  /opt/python/yaml/  ...   ← layer gradeant-python-deps
```

A layer zip must have a top folder named `python/`, which `build.py` creates. So **always upload the zips from `dist/`**. Never zip the `gradeant_common` folder yourself: without the `python/` folder, Lambda can't find it.

#### Step 1: make the zip files

```bash
cd python-backend
python build.py --verify
```

You upload **ZIP files only**, not folders or `.py` files. That is 11 zips in total, all in `python-backend/dist/`:

```
dist/
├── layer-gradeant-python-deps.zip      ← upload as a LAYER
├── layer-gradeant-python-common.zip    ← upload as a LAYER  (this is gradeant_common)
├── gradeant-auth.zip                   ← upload as a FUNCTION
├── gradeant-grades.zip                 ← upload as a FUNCTION
├── gradeant-corrections.zip            ← upload as a FUNCTION
├── gradeant-marks.zip                  ← upload as a FUNCTION
├── gradeant-workflow.zip               ← upload as a FUNCTION
├── gradeant-admin.zip                  ← upload as a FUNCTION
├── gradeant-contact.zip                ← upload as a FUNCTION
├── gradeant-upload.zip                 ← upload as a FUNCTION
└── gradeant-health.zip                 ← upload as a FUNCTION
```

#### Step 2: upload the 2 layers (first, and only once)

1. Open the AWS console. In the top-right Region selector, choose **Europe (Stockholm) eu-north-1**.
2. Search for **Lambda** and open it.
3. Left menu: **Layers**, then **Create layer**.
4. Fill in:
   - **Name:** `gradeant-python-common`
   - **Upload a .zip file:** `layer-gradeant-python-common.zip`
   - **Compatible architectures:** tick **arm64**
   - **Compatible runtimes:** **Python 3.13**
5. Click **Create**.
6. Repeat steps 3–5 with **Name** `gradeant-python-deps` and the file `layer-gradeant-python-deps.zip` (about 23 MB, under the console's 50 MB limit).

You now have two layers. That is where `gradeant_common` and the libraries live on AWS.

#### Step 3: upload the 9 functions (the same steps 9 times)

For example, **auth**:

1. Left menu: **Functions**, then **Create function**.
2. Choose **Author from scratch** and fill in:
   - **Function name:** `gradeant-auth`
   - **Runtime:** **Python 3.13**
   - **Architecture:** **arm64**
3. Click **Create function**. AWS also creates an IAM role for it, named something like `gradeant-auth-role-xxxx`.
4. **Code** tab: **Upload from**, then **.zip file**. Choose `gradeant-auth.zip`, then **Save**.
5. Same tab, scroll to **Runtime settings**, then **Edit**. Set **Handler** to `auth.lambda_handler`, then **Save**.
6. Scroll to **Layers**, then **Add a layer**. Choose **Custom layers**, then `gradeant-python-common` (latest version), then **Add**.
7. **Add a layer** again: **Custom layers**, then `gradeant-python-deps` (latest version), then **Add**.
8. **Configuration** tab, then **General configuration**, then **Edit**: set the memory and timeout from the table below, then **Save**.

Do the same for the other eight, changing only the name, zip, handler, memory and timeout. **Every function gets both layers.**

| Function name | Zip to upload | Handler | Memory | Timeout |
|---|---|---|---|---|
| `gradeant-auth` | `gradeant-auth.zip` | `auth.lambda_handler` | 1024 MB | 29 s |
| `gradeant-grades` | `gradeant-grades.zip` | `grades.lambda_handler` | 512 MB | 29 s |
| `gradeant-corrections` | `gradeant-corrections.zip` | `corrections.lambda_handler` | 512 MB | 29 s |
| `gradeant-marks` | `gradeant-marks.zip` | `marks.lambda_handler` | 512 MB | 29 s |
| `gradeant-workflow` | `gradeant-workflow.zip` | `workflow.lambda_handler` | 1024 MB | **300 s** |
| `gradeant-admin` | `gradeant-admin.zip` | `admin.lambda_handler` | 1024 MB | 29 s |
| `gradeant-contact` | `gradeant-contact.zip` | `contact.lambda_handler` | 256 MB | 15 s |
| `gradeant-upload` | `gradeant-upload.zip` | `upload.lambda_handler` | 512 MB | 15 s |
| `gradeant-health` | `gradeant-health.zip` | `health.lambda_handler` | 256 MB | 10 s |

#### Step 4: environment variables (every function)

On each function, open **Configuration**, then **Environment variables**, then **Edit**, and add these. They hold only ARNs and names, never a secret value.

| Key | Value |
|---|---|
| `APP_ENV` | `production` |
| `AWS_DATA_REGION` | `eu-north-1` |
| `CORS_ALLOWED_ORIGINS` | `https://app.gradeant.io,https://grade-ant.com,https://www.grade-ant.com` |
| `JWT_SECRET_ARN` | the ARN of the `gradeant/jwt` secret (§4.1) |
| `SESSION_TOKEN_SECRET_ARN` | the ARN of the session-key secret (§4.1); needed by `gradeant-admin` |
| `PROFILE_PHOTO_BUCKET` | your profile-photo bucket name; needed by `gradeant-auth` |
| `SES_SENDER_EMAIL` | your verified sender (for example `no-reply@gradeant.io`), or leave it out for no email |
| `CONTACT_INBOX` | `contact@gradeant.com` |
| `GRADING_FUNCTION_NAME` | your grading-pipeline Lambda's name; needed by `gradeant-workflow` |
| `SNS_TOPIC_ARN` | optional: "grades published" topic |
| `GEMINI_MODEL` | `gemini-3.5-flash-preview` |
| `UPLOAD_BUCKET` | leave it out for now; `/uploadpdf` stays off until it's set (see §2) |

#### Step 5: permissions (every function)

Each function's role starts with log-writing only. Add what the function needs:

1. Open the function's **Configuration** tab, then **Permissions**.
2. Click the **role name** link. This opens IAM.
3. Choose **Add permissions**, then **Create inline policy**, then **JSON**.
4. Paste the policy below. Replace `ACCOUNT_ID` with your 12-digit AWS account ID, and `JWT_SECRET_ARN` (and the other placeholders) with your ARNs.
5. **Next**, name the policy (for example `gradeant`), then **Create policy**.

**`gradeant-grades`, `gradeant-corrections`, `gradeant-marks`:**

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

**`gradeant-workflow`:** the same as above, plus these statements, added inside `"Statement": [ ... ]`:

```json
    { "Effect": "Allow", "Action": "lambda:InvokeFunction",
      "Resource": [ "arn:aws:lambda:eu-north-1:ACCOUNT_ID:function:gradeant-workflow",
                    "arn:aws:lambda:eu-north-1:ACCOUNT_ID:function:GRADING_FUNCTION_NAME" ] },
    { "Effect": "Allow", "Action": "ses:SendEmail", "Resource": "*",
      "Condition": { "StringEquals": { "ses:FromAddress": "SES_SENDER_EMAIL" } } },
    { "Effect": "Allow", "Action": "sns:Publish", "Resource": "SNS_TOPIC_ARN" }
```

(Leave out the SES or SNS statement if you don't use that feature.)

**`gradeant-auth`:**

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

**`gradeant-admin`:**

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

**`gradeant-contact`:** sending email only.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "ses:SendEmail", "Resource": "*",
      "Condition": { "StringEquals": { "ses:FromAddress": "SES_SENDER_EMAIL" } } }
  ]
}
```

**`gradeant-upload`:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:Query"],
      "Resource": [ "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users",
                    "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/GradeAnt_Assignments" ] },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": "JWT_SECRET_ARN" },
    { "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": "arn:aws:s3:::UPLOAD_BUCKET/*" }
  ]
}
```

(Add the S3 statement only once you set `UPLOAD_BUCKET`.)

**`gradeant-health`:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "dynamodb:GetItem", "Resource": "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Users" },
    { "Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": "JWT_SECRET_ARN" }
  ]
}
```

These are exactly the permissions `sam deploy` would create; the source is `deploy/template.yaml`.

#### Step 6: check each function works

On `gradeant-health`, open the **Test** tab, create a test event with this JSON, and click **Test**:

```json
{"httpMethod": "GET", "path": "/health/ready", "headers": {}}
```

The result should show `"statusCode": 200`, with `"dynamodb": {"ok": true ...}` and `"jwtSecret": {"ok": true ...}`. If it doesn't, compare the message with this table:

| You see | Fix |
|---|---|
| `No module named 'gradeant_common'` | The `gradeant-python-common` layer isn't attached (step 3). |
| `No module named 'jwt'` or `'yaml'` | The `gradeant-python-deps` layer isn't attached, or the function isn't **arm64**. |
| `"dynamodb": {"ok": false ...}` | The DynamoDB permission is missing (step 5), or `AWS_DATA_REGION` is wrong. |
| `"jwtSecret": {"ok": false ...}` | `JWT_SECRET_ARN` is wrong or missing (step 4), or the secret permission is missing (step 5). |

For any other function, a quick start-up check is the event `{"httpMethod": "OPTIONS", "path": "/getgrades", "headers": {}}`. Use that function's own path; the answer is `"statusCode": 204`.

#### Step 7: connect each function to the website

Follow §4.4: in API Gateway, point each route at its new `gradeant-*` function, then **Deploy API** to `dev`. Do one route at a time and check the website after each.

#### Changing code later

| What you changed | What to upload |
|---|---|
| One function's file, for example `functions/grades.py` | Run `python build.py`, then upload only that function's zip (**Code**, then **Upload from**, then **.zip file**). Nothing else. |
| Anything in `gradeant_common/` or `config/app.yaml` | Run `python build.py`. Go to **Layers**, then `gradeant-python-common`, then **Create version**, and upload the new `layer-gradeant-python-common.zip`. Then, **in every function**: **Layers**, then **Edit**, and pick the new version. |
| `requirements.txt` | The same, for the `gradeant-python-deps` layer. |

Layer versions never change once uploaded, and each function stays on the version it was given until you switch it. That is why a shared-code change needs the second step. `sam deploy` does both steps automatically.

### 4.3 Test before switching traffic

```bash
aws lambda invoke --function-name gradeant-health --region eu-north-1 --cli-binary-format raw-in-base64-out \
  --payload '{"httpMethod":"GET","path":"/health/ready","headers":{}}' out.json && cat out.json
# → statusCode 200 and "checks": {"dynamodb": {"ok": true…}, "jwtSecret": {"ok": true…}}
aws logs tail /aws/lambda/gradeant-health --follow --region eu-north-1
```

### 4.4 Switch API Gateway, one route at a time

1. Go to *API Gateway → `rus9nultj9` → Resources → method → Integration request → Edit*. Choose **Lambda proxy integration** and the new `gradeant-<name>` function.
2. *Deploy API* to stage **dev**.
3. Check the website.

Start with `/health` and `/getgrades`, then `/auth/*`, then the rest. `/contact` and `/health` need new resources on this API. The CLI form of the same steps is in `backend/lambdas/LAMBDAS.md` §7, and it is identical for Python.

**Rollback:** point the route back at the current Lambda and *Deploy API*. Both use the same tables and keys, so nobody is logged out.

---

## 5. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `Runtime.ImportModuleError: No module named 'gradeant_common'` | The `gradeant-python-common` layer isn't attached. |
| `No module named 'jwt'` / `'yaml'` | The `gradeant-python-deps` layer isn't attached, or the function is x86_64. The layer is built for arm64. |
| `{"message":"Internal server error"}` on every call | Startup error. `aws logs tail /aws/lambda/gradeant-<name>` shows it; for example `config/app.yaml` is invalid, or `JWT_SECRET_ARN` is wrong. |
| `Server misconfigured: JWT_SECRET is not set.` | `JwtSecretArn` is empty, or the role can't read the secret. |
| Browser "blocked by CORS policy" | The site's exact origin is missing from `WebOrigins`. |
| AI rubric stays "drafting" | Look at the workflow logs: usually the Gemini key (or a retired model), or the self-invoke permission. |
| `AccessDeniedException` in a log | The role lacks the action. Run `python check_permissions.py` (after `graphify update .`), then fix `lambdas.py` and regenerate the template. |

---

## 6. Grading pipeline (`gradeant-jsonifier`, `gradeant-grader`)

A port of `lambda_python_files/excute_jsonifier.py` + `jsonifier.py` and `grader_lambda.py`. The originals stay untouched as the rollback. Cut-over steps: **DEPLOYMENT.md B15**. Design record: `PIPELINE_PORT_PLAN.md`.

**Flow** (S3 keys as the originals use them; `<root>` = `PIPELINE_ROOT`, default `Data_warehouse`):

1. A student's PDF lands at `<root>/students/assignments/<aid>__<qp>/original_pdf/<student>.pdf`. The S3 event goes through SQS to **`gradeant-jsonifier`**. It reads the question paper (`teachers/QP_Rubric/<aid>__<qp>/QP_pdfs/<qp>.pdf`) and its answer schema (`…/QP_schemas/QP_schemas_answers/<qp>_schema_answer.txt`). It then transcribes the PDF, splits the transcript by question, fills the schema, and writes `Parsed_pdfs/{text,JSON_Seg,JSON_Text}/<student>.txt` and then, **once**, `Parsed_pdfs/JSON/<student>.json`.
2. That JSON's S3 event goes through SQS to **`gradeant-grader`**. It loads the question-paper schema JSON and rubric (from `QPRubricWithAnswers/…` when a solution PDF exists, else `QP_Rubric/…`), pairs questions with answers, and:
   - `GRADE_ON_UPLOAD=true` (default, as today): grades every answer (mark + reasoning, then feedback) and writes the `assignment_grading` rows and the summary row last;
   - `GRADE_ON_UPLOAD=false`: writes the rows ungraded (no marks, no summary). The student corrects the transcription, and the teacher presses **Start grading**.
3. **Start grading:** `gradeant-workflow` invokes `GradingFunctionName` asynchronously with `{assignment_id, qp_id, student_id, method, persona, partial_credit, rubric}`. `gradeant-grader` regrades that student's corrected rows in place, with the assignment's rubric criteria, max marks and persona, then writes the summary.

**What changed from the originals** (each covered by a test):

| Original | Now |
|---|---|
| Failed SQS messages answered `207` and were deleted anyway | `{"batchItemFailures": [...]}`: only failures return to the queue, then go to its DLQ |
| A Gemini error became the answer text, graded as **0 marks** | The message fails and nothing is written. Marks must be a number, capped to `[0, max_marks]`. A feedback failure is still non-fatal, with the original's text. |
| Re-runs added a second set of rows (random ids) | Deterministic row ids (`uuid5` of assignment, question paper, student, question, subpart). A re-run overwrites and deletes the student's stale rows. |
| The JSON was written twice, so each sheet was graded twice | Written once |
| The grader rejected the workflow's "Start grading" invoke (400) | Accepted (regrade) |
| Questions graded one after another (asyncio over blocking calls) | In parallel threads (`GRADER_MAX_WORKERS`, default 4) |
| AWS keys from env vars, `.env` files, `us-east-1` / `Data_warehouse` / `assignmentgrader` defaults | Lambda role only. Gemini key from Secrets Manager. Bucket required (`PIPELINE_BUCKET`), events from any other bucket refused. Root taken from the key. |
| `google-generativeai` (end of support), PDFs uploaded to the Files API | `google-genai` behind `gradeant_common.pipeline.llm`, with PDFs sent inline |
| `/uploadpdf` had no login check and kept user file names | `gradeant-upload` with `UPLOAD_LAYOUT=pipeline`: logged in, owner checks, server-chosen names (`<studentId>.pdf`, `<qp>.pdf`, `<aid>_<qp>_answer.pdf`) |
| A student's "correction" replaced the paper's text, so a rewritten answer would be graded unnoticed | The paper's text is kept once in `ocr_answer`. On "Start grading", an answer whose correction keeps less than `CORRECTION_SIMILARITY_MIN` (0.6) of it is graded **and flagged** (`confidence`, `flag_reason: answer_rewritten_after_ocr`), so the paper goes to the teacher's review queue |
| A repeated SQS message could reset corrected rows, delete a grade, or overwrite a teacher's mark edits | The grader skips a sheet whose rows were already graded, corrected or edited |

**Unchanged on purpose:** every prompt (verbatim, pinned by tests against the original files), the S3 layout, the row fields (plus `graded_at`, and `method`/`persona` on the summary), the AI calls (since the Gemini-only change, every one uses `gemini-3.5-flash-preview`, the `GeminiModel` template parameter).

**Personas.** `balanced` is the original "QRee" persona. `strict` and `lenient` add one sentence each. Those sentences are placeholders marked `TODO(user, plan O5)` in `gradeant_common/pipeline/prompts.py`; replace them with your wording.

### 6.1 The grader on Supabase (free-tier hybrid)

In the hybrid deployment (`config/app.yaml` → `deployment: hybrid`; plan: `backend/FREE_TIER_HYBRID_PLAN.md`) the grading table lives in **Supabase Postgres**, and only the two pipeline workers stay on AWS.

- **Which database:** `config.database()` returns `DATABASE_PROVIDER` if set, otherwise the YAML choice in production, otherwise `dynamodb`. With `supabase`, `gradeant_common/grading.py` reads through Supabase's REST API (`gradeant_common/pg_rest.py`, `httpx2`, already in the pipeline layer). It writes only through the two SQL functions in `backend/sql/002_pipeline_api.sql`: `grading_replace_rows` (details, then the summary last) and `grading_patch` (regrade). Each runs as one transaction. `dynamo.table()` refuses to run on `supabase`, so nothing can write to DynamoDB by mistake.
- **"Start grading" through SQS** (`grading: sqs`): the Node API sends the regrade payload as a message to the grader's own queue. The grader tells it apart from an S3 event by its body. Failures go back to the queue, then to its DLQ.
- **Pipeline-only stack:** `deploy/pipeline-template.yaml`, written by `python deploy/generate_template.py` together with `template.yaml`. It holds the two workers, their layers, queue mappings and failure queue, with no API functions, session secret or photo bucket. Deploy it **instead of** `template.yaml`, because the function names are the same.
- **Free-tier secrets:** `GeminiApiKey` and `SupabaseServiceRoleKey` are `NoEcho` parameters stored as encrypted Lambda environment variables (no Secrets Manager cost). `GeminiSecretArn` still works on the all-AWS deployment.
- **Permissions follow the database:** the grader's DynamoDB permission exists only while `DatabaseProvider` isn't `supabase`. The service-role key itself can only read `assignment_grading` and call the two functions (`backend/sql/003_supabase_access.sql`).
- **Tests:** `tests/test_pipeline_supabase.py` runs the grader against a fake Supabase REST API: request shapes, paging, retries, upload grading, a repeated message skipped, an outage failing the message, and a regrade through SQS with a rewritten answer flagged. `test_live_supabase_roundtrip` runs against a staging project when `SUPABASE_TEST_URL` and `SUPABASE_TEST_KEY` are set. With `SUPABASE_TEST_ANON_KEY` it also checks that the public key can't read the table.
