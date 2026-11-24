# Grading pipeline on AWS Lambda: console setup, step by step (free-tier hybrid)

Deploy the two pipeline workers by hand in the AWS console. Do the **grader first**, then the
**jsonifier**. Each part ends with a test, so you know it works before you move on.

Region for everything: **eu-north-1 (Stockholm)**. Check the region picker (top right) on every
page.

```
student uploads PDF ──► Render /uploadpdf ──► presigned PUT ──► S3 assignmentgrader
                                                                   …/original_pdf/<student>.pdf
                                                                          │ S3 event (.pdf)
                                                                          ▼
                                                              SQS jsonifier queue
                                                                          ▼
                                     Gemini ◄──────────────── Lambda gradeant-jsonifier
                                                                          │ writes
                                                                          ▼
                                                     …/Parsed_pdfs/JSON/<student>.json
                                                                          │ S3 event (.json)
                                                                          ▼
teacher presses "Start grading" ─► Render ─► SQS Grader ◄─────────────────┘
                                                 ▼
                     Gemini ◄────────── Lambda gradeant-grader ──► Supabase gradeant.assignment_grading
                                                                          ▼
                                                               web app (Netlify) shows grades
```

No API Gateway route and no upload Lambda are needed. Render checks logins, hands out the
upload links and sends the "Start grading" messages.

---

## 0. Before you start

Have these ready:

| Item | Where to get it |
|---|---|
| The built zips | `python-backend/dist/` (run `python build.py --verify` in `python-backend/` if you changed code since 19 Nov) |
| AWS account ID (12 digits) | Console top right → your name |
| Gemini API key | Google AI Studio → API keys. Use a **paid-tier** key: free-tier prompts may be used for training. |
| Supabase service_role key | Supabase → `gradeant-eu` → Project Settings → API Keys → **service_role** (secret) |
| Supabase URL | `https://dlbjcqrnnbyguewgwntz.supabase.co` |
| Names of the OLD pipeline Lambdas | Lambda → Functions (probably `grader_lambda` and `excute_jsonifier`) |
| Name of the OLD jsonifier queue | S3 → `assignmentgrader` → Properties → Event notifications: the queue that gets `.pdf` files |

Files you will upload:

| File | Size | Becomes |
|---|---|---|
| `layer-gradeant-pipeline-deps.zip` | ≈ 52 MB | Layer: Gemini SDK, PyMuPDF, boto3, PyYAML |
| `layer-gradeant-python-common.zip` | ≈ 40 KB | Layer: shared GradeAnt code, prompts, `config/app.yaml` |
| `gradeant-grader.zip` | ≈ 4 KB | Function `gradeant-grader` |
| `gradeant-jsonifier.zip` | ≈ 1 KB | Function `gradeant-jsonifier` |

Everything is built for **Python 3.13 on arm64**. Pick exactly that everywhere, or the function
fails with `ImportError` / `Runtime.ImportModuleError`.

The teacher-side files the pipeline reads must already be in S3 (the same layout the old
pipeline used):

| File | Read by |
|---|---|
| `Data_warehouse/teachers/QP_Rubric/<aid>__<qp>/QP_pdfs/<qp>.pdf` | jsonifier |
| `…/QP_Rubric/<aid>__<qp>/QP_schemas/QP_schemas_answers/<qp>_schema_answer.txt` | jsonifier |
| `<base>/QP_schemas/QP_schemas_json/<qp>_schema.json` | grader (required) |
| `<base>/rubric_jsons/<qp>_rubric.json` | grader (optional) |

`<base>` is `teachers/QPRubricWithAnswers/<aid>__<qp>` when
`solution_pdfs/<aid>_<qp>_answer.pdf` exists there. Otherwise it is `teachers/QP_Rubric/<aid>__<qp>`.

---

## Part A: shared layers (once, used by both functions)

### A1. Put the big layer in S3
The console only accepts direct zip uploads up to 50 MB.

1. **S3 → Buckets → `assignmentgrader` → Create folder** `lambda-layers` (outside
   `Data_warehouse/`, so it triggers nothing).
2. Open it → **Upload** → `layer-gradeant-pipeline-deps.zip` → **Upload**.
3. Click the uploaded file and copy its **Object URL**
   (`https://assignmentgrader.s3.eu-north-1.amazonaws.com/lambda-layers/layer-gradeant-pipeline-deps.zip`).

### A2. Create the two layers
**Lambda → Layers (left menu) → Create layer**, twice:

| Field | Layer 1 | Layer 2 |
|---|---|---|
| Name | `gradeant-pipeline-deps` | `gradeant-python-common` |
| Upload | **Upload a file from Amazon S3** → paste the URL from A1 | **Upload a .zip file** → `layer-gradeant-python-common.zip` |
| Compatible architectures | **arm64** | **arm64** |
| Compatible runtimes | **Python 3.13** | **Python 3.13** |

Click **Create** each time. Each gets **version 1**.

---

## Part B: the GRADER

### B1. Get the `Grader` queue ready
**SQS → Queues → `Grader`**

1. **Copy the ARN and the URL** from the Details box. You need them in B4 and B9.
2. **Dead-letter queue:** if there isn't one yet, **Create queue** → Standard → name
   `Grader-dlq` → Message retention **14 days** → Create.
3. Back on `Grader` → **Edit**:
   - **Visibility timeout: 5400 seconds** (90 min, which is 6 × the 15-minute Lambda timeout).
     AWS refuses the trigger in B7 if this is below 900.
   - **Message retention period: 4 days** (the default is fine).
   - **Dead-letter queue** section → **Enabled** → choose `Grader-dlq` → **Maximum receives: 3**.
   - **Encryption:** SSE-SQS (Amazon SQS key) is fine. If it uses a **customer KMS key**, S3
     can't publish to it without a KMS key policy. Switch it to SSE-SQS.
   - **Save**.

### B2. Create the function
**Lambda → Functions → Create function → Author from scratch**

| Field | Value |
|---|---|
| Function name | `gradeant-grader` |
| Runtime | **Python 3.13** |
| Architecture | **arm64** |
| Permissions | **Create a new role with basic Lambda permissions** (adds CloudWatch Logs) |

**Create function**, then:

1. **Code** tab → **Upload from → .zip file** → `gradeant-grader.zip` → Save.
2. **Code** tab → scroll to **Runtime settings → Edit** → Handler **`grader.lambda_handler`** → Save.
3. **Code** tab → bottom → **Layers → Add a layer → Custom layers**:
   - first `gradeant-pipeline-deps`, version 1 → Add
   - then `gradeant-python-common`, version 1 → Add
   
   Keep this order: deps first, common second.

### B3. General configuration and environment
**Configuration → General configuration → Edit**: Memory **1024 MB**, Ephemeral storage
**512 MB**, Timeout **15 min 0 sec** → Save.

**Configuration → Environment variables → Edit → Add**:

| Key | Value |
|---|---|
| `APP_ENV` | `production` |
| `AWS_DATA_REGION` | `eu-north-1` |
| `PIPELINE_BUCKET` | `assignmentgrader` |
| `PIPELINE_ROOT` | `Data_warehouse` |
| `GEMINI_API_KEY` | your Gemini key |
| `GEMINI_MODEL` | `gemini-3.5-flash-preview` |
| `DATABASE_PROVIDER` | `supabase` |
| `SUPABASE_URL` | `https://dlbjcqrnnbyguewgwntz.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | the service_role key |
| `GRADE_ON_UPLOAD` | `false` (rows wait for the teacher's **Start grading**) or `true` (grade as soon as a sheet is read) |
| `GRADER_MAX_WORKERS` | `4` (parallel Gemini calls per student) |

Save. Lambda encrypts these at rest. Only people with Lambda console access can read them.

> Optional: put the Gemini key in Secrets Manager (about $0.40/month) and set
> `GEMINI_API_KEY_ARN` instead of `GEMINI_API_KEY`. The role then also needs
> `secretsmanager:GetSecretValue` on that secret. The same works for `SUPABASE_SERVICE_ROLE_KEY_ARN`.

### B4. Permissions (IAM role)
**Configuration → Permissions → click the role name** (opens IAM) → **Add permissions → Create
inline policy → JSON**. Replace `ACCOUNT_ID`, then paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "PipelineFiles",
      "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::assignmentgrader/Data_warehouse/*" },
    { "Sid": "MissingFileIs404Not403",
      "Effect": "Allow", "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::assignmentgrader",
      "Condition": { "StringLike": { "s3:prefix": "Data_warehouse/*" } } },
    { "Sid": "ReadGraderQueue",
      "Effect": "Allow",
      "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"],
      "Resource": "arn:aws:sqs:eu-north-1:ACCOUNT_ID:Grader" }
  ]
}
```

**Next** → name `gradeant-grader-pipeline` → **Create policy**. Leave the
`AWSLambdaBasicExecutionRole-…` policy that was already there.

### B5. Log retention
**CloudWatch → Log groups**. The group `/aws/lambda/gradeant-grader` appears after the first run.
If it isn't there yet, **Create log group** with that exact name. Then **Actions → Edit retention
setting → 14 days**. Logs never contain answer text.

### B6. Test the grader before connecting anything
**Test** tab → **Create new event** → paste each event → **Test**:

**Test 1: code and layers load** (event name `empty`):
```json
{ "Records": [] }
```
Expect: `{"batchItemFailures": []}`. If you get `Runtime.ImportModuleError`, the runtime,
architecture or layer order is wrong (B2).

**Test 2: bucket setting** (event name `skip-file`):
```json
{ "Records": [ { "messageId": "t2",
  "body": "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/test/readme.txt\"}}}]}" } ] }
```
Expect: `{"batchItemFailures": []}` and a log line `grader: skipped (not a parsed answer sheet)`.
If it says `Event from unexpected bucket`, fix `PIPELINE_BUCKET`.

**Test 3: Supabase connection** (event name `supabase`):
```json
{ "assignment_id": "deploy-test", "qp_id": "deploy-test", "student_id": "nobody" }
```
Expect an **error**: `PipelineInputError: No answer rows for student nobody on
deploy-test/deploy-test`. That error is the correct result: Supabase answered and found nothing.
Other results:
- `401` / `Invalid API key`: wrong `SUPABASE_SERVICE_ROLE_KEY`
- `404` / `PGRST106` / schema error: Supabase → Project Settings → API → Exposed schemas must include `gradeant`
- timeout: wrong `SUPABASE_URL`

### B7. Move the queue trigger from the old grader to the new one
Two functions must never read the same queue: they would compete for messages.

1. **Lambda → old grader (`grader_lambda`) → Configuration → Triggers** → tick the SQS
   `Grader` trigger → **Edit** → untick **Activate trigger** → Save. (Or **Delete** it.
   Disabling keeps the rollback one click away.)
2. **Lambda → `gradeant-grader` → Add trigger → SQS**:

| Field | Value |
|---|---|
| SQS queue | `Grader` |
| Activate trigger | ✔ |
| Batch size | **1** |
| Batch window | 0 |
| Maximum concurrency | **5** (caps parallel graders and so Gemini rate and cost) |
| Report batch item failures | ✔ **required**: the code reports failed messages one by one |

**Add**. The trigger shows **Enabled** after a few seconds.

### B8. S3 → Grader notification (parsed answer sheets)
**S3 → `assignmentgrader` → Properties → Event notifications.**

If a notification already sends parsed `.json` files to `Grader` (from the old pipeline), keep it
and check it matches below. Otherwise **Create event notification**:

| Field | Value |
|---|---|
| Name | `parsed-json-to-grader` |
| Prefix | `Data_warehouse/students/assignments/` |
| Suffix | `.json` |
| Event types | **All object create events** |
| Destination | **SQS queue** → `Grader` |

If the save fails with *"Unable to validate the following destination configurations"*, the
queue doesn't allow S3 to send to it. Go to **SQS → `Grader` → Access policy → Edit** and add this
statement inside `"Statement": [ … ]` (replace `ACCOUNT_ID`):

```json
{ "Sid": "S3SendsParsedSheets",
  "Effect": "Allow",
  "Principal": { "Service": "s3.amazonaws.com" },
  "Action": "sqs:SendMessage",
  "Resource": "arn:aws:sqs:eu-north-1:ACCOUNT_ID:Grader",
  "Condition": {
    "ArnLike": { "aws:SourceArn": "arn:aws:s3:::assignmentgrader" },
    "StringEquals": { "aws:SourceAccount": "ACCOUNT_ID" } } }
```

Save, then create the notification again. S3 sends one `s3:TestEvent` to the queue, and the
grader acknowledges and skips it.

### B9. Connect Render (the "Start grading" button)
1. **IAM → Users → `gradeant-render`** (create it per `backend/deploy/FREE_TIER_HYBRID.md` §3 if it
   doesn't exist yet). Its policy must include:
   ```json
   { "Effect": "Allow", "Action": "sqs:SendMessage",
     "Resource": "arn:aws:sqs:eu-north-1:ACCOUNT_ID:Grader" }
   ```
2. **Render → `gradeant-api` → Environment**. These must be set:
   - `GRADING_QUEUE_URL` = the `Grader` queue **URL** from B1 (not the ARN)
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` = the `gradeant-render` key
   - `AWS_DATA_REGION` = `eu-north-1`

   Save. Render redeploys by itself.

### B10. Test the grader end to end
You need one assignment whose teacher files exist in S3 (§0) and one parsed sheet.

1. **Parsed sheet → rows.** Upload a parsed answer JSON (for example one the old pipeline made) to
   `Data_warehouse/students/assignments/<aid>__<qp>/Parsed_pdfs/JSON/<student_id>.json`.
   Within about 30 s, **CloudWatch → `/aws/lambda/gradeant-grader`** shows
   `grader: answer sheet processed`, and **Supabase → Table Editor → `gradeant.assignment_grading`**
   has that student's rows: ungraded with `GRADE_ON_UPLOAD=false`, graded with a `summary` row
   with `true`.
2. **Start grading → grades.** In https://gradeant.netlify.app, the student submits their
   corrections, then the teacher presses **Start grading**. The log shows `grader: regraded` and
   the roster shows *AI graded*, or *Queued for review* for a rewritten answer.
3. **No failures:** no `grader: message failed` lines, **SQS → `Grader-dlq`** shows 0 messages
   available.

A repeated message for the same sheet logs `skipped (answers already graded/corrected)` and
never overwrites a student's corrections or a teacher's edits.

---

## Part C: the JSONIFIER

### C1. Get the jsonifier queue ready
**SQS → Queues → the old jsonifier queue** (found in §0). If there isn't one, **Create queue** →
Standard → name `Jsonifier`.

Then, as in B1:
1. Copy its **ARN**.
2. Create `Jsonifier-dlq` (Standard, 14 days retention).
3. **Edit** the queue: **Visibility timeout 5400 s**, **Dead-letter queue** → `Jsonifier-dlq`,
   **Maximum receives 3**, SSE-SQS encryption → Save.

Below, `JSONIFIER_QUEUE` stands for that queue's name.

### C2. Create the function
**Lambda → Create function → Author from scratch**: name `gradeant-jsonifier`, **Python 3.13**,
**arm64**, *Create a new role with basic Lambda permissions* → Create. Then:

1. **Upload from → .zip file** → `gradeant-jsonifier.zip` → Save.
2. **Runtime settings → Edit** → Handler **`jsonifier.lambda_handler`** → Save.
3. **Layers → Add a layer**: `gradeant-pipeline-deps` v1, then `gradeant-python-common` v1.

### C3. General configuration and environment
**General configuration → Edit**: Memory **2048 MB** (PyMuPDF and the PDF in memory), Ephemeral
storage **512 MB**, Timeout **15 min** → Save.

**Environment variables:**

| Key | Value |
|---|---|
| `APP_ENV` | `production` |
| `AWS_DATA_REGION` | `eu-north-1` |
| `PIPELINE_BUCKET` | `assignmentgrader` |
| `PIPELINE_ROOT` | `Data_warehouse` |
| `GEMINI_API_KEY` | your Gemini key |
| `GEMINI_MODEL` | `gemini-3.5-flash-preview` |

The jsonifier doesn't touch the database, so it gets no Supabase settings.

### C4. Permissions
Role → **Create inline policy → JSON** (replace `ACCOUNT_ID` and `JSONIFIER_QUEUE`), name
`gradeant-jsonifier-pipeline`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "PipelineFiles",
      "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::assignmentgrader/Data_warehouse/*" },
    { "Sid": "MissingFileIs404Not403",
      "Effect": "Allow", "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::assignmentgrader",
      "Condition": { "StringLike": { "s3:prefix": "Data_warehouse/*" } } },
    { "Sid": "ReadJsonifierQueue",
      "Effect": "Allow",
      "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"],
      "Resource": "arn:aws:sqs:eu-north-1:ACCOUNT_ID:JSONIFIER_QUEUE" }
  ]
}
```

Set log retention on `/aws/lambda/gradeant-jsonifier` to 14 days, as in B5.

### C5. Test before connecting
- `{ "Records": [] }` → `{"batchItemFailures": []}`
- the `skip-file` event from B6 → log line `jsonifier: skipped (not a student answer sheet)`

### C6. Move the queue trigger
1. Old jsonifier (`excute_jsonifier`) → Configuration → Triggers → its SQS trigger → **Edit →
   untick Activate trigger** → Save.
2. `gradeant-jsonifier` → **Add trigger → SQS** → `JSONIFIER_QUEUE`, batch size **1**, batch
   window 0, maximum concurrency **5**, ✔ **Report batch item failures** → Add.

### C7. S3 → jsonifier notification (uploaded PDFs)
**S3 → `assignmentgrader` → Properties → Event notifications.** Keep the existing `.pdf` one if
it matches. Otherwise create:

| Field | Value |
|---|---|
| Name | `answer-pdf-to-jsonifier` |
| Prefix | `Data_warehouse/students/assignments/` |
| Suffix | `.pdf` |
| Event types | All object create events |
| Destination | SQS queue → `JSONIFIER_QUEUE` |

The same validation error and fix apply as in B8, with the jsonifier queue ARN and Sid
`S3SendsAnswerPdfs`.

The two notifications don't overlap (`.pdf` vs `.json`). The jsonifier's own `.txt` working files
trigger nothing.

### C8. Connect the browser upload (Render + bucket CORS)
1. `gradeant-render` user policy: `s3:PutObject` on
   `arn:aws:s3:::assignmentgrader/Data_warehouse/students/assignments/*/original_pdf/*` (plus the
   teacher folders, see `backend/deploy/render-iam-policy.json`).
2. **S3 → `assignmentgrader` → Permissions → Cross-origin resource sharing (CORS) → Edit**:
   ```json
   [ { "AllowedOrigins": ["https://gradeant.netlify.app"],
       "AllowedMethods": ["PUT"],
       "AllowedHeaders": ["content-type"],
       "MaxAgeSeconds": 3000 } ]
   ```
   If profile photos use this bucket too, add `"GET"` to `AllowedMethods`.
3. Keep **Block all public access: On**. Uploads go through one-time presigned links.
4. Render env: `UPLOAD_LAYOUT=pipeline`, `UPLOAD_BUCKET=assignmentgrader`,
   `PIPELINE_BUCKET=assignmentgrader`, `PIPELINE_ROOT=Data_warehouse`.

### C9. Test the whole pipeline end to end
1. On https://gradeant.netlify.app a **student uploads** an answer-sheet PDF for an assignment
   whose teacher files exist.
2. S3 shows `…/<aid>__<qp>/original_pdf/<student>.pdf`.
3. `/aws/lambda/gradeant-jsonifier` logs `jsonifier: answer sheet parsed`. This takes 1 to 3 minutes
   for a few pages. Then `Parsed_pdfs/JSON/<student>.json` appears.
4. That file triggers the grader within seconds. Continue with B10 step 1 onwards.
5. Both dead-letter queues stay at 0.

If the browser shows a **CORS error** on upload, the origin in C8.2 doesn't match the site
address exactly (no trailing `/`).

---

## Part D: monitoring (recommended, free tier)

1. **SNS → Topics → Create topic** (Standard) `gradeant-alerts` → **Create subscription** →
   Email → your address → confirm the email.
2. **CloudWatch → Alarms → Create alarm**, three times:
   - Lambda → By function name → `gradeant-grader` → **Errors**, Sum, 5 min, ≥ 5 → notify
     `gradeant-alerts`
   - the same for `gradeant-jsonifier`
   - SQS → `Grader-dlq` and `Jsonifier-dlq` → **ApproximateNumberOfMessagesVisible**, Max, ≥ 1 → notify

Ten alarms are free. To retry a dead-lettered message after fixing the cause: **SQS → the DLQ
→ Start DLQ redrive** → back to the source queue.

## Common errors

| Symptom (CloudWatch log) | Cause | Fix |
|---|---|---|
| `Runtime.ImportModuleError` | wrong runtime/arch, missing layer, wrong handler | Python 3.13 + arm64, deps layer first, handler name |
| `Event from unexpected bucket` | `PIPELINE_BUCKET` doesn't match | set `assignmentgrader` |
| `PIPELINE_BUCKET is not set` | env var missing | B3 / C3 |
| `AccessDenied` on S3 | inline policy missing or wrong prefix | B4 / C4 |
| `Question-paper schema not found` / `Question-paper PDF not found` | teacher files missing for that assignment | upload them to the §0 paths |
| `LLMError` / `429` | Gemini key wrong or rate-limited | check key and tier; lower Maximum concurrency |
| `Task timed out after 900 seconds` | very long PDF | jsonifier memory → 3008 MB (more CPU) |
| Messages pile up, nothing in logs | trigger disabled, or old Lambda still attached | B7 / C6 |

## Cost (MVP, about 50 students × 5 assignments a month)
Lambda, SQS, S3 requests and CloudWatch all stay inside the AWS free tier. S3 storage of a few GB
is about $0.05 to $0.15. **Gemini Flash is the main cost**, roughly a few cents per answer sheet
(read + grade), so about $5 to $15 a month at that volume.

## Updating later
1. `python build.py --verify` in `python-backend/`.
2. **Function code only:** function → Code → Upload from .zip → the new `gradeant-*.zip`.
3. **Shared code or prompts:** Layers → `gradeant-python-common` → **Create version** → upload the
   new zip. Then, on **both** functions: Layers → Edit → choose the new version.
4. **Dependencies:** the same for `gradeant-pipeline-deps`, through S3 (A1).

## Rollback
1. Disable the two new SQS triggers (`gradeant-grader`, `gradeant-jsonifier`).
2. Re-activate the old Lambdas' triggers.

The old Lambdas write to DynamoDB, so the grades they produce won't show in the hybrid web app,
which reads Supabase.
