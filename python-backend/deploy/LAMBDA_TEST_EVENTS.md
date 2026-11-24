# Lambda test events: grader and jsonifier

Console test events for the two pipeline Lambdas, with the result each one should give and what
every error code means. They use made-up ids (`deploy-test`, `nobody`), so they never change real
data. A console test is a direct invoke and doesn't go through SQS, so a failed test leaves nothing
in the queue or the dead-letter queue.

**How to run:** Lambda → function → **Test** tab → **Create new event** → name → paste JSON →
**Save** → **Test**. The green or red box shows the **Status**, the **Response** and the
**Log output**. Full logs are in CloudWatch → `/aws/lambda/<function name>`.

---

## How to read the result

| What you see | Code | Meaning |
|---|---|---|
| **Status: Succeeded**, response `{"batchItemFailures": []}` | **200** | Every message was handled. ✅ |
| **Status: Succeeded**, response lists `{"itemIdentifier": "…"}` | **200** (partial failure) | The function ran, but that message failed. In production SQS retries it, up to 3 times, then sends it to the DLQ. Some tests below expect this. |
| **Status: Failed**, `errorType` shown | **200** with `FunctionError: Unhandled` | The code raised an exception. A direct invoke (G4) is expected to do this. Anywhere else, read `errorMessage`. |
| `Runtime.ImportModuleError` | function error | Wrong handler, runtime or architecture, or layers in the wrong order. |
| `Runtime.HandlerNotFound` | function error | Handler name is wrong. It must be `grader.lambda_handler` / `jsonifier.lambda_handler`. |
| `Task timed out after 3.00 seconds` | function error | Timeout was left at the default. Set it to **15 min**. |
| `Runtime.OutOfMemory` / `signal: killed` | function error | Not enough memory. Use grader **1024 MB** and jsonifier **2048 MB**. |
| `ValueError: unsupported event shape` | function error | The test JSON isn't one of the shapes below. Check you copied it whole. |
| **403** `AccessDenied` (S3) | AWS error | The role has no S3 permission for `assignmentgrader/Data_warehouse/*`. |
| **404** `NoSuchKey` (S3) | AWS error | The file isn't there, but access works. Some tests expect this. |
| **401** `Invalid API key` (Supabase) | HTTP 401 | `SUPABASE_SERVICE_ROLE_KEY` is wrong. Use the legacy `service_role` key (`eyJ…`). |
| **403** `permission denied for schema gradeant` (Supabase) | HTTP 403 | You used the anon or publishable key instead of the service_role key. |
| **404** / **406** `PGRST106` (Supabase) | HTTP 404/406 | Supabase → Project Settings → API → **Exposed schemas** must include `gradeant`. |
| **429** (Supabase or Gemini) | HTTP 429 | Rate limited. The code retries; in production the message goes back to the queue. |
| **400** `API key not valid` (Gemini) | HTTP 400 | `GEMINI_API_KEY` is wrong. |
| **5xx** (Supabase or Gemini) | HTTP 5xx | Provider outage. The code retries; in production the message goes back to the queue. |
| `Event from unexpected bucket` | `PipelineInputError` | `PIPELINE_BUCKET` isn't `assignmentgrader`. |
| `PIPELINE_BUCKET is not set.` | `PipelineInputError` | Environment variable missing. |
| `SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required…` | `PgRestError` | One of the two Supabase variables is missing. |
| `Gemini API key not configured…` | `LLMError` | `GEMINI_API_KEY` is missing. |
| `KMS.AccessDeniedException` | AWS error | The queue or bucket uses a customer KMS key. Switch it to SSE-SQS / SSE-S3, or add `kms:Decrypt`. |

---

## Grader (`Grader_hybrid`)

### G1 `empty`: code and layers load
```json
{ "Records": [] }
```
- ✅ **200** `{"batchItemFailures": []}`
- ❌ `Runtime.ImportModuleError` / `HandlerNotFound` → see the table above.

### G2 `s3-test-event`: the message S3 sends when you create a notification
```json
{ "Records": [ { "messageId": "g2", "body": "{\"Service\":\"Amazon S3\",\"Event\":\"s3:TestEvent\",\"Bucket\":\"assignmentgrader\"}" } ] }
```
- ✅ **200** `{"batchItemFailures": []}` (acknowledged and ignored)

### G3 `skip-file`: bucket setting
```json
{ "Records": [ { "messageId": "g3", "body": "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/test/readme.txt\"}}}]}" } ] }
```
- ✅ **200** `{"batchItemFailures": []}` + log `grader: skipped (not a parsed answer sheet)`
- ❌ **200** with `g3` in failures + log `Event from unexpected bucket` → fix `PIPELINE_BUCKET`.

### G4 `supabase`: Supabase URL and key (direct invoke)
```json
{ "assignment_id": "deploy-test", "qp_id": "deploy-test", "student_id": "nobody" }
```
- ✅ **Status: Failed** with `PipelineInputError: No answer rows for student nobody on deploy-test/deploy-test`.
  This error means Supabase answered and found nothing.
- ❌ **401** → wrong key · **403** → not the service_role key · **404/406 PGRST106** → schema not
  exposed · timeout → wrong `SUPABASE_URL` (see the table above).

### G5 `start-grading-sqs`: the message Render sends on "Start grading"
```json
{ "Records": [ { "messageId": "g5", "body": "{\"assignment_id\":\"deploy-test\",\"qp_id\":\"deploy-test\",\"student_id\":\"nobody\",\"method\":\"ai\",\"persona\":\"balanced\",\"partial_credit\":true,\"rubric\":[]}" } ] }
```
- ✅ **200** `{"batchItemFailures": [{"itemIdentifier": "g5"}]}` + log
  `grader: message failed … No answer rows`. The failure is expected and proves the queue-message path works.
- ❌ any Supabase code → same fixes as G4.

### G6 `parsed-sheet-missing`: S3 permissions plus Supabase together
```json
{ "Records": [ { "messageId": "g6", "body": "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/students/assignments/deploy-test__deploy-test/Parsed_pdfs/JSON/nobody.json\"}}}]}" } ] }
```
- ✅ **200** with `g6` in failures + log `Question-paper schema not found`
  (S3 answered **404**, which means permissions are fine).
- ❌ **403 AccessDenied** → add the S3 statements to the grader role.

### G7 `bad-body`: a broken message goes to failures, it doesn't crash the function
```json
{ "Records": [ { "messageId": "g7", "body": "not json" } ] }
```
- ✅ **200** `{"batchItemFailures": [{"itemIdentifier": "g7"}]}` + log `SQS message body is not JSON`

---

## Jsonifier

### J1 `empty`
```json
{ "Records": [] }
```
- ✅ **200** `{"batchItemFailures": []}`

### J2 `skip-file`
```json
{ "Records": [ { "messageId": "j2", "body": "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/test/readme.txt\"}}}]}" } ] }
```
- ✅ **200** `{"batchItemFailures": []}` + log `jsonifier: skipped (not a student answer sheet)`
- ❌ `Event from unexpected bucket` → fix `PIPELINE_BUCKET`.

### J3 `s3-test-event`
```json
{ "Records": [ { "messageId": "j3", "body": "{\"Service\":\"Amazon S3\",\"Event\":\"s3:TestEvent\",\"Bucket\":\"assignmentgrader\"}" } ] }
```
- ✅ **200** `{"batchItemFailures": []}`

### J4 `pdf-missing`: S3 permissions
```json
{ "Records": [ { "messageId": "j4", "body": "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/students/assignments/deploy-test__deploy-test/original_pdf/nobody.pdf\"}}}]}" } ] }
```
- ✅ **200** with `j4` in failures + log `NoSuchKey` (**404**: the file isn't there, but access works)
- ❌ **403 AccessDenied** → the jsonifier role is missing the S3 policy or `s3:ListBucket`.

### J5 `bad-body`
```json
{ "Records": [ { "messageId": "j5", "body": "not json" } ] }
```
- ✅ **200** `{"batchItemFailures": [{"itemIdentifier": "j5"}]}`

---

## Checklist

| Test | Grader | Jsonifier |
|---|---|---|
| Code + layers load | G1 | J1 |
| S3 test event ignored | G2 | J3 |
| Bucket setting | G3 | J2 |
| Supabase key/URL | G4 | (no database) |
| "Start grading" message | G5 | — |
| S3 permissions | G6 | J4 |
| Bad message handled | G7 | J5 |

---

## Demo records: a real run with Gemini and Supabase

The folder `python-backend/deploy/demo-records/` mirrors the bucket. Every file goes to the **same
path** in `assignmentgrader`. Ids: assignment `demo-a1`, question paper `demo-qp1`, students
`demo-student1/2/3`. Total 10 marks: Q1 [2], Q2a [2], Q2b [2], Q3 [4]. There's no solution PDF, so
the grader uses the `teachers/QP_Rubric/...` files and grades without standard answers.

| File (S3 key, under `Data_warehouse/`) | Used by | What it is |
|---|---|---|
| `teachers/QP_Rubric/demo-a1__demo-qp1/QP_pdfs/demo-qp1.pdf` | jsonifier | question paper PDF |
| `teachers/QP_Rubric/demo-a1__demo-qp1/QP_schemas/QP_schemas_answers/demo-qp1_schema_answer.txt` | jsonifier | empty answer schema it fills in |
| `teachers/QP_Rubric/demo-a1__demo-qp1/QP_schemas/QP_schemas_json/demo-qp1_schema.json` | grader | questions + max marks (**required**) |
| `teachers/QP_Rubric/demo-a1__demo-qp1/rubric_jsons/demo-qp1_rubric.json` | grader | rubric per question (optional) |
| `students/assignments/demo-a1__demo-qp1/Parsed_pdfs/JSON/demo-student1.json` | grader | good answers, expect about 7–9/10 |
| `students/assignments/demo-a1__demo-qp1/Parsed_pdfs/JSON/demo-student2.json` | grader | weak answers (Q1 wrong, Q2a blank), expect about 4–6/10 |
| `students/assignments/demo-a1__demo-qp1/original_pdf/demo-student3.pdf` | jsonifier | typed answer sheet |

### Step 1: upload the teacher files first
S3 → `assignmentgrader` → **Upload** → drag in the `Data_warehouse` folder from `demo-records`
**without the `students` folder** (S3 keeps the folder paths). Uploading student files before
teacher files makes the run fail with `Question-paper schema not found`.

If the S3 → queue notifications are already active, uploading a student file **starts the
pipeline by itself**. That's fine. Otherwise use the test events below.

### Step 2: grader with a parsed sheet (Gemini + Supabase write)
Upload `demo-student1.json` to its path. Without a notification, run this test event (`demo-upload`):
```json
{ "Records": [ { "messageId": "d1", "body": "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/students/assignments/demo-a1__demo-qp1/Parsed_pdfs/JSON/demo-student1.json\"}}}]}" } ] }
```
- ✅ **200** `{"batchItemFailures": []}` + log `grader: answer sheet processed … questions: 4`
  - `GRADE_ON_UPLOAD=false` → `graded: false`. Supabase gets 4 ungraded rows.
  - `GRADE_ON_UPLOAD=true` → `graded: true, total_marks: …`. You get 4 graded rows plus 1 `summary` row.
- Running it a second time → log `skipped (answers already graded)` (only if graded). That protects
  student corrections and teacher edits.
- ❌ `Question-paper schema not found` → step 1 is missing or has the wrong path.
- ❌ Gemini **400/403** → `GEMINI_API_KEY` · **404** model → `GEMINI_MODEL` name.

### Step 3: "Start grading" (regrade with Gemini)
After step 2 (rows exist), run the message Render would send (`demo-start-grading`):
```json
{ "Records": [ { "messageId": "d2", "body": "{\"assignment_id\":\"demo-a1\",\"qp_id\":\"demo-qp1\",\"student_id\":\"demo-student1\",\"method\":\"ai\",\"persona\":\"balanced\",\"partial_credit\":true,\"rubric\":[]}" } ] }
```
- ✅ **200** `{"batchItemFailures": []}` + log `grader: regraded … questions: 4, total_marks: …`.
  Every row becomes `status: graded` and a `summary` row is written.
- Repeat steps 2 and 3 with `demo-student2` and compare: its total should be clearly lower.

### Step 4: jsonifier with a real PDF (Gemini, 1–3 min)
Upload `demo-student3.pdf` to `.../original_pdf/`. Without a notification, run (`demo-pdf`):
```json
{ "Records": [ { "messageId": "d3", "body": "{\"Records\":[{\"s3\":{\"bucket\":{\"name\":\"assignmentgrader\"},\"object\":{\"key\":\"Data_warehouse/students/assignments/demo-a1__demo-qp1/original_pdf/demo-student3.pdf\"}}}]}" } ] }
```
- ✅ **200** `{"batchItemFailures": []}` + log `jsonifier: answer sheet parsed`. S3 now has
  `Parsed_pdfs/text/`, `JSON_Seg/`, `JSON_Text/` `.txt` files and `Parsed_pdfs/JSON/demo-student3.json`.
  With the `.json` notification active, the grader starts by itself (check its log).
- Running it again → `skipped (already parsed)`.
- ❌ `Question-paper PDF not found` / `Question-paper answer schema not found` → step 1.
- ❌ `Task timed out` → timeout below 15 min · `Runtime.OutOfMemory` → memory below 2048 MB.

### Check the result in Supabase
Supabase → **SQL Editor**:
```sql
select student_id, "type", item->>'question_number' as q, item->>'subpart' as sub,
       item->>'status' as status, item->>'marks' as marks, item->>'total_marks' as total
from gradeant.assignment_grading
where assignment_id = 'demo-a1'
order by student_id, "type", q, sub;
```

### Clean up afterwards
```sql
delete from gradeant.assignment_grading where assignment_id = 'demo-a1';
```
S3 → `assignmentgrader` → delete the two `demo-a1__demo-qp1` folders (teacher and student). The
demo rows aren't linked to an assignment in the app, so they don't appear on any teacher or
student page.

---

**Full app test:** after the demo records work, do the end-to-end test in
`AWS_CONSOLE_PIPELINE.md` B10 / C9 with a real assignment created in https://gradeant.netlify.app:
a student uploads a PDF, then the teacher clicks **Start grading** in the app.
