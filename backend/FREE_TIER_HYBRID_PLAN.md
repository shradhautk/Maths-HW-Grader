# Free-tier hybrid deployment (Netlify + Render + Supabase + AWS pipeline): implementation plan

Status: **APPROVED 2026-11-18 with the user's changes in §12a** (written 2026-11-17). It supersedes the target in `SUPABASE_NETLIFY_PLAN.md`, which is kept for reference. That plan's Postgres schema, driver seam and migration design are reused here.
Work style: finish one check-in group, run its done-checks, report, wait for "go". Never commit. The all-AWS path stays working and remains the rollback.

## 0. Summary

1. **Frontend:** `web/` on **Netlify Free** as a static build, SPA fallback, API base URL pointing at Render.
2. **API:** the Node `backend/` on a **Render Free** web service as a plain long-running Express server, with the same API paths as today (no `/api` prefix: separate origin).
3. **Database:** **Supabase Free** Postgres replaces **all** DynamoDB tables. Node uses a `pg` driver. The Python grader Lambda uses **PostgREST over HTTPS**: httpx is already in its layer and it needs no new native wheel. Both sides use the same rows: key columns plus `item jsonb`.
4. **AWS keeps only the grading pipeline:** the S3 bucket for PDFs, S3 events → SQS → `gradeant-jsonifier` and `gradeant-grader` (Gemini **Flash** only), their DLQs, and the grader's async-invoke path for "Start grading". The Python API Lambdas are not deployed.
5. **One YAML line switches it:** `deployment: aws | hybrid` in `config/app.yaml`, read by Node, the Python pipeline and the web build. Secrets live only in Render, Netlify and Lambda env vars. A one-time, idempotent DynamoDB → Supabase script moves the data, with dry-run, counts, checksums and a reverse mode for rollback.

---

## 1. Findings: what changed since the first plan (verified 2026-11-17)

### 1.1 What the Python pipeline does with the database (only `gradeant-grader`; `gradeant-jsonifier` has `tables: []` in `lambdas.py`)

| Call site | Today (DynamoDB) | Needed on Supabase |
|---|---|---|
| `grading.student_rows(sid, assignment_id, qp_id, type?)` used by `functions/grader.py` (`progressed_rows`, `regrade`) and `pipeline/grading.write_student` | Query `student_id-index` + FilterExpression | `GET …/assignment_grading?select=item&student_id=eq.…&assignment_id=eq.…&qp_id=eq.…`, with `question_number`/`type` filtered in Python (same semantics as the Node driver) |
| `pipeline/grading.write_student` | `batch_writer` delete stale ids + put rows, then put summary **last** | RPC `gradeant.grading_replace_rows(rows, delete_ids)` (one transaction), then the same RPC with `[summary]` |
| `functions/grader.regrade` | `update_item` with `SET marks…, #s = :s[, confidence…, flag_reason…] REMOVE flag_resolved_at, flag_resolved_by, flag_resolution` (no condition) | RPC `gradeant.grading_patch(id, set jsonb, remove text[])`, which does `item = (item \|\| set) - remove` atomically |
| `functions/grader.regrade` summary | `put_item(summary)` last | `grading_replace_rows([summary], [])` |
| `dynamo.to_dynamo` (float → Decimal, drop `None`) | needed by DynamoDB | the Supabase path drops `None` only and keeps floats |

The Python API modules (`gradeant_common/{auth,ownership,…}`, `functions/{auth,grades,…}`) are not deployed in the hybrid. `dynamo.table()` will refuse with a clear error when `database: supabase`, so nothing can write to DynamoDB by accident.

### 1.2 Gemini models: both Flash, but the defaults need attention
- `gradeant_common/pipeline/llm.py`: `GRADER_MODEL` defaults to **`gemini-2.5-flash`** and `JSONIFIER_MODEL` to **`gemini-3-flash-preview`**. **Both are Flash models.**
- **`gemini-2.5-flash` is deprecated:** "Starting December 1, 2026, its requests route to `gemini-3.5-flash-lite`" ([Gemini pricing](https://aistudio.google.com/docs/pricing)). That is two weeks away, and the grading model would silently change, so pin a GA Flash model instead (open decision O3).
- `gemini-3-flash-preview` is a *preview* model. Preview models change and get retired, so pin a GA Flash model for parsing too (O3).

### 1.3 Gemini free tier: privacy and terms ([terms](https://aistudio.google.com/docs/terms), [pricing](https://aistudio.google.com/docs/pricing))
- **Unpaid Services:** content is used "to provide, improve, and develop Google products and services", and "human reviewers may read, annotate, and process your API input and output". Also: "**Do not submit sensitive, confidential, or personal information to the Unpaid Services.**" The pricing table's "Used to improve our products" column reads **Yes** (Free) and **No** (Paid).
- **EEA, Switzerland and UK:** developers must "use only Paid Services when making API Clients available to users in the European Economic Area" (Switzerland and the UK are also named).
- **Age:** "You may not offer API Clients … directed towards or … likely to be accessed by individuals under the age of 18." That clause applies to paid usage too, and it matters for a school product. This is a **legal question for the user, not something the plan can solve** (risk R1).
- **Consequence:** student answer sheets are personal data, so the free Gemini tier is not acceptable for real students. Recommendation O2: enable billing on the Gemini API project (paid Flash rates are pennies per paper, see §10). This is the one unavoidable cost.

### 1.4 Hosting constraints that change the design
- **Render Free blocks outbound SMTP:** "Render blocks outbound traffic on port 25 for all services", and **ports 465 and 587 are also blocked for Free web services** ([Render free](https://render.com/docs/free)). So `email: smtp` cannot work from a free Render service. The options are `log`, `ses` (the SES API is HTTPS on 443, so it is not blocked) or an HTTP email API (O5).
- **Render Free spins down** after 15 min without traffic, a wake-up takes "about one minute", there are 750 instance hours per workspace per month and no persistent disk. Render's own docs say "Do not use them for production applications" ([Render free](https://render.com/docs/free)).
- **Separate origins (Netlify ↔ Render):** no path collision, so the API keeps today's paths (`/auth/*`, `/workflow`, `/getgrades`, `/uploadpdf`, `/health`, …) with `API_BASE_PATH` empty, and CORS lists the Netlify origin.
- **Supabase direct connections are IPv6** unless the project has the IPv4 add-on, while "the shared pooler is IPv4-only on every plan" ([connecting](https://supabase.com/docs/guides/database/connecting-to-postgres)). Node on Render uses the **session pooler (port 5432)**, a good fit for a long-running server. Free Nano compute allows **60 direct connections / 200 pooler clients** ([compute](https://supabase.com/docs/guides/platform/compute-and-disk)).
- **PostgREST needs the schema exposed:** add `gradeant` to "Exposed schemas" and grant to roles; requests send `Accept-Profile` / `Content-Profile: gradeant` ([custom schemas](https://supabase.com/docs/guides/api/using-custom-schemas)). Supabase's page grants ALL to `anon` and `authenticated` too. **This plan deliberately grants only `service_role`**, and RLS stays on with no policies, so the public anon key can read nothing.
- **httpx is already in the pipeline layer**, as a dependency of google-genai. It shows up in `python-backend/dist/layer-gradeant-pipeline-deps.zip`. PostgREST therefore needs no new wheel; the plan still pins `httpx` explicitly (step 3.1).

### 1.5 What Node on Render still needs from AWS
- **Presigned S3 PUT URLs** for answer sheets, question papers and answer keys, in the **pipeline layout**. `UPLOAD_LAYOUT=pipeline` exists only in Python (`python-backend/functions/upload.py`), so Node must port it (step 4.1).
- **Profile photos:** `src/services/auth.service.js` presigns PUT/GET and does head/delete on `PROFILE_PHOTO_BUCKET` (S3; kept).
- **"Start grading":** `src/aws/lambda.js` sends an async `Invoke` (`InvocationType: 'Event'`) to `GRADING_FUNCTION_NAME`, and the grader already accepts that payload (§1.3 of `PIPELINE_PORT_PLAN.md`).
- **Optional:** SES `SendEmail` (O5).
- Node has no other AWS reads: it never fetches pipeline outputs from S3, and question text comes from `assignment_grading` rows.

---

## 2. Target architecture

```
  Browser ── https://<site>.netlify.app (Netlify Free: static web/dist, /* → /index.html)
     │
     │  fetch https://gradeant-api.onrender.com/...   (same paths as today; CORS = Netlify origin)
     ▼
  Render Free web service (Frankfurt) ── node index.js --mode=production (Express, long-running)
     │  pg pool (≤5) ── Supabase session pooler :5432 (IPv4)
     │  AWS SDK with IAM user "gradeant-render" (least privilege):
     │     ├─ presign s3:PutObject  → S3 pipeline bucket (eu-north-1)
     │     ├─ presign get/put/delete → profile-photos/*
     │     ├─ lambda:InvokeFunction (Event) → gradeant-grader   ("Start grading")
     │     └─ (optional) ses:SendEmail
     │  Gemini API (Flash, paid tier): rubric drafts + teacher assistant  (ai: gemini)
     ▼
  Supabase Free (Frankfurt, eu-central-1): Postgres schema gradeant (13 tables, pk + item jsonb)
     ▲   RLS on, no policies; only service_role/postgres may touch it
     │   PostgREST https://<ref>.supabase.co/rest/v1 (Accept/Content-Profile: gradeant)
     │
  AWS eu-north-1 (pipeline only)
     Browser PUT ──► S3 bucket ── S3 event ──► SQS jsonifier queue ──► λ gradeant-jsonifier (Gemini Flash)
                                                  (writes Parsed_pdfs/JSON/<student>.json)
                     S3 event (JSON) ──► SQS grader queue ──► λ gradeant-grader (Gemini Flash)
     Render ── async Invoke (regrade payload) ─────────────────────► λ gradeant-grader
                     failures → each queue's DLQ / async on-failure destination
     gradeant-grader ── httpx + service-role key ──► Supabase PostgREST (rows, summary)
```

---

## 3. Config design

### 3.1 `config/app.yaml` (appended; existing sections unchanged)

```yaml
# ---------------------------------------------------------------------------
#  Deployment target: the ONE line to change. Secrets never go here; each
#  provider reads its keys from environment variables (backend/.env.example,
#  python-backend/README.md). Moving existing data is a separate one-time step:
#  backend/scripts/migrate-dynamodb-to-supabase.mjs.
#  The Python pipeline Lambdas read this file from their common layer:
#  rebuild + redeploy the layer after changing it (or set DATABASE_PROVIDER).
# ---------------------------------------------------------------------------
deployment: aws                      # aws | hybrid

deployments:
  aws:                               # today: everything on AWS
    providers:
      database: dynamodb             # dynamodb | supabase   (Node API AND Python grader)
      ai: bedrock                    # bedrock | gemini | off
      email: ses                     # ses | smtp | log
      secrets: aws                   # aws | env
  hybrid:                            # Netlify (web) + Render (API) + Supabase (DB) + AWS (pipeline)
    providers:
      database: supabase
      ai: gemini
      email: log                     # see open decision O5 (smtp is blocked on Render Free)
      secrets: env
    urls:                            # merged over urls.production
      web: https://gradeant.netlify.app
      api: https://gradeant-api.onrender.com
      upload: ""
      corsOrigins: [https://gradeant.netlify.app, https://grade-ant.com, https://www.grade-ant.com]
```

Storage (S3) and the grading hand-off (Lambda invoke) are AWS in both targets, so they are not providers. The existing `STORAGE_DRIVER` / `GRADING_DRIVER` env behaviour stays as it is.

**Validation** (Node `validateAppConfig`; Python `config._validate` gets the matching subset):
- `deployment` is missing (meaning `aws` with the providers above, so old YAML and the parity fixtures are unchanged) or names a key of `deployments`.
- Provider values must be in their lists.
- `urls` overrides are limited to `web/api/upload/corsOrigins`.
- `email: smtp` together with `deployment: hybrid` is a **warning**, not an error ("Render Free blocks SMTP ports").

**Driver resolution** (Node `config/index.js`, same precedence rule as plan 1): an explicit `*_DRIVER` env var wins, then (in production mode) `deployments[deployment].providers`, then the dev default (`memory`, `canned`, `log`).

| Provider | Node `config.drivers` | Python |
|---|---|---|
| database dynamodb / supabase | `db`: `dynamodb` / `postgres` (tests: `pglite`) | `config.database()` returns `dynamodb` / `supabase` (env `DATABASE_PROVIDER` overrides YAML) |
| ai bedrock / gemini / off | `ai`: `bedrock` / `gemini` / `canned` | – (the pipeline always uses Gemini) |
| email ses / smtp / log | `mail`: `ses` / `smtp` / `log` | – |
| secrets aws / env | `secrets`: `aws` / `env` | `secrets._resolve` already prefers the plain env value over the ARN (no change) |

Step 1.3 deletes the `*_DRIVER=` lines from `backend/.env.production`, because file values would otherwise override the YAML. `web/vite.config.ts` applies the same `urls` merge for production builds.

### 3.2 Env vars (🔒 = secret; set in each host's dashboard, never in YAML or committed files)

| Where | Variables |
|---|---|
| **Render** (Node) | `APP_ENV=production`, `HOST=0.0.0.0` (Render routes to `PORT`, which it sets), 🔒`SUPABASE_DB_URL` (session pooler `…pooler.supabase.com:5432/postgres?sslmode=require`), `PG_POOL_MAX=5`, 🔒`JWT_SECRET`, 🔒`SESSION_TOKEN_SECRET`, `AWS_DATA_REGION=eu-north-1`, 🔒`AWS_ACCESS_KEY_ID`, 🔒`AWS_SECRET_ACCESS_KEY` (IAM user §6.3), `UPLOAD_BUCKET`, `UPLOAD_LAYOUT=pipeline`, `PIPELINE_ROOT=Data_warehouse`, `PROFILE_PHOTO_BUCKET`, `GRADING_FUNCTION_NAME=gradeant-grader`, 🔒`GEMINI_API_KEY`, `GEMINI_MODEL` (O3), `SES_SENDER_EMAIL` (if `email: ses`) |
| **Netlify** (build only) | `NODE_VERSION=22`. Nothing secret: the API URL comes from `app.yaml` |
| **Lambda `gradeant-grader`** | `DATABASE_PROVIDER=supabase` (or via YAML), `SUPABASE_URL`, 🔒`SUPABASE_SERVICE_ROLE_KEY`, 🔒`GEMINI_API_KEY` (plain env instead of `GEMINI_API_KEY_ARN`), `GRADER_MODEL`, `PIPELINE_BUCKET`, `PIPELINE_ROOT`, `GRADE_ON_UPLOAD`, … (unchanged) |
| **Lambda `gradeant-jsonifier`** | 🔒`GEMINI_API_KEY`, `JSONIFIER_MODEL`, `PIPELINE_BUCKET`, `PIPELINE_ROOT` |

**Secrets on Lambda:**
- Lambda env vars are encrypted at rest with the AWS-managed KMS key, at no extra cost, so the free MVP needs no Secrets Manager. Secrets Manager costs $0.40 per secret per month (figure from the brief; **unverified** here).
- The trade-off is that anyone with `lambda:GetFunctionConfiguration` sees the values, so keep the AWS account's IAM users few.
- In `deploy/generate_template.py` these become `NoEcho` parameters. `gradeant_common/secrets.py` already reads `GEMINI_API_KEY` before `GEMINI_API_KEY_ARN`, so no code change is needed.

---

## 4. Postgres schema (reused from plan 1, plus RPC functions and PostgREST grants)

**Tables:** unchanged from `SUPABASE_NETLIFY_PLAN.md` §4, as `backend/sql/001_schema.sql`. Schema `gradeant` has 13 tables, each with key columns plus `item jsonb`. On `assignment_grading` the columns `student_id`, `"type"`, `assignment_id` and `qp_id` are generated from the item and indexed. `assignments` has a generated, indexed `teacher_id`; `submissions` has the composite key and a `student_id` index. RLS is enabled on every table with no policies, and anon/authenticated are revoked.

**New `backend/sql/002_pipeline_api.sql`** (used by the Python grader; Node uses tables directly):

```sql
-- Replace/insert grading rows and delete stale ones in ONE transaction.
create or replace function gradeant.grading_replace_rows(p_rows jsonb, p_delete text[] default '{}')
returns integer language plpgsql security invoker set search_path = '' as $$
begin
  delete from gradeant.assignment_grading where evaluation_id = any(p_delete);
  insert into gradeant.assignment_grading (evaluation_id, item)
    select r->>'evaluation_id', r from jsonb_array_elements(p_rows) as r
  on conflict (evaluation_id) do update set item = excluded.item;
  return jsonb_array_length(p_rows);
end $$;

-- DynamoDB "SET a=:v,… REMOVE x,y" without a condition (an upsert, like UpdateItem).
create or replace function gradeant.grading_patch(p_id text, p_set jsonb, p_remove text[] default '{}')
returns void language sql security invoker set search_path = '' as $$
  insert into gradeant.assignment_grading (evaluation_id, item)
    values (p_id, jsonb_build_object('evaluation_id', p_id) || p_set)
  on conflict (evaluation_id) do update
    set item = (gradeant.assignment_grading.item || p_set) - p_remove;
$$;

-- Only the server roles may use the schema through PostgREST.
grant usage on schema gradeant to service_role;
grant select, insert, update, delete on gradeant.assignment_grading to service_role;
revoke execute on function gradeant.grading_replace_rows(jsonb, text[]), gradeant.grading_patch(text, jsonb, text[])
  from public, anon, authenticated;
grant execute on function gradeant.grading_replace_rows(jsonb, text[]), gradeant.grading_patch(text, jsonb, text[])
  to service_role;
```

Dashboard step: API settings → **Exposed schemas** += `gradeant`. Whether `service_role` bypasses RLS through PostgREST is documented Supabase behaviour but **unverified** here; checked in step 3.5.

Both sides read the same rows. The Node driver reads `item` and the Python path reads `select=item`. Both write whole items with `None`/`undefined` removed. Numbers are JSON numbers on both sides, and Python's `from_dynamo` passes plain ints and floats through unchanged.

---

## 5. Adapters

### 5.1 Node (from plan 1 §5.1; only the database part is reused)
- **`src/data/`:** `index.js` (facade), `tables.js`, `dynamodb.js` (today's code moved verbatim), `expressions.js` (the §1.3 grammar of plan 1) and `postgres.js` (`pg` or PGlite). `src/aws/dynamodb.js` becomes a re-export shim, and the call sites change to `table('users')` and the like.
- **Postgres pool for Render:** `max: PG_POOL_MAX (5)`, `idleTimeoutMillis: 30000`, `ssl` required, on the session pooler.
- **Correction to plan 1 §1.3:** the result of `p <> :v` when `p` is *absent* must be **copied from DynamoDB, not assumed**. The differential test (step 2.4) runs the same filter on dynalite and on PGlite, and the evaluator follows what dynalite does. Detail rows without a `type` attribute depend on it (`studentRows(..., type:'notSummary')`).
- **AI (`src/aws/bedrock.js` `askClaude`):** add a `gemini` branch using `@google/genai` (new dependency) with model `GEMINI_MODEL` (Flash, O3) and the same contract: text, or `null` on a block. `outputFormat.schema` maps to `config.responseMimeType: 'application/json'` + `responseJsonSchema` (field name **unverified**; fall back to the existing JSON-in-prompt path). `effort` is ignored.
  - `ai: off` maps to `canned`. Rubric drafting then keeps the default rubric (the teacher writes the criteria), and the assistant replies "not connected". That is today's dev behaviour.
- **Email:** `log` keeps the existing driver; `ses` keeps the existing code with the IAM user. `smtp` (nodemailer) is implemented but documented as blocked on Render Free.
- **Upload (step 4.1):** port `UPLOAD_LAYOUT=pipeline` from `python-backend/functions/upload.py`. The key layout, server-chosen names (`<studentId>.pdf`, `<qp>.pdf`, `<aid>_<qp>_answer.pdf`), metadata unwrap and ownership checks match Python's, and the tests mirror `python-backend/tests/test_upload_pipeline.py`'s presign cases.
- **Health:** `checks.database` on Postgres (`select 1`) and `checks.dynamodb` on AWS (unchanged).

### 5.2 Python pipeline (grader only)
- **New module `gradeant_common/pg_rest.py`** (~120 lines, httpx only):
  - `_client()`: a cached `httpx.Client(base_url=f'{SUPABASE_URL}/rest/v1', headers={'apikey': key, 'Authorization': f'Bearer {key}', 'Accept-Profile': 'gradeant', 'Content-Profile': 'gradeant'}, timeout=15)`.
  - Retries 3× with backoff on 429/5xx/connect errors, and raises `PgRestError` otherwise (the SQS message fails and is retried).
  - `select(table, filters: dict, select='item')` uses `col=eq.<value>` params and follows pagination with `Range`/`limit`+`offset` until a short page. PostgREST caps rows per response, so the default `max_rows` must be **verified**.
  - `rpc(name, args)` is `POST /rpc/<name>`.
  - `set_transport(t)` lets tests inject `httpx.MockTransport`.
- **`gradeant_common/grading.py`** branches on `config.database()`:
  - `student_rows(...)` on supabase: `select('assignment_grading', {student_id, assignment_id?, qp_id?})`, then `question_number`/`type` filtered in Python with the same rule as the Node evaluator.
  - New `patch_row(evaluation_id, set_fields, remove=())`. On dynamodb it builds `SET #k0 = :v0, … [REMOVE …]` and calls `update_item` (equivalent to today's expression). On supabase it calls `rpc('grading_patch', …)`.
  - New `replace_rows(rows, delete_ids)`. On dynamodb it is today's `batch_writer` block, moved verbatim. On supabase it calls `rpc('grading_replace_rows', …)`.
  - `summary_rows` and `assignment_rows` are not used by the pipeline; on supabase they raise `NotImplementedError`.
- **`pipeline/grading.write_student`** calls `replace_rows(rows, stale)` and then `replace_rows([summary], [])`, so the summary is still written **last**. **`functions/grader.regrade`** calls `patch_row(...)` and `replace_rows([summary], [])`. Rows pass through `dynamo.to_dynamo` only on the DynamoDB path; on supabase they pass through `clean_none()`.
- **`gradeant_common/dynamo.table(key)`** raises `RuntimeError('DynamoDB is not used when database: supabase')` when supabase is selected, so anything left unconverted fails loudly.
- **Config:** `config.database()` returns `DATABASE_PROVIDER` env, else the YAML `deployments[deployment].providers.database`, else `dynamodb`.
- **Packaging:**
  - `pipeline-requirements.txt` pins `httpx` explicitly, to the version already in the layer.
  - `lambdas.py` grader entry: the env gains `DATABASE_PROVIDER`, `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (NoEcho param), and the `dynamodb` IAM statement is dropped when the template is built with `--database supabase`.
  - `deploy/generate_template.py` gains `--only grader,jsonifier` for a **pipeline-only stack** that holds the two workers, their SQS queues and DLQs, the S3 notification, the async on-failure destination and their two layers.
  - `check_permissions.py` must still pass after a graph refresh. pg_rest calls no AWS service, so the grader's needed AWS services become {s3} plus secrets only when an ARN is used.

---

## 6. Setup per host

### 6.1 Netlify Free (web)
`web/netlify.toml`:
```toml
[build]
  base = "web"
  command = "npm ci && npm run build"   # vite reads ../config/app.yaml (repo is fully cloned)
  publish = "dist"
[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```
There are no functions and no secrets. The production API URL comes from `deployments.hybrid.urls.api`. Use the default `*.netlify.app` domain, or point `app.gradeant.io` at it at cut-over.

### 6.2 Render Free (API)
`render.yaml` (Blueprint; fields per [blueprint spec](https://render.com/docs/blueprint-spec)):
```yaml
services:
  - type: web
    name: gradeant-api
    runtime: node
    plan: free
    region: frankfurt
    rootDir: backend
    buildCommand: npm ci --omit=dev
    startCommand: node index.js --mode=production
    healthCheckPath: /health
    envVars:
      - { key: APP_ENV, value: production }
      - { key: HOST, value: 0.0.0.0 }
      - { key: NODE_VERSION, value: "22" }
      - { key: SUPABASE_DB_URL, sync: false }
      - { key: JWT_SECRET, sync: false }
      # … every 🔒 variable from §3.2 with sync: false
```
- **Config file location:** the Blueprint spec doesn't say whether files outside `rootDir` are present at runtime (**unverified**). Step 4.4 checks that `../config/app.yaml` resolves. The fallback is a `buildCommand` that copies it in (`cp ../config/app.yaml config/app.yaml`), which `appConfig.findAppConfigFile` already looks for as `<backend>/config/app.yaml`, or `APP_CONFIG_FILE`.
- **Cold starts:** expect about a minute after 15 idle minutes. Mitigation (O6): an external uptime pinger on `/health/ready` every 10–14 min (for example the free GitHub Actions cron or a free uptime monitor). One always-on service uses about 744 h, under the 750 h workspace allowance. The same ping also keeps the Supabase project from pausing (whether a query counts as "activity" is **unverified**).

### 6.3 AWS (pipeline only, eu-north-1)
- **Deploy:** `python build.py --verify`, then `deploy/generate_template.py --only grader,jsonifier --database supabase`, then `sam deploy` (existing flow, pipeline-only stack). The existing S3 → SQS notifications, queues and DLQs keep working.
- **IAM user `gradeant-render`** (access keys go into Render env only; rotate every 90 days):
```json
{ "Version": "2012-10-17", "Statement": [
  { "Sid": "PresignPipelineUploads", "Effect": "Allow", "Action": "s3:PutObject", "Resource": [
      "arn:aws:s3:::<PIPELINE_BUCKET>/Data_warehouse/students/assignments/*/original_pdf/*",
      "arn:aws:s3:::<PIPELINE_BUCKET>/Data_warehouse/teachers/QP_Rubric/*/QP_pdfs/*",
      "arn:aws:s3:::<PIPELINE_BUCKET>/Data_warehouse/teachers/QPRubricWithAnswers/*/solution_pdfs/*" ] },
  { "Sid": "ProfilePhotos", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
    "Resource": "arn:aws:s3:::<PROFILE_PHOTO_BUCKET>/profile-photos/*" },
  { "Sid": "StartGrading", "Effect": "Allow", "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:eu-north-1:<ACCOUNT>:function:gradeant-grader" }
]}
```
  - Add `ses:SendEmail` on the sender identity only if O5 = ses.
  - `HeadObject` is covered by `s3:GetObject`. Without `s3:ListBucket`, a missing photo heads to 403 instead of 404. The photo `confirm` path treats both as "not uploaded" (test this).
  - Pin the exact layout prefixes to `PIPELINE_ROOT` (`Data_warehouse`).
- **Why async invoke instead of an SQS message for "Start grading":** it is zero new code. `src/aws/lambda.js` already does it, the grader already accepts the regrade payload, and Lambda async invokes retry twice and then go to the existing on-failure destination. An SQS message would need a new body shape in `functions/grader.py` (it expects S3-event bodies) and an extra queue. Option recorded in O7.
- **CloudWatch Logs:** set 14-day retention on the two log groups to keep storage near zero.

### 6.4 Supabase Free
- Create the project in **Frankfurt (eu-central-1)**, the same region as Render. A second free project serves as staging; two free projects are allowed ([billing](https://supabase.com/docs/guides/platform/billing-on-supabase)).
- Run `node backend/scripts/sql-apply.mjs` for `001_schema.sql` and `002_pipeline_api.sql`, expose `gradeant` in the API settings, and keep the anon key unused (the web never talks to Supabase).
- The Free plan has **no backups** ([pricing](https://supabase.com/pricing)). Mitigation O8: a weekly `pg_dump` via GitHub Actions cron to an encrypted object in the S3 bucket, using a separate key with `s3:PutObject` on `backups/*` only.

### 6.5 Regions
| Hop | Choice | Why |
|---|---|---|
| Render ↔ Supabase | both **Frankfurt** | the API makes many sequential queries per request, so this hop dominates latency |
| Lambda (eu-north-1) ↔ Supabase (Frankfurt) | cross-region HTTPS (~25–30 ms, **unverified**) | a few calls per paper; moving the pipeline is out of scope |
| Browser ↔ Render | Frankfurt | Render's EU region (`oregon/ohio/virginia/frankfurt/singapore`) |
| Netlify | global CDN | static only |

---

## 7. Data migration (DynamoDB → Supabase)

**Script:** `backend/scripts/migrate-dynamodb-to-supabase.mjs` (the table part of plan 1 §8; there is no storage part, because S3 stays):

```
node scripts/migrate-dynamodb-to-supabase.mjs [--dry-run] [--tables all|users,grading,…] [--verify-only] [--reverse]
```
- **Source:** the operator's AWS profile (read-only: `dynamodb:Scan` on the 13 tables). **Target:** `SUPABASE_DB_URL` (session pooler or direct).
- Scan with pagination and unmarshall exactly as the app does. Sets become sorted arrays and Binary becomes base64, both counted in the report. Upsert in batches of 500: `insert … on conflict (pk) do update set item = excluded.item`. This is idempotent.
- **Verification:** per-table counts plus a per-item sha256 of canonical JSON (sorted keys) on each side, written to `backend/.data/migration-report.json`. The script exits non-zero on any mismatch.
- **`--reverse`:** Postgres → DynamoDB `PutItem`, for rollback after writes.
- **Size check:** Free is 500 MB per database ([billing](https://supabase.com/docs/guides/platform/billing-on-supabase)). The dry-run prints the total item bytes, and the script stops if the projected size exceeds 400 MB.

**Cut-over runbook** (goes into `backend/deploy/FREE_TIER_HYBRID.md`):
1. Create staging: a second Supabase project, a Render service and a Netlify site with `deployment: hybrid` on a branch, plus a *staging* pipeline-only stack (or the production one pointed at staging Supabase during the rehearsal). Smoke-test it: signup, login, create assignment, upload student and teacher PDFs, jsonifier → grader rows, correction, Start grading, publish, admin invite (email logged).
2. Rehearse: run the migration and `--verify-only` against staging with a copy of production data.
3. Freeze: set `permissions.upload.*: []` on the current AWS API and deploy it, wait for both SQS queues to be empty and the DLQs to be checked, then run the migration and `--verify-only` (0 mismatches required).
4. Switch:
   - Redeploy the grader with `DATABASE_PROVIDER=supabase` and the Supabase keys.
   - Deploy Render and Netlify with `deployment: hybrid` merged.
   - Point `app.gradeant.io` at Netlify (optional).
   - Re-enable uploads.
5. Watch for a week: Render logs, `/health/ready`, grader errors and DLQ depth, Supabase DB size.

**Rollback:**
- **Before writes on Supabase:** revert the grader env to `dynamodb` and point users back to the AWS web/API.
- **After writes:** run `--reverse`, verify, then do the same.
- Keep the DynamoDB tables (on-demand, near $0 when idle) for at least 30 days.

---

## 8. Steps (5 check-in groups)

Standard done-checks for every group:
- `cd backend && npm test`: baseline 80 tests with one pre-existing Windows SIGTERM failure.
- `npm run check:parity`: **0 differences** on the AWS/memory path.
- `npm run check:lambdas`.
- `cd python-backend && python -m pytest`: all green.
- `python build.py --verify`.

### Group 1: config switch + Node database seam (no behaviour change)
| # | Step | Files |
|---|---|---|
| 1.1 | `deployment`/`deployments` parse, validate, URL merge (§3.1) | `backend/src/config/appConfig.js`, `config/app.yaml` (append, `deployment: aws`) |
| 1.2 | Driver resolution, `ai: gemini\|off`, `email: log`, `secrets: env`, env-var validation per provider | `backend/src/config/index.js` |
| 1.3 | Remove `*_DRIVER` lines; document vars | `backend/.env.production`, `backend/.env.example`, `backend/README.md` |
| 1.4 | `src/data/` seam (DynamoDB code moved verbatim; `table(key)` call sites) | `backend/src/data/*`, `src/aws/dynamodb.js`, `src/repositories/*.js`, `src/dev/seed.js` |
| 1.5 | Python `config.database()` + YAML subset validation | `python-backend/gradeant_common/config.py`, `tests/test_config_providers.py` (new) |
| 1.6 | Web build URL merge | `web/vite.config.ts` |

Done-checks: the standard set, plus new appConfig tests (missing block means aws; an unknown provider is rejected; the hybrid URL merge works; the smtp+hybrid warning appears), plus the `web` build with `deployment: aws` embedding a byte-identical `__APP_CONFIG__`.

### Group 2: Node Postgres driver + schema
| # | Step | Files |
|---|---|---|
| 2.1 | Expression parser/evaluator | `backend/src/data/expressions.js` |
| 2.2 | SQL + apply script | `backend/sql/001_schema.sql`, `backend/sql/002_pipeline_api.sql`, `backend/scripts/sql-apply.mjs` |
| 2.3 | `postgres`/`pglite` driver; `npm run test:pg` | `backend/src/data/postgres.js`, `backend/package.json` (+`pg`, dev +`@electric-sql/pglite`), `test/helpers.js` |
| 2.4 | Differential test dynalite ↔ PGlite (every repository op, CCF, upsert-update, `if_not_exists` null vs absent, `<>` on absent, REMOVE, ordering) | `backend/test/dataDrivers.test.js` |
| 2.5 | **Shared-row test:** call `gradeant.grading_replace_rows` / `grading_patch` in PGlite with rows shaped like the Python pipeline's, then read them back through the Node driver and the workflow services (`listSubmissions`, `getGradingRun`, `getsingle grade`) | `backend/test/sharedRows.test.js` |

Done-checks: the standard set; `npm run test:pg` passes the same count as memory; the differential and shared-row tests are green; the expression-coverage test finds no `UnsupportedExpression`.

### Group 3: Python grader on Supabase
| # | Step | Files |
|---|---|---|
| 3.1 | `pg_rest.py`; pin `httpx` | `python-backend/gradeant_common/pg_rest.py`, `pipeline-requirements.txt`, `requirements-dev.txt` |
| 3.2 | `grading.py` provider branch: `student_rows`, `patch_row`, `replace_rows`; `dynamo.table` guard | `gradeant_common/grading.py`, `gradeant_common/dynamo.py` |
| 3.3 | Callers use `replace_rows` / `patch_row` | `gradeant_common/pipeline/grading.py`, `functions/grader.py` |
| 3.4 | Tests: the existing moto tests are unchanged (DynamoDB path). A new `tests/test_pipeline_supabase.py` uses `httpx.MockTransport` to assert exact PostgREST requests (paths, `eq.` filters, profile headers, RPC bodies, pagination, retries), runs `test_end_to_end_upload_to_summary` again with `DATABASE_PROVIDER=supabase` against an in-memory fake of the two RPCs, and checks that a PostgREST 5xx after the retries fails the SQS message (`batchItemFailures`) | `python-backend/tests/test_pipeline_supabase.py` |
| 3.5 | Optional live test against the staging Supabase project (`SUPABASE_TEST_URL`/`_KEY`; skipped otherwise): write, read, patch, and **anon key denied** | same file |
| 3.6 | Packaging: grader env/params, `--only` pipeline stack, `--database supabase` drops the DynamoDB IAM; README + DEPLOYMENT section | `lambdas.py`, `deploy/generate_template.py`, `deploy/template.yaml` (regenerated), `README.md`, `DEPLOYMENT.md` |

Done-checks:
- The standard set, including `python build.py --verify`, which unpacks the grader with its layers and imports `httpx` and `gradeant_common.pg_rest`.
- `tests/test_build_registry.py` green, and `check_permissions.py` green after the documented graph refresh.
- The generated pipeline-only template contains exactly the two workers and their queues.

### Group 4: Render + Netlify + Node hybrid features
| # | Step | Files |
|---|---|---|
| 4.1 | `UPLOAD_LAYOUT=pipeline` in Node `/uploadpdf` (port of Python's) | `backend/src/services/upload.service.js`, `src/config/index.js`, `backend/test/upload.test.js` |
| 4.2 | `ai: gemini` branch (`@google/genai`, Flash) | `backend/src/aws/bedrock.js`, `backend/package.json`, `backend/test/bedrock.test.js` (+gemini cases with a fake client) |
| 4.3 | Health `database` check; Render binds `HOST=0.0.0.0` | `src/controllers/health.controller.js` |
| 4.4 | `render.yaml`, `web/netlify.toml`; deploy staging; confirm the config-file location on Render | `render.yaml`, `web/netlify.toml` |
| 4.5 | IAM user + policy; staging smoke list (§7 step 1); confirm a browser PUT to the presigned URL from the Netlify origin (S3 bucket **CORS** must allow the Netlify origin for `PUT`; update the bucket CORS) | AWS console/CLI; `backend/deploy/FREE_TIER_HYBRID.md` |

Done-checks:
- The standard set.
- The staging smoke list passes end to end: upload → jsonifier → grader writes rows in Supabase → the web shows them → Start grading → regrade → publish.
- Measure and record the cold-start time once.
- `/health/ready` is 200 with `database` ok.

### Group 5: migration + cut-over
| # | Step | Files |
|---|---|---|
| 5.1 | Migration script (forward, verify, reverse, size guard) | `backend/scripts/migrate-dynamodb-to-supabase.mjs` |
| 5.2 | Tests: dynalite → PGlite; a re-run changes 0 rows; a tampered item is detected; reverse restores it | `backend/test/migrate.test.js` |
| 5.3 | Runbook | `backend/deploy/FREE_TIER_HYBRID.md`, link in `backend/README.md` |
| 5.4 | Rehearsal on staging with a copy of production data; then cut-over per §7 | – |

Done-checks: the standard set, a rehearsal report with 0 mismatches, and the user's approval of the runbook before production cut-over.

---

## 9. Free-tier limits and their effect

| Service | Limit (source) | Effect on GradeAnt | Mitigation |
|---|---|---|---|
| **Supabase Free** | 500 MB DB, 1 GB storage, 5 GB egress, 2 free projects ([billing](https://supabase.com/docs/guides/platform/billing-on-supabase)); "paused after 1 week of inactivity", no backups ([pricing](https://supabase.com/pricing)); Nano: 60 direct / 200 pooler connections ([compute](https://supabase.com/docs/guides/platform/compute-and-disk)) | Rough estimate (**unverified**): ~20 rows × ~5 KB per paper ≈ 100 KB, so about 4–5k papers fit. A pause takes the API down until it is restored; data could be lost without backups | Size guard + monitoring; pinger (O6); weekly `pg_dump` (O8); Pro ($25/mo) when it becomes real |
| **Render Free** | spins down after 15 min idle, ~1 min wake-up, 750 h/workspace/month, no persistent disk, SMTP ports blocked, may restart at any time, "do not use for production" ([free](https://render.com/docs/free)) | The first request after idle takes about a minute (login looks broken). Email can't use SMTP. In-memory state (contact-form rate limiter) resets | Pinger (O6); `email: log` or `ses` (O5); the app is stateless, so a restart is harmless |
| **Netlify Free** | 300 credits/month; bandwidth 20 credits/GB, 2 credits per 10k requests ([pricing](https://www.netlify.com/pricing/)); commercial projects allowed per Netlify's announcement ([blog](https://www.netlify.com/blog/introducing-netlify-free-plan/)); ~15 credits per production deploy (third-party figure, **unverified**) | A static SPA of a few MB, so 300 credits is roughly 10 GB of bandwidth, or fewer if there are many deploys | Deploy only from `main`; watch credits |
| **AWS Lambda** | 1M requests + 400,000 GB-s per month ([pricing](https://aws.amazon.com/lambda/pricing/)); whether it is always free under the new model is **unverified** | Per paper: jsonifier 2 GB × ~60 s + grader 1 GB × ~60 s ≈ 180 GB-s (**estimate**), so about 2,200 papers/month free | Fine for an MVP |
| **AWS SQS** | "All customers can make 1 million Amazon SQS requests for free each month" ([pricing](https://aws.amazon.com/sqs/pricing/)) | The Lambda event-source mappings **long-poll even when idle**. Two queues may use a large share of 1M requests a month (**unverified**; check Billing after week 1) | Cents if exceeded |
| **AWS S3** | accounts created after 2025-07-15 get credit-based free tier ($200, 6-month free plan) ([S3 pricing](https://aws.amazon.com/s3/pricing/), [free tier](https://aws.amazon.com/free/)); the existing production account is likely past its free tier (**unverified**) | A few GB of PDFs | Cents/month |
| **Lambda env secrets** | AWS-managed KMS key encryption at no charge (**unverified** wording) | Avoids Secrets Manager ($0.40/secret/month, per the brief, **unverified**) | – |
| **Gemini API Free** | free-tier rate limits are not listed on the docs page; check the AI Studio rate-limit page (**unverified**). Free content **is** used to improve Google products, with human review; "do not submit … personal information"; EEA/UK/CH must use Paid ([terms](https://aistudio.google.com/docs/terms)) | **Not usable for real student data** | Paid tier (O2) |

## 10. Cost (MVP)

| Item | Monthly |
|---|---|
| Netlify Free | $0 |
| Render Free | $0 |
| Supabase Free | $0 |
| AWS Lambda + SQS (within free allowances) | $0 to < $1 |
| AWS S3 storage + requests (few GB) | < $1 |
| CloudWatch Logs (14-day retention, low volume) | ~$0 (**unverified**) |
| DynamoDB kept 30 days for rollback (on-demand, idle) | ~$0 storage cents |
| **Gemini paid tier (recommended)**: `gemini-2.5-flash` $0.30 in / $2.50 out per 1M tokens; `gemini-3-flash-preview` $0.50 / $3.00; GA `gemini-3.6-flash` $0.75 / $3.75 until 2026-12-31, then $1.50 / $7.50 ([pricing](https://aistudio.google.com/docs/pricing)) | Estimate (**unverified** token counts): ~30k in + 8k out per paper across parse + grade ≈ $0.02–0.06 per paper. **500 papers ≈ $10–30** |
| **Total** | **≈ $0 platform + Gemini usage (~$10–30 per 500 papers)** |

---

## 11. Risks and rollback

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Gemini API terms vs a school product:** clients "directed towards … individuals under the age of 18" are not allowed; the free tier forbids personal data; EEA/UK/CH need Paid | **The user must review this** (legal). Use the paid tier. Do not process real student papers until it is cleared |
| R2 | `gemini-2.5-flash` reroutes to `gemini-3.5-flash-lite` on 2026-12-01; the preview parse model may be retired | Pin GA Flash models (O3), with an A/B on 10 papers against the current marks before switching |
| R3 | Render cold start (~1 min) and "not for production" | Pinger (O6); move to the $7 tier or similar when piloting with real classes (**price unverified**) |
| R4 | Supabase pause or data loss on Free (no backups) | Pinger; weekly `pg_dump` to S3 (O8); Pro when real |
| R5 | Driver semantics drift (CCF, upsert, `<>` on absent, ordering) | Differential + shared-row tests; parity unchanged on the AWS path |
| R6 | Service-role key in Lambda env and Render env (full DB access) | Least-privilege grants to `service_role` on the one table it uses through PostgREST (Node uses the pooler user); rotate keys; few IAM users can read Lambda config |
| R7 | Leaked IAM user key on Render | Policy limited to 3 prefixes + 1 function; rotation; CloudTrail |
| R8 | Cross-region Lambda → Supabase latency or outages | httpx retries; a failed message goes back to SQS and then the DLQ |
| R9 | S3 bucket CORS doesn't allow the Netlify origin, so browser uploads fail | Step 4.5 updates the CORS and tests from the real origin |
| R10 | 500 MB limit reached | Size guard in the migration; a monthly size check; upgrade path |

**Rollback:** see §7. The AWS-only deployment (Node API on AWS, DynamoDB, Python API Lambdas) is left untouched for 30 days, and switching back means `deployment: aws` plus the grader `DATABASE_PROVIDER=dynamodb` plus `--reverse` if writes happened.

## 12. Open decisions (with recommendations)

| # | Decision | Recommendation |
|---|---|---|
| O1 | Python → Supabase client: **PostgREST over HTTPS (httpx, already in the layer)** vs psycopg via Supavisor (native arm64 wheel, connection churn from Lambda concurrency) | **PostgREST + 2 RPC functions.** No new native dependency, no connection limits, atomic writes through the RPCs |
| O2 | Gemini tier for the pipeline and Node AI | **Paid tier** (billing enabled). The free tier uses content to improve Google's products and forbids personal data; EEA/UK/CH require Paid |
| O3 | Flash model IDs (`gemini-2.5-flash` reroutes on 2026-12-01; `gemini-3-flash-preview` is a preview) | **Pin one GA Flash model** for `GRADER_MODEL`, `JSONIFIER_MODEL` and Node `GEMINI_MODEL` (e.g. `gemini-3.6-flash`, availability **unverified**) after a 10-paper A/B against current marks. Until then leave the defaults, and the deadline is 2026-12-01 |
| O4 | Node AI features (rubric drafting, teacher assistant) | **`ai: gemini`** (Flash, paid tier) through the existing seam. `ai: off` is the fallback (default rubric, assistant disabled) |
| O5 | Email on Render Free (SMTP blocked) | **`log` for the MVP** (invite links shown to the admin, no reminder mails). Use `ses` through the IAM user when real invites are needed: SES is HTTPS and costs about $0.10 per 1k emails (**unverified**), but needs SES production access |
| O6 | Keep-alive pinger for Render and Supabase | **Yes**, `/health/ready` every 10–14 min; this also keeps Supabase active. Check Render's terms on keep-alive (**unverified**) |
| O7 | "Start grading" hand-off: Lambda async invoke vs SQS message | **Async invoke** (existing code, retries + on-failure destination). SQS would need a new body shape in the grader |
| O8 | Backups on Supabase Free | **Weekly `pg_dump` via GitHub Actions to S3 `backups/`** (encrypted, own key) |
| O9 | Regions | **Render Frankfurt + Supabase Frankfurt**; the pipeline stays in eu-north-1 |
| O10 | Profile photos | **Keep on S3** (existing code, one prefix in the IAM policy). The alternative is to disable them in the MVP |
| O11 | Supabase Auth instead of the app JWT | **Not in this plan.** Keep the HS256 JWT over `users` |

## 12a. User decisions (2026-11-18), these override §12

| # | Decision | What it changes in the steps |
|---|---|---|
| O1 | Accept: PostgREST + two **restricted** RPC functions | As planned (execute granted to `service_role` only) |
| O2 | Accept: **paid** Gemini API for identifiable student data | No real student papers on the free tier |
| O3 | Accept: benchmark **10 representative papers** before pinning one GA Flash model | Defaults stay until the benchmark; deadline 2026-12-01 |
| O4 | Accept: Node AI on the **same pinned Flash model**; `ai: off` if unavailable | Step 4.2 |
| O5 | Accept: email **log-only** for the MVP | — |
| O6 | Accept, **but monitor actual uptime and avoid excessive pinging** | The pinger uses the longest interval that keeps Render warm (≈14 min) and records uptime/latency; documented in the runbook |
| O7 | **Change: SQS between the API and the grading worker** (durable queue, retries, DLQ, job lifecycle) instead of direct async invoke | New queue `gradeant-grading-jobs` (+ DLQ). Node `startGrading` sends `sqs:SendMessage` with today's regrade payload as the body (the IAM user gets `sqs:SendMessage` on that queue instead of `lambda:InvokeFunction`). The grader's SQS handler accepts **two body shapes**: S3 event (upload) and regrade job (`assignment_id`/`qp_id`/`student_id`), each failing via `batchItemFailures`. The direct-invoke path stays for the all-AWS deployment (`grading: lambda \| sqs`) |
| O8 (as built, 2026-11-18) | Logical backups instead of `pg_dump` | `backend/scripts/backup.mjs` (no client-version coupling) + `.github/workflows/supabase-backup.yml` weekly: dump → restore test in PGlite from `backend/sql` (counts + checksums) → private encrypted S3, lifecycle 56 days; `restore --yes` for disaster recovery |
| Migration (2026-11-18) | **Dropped by the user:** no DynamoDB → Supabase migration | The hybrid starts with an empty database; §7 and group 5's migration steps are not built. Rollback = switch back (the stores aren't synchronised) |
| O7 (as built, group 3) | One queue instead of two | No separate `gradeant-grading-jobs` queue: Node sends the regrade job to the **grader's existing queue** (its DLQ covers both message types; `sqs.regrade_job` tells them apart by body). The Render IAM user gets `sqs:SendMessage` on that queue only |
| O8 | **Improve: weekly backups + retention + restore testing** | Weekly `pg_dump` to S3 `backups/` with an S3 lifecycle rule (keep e.g. 8 weekly copies), plus a scheduled **restore test** into a scratch Postgres that checks row counts; documented in the runbook |
| O9 | Accept **with caveat**: Frankfurt (Render, Supabase) + Stockholm (AWS) **subject to data-residency verification** | Runbook item: confirm with the school which regions personal data may be stored/processed in before cut-over |
| O10 | Accept: profile photos on S3, **private bucket + presigned URLs only** | Bucket Block Public Access on; no public URLs |
| O11 | Accept for now: keep the app's own login tokens **with proper security controls** | Rotate `JWT_SECRET` (old one is in git history), keep token expiry, secrets only in host env vars, CORS allowlist = Netlify origin |

## 13. Sources
- Render Free (spin-down 15 min, ~1 min wake, 750 h, no disk, SMTP 25/465/587 blocked, not for production): https://render.com/docs/free
- Render Blueprint spec (`plan: free`, regions, `rootDir`, `sync: false`): https://render.com/docs/blueprint-spec
- Supabase billing (2 free projects, 500 MB DB, 1 GB storage, 5 GB egress): https://supabase.com/docs/guides/platform/billing-on-supabase
- Supabase pricing (Free paused after 1 week inactivity; no backups on Free): https://supabase.com/pricing
- Supabase compute (Nano: 60 direct, 200 pooler clients): https://supabase.com/docs/guides/platform/compute-and-disk
- Supabase connections (pooler IPv4, direct IPv6; transaction vs session): https://supabase.com/docs/guides/database/connecting-to-postgres
- Supabase custom schemas (Exposed schemas, grants, Accept/Content-Profile): https://supabase.com/docs/guides/api/using-custom-schemas
- Netlify pricing (Free 300 credits; credit rates): https://www.netlify.com/pricing/ ; Free plan commercial use: https://www.netlify.com/blog/introducing-netlify-free-plan/ ; credit-based plans: https://www.netlify.com/changelog/netlify-pricing-update-introducing-credit-based-plans/
- AWS Lambda pricing (1M requests, 400,000 GB-s free): https://aws.amazon.com/lambda/pricing/
- AWS SQS pricing (1M free requests/month): https://aws.amazon.com/sqs/pricing/
- AWS S3 pricing (credit-based free tier for customers from 2025-07-15): https://aws.amazon.com/s3/pricing/ ; AWS Free Tier: https://aws.amazon.com/free/
- Gemini API terms (unpaid data use, human review, no personal info, EEA/UK/CH paid-only, 18+): https://aistudio.google.com/docs/terms
- Gemini pricing (Flash prices; free content used to improve products; `gemini-2.5-flash` deprecated and rerouted 2026-12-01): https://aistudio.google.com/docs/pricing
- Gemini rate limits (free-tier per-model limits not listed on the docs page; see AI Studio): https://aistudio.google.com/docs/rate-limits
- **Unverified (checked during build):** Render `rootDir` checkout contents; whether a DB query counts as Supabase "activity"; Supabase PostgREST `max_rows` default; service_role RLS bypass via PostgREST; `@google/genai` structured-output field name; Lambda/SQS free tier "always free" status and idle long-poll request volume; Secrets Manager and SES prices; Netlify per-deploy credit cost; free-tier Gemini rate limits; Frankfurt ↔ Stockholm latency; availability of `gemini-3.6-flash` on the API.
