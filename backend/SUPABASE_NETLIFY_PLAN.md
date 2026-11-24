# Supabase + Netlify deployment option: implementation plan

Status: **PLAN, awaiting approval** (written 2026-11-17). Nothing in this file has been built yet.
Work style: finish one check-in group, run its done-checks, report, wait for "go". Never commit. The AWS path (Node on Lambda, `python-backend/`, `lambda_python_files/`) must keep working unchanged at every step.

## 0. Summary

1. Add a second deployment target to the **Node backend**. The web app and the API run on **Netlify**: the SPA is served from `web/dist`, and one Netlify Function wraps the existing Express app under `/api`. Data lives on **Supabase**: Postgres replaces DynamoDB, Storage replaces S3, and Queues (pgmq) replace SQS and the Lambda async invoke.
2. **One YAML line switches it.** `config/app.yaml` gets `deployment: aws | netlify`, plus a `deployments:` block that holds each target's `providers` and URL overrides. The default is `aws`, so nothing changes until someone switches. Secrets only ever come from env vars.
3. The data layer gets a driver seam. The current DynamoDB code moves behind it unchanged. A Postgres driver stores each item whole in a `jsonb` column and evaluates the small DynamoDB expression subset the services use, so services keep their exact semantics (conditional checks become 404/409, `if_not_exists`, upsert-on-update).
4. Storage, queue, email, AI and secrets get a branch inside the modules that already switch on `config.drivers`: Supabase Storage signed URLs, pgmq plus a Netlify Background Function worker, SMTP, the Anthropic API (`claude-opus-5-5`) and plain env secrets.
5. A one-time, idempotent `migrate-to-supabase.mjs` copies every DynamoDB table and S3 bucket, with dry-run, counts, per-item checksums and an optional reverse sync for rollback. The grading pipeline is ported to Node and runs in Netlify Background Functions (open decision O1).

---

## 1. Findings (verified by reading the code, 2026-11-17)

### 1.1 How the Node backend touches AWS today

| Concern | Module | Existing switch | Used by |
|---|---|---|---|
| DynamoDB | `src/aws/dynamodb.js`: `table(getName)` returns `{get, put, update, delete, queryAll, scanAll}`, plus `clean`, `pingDatabase`, `isMissingIndexError` | `DB_DRIVER=memory\|dynamodb` (memory = in-process dynalite) | `src/repositories/*.js`, `src/dev/seed.js`, `test/helpers.js`, `index.js`, `worker.js`, `lambdas/adapter.js`, `controllers/health.controller.js` |
| S3 | `src/aws/s3.js`: `storage()` returns `{presignPut(bucket,key,type,ttl,len), presignGet, head, delete}` | `STORAGE_DRIVER=local\|s3` | `auth.service.js` (profile photo), `upload.service.js` (`/uploadpdf`) |
| SES | `src/aws/ses.js` `sendEmail` | `MAIL_DRIVER=log\|ses` | admin invites, contact form, `remindStudent` |
| SNS | `src/aws/sns.js` `publishNotification` | `NOTIFY_DRIVER=log\|sns` | `publishGrades` |
| SQS | `src/aws/sqs.js` + `services/jobs.service.js` + `worker.js` | `JOBS_DRIVER=inline\|sqs` | `setGradingMethod` → `draftRubric` job |
| Lambda invoke | `src/aws/lambda.js` `triggerGrading` | `GRADING_DRIVER=log\|lambda` | `startGrading` |
| Bedrock | `src/aws/bedrock.js` `askClaude` (Mantle client + refusal-fallback middleware) | `AI_DRIVER=canned\|bedrock` | `workflow/assistant.js` |
| Secrets | `src/aws/secrets.js` (value from env, or Secrets Manager ARN) | none | `lib/tokens.js`, `middleware/auth.js`, health |

Production defaults come from `.env.production`, which **sets every `*_DRIVER` explicitly** (`DB_DRIVER=dynamodb` … `GRADING_DRIVER=lambda`). Env-file values land in `process.env`, so as written they would override any YAML provider choice. Step 1.3 removes them.

`config/app.yaml` is read by `backend/src/config/appConfig.js`, by `python-backend/gradeant_common/config.py` and by `web/vite.config.ts`. **Verified:** the Python `_validate` and the Node `validateAppConfig` both ignore unknown top-level keys, so adding `deployment:` and `deployments:` breaks neither. Vite copies only `urls/api/pages/roles` into the bundle, so provider names never reach the browser.

### 1.2 Data-access inventory: each repository function → DynamoDB operation → Postgres equivalent

Notation: `T(pk)` is a GetItem/PutItem on the base key. "Upsert-update" means a DynamoDB UpdateItem with **no condition**. That call **creates** the item when it is missing, and the Postgres driver must do the same.

| Repository / call site | DynamoDB operation | Behaviour to preserve | Postgres equivalent (driver) |
|---|---|---|---|
| **users** `getUserByEmail` | GetItem(email) | – | `select item from gradeant.users where email=$1` |
| `putUser` | PutItem | overwrite | `insert … on conflict (email) do update set item=excluded.item` |
| `createUser` | PutItem + `attribute_not_exists(email)` | taken → CCF → admin `completeFirstLogin` answers **409** | `insert … on conflict do nothing returning 1`, no row → throw CCF |
| `setLastLogin`, `setPassword`, `setPhotoKey`, `setGradingSettings` | upsert-update `SET …` | creates a stray item if missing (today's behaviour) | tx: `select … for update`, apply SET in JS, upsert |
| `setTempPassword` | UpdateItem SET + `#uid = :uid` | mismatch/missing → CCF → **404** "account no longer exists" | tx + JS condition → CCF |
| `setAccountStatus` | UpdateItem SET + `attribute_exists(email)`; CCF swallowed | missing → no-op | same generic path (CCF swallowed by caller) |
| `listDirectory` | Scan + Projection (`user_id,email,name,userRole,classes`) | – | `select item from users order by email`, JS projection |
| **grading** `getRow` | GetItem(evaluation_id) | – | by PK |
| `updateRow(id, params)` | UpdateItem with caller expressions, see 1.3 | `#qn = :qn` / `#type = :summary` CCF → per-row error text (`correctanswer`, `updatedb`) or **404** (`updateStudentGrade`); `if_not_exists(teacher_id\|ocr_answer)`; upsert when no condition (`grades.service`, `acceptFlagged`) | generic tx update |
| `studentRows(sid, filters)` | Query `student_id-index` + Filter (`=`, `<>`) | – | `where student_id=$1` + JS filter |
| `summaryRows(aid, qp)` | Query `type-index`='summary' + Filter aid/qp | – | `where type='summary' and assignment_id=$1 and qp_id=$2` |
| `assignmentRows(aid, qp)` | Query optional `assignment_id-index`; missing index → `null` → caller falls back | – | `where assignment_id=$1 and qp_id=$2 and student_id is not null` (never null on PG) |
| **workflow** `getAssignment` / `putAssignment` | Get / Put | – | by PK |
| `updateAssignment` | UpdateItem: `confirmQuestions` (`if_not_exists(qp_status)`), `setGradingMethod` (`SET … REMOVE rubric`), `approveRubric`, `startGrading`, `publishGrades`, `assistant.draftRubric` (`qp_status = :generating`, CCF swallowed) | as DynamoDB | generic tx update |
| `scanAssignments` | Scan | – | `select item … order by assignment_id` |
| `assignmentsByTeacher` | Query `teacher_id-index` | – | `where teacher_id=$1` |
| `publishedAssignments` | Scan + Filter `content_status = :p` | – | scan + JS filter (or `where item->>'content_status'='published'`) |
| `getSubmission` | Get(assignment_id, student_id) | – | composite PK |
| `submissionsForAssignment` | Query PK `assignment_id` | **sorted by SK `student_id` ascending** | `where assignment_id=$1 order by student_id` |
| `updateSubmission` | upsert-update: `putSubmissionStatus` (dynamic SET, **creates submission rows**), `remindStudent` (`if_not_exists(#s)`), `requestRecheck` | as DynamoDB | generic tx update |
| `getAnnouncement` / `putAnnouncement` / `deleteAnnouncement` / `scanAnnouncements` | Get / Put / Delete / Scan | – | by PK / scan |
| `announcementsByAuthor` | Scan + Filter `author_id = :a` | – | scan + JS filter |
| `announcementsForClasses(classes)` | Scan + Filter `class_name IN (…)` | DynamoDB caps IN at 100 operands | JS filter (no cap; harmless) |
| **admin** (8 tables via `table()`) `adminUsers/resetRequests/content/tickets/auditLogs/settings/invites/analytics` | Get, Put, Delete, Scan + `equalsFilter` (=, AND), upsert-update via `setFields` (`SET #f0 = :v0, …`) | `invites.update consumed`, `adminUsers.update` upsert | by PK / scan + JS filter / generic update |
| `pendingResetRequestsFor` | Scan + Filter (user_id, status) | – | scan + JS filter |
| **health** `pingDatabase` | GetItem `__healthcheck__` | check key `dynamodb` | `select 1`; check key `database` on Postgres (keep `dynamodb` on AWS) |
| **dev** `schema.js` / `seed.js` | CreateTable / Put | local only | `backend/sql/*.sql` + the same seed through the facade |

There is no BatchWrite, TransactWrite, `ADD`, `list_append` or Sets in the Node code. Only the Python pipeline uses BatchWriteItem.

### 1.3 The complete DynamoDB expression grammar the Node code uses

The Postgres driver needs only this subset, and an unknown construct must throw `UnsupportedExpression` (tested):

- **UpdateExpression:** `SET p = :v[, …]`, `SET p = if_not_exists(p, :v)`, `REMOVE p` (only after SET in one expression), where `p` is a name or `#alias`. The keyword appears as both `SET` and `set`.
- **ConditionExpression:** `attribute_not_exists(p)`, `attribute_exists(p)`, `p = :v` (unquoted `qp_status` and aliased `#qn`/`#type`/`#uid`).
- **FilterExpression:** `p = :v`, `p <> :v`, `p IN (:a, :b, …)`, joined by `AND`.
- **KeyConditionExpression:** `p = :v` (one hash key; no range conditions are used).
- **ProjectionExpression:** comma list of names/aliases.
- **IndexName:** `student_id-index`, `type-index`, `${GRADING_ASSIGNMENT_INDEX}`, `teacher_id-index`, submissions `student_id-index` (declared in `dev/schema.js`, never queried in Node).

Semantics to copy exactly: a missing attribute is "absent" (so `<>` is true and `=` is false); `if_not_exists` treats only *absent* as missing (a stored `null` counts as present); `removeUndefinedValues` drops `undefined` from items; and a failed condition throws `Error{name:'ConditionalCheckFailedException'}` so `lib/errors.isConditionalCheckFailed` keeps working.

### 1.4 Other findings that shape the plan

- **The web routes collide with API paths on one origin:** `/contact` is both a React page and `api.contact`; the admin console and `/auth/*.html` are similar. So on Netlify the API is mounted under **`/api`**, using the existing `API_BASE_PATH` (`config.server.basePath`), and `urls.api` becomes `https://<site>/api`. The paths *below* the base (`/auth/login`, `/workflow`, `/getgrades`, `/uploadpdf`, `/health`, …) are unchanged, exactly as on API Gateway's `/dev` stage today.
- **`/uploadpdf` key layout:** Node's `UPLOAD_KEY_TEMPLATE` cannot express the pipeline layout that Python implements as `UPLOAD_LAYOUT=pipeline` (`python-backend/functions/upload.py`). The Netlify variant needs the pipeline layout in Node (Group 5).
- **The browser PUT contract:** `web/src/lib/account.ts` and `web/src/lib/upload.ts` do `fetch(url, {method:'PUT', headers:{'Content-Type'}, body:file})`. The Supabase signed upload URL must accept that raw PUT. This is **unverified** and is spiked in step 3.1.
- **Size enforcement:** S3 presigned URLs sign `ContentLength`. Supabase signed upload URLs do not (unverified in docs), so size limits move to **per-bucket `file_size_limit` + `allowed_mime_types`**, plus the existing `head()` check on photo confirm.
- **Gemini pipeline:** today this exists only in Python (`gradeant_common/pipeline/*`, ~1,250 lines incl. `functions/{jsonifier,grader}.py`), and its prompts are pinned verbatim by tests. Contract: `python-backend/README.md` §6 and `PIPELINE_PORT_PLAN.md` §1.3 (two grader entry shapes: an S3/SQS upload event, and the workflow regrade payload `{assignment_id, qp_id, student_id, method, persona, partial_credit, rubric}`).
- **The esbuild bundler would break `config/index.js`:** `ROOT` comes from `import.meta.url` (`../..`), and the `.env` files, `package.json` (health) and `config/app.yaml` are all resolved from it. Netlify's default bundler keeps the file tree. esbuild is opt-in (`node_bundler = "esbuild"`), so we **do not** opt in (see §6).

---

## 2. Target architecture

```
                     Browser (React SPA)
                        │  same origin: https://app.gradeant.io
          ┌─────────────┴───────────────────────────────────────────────┐
          │ NETLIFY                                                     │
          │  CDN: web/dist (SPA, /* → /index.html)                      │
          │  fn  api          path /api, /api/*   (sync, ≤60 s)         │──┐
          │       └ boots backend Express once/instance on 127.0.0.1:0 │  │
          │         and proxies Request → fetch → Response             │  │
          │  fn  worker       background:true     (≤15 min)            │◄─┤ kick (POST + X-Worker-Secret)
          │       └ drains pgmq: jobs (draftRubric), grading, pipeline │  │
          │  fn  dispatch     schedule "* * * * *" (≤30 s)             │──┘ kicks worker if queues non-empty
          └───────┬──────────────────────────────┬──────────────────────┘
                  │ pg (Supavisor txn pool :6543)│ supabase-js (service role, server only)
          ┌───────┴──────────────────────────────┴──────────────────────┐
          │ SUPABASE                                                    │
          │  Postgres schema gradeant: 13 tables (pk cols + item jsonb) │
          │  pgmq queues: gradeant_jobs, gradeant_grading,             │
          │               gradeant_pipeline (+ archive = DLQ)          │
          │  Storage buckets (private): uploads, profile-photos        │
          │    trigger on storage.objects INSERT (uploads/original_pdf)│
          │      → pgmq.send('gradeant_pipeline', {...})               │
          │  RLS on, no policies; schema not exposed to the Data API   │
          └─────────────────────────────────────────────────────────────┘
   External: Anthropic API (claude-opus-5-5, assistant + rubric drafts)
             Google Gemini (pipeline transcription + grading)
             SMTP provider (invites, reminders, contact form)
```

The AWS target stays exactly as today: Lambda Web Function or per-path Lambdas (`backend/deploy`, `backend/lambdas`), the Python Lambdas, DynamoDB, S3, SQS, SES, SNS and Bedrock.

---

## 3. Provider config design

### 3.1 `config/app.yaml` additions (appended; existing sections unchanged)

```yaml
# ---------------------------------------------------------------------------
#  Deployment target: the ONE line to change to move between AWS and
#  Supabase + Netlify. Secrets never go here: each provider reads its keys
#  from environment variables (listed in backend/.env.example).
#  Moving existing data is a separate one-time step:
#  backend/scripts/migrate-to-supabase.mjs (see backend/SUPABASE_NETLIFY_PLAN.md §8).
#  Applies in production mode. Development/test keep the local emulators
#  unless a *_DRIVER env var says otherwise.
# ---------------------------------------------------------------------------
deployment: aws                      # aws | netlify

deployments:
  aws:
    providers:
      database: dynamodb             # dynamodb | supabase
      storage: s3                    # s3 | supabase
      queue: sqs                     # sqs | supabase   (background jobs + grading hand-off)
      email: ses                     # ses | smtp
      notify: sns                    # sns | none
      ai: bedrock                    # bedrock | anthropic
      secrets: aws                   # aws | env
    # urls: {}                       # no overrides: urls.production is the AWS deployment
  netlify:
    providers:
      database: supabase
      storage: supabase
      queue: supabase
      email: smtp
      notify: none
      ai: anthropic
      secrets: env
    urls:                            # merged over urls.production when this deployment is active
      api: https://app.gradeant.io/api
      upload: ""
```

Rules (enforced by `validateAppConfig`; every problem is listed, as today):
- `deployment` is missing, or it names a key of `deployments`. If it is missing, the whole block defaults to the `aws` providers above, so old YAML files and the parity fixtures keep working.
- Every provider value must be in its allowed list.
- **Combination rules:** `queue: supabase` requires `database: supabase` (pgmq lives in that Postgres). Any AWS provider (dynamodb/s3/sqs/ses/sns/bedrock/secrets:aws) under `deployment: netlify` requires the `GRADEANT_AWS_*` credentials (see 3.3). Netlify's runtime reserves the AWS Lambda variables (["AWS's reserved environment variables are available for read-only access"](https://docs.netlify.com/build/functions/environment-variables/)), so the SDK would otherwise pick up Netlify's own credentials. Whether `AWS_ACCESS_KEY_ID` is actually populated there is **unverified**.
- `deployments.<x>.urls` may override only `web`, `api`, `upload` and `corsOrigins`. These are merged over `urls.production`.

The validator returns `{…existing, deployment, providers, urlsEffective}`.

### 3.2 Driver resolution (`backend/src/config/index.js`)

| Provider | → `config.drivers` | Dev/test default (unchanged) | Env override (local/testing only) |
|---|---|---|---|
| database dynamodb / supabase | `db`: `dynamodb` / `postgres` | `memory` | `DB_DRIVER=memory\|dynamodb\|postgres\|pglite` |
| storage s3 / supabase | `storage`: `s3` / `supabase` | `local` | `STORAGE_DRIVER` |
| queue sqs | `jobs`: `JOBS_DRIVER` (default `inline`, as production today), `grading`: `lambda` | `inline` / `log` | `JOBS_DRIVER`, `GRADING_DRIVER` |
| queue supabase | `jobs`: `pgmq`, `grading`: `pgmq` | – | same |
| email ses / smtp | `mail`: `ses` / `smtp` | `log` | `MAIL_DRIVER` |
| notify sns / none | `notify`: `sns` / `log` | `log` | `NOTIFY_DRIVER` |
| ai bedrock / anthropic | `ai`: `bedrock` / `anthropic` | `canned` | `AI_DRIVER` |
| secrets aws / env | `secrets`: `aws` / `env` | `env` | `SECRETS_DRIVER` |

Precedence: an explicit `*_DRIVER` env var wins, then (in production mode) `deployments[deployment].providers`, then the dev default. Step 1.3 deletes the seven `*_DRIVER=` lines from `backend/.env.production`; they equal the `aws` providers, so behaviour is identical. The AWS templates also get grepped: `backend/deploy/template.yaml`, `backend/lambdas/template.yaml`, `.github/workflows/aws.yml`. Any `*_DRIVER` set there to the AWS value is harmless, and the grep result gets reported.

`config.app.urls` is the effective URL set. CORS origins and `mail.appBaseUrl` read it. `web/vite.config.ts` applies the same merge for `mode === 'production'` (a ~10-line change), so `npm run build` targets whichever deployment is active.

### 3.3 Env vars per provider (secrets marked 🔒; all set in the Netlify UI with Functions scope, never in netlify.toml)

| Provider | Variables |
|---|---|
| all on Netlify | `APP_ENV=production`, `API_BASE_PATH=/api` |
| database supabase | 🔒`SUPABASE_DB_URL` (transaction pooler, port **6543**, `?sslmode=require`), `PG_POOL_MAX` (default 1), `PG_SCHEMA` (default `gradeant`) |
| storage supabase | `SUPABASE_URL`, 🔒`SUPABASE_SERVICE_ROLE_KEY`, `UPLOAD_BUCKET` (e.g. `uploads`), `PROFILE_PHOTO_BUCKET` (e.g. `profile-photos`). Existing `UPLOAD_*` / `PIPELINE_ROOT` stay |
| queue supabase | uses `SUPABASE_DB_URL`; `PGMQ_JOBS_QUEUE=gradeant_jobs`, `PGMQ_GRADING_QUEUE=gradeant_grading`, `PGMQ_PIPELINE_QUEUE=gradeant_pipeline`, `PGMQ_MAX_ATTEMPTS=5`, `WORKER_URL` (default `${URL}/.netlify/functions/worker`; `URL` is a Netlify read-only var), 🔒`WORKER_SHARED_SECRET` |
| email smtp | `SMTP_HOST`, `SMTP_PORT` (587), `SMTP_SECURE` (false = STARTTLS), `SMTP_USER`, 🔒`SMTP_PASSWORD`. Existing `SES_SENDER_EMAIL`/`INVITE_SENDER_EMAIL`/`CONTACT_*` are reused as From addresses (renaming them is optional, O7) |
| ai anthropic | 🔒`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL_ID` (default `claude-opus-5-5`), `ANTHROPIC_FALLBACK_MODEL_ID` (default `claude-opus-5`). **Set the key explicitly:** on credit-based plans Netlify auto-injects AI-gateway keys ([source](https://docs.netlify.com/build/functions/environment-variables/#ai-gateway-environment-variables)), which would silently bill AI through Netlify credits |
| secrets env | 🔒`JWT_SECRET`, 🔒`SESSION_TOKEN_SECRET` (`*_ARN` rejected when `secrets: env`) |
| pipeline (Group 5) | 🔒`GEMINI_API_KEY`, `PIPELINE_ROOT` (default `Data_warehouse`), `GRADE_ON_UPLOAD`, `GRADER_MAX_WORKERS`, `CORRECTION_SIMILARITY_MIN` (same names as Python) |
| AWS provider used from Netlify (mixed setups only) | 🔒`GRADEANT_AWS_ACCESS_KEY_ID`, 🔒`GRADEANT_AWS_SECRET_ACCESS_KEY`, `AWS_DATA_REGION` |

`validate()` (production only) adds one "X is required for provider Y" problem per missing variable. Examples: `SUPABASE_DB_URL` for database supabase, and `JWT_SECRET` of at least 32 characters for secrets env.

### 3.4 Python (open decision O6)
Recommended: a 6-line guard in `gradeant_common/config.py`. If `app.yaml` has `deployment` and it is not `aws`, the startup raises "Python Lambdas only support deployment: aws". That stops a mis-deploy that would read config meant for Supabase while writing to DynamoDB. With it, `python-backend` tests stay green; without it, Python is untouched.

---

## 4. Postgres schema

**Recommendation: key and index columns, plus the whole item in `item jsonb`.** GSI attributes are `GENERATED ALWAYS … STORED` columns over `item`.
Why not fully typed tables:
- The rows are schemaless and nested (rubric arrays, `classes`, `recheck`, `gradeant_settings`). Services add attributes freely.
- The services' semantics depend on *absent vs null* (`if_not_exists`, `attribute_exists`). Only a document column keeps that exactly.
- Migration and verification become a 1:1 copy with a per-item checksum.
- The Python pipeline row shape stays authoritative.
- Generated columns keep the index columns consistent with `item` without double writes.

Typed columns can come later, table by table, without changing the driver contract. Each PK column is also present inside `item`.

`backend/sql/001_schema.sql`:

```sql
create schema if not exists gradeant;

-- one helper pattern per table: pk columns + item jsonb (+ generated index columns)
create table gradeant.users           (email text primary key,            item jsonb not null);
create table gradeant.assignment_grading (
  evaluation_id text primary key,
  item          jsonb not null,
  student_id    text generated always as (item->>'student_id')    stored,
  "type"        text generated always as (item->>'type')          stored,
  assignment_id text generated always as (item->>'assignment_id') stored,
  qp_id         text generated always as (item->>'qp_id')         stored
);
create index grading_student_idx    on gradeant.assignment_grading (student_id);
create index grading_type_idx       on gradeant.assignment_grading ("type", assignment_id, qp_id);
create index grading_assignment_idx on gradeant.assignment_grading (assignment_id, qp_id);

create table gradeant.assignments (
  assignment_id text primary key, item jsonb not null,
  teacher_id text generated always as (item->>'teacher_id') stored);
create index assignments_teacher_idx on gradeant.assignments (teacher_id);

create table gradeant.submissions (
  assignment_id text not null, student_id text not null, item jsonb not null,
  primary key (assignment_id, student_id));
create index submissions_student_idx on gradeant.submissions (student_id);

create table gradeant.announcements  (announcement_id text primary key, item jsonb not null);
create table gradeant.admin_users    (user_id     text primary key, item jsonb not null);
create table gradeant.reset_requests (request_id  text primary key, item jsonb not null);
create table gradeant.content        (post_id     text primary key, item jsonb not null);
create table gradeant.tickets        (ticket_id   text primary key, item jsonb not null);
create table gradeant.audit_logs     (log_id      text primary key, item jsonb not null);
create table gradeant.settings       (setting_key text primary key, item jsonb not null);
create table gradeant.invites        ("token"     text primary key, item jsonb not null);
create table gradeant.analytics      (metric_key  text primary key, item jsonb not null);

-- Lock everything down: the browser never talks to Postgres.
do $$ declare t text; begin
  for t in select tablename from pg_tables where schemaname = 'gradeant' loop
    execute format('alter table gradeant.%I enable row level security', t);  -- no policies = deny
  end loop; end $$;
revoke all on schema gradeant from anon, authenticated;
revoke all on all tables in schema gradeant from anon, authenticated;
```

The server connects as the `postgres` role over the pooler, and that role bypasses RLS. `gradeant` is **not** added to the Data API's exposed schemas: it stays off PostgREST, and RLS is the second layer. GSIs are sparse in DynamoDB, and `where col = $1` excludes NULLs the same way.

`backend/sql/002_queues_storage.sql`:
- `create extension if not exists pgmq;` then `select pgmq.create('gradeant_jobs'), pgmq.create('gradeant_grading'), pgmq.create('gradeant_pipeline');`.
- Leave "expose Queues via PostgREST" **off**.
- Buckets go through `insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)`: `uploads` (private, 10 MB, `{application/pdf}`) and `profile-photos` (private, 5 MB, `{image/jpeg,image/png,image/webp}`). The column names are **unverified**; the fallback is to create the buckets in the dashboard or with `supabase-js` `createBucket`.

`backend/sql/003_pipeline_trigger.sql` (Group 5): `after insert on storage.objects` where `bucket_id='uploads' and name like '%/original_pdf/%.pdf'` runs `pgmq.send('gradeant_pipeline', jsonb_build_object('kind','pdf','bucket',bucket_id,'key',name))`. Whether user triggers are allowed on `storage.objects` is **unverified**. The fallback is a `confirm` action on `/uploadpdf` that the web calls after a successful PUT (a small web change).

Migrations run with `node backend/scripts/sql-apply.mjs` (plain `pg`, applies files in order, records them in `gradeant.schema_migrations`). The Supabase CLI is not required.

---

## 5. Adapter layer

### 5.1 Database (the only real refactor)

```
backend/src/data/
  tables.js       logical key → { dynamoName: () => config.tables[key], pg: 'gradeant.users',
                                   keys: ['email'], indexes: { 'student_id-index': 'student_id', … } }
  index.js        facade: table(key), initDatabase, closeDatabase, pingDatabase, clean,
                  isMissingIndexError — picks the driver from config.drivers.db (lazy import)
  dynamodb.js     today's src/aws/dynamodb.js body, moved verbatim (memory + dynamodb drivers)
  expressions.js  parser + evaluator for the §1.3 grammar (pure; no I/O)
  postgres.js     pg Pool or PGlite; implements the same 6 table methods
src/aws/dynamodb.js  → becomes `export * from '../data/index.js'` (shim: worker.js, lambdas/adapter.js,
                       test/helpers.js and graph checks keep their imports)
```

**Interface (unchanged shape):**
`table(key) → { name, get(Key, extra?), put(Item, extra?), update(params), delete(Key, extra?), queryAll(params), scanAll(params?) }`

There is one call-site change. `table(() => config.tables.users)` becomes `table('users')` in the 4 repositories and `dev/seed.js`. The logical keys are the existing `config.tables` keys (`users, grading, assignments, submissions, announcements, adminUsers, resetRequests, content, tickets, auditLogs, settings, invites, analytics`). The DynamoDB driver resolves `config.tables[key]` lazily, as today, so tests that reconfigure table names still work.

**Postgres driver semantics:**
- `get(Key)` → `select item … where <pk> = $n`, returning `item` or `undefined`.
- `put(Item, {ConditionExpression?})`: with no condition it upserts. `attribute_not_exists(pk)` becomes `on conflict do nothing returning 1`, and if no row comes back it throws CCF. Any other condition goes through the update transaction path.
- `update({Key, UpdateExpression, ConditionExpression?, ExpressionAttributeNames, ExpressionAttributeValues})` runs one transaction:
  1. `select item … for update`; a missing row means the current item is `undefined`.
  2. Evaluate the condition against the current item. False → `rollback`, then throw `Object.assign(new Error('The conditional request failed'), {name:'ConditionalCheckFailedException'})`.
  3. Apply SET/`if_not_exists`/REMOVE to a deep copy, starting from `{...Key}` when missing, which is DynamoDB's upsert.
  4. `clean`-equivalent removal of `undefined`, then `insert … on conflict (pk) do update set item = excluded.item`, then `commit`.
- `delete(Key)` → `delete … where pk`.
- `queryAll({IndexName?, KeyConditionExpression, FilterExpression?})` parses `col = :v`. The column comes from the PK (no IndexName) or `indexes[IndexName]`; an unknown index throws `{name:'ValidationException'}`, so `isMissingIndexError` keeps working. The query is `select item … where col=$1 order by <pk>`, followed by the JS filter. A base-table query on `submissions` orders by `student_id`, matching the DynamoDB sort key.
- `scanAll({FilterExpression?, ProjectionExpression?})` → `select item … order by <pk>`, then the JS filter and JS projection.
- `pingDatabase` → `select 1`.

Transactions over the Supavisor **transaction** pooler are fine: one client per transaction. `pg` uses unnamed statements unless `name` is passed, and it never is. Supabase says to turn prepared statements off in transaction mode ([source](https://supabase.com/docs/guides/database/connecting-to-postgres)); that `pg`'s unnamed extended-protocol statements are fine there is **unverified** and gets a check in step 2.6. Pool: `max: PG_POOL_MAX (1)`, `idleTimeoutMillis: 10000`, `ssl: {rejectUnauthorized: false}` (or the Supabase CA).

**Why `pg` and not supabase-js for data:** PostgREST cannot do multi-statement transactions or `FOR UPDATE`, and the conditional updates need both. supabase-js is used for Storage only.

**Tests:** `@electric-sql/pglite` (Postgres in WASM, devDependency) backs `DB_DRIVER=pglite`. It needs no Docker on Windows and supports jsonb and generated columns. `initDatabase()` applies `sql/001_schema.sql` to it (skipping the RLS/`revoke` statements, which need Supabase roles) and seeds through the facade, exactly like `memory`.

### 5.2 Storage: `src/aws/s3.js` `storage()` gains `supabase` → `src/supabase/storage.js` (lazy import)
| Method | Supabase implementation |
|---|---|
| `presignPut(bucket,key,type,ttl,len)` | `storage.from(bucket).createSignedUploadUrl(key, {upsert:true})` returns `signedUrl`. Validity is fixed at 2 h ([source](https://supabase.com/docs/reference/javascript/storage-from-createsigneduploadurl)), so `ttl` is ignored and documented. Size and type are enforced by the bucket limits |
| `presignGet(bucket,key,ttl)` | `createSignedUrl(key, ttl)` |
| `head(bucket,key)` | `storage.from(bucket).info(key)` → `{contentLength: size, contentType}`. Missing → `{name:'NotFound'}`. `info()` availability in the pinned storage-js is **unverified**; the fallback is `list(dir, {search: file})` |
| `delete(bucket,key)` | `remove([key])` |

### 5.3 Queue and background work
- `services/jobs.service.js` `enqueueJob`: `pgmq` → `src/supabase/queue.js` `send(PGMQ_JOBS_QUEUE, job)` + `kickWorker()`.
- `src/aws/lambda.js` `triggerGrading`: `pgmq` → `send(PGMQ_GRADING_QUEUE, payload)` + `kickWorker()`. The rubric payload rides in the queue, never in the background request body (256 KB limit).
- `kickWorker()` does `fetch(WORKER_URL, {method:'POST', headers:{'x-worker-secret'}})`, awaited with a 3 s timeout and errors only logged. The background function answers 202 immediately.
- **`netlify/functions/worker.mjs`** (`config.background = true`):
  1. Check the secret (constant-time).
  2. Loop until a queue is empty or 13 min have elapsed: `select * from pgmq.read($q, 900, 1)` (vt = 15 min) over the queues in the order grading → pipeline → jobs.
  3. Run the handler: `runJob`, the grader regrade, or the jsonifier/grader upload path.
  4. On success, `pgmq.delete`. On failure leave the message: it reappears after vt and `read_ct` increments. When `read_ct > PGMQ_MAX_ATTEMPTS`, `pgmq.archive` it and log `level:error` (the DLQ equivalent).
  5. **Always return normally.** Netlify retries failed background invocations after 1 and 2 min ([source](https://docs.netlify.com/build/functions/background-functions/)), and pgmq already owns retries.

  pgmq SQL names (`pgmq.send/read/delete/archive`, `read_ct`) come from the pgmq README. The Supabase docs page shows the `pgmq_public` wrappers ([source](https://supabase.com/docs/guides/queues/api)); the exact signatures in the installed version are **unverified**.
- **`netlify/functions/dispatch.mjs`** (`config.schedule = '* * * * *'`, 30 s limit) runs `select queue_name, queue_length from pgmq.metrics_all()`. If anything is waiting, it calls `kickWorker()`. This is the safety net for storage-trigger messages and lost kicks. The 1-minute minimum interval is **unverified**. Scheduled functions run only on the published deploy ([source](https://docs.netlify.com/build/functions/scheduled-functions/)).
- The AWS path (`sqs` / `lambda`) is unchanged.

### 5.4 Email: `src/aws/ses.js` `sendEmail` gains `smtp` → `src/mail/smtp.js`
This uses `nodemailer` (new dependency, lazy import) with the same `{from,to,subject,html,text,replyTo}` contract. `log` stays the dev default.

### 5.5 Notify: `notify: none` maps to the `log` driver
The SNS topic's consumers are outside this repo, so this is open decision O5.

### 5.6 AI: `src/aws/bedrock.js` `askClaude` gains `anthropic` (rename of the module optional, O7)
- The client is `new Anthropic({ apiKey })` from the already-installed `@anthropic-ai/sdk`, with model `config.anthropic.modelId` (`claude-opus-5-5`).
- Refusal fallback: on the direct API use the **server-side** `fallbacks: [{model: ANTHROPIC_FALLBACK_MODEL_ID}]` with beta `server-side-fallback-2026-07-01`. That needs `client.beta.messages.create`. The Bedrock client-side middleware stays on the Bedrock branch only.
- `output_config: {effort, format}` and the existing `BadRequestError` → JSON-in-prompt retry are kept as-is.
- `stop_reason === 'refusal'` → `null` (same contract).
- Opus 5.5 defaults to effort `medium`. The callers already pass `effort`; check that each call site sets it.

### 5.7 Secrets: `src/aws/secrets.js` `resolve()`
With `secrets: env`, it returns the env value and never constructs the Secrets Manager client.

### 5.8 Health
`/health/ready` reports `checks.database` on Postgres and keeps `checks.dynamodb` on AWS. Under `providers.storage = supabase` it adds a `storage` check (`getBucket(UPLOAD_BUCKET)`), and under `queue = supabase` a `queue` check (`pgmq.metrics`).

---

## 6. Netlify packaging

**`netlify.toml`** (repo root; no secrets, and no `[build.environment]` for runtime values, because "Environment variables declared in netlify.toml are not available to serverless functions": [source](https://docs.netlify.com/build/functions/environment-variables/)):

```toml
[build]
  command = "npm ci --prefix backend --omit=dev && npm ci --prefix web && npm run build --prefix web"
  publish = "web/dist"

[functions]
  directory = "netlify/functions"
  # Default bundler (not esbuild): keeps the file tree so config/index.js's import.meta.url-relative
  # ROOT still finds backend/.env.production, backend/package.json and ../config/app.yaml.
  included_files = ["config/app.yaml", "backend/package.json", "backend/.env.production", "backend/sql/**"]

[[redirects]]          # SPA fallback; /api/* is claimed by the api function's config.path
  from = "/*"
  to = "/index.html"
  status = 200
```

Whether a function's `config.path` takes precedence over the `/*` rewrite is **unverified**; done-check 4.4 tests it. The fallback is an explicit `[[redirects]] from="/api/*" to="/.netlify/functions/api/:splat" status=200` placed first, with the function's `path` removed.

**`netlify/functions/api.mjs`** reuses the proven pattern from `backend/lambdas/adapter.js`. It starts the real Express app once per instance on `127.0.0.1:0` and forwards each request:

```js
import { startServer } from '../../backend/netlify/server.js'   // initDatabase() + createApp().listen(0)
export default async (req, context) => {
  const base = await startServer()                                   // cached promise
  const url = new URL(req.url)
  const headers = new Headers(req.headers)
  headers.set('x-forwarded-for', context.ip ?? '')
  headers.set('lambda-web-request-id', context.requestId ?? '')     // existing log field
  const res = await fetch(base + url.pathname + url.search, {
    method: req.method, headers, body: ['GET', 'HEAD'].includes(req.method) ? undefined : req.body,
    duplex: 'half', redirect: 'manual' })
  return new Response(res.body, { status: res.status, headers: res.headers })
}
export const config = { path: ['/api', '/api/*'] }
```

- Express mounts at `API_BASE_PATH=/api`, so `/api/auth/login`, `/api/workflow`, `/api/getgrades`, `/api/uploadpdf` and `/api/health` keep their sub-paths.
- It runs in production mode because `APP_ENV=production` is set in the Netlify UI (`AWS_LAMBDA_HTTP_ENDPOINT` is not set there, so `pickMode` would otherwise fall back to development).
- Limits ([source](https://docs.netlify.com/build/functions/configuration/)): 60 s sync, 6 MB buffered payload, 1024 MB, region `cmh` by default (changing region needs Pro), Node from the build version (pin `NODE_VERSION=22` in the UI).

**Other entries:** `netlify/functions/worker.mjs` (background) and `netlify/functions/dispatch.mjs` (scheduled), both importing from `backend/netlify/` and `backend/src/`.

**Web:** `urls.production.web` is the Netlify site URL or custom domain, and the netlify deployment's `urls.api` is `https://<same host>/api`. CORS is same-origin, but `corsOrigins` still lists the marketing site.

**AWS packaging untouched:** `backend/scripts/package.mjs` and `package-lambdas.mjs` copy `src/`. The new modules are lazy-imported, so the AWS zips grow by the `pg` / `@supabase/supabase-js` / `nodemailer` files but never load them. The decision on that size cost is open (O8). `npm run check:lambdas` (IAM vs code graph) must still pass, and new non-AWS modules need no IAM.

---

## 7. Pipeline decision (open decision O1)

| | **A. Port to Node, run in Netlify Background Functions (recommended)** | B. Keep the Python pipeline as a container worker polling pgmq | C. Hybrid: pipeline stays on AWS Lambda |
|---|---|---|---|
| Platforms | Supabase + Netlify only (the user's goal) | adds a third host (Fly.io/Render/Cloud Run) | AWS stays in the loop: not a real move |
| Fidelity | Prompts copied verbatim from `gradeant_common/pipeline/prompts.py` and pinned by the same golden tests. PDF text comes from `unpdf` (pdf.js) instead of PyMuPDF; that text is only the QP context and the fallback transcript, while the main transcription sends the PDF inline to Gemini | identical code | identical |
| Work | ~1,200 lines to port + tests (mirrors Python groups A–C) | Python adapters for Postgres rows + Storage + a poll loop, a Dockerfile, a new host | cross-cloud credentials and events |
| Limits | 15 min per background run, enough for one student (Python grades questions in parallel, 4 workers). 1024 MB is enough for 10 MB PDFs | none | none |
| Maintenance | the pipeline exists twice (Python for AWS, Node for Netlify) until AWS is retired | one pipeline | one pipeline |

**Recommendation A.** It is the only option that keeps the variant to two platforms and makes the YAML switch complete. Fidelity risk is contained by the golden tests and an A/B run on 10 real papers (step 5.6). If A/B shows material grade drift, fall back to B, keeping the same queue contract and the same worker message shapes.

**Node pipeline layout:** `backend/src/pipeline/{paths,prompts,llm,pdf,jsonify,grading}.js` port the Python modules one-to-one. `llm.js` uses `@google/genai` with `GEMINI_API_KEY`, inline PDF, the same models `gemini-3-flash-preview` / `gemini-2.5-flash`, and the same retry/strict-parse rules (D4). Rows get deterministic uuid5 ids (D2) and are written through `table('grading')`. The worker then handles:
- `{kind:'pdf', key}` → jsonify. It writes the `Parsed_pdfs/...` objects to Storage, then directly `send(pipeline, {kind:'json', key})`; no second storage event is needed.
- `{kind:'json'}` → grade on upload, honouring `GRADE_ON_UPLOAD`.
- grading-queue messages → regrade (the §1.3 shape).

Node `/uploadpdf` gains `UPLOAD_LAYOUT=pipeline`, ported from `python-backend/functions/upload.py`.

---

## 8. Data migration

**`backend/scripts/migrate-to-supabase.mjs`** is one-time but safely re-runnable. Every write is an upsert, so it is idempotent.

```
node scripts/migrate-to-supabase.mjs [--dry-run] [--tables all|users,grading,…] [--storage] [--verify-only]
                                     [--reverse]   # Postgres → DynamoDB, for rollback (O4)
```

- **Source:** AWS credentials from the operator's shell (read-only IAM: `dynamodb:Scan`, `s3:ListBucket`, `s3:GetObject`), with table and bucket names from the existing env vars.
- **Target:** `SUPABASE_DB_URL_DIRECT` (session or direct connection on port 5432, better for long batches), plus `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
- **Tables:**
  1. Scan in pages (`Limit 500`), unmarshalled with the app's own options so numbers come back as JS numbers.
  2. Convert DynamoDB Sets to sorted arrays and Binary to base64, counting each in the report (none are expected).
  3. Upsert in batches of 500 inside a transaction: `insert … on conflict (pk) do update set item = excluded.item`.
  4. `--dry-run` scans and transforms only, then prints the counts.
- **Storage:**
  1. List `UPLOAD_BUCKET` (all of `PIPELINE_ROOT/`) and `PROFILE_PHOTO_BUCKET`.
  2. For each object, `head` the target; skip it if the size is equal.
  3. Otherwise stream `GetObject` into `upload(key, body, {contentType, upsert:true})`, computing sha256 on the way through.
  4. Keys are unchanged, so `users.photo_key` and every pipeline path keep working.
  5. Objects over the bucket limit are reported, not copied. The Free plan caps files at 50 MB ([source](https://supabase.com/docs/guides/storage/uploads/file-limits)).
- **Verification** (`--verify-only` runs it alone) writes `.data/migration-report.json`:
  - Per table: the source count equals the target count.
  - Per item: sha256 of canonical JSON (sorted keys) on each side, listing up to 50 mismatching keys.
  - Per object: count, total bytes and per-object size; sha256 is re-downloaded for a 5 % sample plus every profile photo.
  - The script exits non-zero on any mismatch.

**Cut-over runbook** (goes into `backend/deploy/SUPABASE_NETLIFY.md` in Group 6):
1. Create the Supabase project (Pro, in the region of O3). Run `sql-apply` for 001–003, then create the buckets and queues.
2. Create the Netlify site from the repo and set the §3.3 env vars. Deploy a **branch** with `deployment: netlify` on a Supabase *staging* project. Smoke-test it: signup, login, create assignment, upload, grade, publish, admin invite email.
3. Rehearsal: run the migration and `--verify-only` against staging using a copy of production data.
4. Freeze window, announced to schools:
   - Stop new uploads by setting `permissions.upload.*: []` in the AWS deploy's YAML and redeploying.
   - Wait for the SQS queues and DLQs to drain.
   - Run the migration, then `--verify-only`. It must show 0 mismatches.
5. Merge `deployment: netlify` to main and deploy. Point `app.gradeant.io` at Netlify, keeping the low DNS TTL set a day before. Re-enable uploads in YAML.
6. Watch for 72 h: Netlify function logs, `/api/health/ready`, and the pgmq archive count.

**Rollback:**
- **Before any writes on Supabase:** point DNS back to AWS. DynamoDB and S3 are untouched.
- **After writes:** run `migrate-to-supabase.mjs --reverse`, which upserts Postgres items to DynamoDB and copies new Storage objects to S3 (same canonical checksum verify). Then point DNS back.
- Keep the AWS stack running and paid for at least 30 days.

---

## 9. Steps (6 check-in groups)

Every group's done-checks include `cd backend && npm test`. The baseline is 80 tests with exactly one pre-existing failure, the Windows-only SIGTERM test, and the same count must hold. They also include `npm run check:parity` with **0 differences** (211 cases, AWS/memory path) and `npm run check:lambdas`.

### Group 1: config switch + database seam (no behaviour change)
| # | Step | Files |
|---|---|---|
| 1.1 | `deployment`/`deployments` parsing, validation, URL merge; export `providers` | `backend/src/config/appConfig.js`, `config/app.yaml` (append §3.1 block, `deployment: aws`) |
| 1.2 | Driver resolution table §3.2; new driver values accepted; per-provider env validation; `config.supabase`, `config.pgmq`, `config.smtp`, `config.anthropic` sections | `backend/src/config/index.js` |
| 1.3 | Remove the 7 `*_DRIVER=` lines; document the new vars | `backend/.env.production`, `backend/.env.example`, `backend/README.md` |
| 1.4 | Create `src/data/{index,tables,dynamodb}.js`; `src/aws/dynamodb.js` becomes a shim; `table(key)` call sites | `backend/src/data/*`, `src/aws/dynamodb.js`, `src/repositories/*.js`, `src/dev/seed.js` |
| 1.5 | Web build honours the deployment URL merge | `web/vite.config.ts` |
| 1.6 | (if O6 = yes) Python guard + test | `python-backend/gradeant_common/config.py`, `python-backend/tests/test_config.py` |

Done-checks:
- New `test/appConfig.test.js` cases: a missing block falls back to aws; an unknown provider is rejected; `queue: supabase` + `database: dynamodb` is rejected; the netlify URL merge works; production env-var problems are listed.
- The standard suite passes unchanged: npm test, parity at 0, check:lambdas.
- `npm run package` and `npm run package:lambdas` still build.
- `web: npm run build` with `deployment: aws` produces a bundle whose embedded config is byte-identical to before (diff `__APP_CONFIG__`).

### Group 2: Postgres driver
| # | Step | Files |
|---|---|---|
| 2.1 | Expression parser/evaluator for §1.3, with `UnsupportedExpression` | `backend/src/data/expressions.js` |
| 2.2 | Schema SQL + `sql-apply` script | `backend/sql/001_schema.sql`, `backend/scripts/sql-apply.mjs` |
| 2.3 | `postgres` (pg) + `pglite` driver per §5.1 | `backend/src/data/postgres.js`, `backend/package.json` (+`pg`, dev +`@electric-sql/pglite`) |
| 2.4 | Run the whole suite on Postgres: `npm run test:pg` (= `DB_DRIVER=pglite npm test`) | `backend/package.json`, `test/helpers.js` (driver from env) |
| 2.5 | Differential test: the same script of ~60 operations (every §1.2 row, incl. CCF, upsert-update, `if_not_exists` on null vs absent, REMOVE, ordering) runs on `memory` (dynalite) and on `pglite`, and the results are compared | `backend/test/dataDrivers.test.js` |
| 2.6 | Optional live check against a Supabase dev project through the **6543** pooler (`SUPABASE_TEST_DB_URL`; skipped when unset) | same test file |

Done-checks: `npm test` (memory) unchanged, `npm run test:pg` with the same pass count, the differential test green, parity at 0, and an expression-coverage test that drives every service through the suite and fails on any `UnsupportedExpression`.

### Group 3: Supabase Storage, queues, SMTP, Anthropic, env secrets
| # | Step | Files |
|---|---|---|
| 3.1 | **Spike:** raw `PUT` (Content-Type header, raw body) to a `createSignedUploadUrl` URL from Node and the browser, plus a bucket size/MIME rejection. If raw PUT fails, switch the web to `uploadToSignedUrl`-compatible form data (record the decision) | scratch script only |
| 3.2 | Storage driver §5.2 | `backend/src/supabase/storage.js`, `src/aws/s3.js` (switch), `package.json` (+`@supabase/supabase-js`) |
| 3.3 | Queue module + `kickWorker`; jobs/grading branches | `backend/src/supabase/queue.js`, `src/services/jobs.service.js`, `src/aws/lambda.js`, `backend/sql/002_queues_storage.sql` |
| 3.4 | SMTP mail; notify `none` | `backend/src/mail/smtp.js`, `src/aws/ses.js`, `package.json` (+`nodemailer`) |
| 3.5 | Anthropic branch §5.6 | `backend/src/aws/bedrock.js` |
| 3.6 | env secrets; health checks §5.8 | `src/aws/secrets.js`, `src/controllers/health.controller.js` |

Done-checks:
- Unit tests with fakes: the storage driver maps calls and `NotFound`; queue `send`/`kick` happen on `startGrading` and `setGradingMethod`; nodemailer is called with the same fields as SES; the Anthropic branch sends `model:'claude-opus-5-5'`, the `fallbacks` beta and `output_config`, and maps a refusal to `null` (`setClaudeOverride` untouched).
- With `SUPABASE_TEST_*` set, an integration test does upload → head → signed GET → delete, and pgmq send → read → delete.
- `npm test`, `test:pg` and parity at 0.

### Group 4: Netlify packaging
| # | Step | Files |
|---|---|---|
| 4.1 | `startServer()` (init DB + listen on :0, cached) | `backend/netlify/server.js` |
| 4.2 | Function entries | `netlify/functions/api.mjs`, `worker.mjs`, `dispatch.mjs` |
| 4.3 | `netlify.toml` §6 | `netlify.toml` |
| 4.4 | Local run: `npx netlify dev` with a `.env` holding the Supabase dev project. Verify `/api/health/ready` (database/storage/queue ok), that `/contact` serves the SPA while `/api/contact` serves the API, and that the function path beats the SPA rewrite | – |
| 4.5 | Deploy a preview branch with `deployment: netlify` and run the full smoke list (runbook step 2) | – |
| 4.6 | Worker tests: a fake queue makes a background run drain messages; a failure leaves the message; `read_ct > max` → archive; a bad secret → 401 | `backend/test/netlifyWorker.test.js` |

Done-checks:
- The preview passes the smoke list, minus pipeline grading if Group 5 isn't done yet; startGrading then just queues.
- Cold start of `/api/health` is under 3 s. Log a one-off measurement.
- The AWS zips still build, and parity is at 0.

### Group 5: pipeline in Node (if O1 = A)
| # | Step | Files |
|---|---|---|
| 5.1 | `UPLOAD_LAYOUT=pipeline` for Node `/uploadpdf` (port of Python's, same keys and checks) | `backend/src/services/upload.service.js`, `config/index.js` |
| 5.2 | Port `paths`, `prompts` (verbatim), `pdf` (`unpdf`), `llm` (`@google/genai`) | `backend/src/pipeline/*.js`, `package.json` |
| 5.3 | Port `jsonify` + `grading` (uuid5 ids, strict parse, parallel 4, skip-if-already-graded, ocr similarity flag) | `backend/src/pipeline/{jsonify,grading}.js` |
| 5.4 | Worker handlers for `pipeline` and `grading` messages | `backend/netlify/handlers.js` |
| 5.5 | Storage trigger (or the confirm fallback) | `backend/sql/003_pipeline_trigger.sql` |
| 5.6 | Golden tests: prompts byte-equal to Python's (read `python-backend/gradeant_common/pipeline/prompts.py` fixtures); the Python test cases mirrored with a stub LLM; an end-to-end upload → rows → summary on pglite + fake storage. **A/B:** 10 real papers through Python (AWS) and Node (Netlify preview), comparing marks per question | `backend/test/pipeline*.test.js` |

Done-checks: golden and mirrored tests green; the A/B report shows a total-marks difference within ±5 % for each paper and no question flipped from full marks to 0. If that fails, stop and revisit O1. Python tests remain untouched and green.

### Group 6: migration + runbook
| # | Step | Files |
|---|---|---|
| 6.1 | Migration script §8 (forward, verify, reverse) | `backend/scripts/migrate-to-supabase.mjs` |
| 6.2 | Tests: dynalite source → pglite target + local-folder storage. A second run changes 0 rows; a tampered item is detected; reverse restores it | `backend/test/migrate.test.js` |
| 6.3 | Runbook + env checklist | `backend/deploy/SUPABASE_NETLIFY.md`, `backend/README.md` (link) |
| 6.4 | Rehearsal on Supabase staging with a copy of production data; report attached | – |

Done-checks: tests green, the rehearsal report shows 0 mismatches, the runbook is reviewed by the user, then the final full regression: `npm test`, `test:pg`, parity at 0, check:lambdas, python-backend tests.

---

## 10. Cost (MVP, one school pilot; prices as published on 2026-11-17)

| Item | Plan | Price | Notes |
|---|---|---|---|
| Supabase | **Pro** (recommended) | from **$25/mo**, incl. $10 compute credit (Micro) | 8 GB disk, 100 GB storage, 250 GB egress, 7-day daily backups, no pausing; PITR +$100/mo ([pricing](https://supabase.com/pricing)) |
| Supabase | Free (staging only) | $0 | 500 MB DB, 1 GB storage, 50 MB file limit, **paused after 1 week inactive**, no backups: not for production |
| Netlify | **Pro** (recommended) | **$20/mo**, 3,000 credits | needed for function **region** selection (to sit next to Supabase) and env-var scopes ([pricing](https://www.netlify.com/pricing/), [region](https://docs.netlify.com/build/functions/configuration/#region)) |
| Netlify | Personal / Free | $9 (1,000 credits) / $0 (300) | functions fixed to `cmh` (Ohio) |
| Netlify credit use (estimate, **unverified model**) | – | – | compute 10 credits/GB-h, bandwidth 20/GB, 2 per 10k requests. 100k API calls × 0.3 s × 1 GB ≈ 8.3 GB-h ≈ 85 credits + 20 for requests. 1,000 papers × ~3 min background ≈ 50 GB-h ≈ 500 credits. 10 GB bandwidth = 200 credits. **≈ 800 credits/mo**, so Personal is borderline and Pro has headroom |
| Anthropic API | pay-as-you-go | `claude-opus-5-5` $4 / $20 per MTok in/out | assistant + rubric drafts only |
| Gemini | unchanged | – | pipeline |
| SMTP | e.g. any transactional provider's free tier | $0–15/mo | O7 |

**MVP total ≈ $45/mo + AI usage**, compared with today's AWS bill (not measured here).

---

## 11. Risks and rollback

| Risk | Mitigation |
|---|---|
| Semantics drift between DynamoDB and the Postgres driver (CCF, upsert-on-update, ordering) | Differential test (2.5); whole suite on pglite; expression-coverage test; parity stays on the AWS path |
| Supabase signed upload URL rejects raw PUT, or ignores size | Spike 3.1 first; bucket `file_size_limit`/MIME; web fallback |
| Latency: services make many sequential queries per request | Same region for Netlify functions and Supabase (O3, needs Netlify Pro); pool of 1 per instance; measure p95 on the preview |
| 60 s sync limit: `askAssistant` with a long Claude answer | Effort `medium`, `max_tokens` cap; streaming response later |
| Cold starts (Express + pg + supabase-js) | Lazy imports; measure in 4.4; keep the Netlify region next to the DB |
| Background function retry runs a job twice | Worker always returns 200; pgmq vt; the grader is idempotent (uuid5, skip-if-graded) |
| Storage trigger not allowed on `storage.objects` | `/uploadpdf` confirm fallback (web change) |
| Netlify AI-gateway auto keys | Set `ANTHROPIC_API_KEY` explicitly; health logs the key source |
| Pipeline fidelity (A) | Golden prompts + A/B; fallback B |
| AWS package grows with new deps | Lazy import; optional exclusion in `package.mjs` (O8) |
| Data loss at cut-over | Freeze uploads, drain SQS, verify 0 mismatches, keep AWS 30 days, reverse sync |

---

## 12. Open decisions (with recommendations)

| # | Decision | Recommendation |
|---|---|---|
| O1 | Pipeline: A Node port in Netlify Background Functions / B Python container worker / C stay on AWS | **A**, with B as fallback if the A/B test fails |
| O2 | Schema: `item jsonb` + key/generated columns, or fully typed | **jsonb + key columns** (§4); type tables later if needed |
| O3 | Region | Supabase **eu-central-1 (Frankfurt) + Netlify `fra`** (Pro) if users are in Europe, which matches today's eu-north-1. Without Netlify Pro: Supabase **us-east-2** to sit next to `cmh` |
| O4 | Build the `--reverse` migration (Postgres → DynamoDB) for rollback after writes | **Yes** (small; same code path) |
| O5 | SNS "grades published" on Supabase: `none` (log), a `notifications` table, or a webhook | **none** now; the consumers of the topic are unknown |
| O6 | Python startup guard against `deployment != aws` | **Yes** (6 lines + test), or leave Python untouched if "unchanged" is strict |
| O7 | Naming: keep `SES_SENDER_EMAIL` and `src/aws/bedrock.js` names, or rename to provider-neutral (`MAIL_FROM`, `src/ai/claude.js`) | **Keep** in this project; rename later |
| O8 | Exclude `pg`/supabase/nodemailer from the AWS Lambda zips | **No** for now (lazy-loaded); revisit if the zip limit gets close |
| O9 | Supabase Auth instead of the app JWT | **Not in this plan.** Keep the HS256 JWT over `users`; it's a possible later step |
| O10 | Worker trigger: dispatcher every minute + direct kicks (as planned), or also a `pg_net` kick from the storage trigger for instant pipeline start | **Dispatcher + direct kicks**; add `pg_net` only if the ≤1 min upload latency matters |

## 13. Sources

- Netlify Functions config (limits: 60 s sync / 15 min background / 30 s scheduled; 6 MB buffered and 256 KB background payload; 1024 MB; region `cmh`; routing `config.path`, `excludedPath`, `preferStatic`; esbuild opt-in; `included_files`; module format): https://docs.netlify.com/build/functions/configuration/
- Netlify Functions API (Request/Response handler, `context.ip`/`requestId`, `config.background`, `schedule`): https://docs.netlify.com/build/functions/api/
- Background Functions (retries after 1 and 2 min; available on Free/Personal/Pro): https://docs.netlify.com/build/functions/background-functions/
- Scheduled Functions (cron, 30 s limit, published deploys only, no URL invocation): https://docs.netlify.com/build/functions/scheduled-functions/
- Netlify env vars for functions (toml vars not available; AWS reserved vars read-only; AI gateway vars): https://docs.netlify.com/build/functions/environment-variables/
- Netlify pricing (Free 300 / Personal $9, 1,000 / Pro $20, 3,000 credits; credit rates): https://www.netlify.com/pricing/
- Supabase Postgres connections (transaction mode :6543 for serverless; prepared statements unsupported there; pooler IPv4): https://supabase.com/docs/guides/database/connecting-to-postgres
- Supabase Storage signed upload URL (valid 2 h): https://supabase.com/docs/reference/javascript/storage-from-createsigneduploadurl
- Supabase Storage file limits (Free 50 MB, Pro up to 500 GB; per-bucket limits): https://supabase.com/docs/guides/storage/uploads/file-limits
- Supabase Queues (pgmq; send/read/pop/archive/delete; visibility timeout): https://supabase.com/docs/guides/queues and https://supabase.com/docs/guides/queues/api
- Supabase pricing: https://supabase.com/pricing
- Claude API: model `claude-opus-5-5` ($4/$20 per MTok); server-side `fallbacks` (beta `server-side-fallback-2026-07-01`) on the direct API, client-side middleware only on Bedrock/Vertex/Foundry. Source: the Anthropic SDK reference, as of 2026-11.
- **Unverified (checked during build):** raw PUT to Supabase signed upload URLs; content-length enforcement; `info()` in storage-js; `storage.buckets` column names; triggers on `storage.objects`; pgmq SQL signatures in the installed version; function-path vs `/*` rewrite precedence; the 1-minute schedule minimum; `pg` behaviour through Supavisor transaction mode; whether Netlify populates `AWS_ACCESS_KEY_ID`; credit consumption model.
