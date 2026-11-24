# Free-tier hybrid: setup runbook

**Web** on Netlify Free · **Node API** on Render Free · **Database** on Supabase Free · **Grading pipeline** on AWS (S3 + SQS + 2 Lambdas, Gemini Flash).
Design and decisions: `backend/FREE_TIER_HYBRID_PLAN.md`. Do the steps in order. Nothing changes for users until step 9.

> **Before any real student paper:**
> - **Gemini terms:** paid tier only for student data, and EEA/UK/CH users must be on the paid tier. Google's terms also bar apps used by under-18s on every tier. This is a legal question for you (plan R1).
> - **Data residency:** confirm with the school that personal data may be stored in Frankfurt (Supabase, Render) and Stockholm (AWS) (plan O9).

## 1. Supabase (database)
1. Create a project on supabase.com: Free plan, region **Frankfurt (eu-central-1)**. Keep the database password safe.
2. Click **Connect** and copy the **Session pooler** URI (IPv4, port 5432). This is `SUPABASE_DB_URL`.
3. Create the tables from your PC:
   ```bash
   cd backend
   SUPABASE_DB_URL='postgresql://…' npm run sql:apply -- --dry-run   # lists 001, 002, 003
   SUPABASE_DB_URL='postgresql://…' npm run sql:apply
   ```
   Type the URI into your own terminal, never into a shared chat. On Windows PowerShell use `$env:SUPABASE_DB_URL='…'; npm run sql:apply`.
4. Go to **Project Settings → API → Exposed schemas** and add `gradeant`. The grader Lambda needs this.
5. Copy the **Project URL** (`SupabaseUrl`) and the **service_role** key (`SupabaseServiceRoleKey`). The service_role key is a secret; it goes only into the Lambda parameters (step 4).

## 2. Data: a fresh start
There is **no data migration** (decided 2026-11-18): the hybrid starts with an empty Supabase database. Teachers, students and admins are created again, through sign-up or admin invites, and assignments start fresh. The DynamoDB data stays where it is, untouched, for as long as you keep the AWS tables.

## 3. AWS: user for Render, bucket CORS
1. **IAM → Users → Create user** `gradeant-render`, with no console access. Attach an inline policy made from `backend/deploy/render-iam-policy.json` after replacing `PIPELINE_BUCKET`, `PROFILE_PHOTO_BUCKET`, `ACCOUNT_ID` and `GRADER_QUEUE_NAME`. It can only:
   - presign uploads into the three pipeline folders;
   - read, write and delete profile photos;
   - send to the grader queue.
2. **Create an access key** for it (type "Application running outside AWS"). It goes into Render (step 5), nowhere else. Rotate it if it ever leaks.
3. **Pipeline bucket → Permissions → CORS:** paste `backend/deploy/pipeline-bucket-cors.json` with your Netlify address. Do the same on the profile-photo bucket, allowing `PUT` and `GET`. Keep **Block all public access** on both buckets; photos are only reachable through presigned URLs.

## 4. AWS: the grading pipeline (Lambdas)
Follow `python-backend/DEPLOYMENT.md` **B15** and **B16** with `deploy/pipeline-template.yaml`:
- **Parameters:** `DatabaseProvider=supabase`, `SupabaseUrl`, `SupabaseServiceRoleKey`, `GeminiApiKey` (all NoEcho, so no Secrets Manager cost), the bucket, and the two queue ARNs.
- **Order:** deploy with `PipelineQueuesEnabled=false` first, then switch the queues over as B15 says.
- **Grader queue settings:** visibility timeout ≥ 5400 s, and a dead-letter queue (maxReceiveCount 3). "Start grading" jobs use this same queue.

## 5. Render (Node API)
1. Commit `render.yaml`, then go to **Render → New → Blueprint** and pick this repository. It creates `gradeant-api` (Free, Frankfurt).
2. Fill in the values marked `sync: false`:
   - `SUPABASE_DB_URL`
   - `JWT_SECRET`: a **new** random value of 32+ characters. The old one is in git history, so everyone signs in again once.
   - `SESSION_TOKEN_SECRET`
   - `GEMINI_API_KEY`
   - the `gradeant-render` access key pair
   - `UPLOAD_BUCKET` = `PIPELINE_BUCKET`
   - `PROFILE_PHOTO_BUCKET`
   - `GRADING_QUEUE_URL`
3. Check the first deploy's log for a config error. The server names every missing setting, for example `GEMINI_API_KEY is required…`.
   - **First-deploy check:** confirm the log does **not** say `config/app.yaml not found`. If it does, Render didn't keep the repository's `config/` folder, so set `APP_CONFIG_FILE=../config/app.yaml`.
4. Open `https://<service>.onrender.com/health/ready`. You should see `"checks": {"database": {"ok": true}, "jwtSecret": {"ok": true}}`.

## 6. Netlify (web)
1. **Add new site → Import from Git**. Base directory `web` (also set in `web/netlify.toml`).
2. Note the site address (`https://<name>.netlify.app`) and put it in `config/app.yaml` (step 7) and in the bucket CORS (step 3.3).

## 7. The switch: `config/app.yaml`
```yaml
deployment: hybrid
deployments:
  hybrid:
    urls:
      web: https://<name>.netlify.app
      api: https://<service>.onrender.com
      corsOrigins: [https://<name>.netlify.app]
```
Commit it. Render redeploys the API with Supabase, Gemini, SQS grading and log-only email, and Netlify rebuilds the web app pointing at Render.

## 8. Keep it awake, and keep an eye on it (plan O6)
- **Render Free** sleeps after 15 idle minutes and takes about a minute to wake. **Supabase Free** pauses after a week without activity.
- **Keep-alive:** a free external uptime monitor (for example UptimeRobot or cron-job.org) calling `GET /health/ready` every 14 minutes keeps both awake and records uptime. That's about 3,100 requests a month, the minimum that works.
  - A GitHub Actions cron is **not** recommended: on a private repository each run bills at least one minute, about 3,100 minutes a month, which is more than the 2,000 free.
- **Watch these:**
  - the monitor's uptime report;
  - Render → Metrics;
  - Supabase → Reports (database size, against the 500 MB limit);
  - the `gradeant-pipeline-errors` CloudWatch alarm;
  - the grader DLQ.

## 8b. Weekly backups with a restore test (plan O8)
Supabase Free keeps no backups, so `.github/workflows/supabase-backup.yml` runs every Monday (about 3 Actions minutes):
1. It dumps every `gradeant` table (`npm run backup -- backup --out …`).
2. It **restore-tests** the file: it loads it into a fresh Postgres built from `backend/sql`, and every table's row count and checksum must match.
3. It uploads the file to a private, encrypted S3 bucket.

A failed run emails you; it's also the first sign that Supabase was paused.

**Setup (once):**
1. **S3:** create a private bucket for backups, for example `gradeant-backups-<account>`, in eu-north-1, with **Block all public access** on and default encryption. Under **Management → Lifecycle rules**, add `backend/deploy/backup-lifecycle.json`: it keeps 8 weekly backups (56 days).
2. **IAM:** create a user `gradeant-backup` with only `backend/deploy/backup-iam-policy.json` (write new files under `backups/`; it can't read or delete). Create an access key for it.
3. **GitHub → Settings → Secrets and variables → Actions:**
   - secrets `SUPABASE_DB_URL`, `BACKUP_AWS_ACCESS_KEY_ID`, `BACKUP_AWS_SECRET_ACCESS_KEY`, and optionally `PG_SSL_CA`;
   - variables `BACKUP_BUCKET` and `BACKUP_REGION`.

   The job skips itself until `BACKUP_BUCKET` is set.
4. **Run it once by hand:** Actions → *Supabase backup* → **Run workflow**. The log should end with `restore test passed` and `stored s3://…`.

**Restoring after a disaster:**
1. Download the latest file from the bucket.
2. Test it locally: `npm run backup -- verify FILE`.
3. Create the tables on the new or emptied project: `npm run sql:apply`.
4. Load the rows: `SUPABASE_DB_URL=… npm run backup -- restore FILE --yes`.

The restore upserts every row; generated index columns are recomputed.

## 9. Smoke test (staging first)
1. A student uploads a PDF in the web app. The browser PUT goes to S3, and the network tab shows 200.
2. `Parsed_pdfs/JSON/<student>.json` appears, then rows appear in Supabase `gradeant.assignment_grading`.
3. The student corrects, the teacher presses **Start grading**, and the roster shows *AI graded* (or *Queued for review* for a rewritten answer).
4. Publish, and the student sees the grade.
5. Note the cold-start time once (the first request after 15+ idle minutes).

## Rollback
- Set `deployment: aws` in `config/app.yaml`.
- Point the web build back with the next Netlify deploy, or use the AWS-hosted web app.
- Set the grader's `DatabaseProvider=dynamodb`.
- The two stores aren't synchronised (no migration): after a rollback the app shows the DynamoDB data again, and whatever was created in Supabase stays only there.
