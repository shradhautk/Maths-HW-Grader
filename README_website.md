# GradeAnt

AI-powered grading for STEM homework: teachers upload a question paper, students upload handwritten answers, GradeFlow™ grades them against a rubric, and the teacher reviews and publishes the results.

## What's in this repo

| Path | What |
|---|---|
| [`web/`](web/) | **The web app** — React 19 + TypeScript + Vite. Marketing site, student / teacher / admin areas. Start here. |
| [`backend/`](backend/) | **The API** — Node.js + Express. Runs locally with no AWS account, and on AWS Lambda Web Functions in production. Replaces the Python Lambdas below. See [`backend/README.md`](backend/README.md). |
| [`lambda_python_files/`](lambda_python_files/), [`admin/lambda_admin.py`](admin/lambda_admin.py) | The original Python Lambdas, kept unchanged as the rollback path and as the reference for the parity check until the Node API has run in production. |
| [`config/app.yaml`](config/app.yaml) | **Main config** — URLs, API paths, roles and permissions, read by both the web app and the API. |
| `.github/workflows/aws.yml`, `Task_definition.json` | CI: builds the React app's image (`docker build -f web/Dockerfile .`) and deploys it to ECS. |
| `extra/` (git-ignored, local only) | Everything not needed to build or run the app: the old static HTML site the React app replaced, design wireframes and schema docs, notes and backups. See `extra/README.md`. [`web/WIREFRAMES.md`](web/WIREFRAMES.md) maps every wireframe screen to its route and file. |

## Run it locally

You need **Node.js 22+**. No AWS account is needed: the API uses an in-process DynamoDB loaded with demo data, local file storage, and logged emails.

```bash
# Terminal 1 — API on http://localhost:8787
cd backend
npm install
npm run dev

# Terminal 2 — web app on http://localhost:5173
cd web
npm install
npm run dev
```

Open http://localhost:5173 and log in (pick the matching role on the login page). Password for every demo account: **`Password123!`**

| Role | Email |
|---|---|
| Admin | `admin@gradeant.dev` |
| Teacher | `teacher@gradeant.dev` |
| Student | `riya@gradeant.dev`, `sam@gradeant.dev`, `li@gradeant.dev` |

How the two environments are selected:

- **API.** `npm run dev` loads `backend/.env.development`, which uses local services. `npm run start:prod`, or running on Lambda, loads `backend/.env.production`, which uses AWS. Every setting is documented in [`backend/.env.example`](backend/.env.example).
- **Web app.** `npm run dev` uses `web/.env.development`, which points at `http://localhost:8787`. `npm run build` uses `web/.env.production`, which points at the deployed API.
- **Overrides.** Use `.env.<mode>.local` (gitignored) to change either. The web app only ever gets public URLs; secrets and AWS credentials stay in the backend.

The original Python dev server still works if you need it: `python lambda_python_files/local_dev_server.py`, with `VITE_API_BASE_URL=http://localhost:8787` in `web/.env.development.local`.

### Other commands

```bash
cd backend
npm test                       # API tests
npm run check:parity           # same requests against the Python Lambdas and the Node API, diffed (needs PYTHON=…)
npm run package -- --verify    # Lambda Web Functions deployment zip, checked like Lambda would run it

cd web
npm run build                  # type-check + production bundle in web/dist/
npm run preview                # serve the production build locally
npm run lint

pytest lambda_python_files/tests   # original Python tests (with lambda_python_files/requirements-dev.txt)
```

## Deploying

**API.** [`backend/README.md`](backend/README.md#deploying-to-aws-lambda-web-functions) covers the execution role and IAM permissions, secrets, packaging, `aws lambda-web` commands, and switching the web app over.

- **Status.** Nothing has been deployed yet; production still runs the Python Lambdas behind API Gateway.
- **Shared state.** Both share the same DynamoDB tables and JWT secret, so the switch and a rollback are just a change of `VITE_API_BASE_URL`.

**Web app.** `web/Dockerfile` builds the app and serves it with NGINX, and `web/nginx.conf` adds the SPA fallback and 301s every old `.html` URL to its new route. The build uses `web/.env.production` unless you pass build args:

```bash
cd web
docker build --build-arg VITE_API_BASE_URL=https://<api-endpoint> -t gradeant-web .
docker run -p 8080:80 gradeant-web
```

> **Note:** the GitHub Action in `.github/workflows/aws.yml` still builds the **root** `Dockerfile` (the legacy static site) on every push to `main`. Point it at `web/` to ship the React app.

**Python Lambdas (current production).** One API Gateway REST API with Lambda proxy integration per route, all sharing the same `JWT_SECRET`:

| Route | File | Tables |
|---|---|---|
| `POST /auth/*` | `lambda_python_files/authetication.py` | `Users`, `GradeAnt_ResetRequests` |
| `POST /getgrades` | `lambda_python_files/get_grades.py` | `assignment_grading`, `Users`, `GradeAnt_Assignments` (read: who owns each assignment) |
| `POST /correctanswer` | `lambda_python_files/correct_questions_answer.py` | `assignment_grading`, `Users`, `GradeAnt_Assignments` (read: who owns each assignment) |
| `POST /updatedb` | `lambda_python_files/update_db.py` | `assignment_grading`, `Users`, `GradeAnt_Assignments` (read: who owns each assignment) |
| `POST /admin` | `admin/lambda_admin.py` | `GradeAnt_*` admin tables, `Users` |
| `POST /workflow` | `lambda_python_files/workflow.py` | `GradeAnt_Assignments`, `GradeAnt_Submissions`, `GradeAnt_Announcements`, `assignment_grading`, `Users`; Claude on Amazon Bedrock |
| `POST /uploadpdf` | *(not in this repo)* | S3 presigned uploads |
