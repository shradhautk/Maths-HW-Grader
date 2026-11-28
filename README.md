# GradeAnt

AI grading for handwritten STEM homework · [grade-ant.com](https://www.grade-ant.com/)

GradeAnt reads a student's handwritten answer sheet (PDF), turns it into structured answers with a vision-language model, grades each step against the teacher's rubric, and explains every error. Teachers review and correct the grades before students see them.

## How grading works

```
student PDF ──► S3 ──► jsonifier (vision-language model) ──► structured answers
            ──► grader (rubric, step by step) ──► grades + explanations ──► teacher review ──► student
```

## What's in this repo

| Folder | What it is |
| --- | --- |
| [`web/`](web/) | React 19 + TypeScript app for students, teachers and admins. See [web/README.md](web/README.md). |
| [`backend/`](backend/) | Node.js + Express API. Runs locally with no AWS account, or on AWS. See [backend/README.md](backend/README.md). |
| [`python-backend/`](python-backend/) | The same API as Python Lambdas, plus the grading pipeline (PDF → JSON → grades) on AWS: API Gateway, DynamoDB, S3, SQS. See [python-backend/README.md](python-backend/README.md) and [DEPLOYMENT.md](python-backend/DEPLOYMENT.md). |
| [`admin/`](admin/), [`config/`](config/) | Admin function and shared configuration. |

## Run it locally

```bash
# Terminal 1: API on http://localhost:8787
cd backend && npm install && npm run dev

# Terminal 2: web app on http://localhost:5173
cd web && npm install && npm run dev
```

## Tests

- `backend/`: `npm test` (node:test)
- `python-backend/`: `pytest` (moto, fake model), plus a parity check that sends the same requests to both backends and compares the replies.
