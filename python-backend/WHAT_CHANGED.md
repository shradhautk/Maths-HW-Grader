# What changed in the grading pipeline, step by step

*Written 2026-11-17. This is the plain-language version. The detailed versions are `README.md` §6, `DEPLOYMENT.md` B15, and `PIPELINE_PORT_PLAN.md`.*

---

## 1. The big picture in one minute

GradeAnt grades handwritten answer sheets in three stages:

```
 ①  UPLOAD                ②  READ THE PDF               ③  GRADE
 Student uploads a PDF ─► the PDF becomes text,     ─► each answer gets marks,
 (gradeant-upload)        then neat JSON per question  reasoning and feedback,
                          (gradeant-jsonifier)         saved to the database
                                                       (gradeant-grader)
```

**Before:** stages ② and ③ ran on old Lambda files in `lambda_python_files/`. They had bugs: students could get **0 marks because the AI had an error**, sheets were **graded twice**, and the teacher's **"Start grading" button never actually worked**.

**Now:** stages ② and ③ are rebuilt inside `python-backend/`, next to the other API functions. They share the same safe building blocks, have tests, and keep the original grading prompts word for word, so grades come out the same.

**The old files are untouched.** If anything goes wrong after launch, you switch back to them (see §7).

---

## 2. Words used in this file

| Word | Simple meaning |
|---|---|
| **Lambda** | A small program on AWS that runs when something happens (a web request, a file upload). |
| **Layer** | A shared bundle of code that many Lambdas use, uploaded once instead of copied into each one. |
| **S3** | AWS file storage. The PDFs and JSON files live in a "bucket" (`assignmentgrader`). |
| **SQS (queue)** | A to-do list. When a file lands in S3, a message goes on the queue and a Lambda picks it up. |
| **DLQ (dead-letter queue)** | Where a message goes after failing several times, so a person can look at it. Nothing is lost. |
| **DynamoDB** | The database. Grades live in the `assignment_grading` table. |
| **Gemini** | Google's AI model, which reads the PDFs and grades the answers (same as before). |
| **Secrets Manager** | Safe storage on AWS for passwords and keys. Code holds only the key's *address* (ARN), never the key itself. |
| **moto** | A fake AWS used in tests, so tests run on your PC with no real AWS and no cost. |
| **Presigned URL** | A one-time upload link: the browser sends the PDF straight to S3 with it. |

---

## 3. What I did, step by step

### Step 0. Health check and safety fixes (in the old files)

| What | Why |
|---|---|
| Ran a health check (`/doctor`) on Claude Code itself | You asked. Everything was fine. |
| **Removed a hard-coded secret login key** from 4 old files (`authetication.py`, `get_grades.py`, `update_db.py`, `correct_questions_answer.py`) | Anyone who could read the code could have created fake logins. Now the key must come from the environment, or the request is refused. |
| Fixed a broken first line in `local_dev_server.py` | 4 stray spaces stopped the file from running. |

**Your action:** that old key is still in git history, so **change (rotate) the login key** on AWS.

### Step 1. Compared the old and new code with the code graph

- Refreshed the code map (graphify) and compared `lambda_python_files/` with `python-backend/`.
- **Finding:** all the website API parts (login, grades, marks, corrections, workflow) were **already up to date** in `python-backend/`.
- **The real gap:** the 3-stage grading pipeline wasn't in `python-backend/` at all.

### Step 2. Wrote the plan (with Fable 5.2)

As you asked, the plan was written by the Fable model: `PIPELINE_PORT_PLAN.md`. It found the bugs listed in §5 and split the work into 4 groups (A–D) with a check-in after each one.

### Step 3. Group A: foundations (nothing visible changes yet)

The shared building blocks the new functions need:

| New / changed file | What it does, simply |
|---|---|
| `gradeant_common/secrets.py` (changed) | Adds `gemini_api_key()`, which fetches the Gemini key from Secrets Manager. No key built into the code. |
| `gradeant_common/pipeline/paths.py` | **One place that knows every S3 file path** (where PDFs, JSON and rubrics live). Same paths as the old code, checked by a test. |
| `gradeant_common/pipeline/s3io.py` | Reads and writes files in S3, and **refuses files from any other bucket**. |
| `gradeant_common/pipeline/llm.py` | Talks to Gemini. Retries when Google is busy, and **raises a clear error instead of returning error text as an "answer"**. Uses Google's new supported library (`google-genai`). |
| `lambdas.py`, `build.py`, `deploy/generate_template.py` (changed) | Taught the build and deploy tools a new kind of function: a **worker**, which is started by a queue instead of a web request. |

**Why:** build safe, tested foundations first, so the later steps are small.

### Step 4. Group B: the grader (`gradeant-grader`)

| New file | What it does |
|---|---|
| `gradeant_common/pipeline/prompts.py` | The grading instructions for the AI, **copied word for word** from the old grader. A test checks they are identical. |
| `gradeant_common/pipeline/grading.py` | The grading logic: pair each question with its answer, ask the AI for marks + reasoning + feedback, and save the rows. |
| `gradeant_common/pipeline/sqs.py` | Reads queue messages, and reports **only the failed ones** back so just those are retried. |
| `functions/grader.py` | The Lambda itself. It handles **two triggers**: |

- **Trigger 1, a parsed answer sheet arrives (queue):** grade it right away (as today), or store it ungraded and wait for the teacher (setting `GRADE_ON_UPLOAD=false`).
- **Trigger 2, the teacher presses "Start grading":** regrade the student's corrected answers using the assignment's rubric, persona and partial-credit setting.

**Why:** the old grader rejected "Start grading" with an error, so that button never worked. Now it does.

**Packaging problem found and fixed:** the pipeline needs big libraries (PyMuPDF, Google's SDK). Mixing them with the API's library bundle put two different copies of the same libraries in one folder. The fix: **workers get their own self-contained bundle** (`gradeant-pipeline-deps`), and each function now lists its own bundles.

### Step 5. Group C: reading PDFs (`gradeant-jsonifier`) and uploads

| New / changed file | What it does |
|---|---|
| `gradeant_common/pipeline/pdf.py` | Pulls plain text out of a PDF (PyMuPDF), as the old code did. |
| `gradeant_common/pipeline/jsonify.py` | PDF → text → answers split by question → JSON in the question paper's format. Prompts **word for word** from the old code (a test even caught a missing space). |
| `functions/jsonifier.py` | The Lambda: picks up each uploaded student PDF from the queue and writes `…/Parsed_pdfs/JSON/<student>.json` **once**, which then starts the grader. |
| `functions/upload.py` (changed) | Merged in the old upload Lambda's job. New setting `UPLOAD_LAYOUT=pipeline` saves files exactly where the pipeline reads them, with **server-chosen file names**. The default (`template`) keeps today's behaviour. |

**Why the upload change matters:** the old upload Lambda had **no login check at all**, and used whatever file name the user picked. Now:
- you must be logged in;
- students upload only for their own classes;
- only the assignment's teacher (or an admin) can upload the question paper or solution.

### Step 6. Group D: docs and final checks

- `README.md`: new **§6 Grading pipeline**; updated tables, counts and settings.
- `DEPLOYMENT.md`: new **B15**, the step-by-step switch-over plan with rollback, plus new settings and troubleshooting rows.
- `PIPELINE_PORT_PLAN.md`: marked as done, with the 4 places where the build differed from the plan.
- Final run: all tests pass on Python 3.13 and 3.10, all 11 functions start from their packages, and every function's permissions match what its code uses.

---

## 4. Every component at a glance

```
python-backend/
├── functions/                      one file = one Lambda
│   ├── auth.py, grades.py, corrections.py, marks.py,
│   │   workflow.py, admin.py, contact.py, health.py      (website API, unchanged)
│   ├── upload.py                   CHANGED: pipeline upload layout, owner checks
│   ├── jsonifier.py                NEW: PDF → JSON worker
│   └── grader.py                   NEW: grading worker
├── gradeant_common/                shared code (a layer)
│   ├── secrets.py                  CHANGED: + Gemini key
│   └── pipeline/                   NEW folder
│       ├── paths.py                every S3 path
│       ├── s3io.py                 S3 read/write, bucket check
│       ├── llm.py                  Gemini client with retries
│       ├── pdf.py                  PDF text
│       ├── jsonify.py              PDF → JSON steps (original prompts)
│       ├── prompts.py              grading prompts (original text) + personas
│       ├── grading.py              grade answers, save rows
│       └── sqs.py                  queue messages, failure reporting
├── lambdas.py                      CHANGED: + grader, jsonifier, upload settings
├── build.py                        CHANGED: 3rd bundle, per-function bundles, worker checks
├── deploy/generate_template.py     CHANGED: queues, failure queue, pipeline permissions
├── deploy/template.yaml            REGENERATED: 11 functions
├── pipeline-requirements.txt       NEW: worker libraries (pinned versions)
├── requirements-dev.txt            CHANGED: + PyMuPDF for tests
├── tests/
│   ├── test_pipeline_grader.py     NEW: foundations (paths, S3, Gemini client)
│   ├── test_pipeline_grading.py    NEW: grader
│   ├── test_pipeline_jsonifier.py  NEW: PDF → JSON
│   ├── test_upload_pipeline.py     NEW: uploads + full end-to-end run
│   └── test_build_registry.py      NEW: template matches lambdas.py
├── README.md, DEPLOYMENT.md        CHANGED
├── PIPELINE_PORT_PLAN.md           NEW: the plan / design record
└── WHAT_CHANGED.md                 NEW: this file
```

---

## 5. Old vs new: what got better

| Problem in the old code | What happens now |
|---|---|
| When the AI had an error, the error text was graded as the answer, so the student got **0 marks** | The job fails safely and is **retried**. Nothing wrong is saved. |
| Failed jobs were **deleted** anyway | Only failed jobs go back to the queue, then to a DLQ for a person to check. |
| Each sheet was **graded twice** (the JSON was written twice) | Written once, graded once. |
| Running a sheet again created **duplicate grade rows** | Each row has a fixed ID, so a re-run **overwrites** it and old duplicates are removed. |
| The **"Start grading"** button did nothing (the grader rejected it) | Works: it regrades the corrected answers with the teacher's rubric and persona. |
| The AI could give **more marks than the maximum** | Marks are capped between 0 and the maximum. |
| Questions graded **one after another** | Graded **in parallel** (4 at a time), so it's faster. |
| AWS keys and the Gemini key read from code/env files | AWS access through the Lambda's role only; Gemini key in Secrets Manager. |
| Upload had **no login check** and used the user's own file names | Login required, owner checks, the server picks the file names. |
| Old Google library (no longer supported) | New supported library (`google-genai`). |
| A student could "correct" their answer into a completely new one, and that would be graded unnoticed | The paper's text is kept (`ocr_answer`). A correction that changes too much is still graded, but the paper goes to the **teacher's review queue**. *(Security review fix.)* |
| A repeated queue message could wipe corrections, delete a grade, or undo a teacher's mark changes | The grader skips any sheet that was already corrected, graded or edited. *(Security review fix.)* |

**Kept exactly the same on purpose:** the AI instructions (prompts), the S3 folder layout, the AI models, and the fields saved for each grade. Grades should look the same as before.

---

## 6. How to check it yourself

```bash
cd python-backend
python -m pytest tests            # 97 tests: everything above, with fake AWS and a fake AI
python build.py --verify          # builds every package and starts each function from it
```

The most complete test is **`tests/test_upload_pipeline.py::test_end_to_end_upload_to_summary`**. It runs the whole journey: student gets an upload link → PDF uploaded → jsonifier makes the JSON → grader saves the answers → teacher presses Start grading → grades and summary saved.

---

## 7. Going live and going back

**Going live** follows `DEPLOYMENT.md` **B15**. In short:
1. Put the Gemini key in Secrets Manager.
2. Check the bucket's settings (2 read-only commands).
3. Deploy with the new queue triggers **off**.
4. Test on a **test assignment**.
5. Turn the old Lambdas' queue triggers **off**, then the new ones **on**. Never both at once.
6. Point "Start grading" at `gradeant-grader`.

**Going back** (any time, no code change): turn the new queue triggers off, turn the old ones back on, and point "Start grading" back at the old grader. Grade rows from both versions have the same fields, so all screens keep working.

---

## 8. Still needed from you

| # | What | Why |
|---|---|---|
| 1 | Wording for the **strict** and **lenient** grading personas | Placeholder text is marked `TODO(user, plan O5)` in `gradeant_common/pipeline/prompts.py`. |
| 2 | **Who creates** the question-paper schema JSON, rubric JSON and schema-answer text? | The new functions only *read* these files; nothing in the repo makes them. |
| 3 | Confirm it's OK that student PDFs go to **Google (Gemini)** | Same as today, but worth a conscious yes. |
| 4 | Run the **2 bucket check commands** (DEPLOYMENT.md B15 step 2) | To learn the bucket's Region and queue wiring before going live. |
| 5 | **Rotate the old login (JWT) key** | The old one is visible in git history. |
| 6 | Check your **`.gitignore`** | It currently ignores `python-backend/`, so none of this shows in git. Remove that line if you want to commit it. |

---

## 9. Update: one AI model for everything (Gemini 3.5 Flash preview)

**What you asked:** use only `gemini-3.5-flash-preview` for all AI, through Google's Gen AI SDK, with the exact prompts from `lambda_python_files`.

**Before:** grading used `gemini-2.5-flash`, reading answer sheets used `gemini-3-flash-preview`, and the teacher assistant plus rubric drafting used Claude on Amazon Bedrock.
**Now:** every AI call uses one model, set in one place (`GEMINI_MODEL`, default `gemini-3.5-flash-preview`).

| Part | What changed |
|---|---|
| Grading and reading answer sheets (Python workers) | Both use `GEMINI_MODEL`. The old `GRADER_MODEL` and `JSONIFIER_MODEL` settings are gone. |
| Teacher assistant and rubric drafting (Python `gradeant-workflow`) | New `gradeant_common/ai.py` calls Gemini (`google-genai`) instead of Claude. Same prompts, same results: a reply, or nothing if the model declines. |
| Teacher assistant and rubric drafting (Node API, both AWS and hybrid) | New `backend/src/ai/index.js` (`@google/genai`). `src/aws/bedrock.js` and the Anthropic packages are removed. |
| Settings | `config/app.yaml`: `ai: gemini` for both `aws` and `hybrid`. Bedrock is no longer an option. |
| AWS templates (Python and Node) | Bedrock parameters and permissions are removed. New `GeminiModel`, `GeminiSecretArn` and `GeminiApiKey` parameters. Functions get read access to the key secret only when you give one. |
| Lambda layers | The Python deps layer no longer carries Anthropic, so it is smaller (25 MB). `gradeant-workflow` also gets the pipeline layer for `google-genai`. |
| Safety check | Production refuses to start if AI is on but no Gemini key is set. |

**Prompts:** unchanged, word for word. The parity check (Node vs Python, 211 comparisons) still matches the originals.

**Things to know:**
- `gemini-3.5-flash-preview` is a **preview** model. Google can change or retire it. Changing it later is only a setting (`GEMINI_MODEL`), with no code change.
- "effort" (a Claude-only setting) has no Gemini equivalent and is ignored.
- `lambda_python_files/` is untouched. It is still your way back.
