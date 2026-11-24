# Grading pipeline port — implementation plan

Status: **IMPLEMENTED 2026-11-17 (Groups A–D).** Written 2026-11-17 as a plan; kept as the design record. Deviations from the plan, found while building:
- Layers are set **per function**, not in `Globals`: SAM appends a function's layers to the global ones, so the workers would otherwise get `gradeant-python-deps` stacked under `gradeant-pipeline-deps` (two installs of httpx/pydantic in one `/opt/python`). `pipeline-requirements.txt` is self-contained, boto3 included. `build.py --verify` unpacks each function's own layers only.
- Pins verified for manylinux2014_aarch64/cp313: boto3 1.44.4, PyYAML 6.0.3, pymupdf 1.26.0, google-genai **3.4.0** (the calls `llm.py` makes were checked in that wheel).
- Uploaded PDF names are made safe (`+`/space → `_` as the original, other characters → `_`), not rejected, so legacy uploads keep working; parsed-JSON names stay strict.
- Grader tests live in `tests/test_pipeline_grading.py` (Group B) next to `tests/test_pipeline_grader.py` (Group A); the end-to-end demo is `tests/test_upload_pipeline.py::test_end_to_end_upload_to_summary`.
- The rest of this file is the plan as approved.
Work in check-in rounds: finish one group, run its done-checks, report, wait for "go".
Never edit anything in `lambda_python_files/` (that is the rollback path). Never commit.

## Summary

- Port the PDF→JSON step (`excute_jsonifier.py` + `jsonifier.py`) and the grader (`grader_lambda.py`, live code from line 1759) into `python-backend` as two worker Lambdas, **`gradeant-jsonifier`** and **`gradeant-grader`**. Both are fed by SQS (S3 events). The grader is also invoked asynchronously by `gradeant-workflow` `startGrading`.
- Shared pipeline code goes in `gradeant_common/pipeline/`: S3 layout, S3 I/O, an LLM client interface (Gemini first), the grading core and SQS helpers. It goes there so `check_permissions.py` can follow it.
- Merge `upload_pdf.py` (`creat_url.py` is the same file) into the existing `functions/upload.py` as a `pipeline` key layout. There will be no second upload Lambda.
- Fix the verified defects. Failed SQS messages go back to the queue through `batchItemFailures`. Re-runs reuse the same row IDs (`uuid5`), so they no longer create duplicate rows. A Gemini error now fails the message instead of becoming 0 marks. The credentials and dotenv in the code are gone; the role and Secrets Manager are used instead. The questions in a submission are graded in parallel. The student ID comes from the upload key that the server builds.
- Packaging: one new layer, `gradeant-pipeline-deps` (PyYAML, PyMuPDF, google-genai). Only the two worker functions use it.

---

## 1. Findings: what the originals actually do (verified)

### 1.1 Contracts

| | `upload_pdf.py` (= `creat_url.py`, byte-identical, `cmp` checked) | `excute_jsonifier.py` + `jsonifier.py` | `grader_lambda.py` (live part, L1759–2548) |
|---|---|---|---|
| Trigger | API Gateway `POST /uploadpdf`. **No auth at all.** | SQS records whose body is an S3 event (`body.Records[0].s3`) | SQS records whose body is an S3 event. Needs `Records`; anything else → 400. **It does NOT accept the workflow invoke payload** (see 1.3). |
| Input | Body JSON, optionally wrapped in `{"metadata": {...}}`: `uploadType` (student/teacher/answer), `assignmentId`, `courseId` (= QP id), `files[{name,type,size}]`, `answerType` (answer only) | PDF key. Skips anything that isn't `.pdf` (returns 400 for that record) | JSON key. Anything that isn't `.json` → 400 for that record |
| S3 bucket | Hard-coded `assignmentgrader` | Comes from the event record (the JsonifierProcessor default is env `S3_BUCKET_NAME` or `assignmentgrader`) | **Ignores the event's bucket.** Uses `config s3_bucket` → env `S3_BUCKET_NAME` → `assignmentgrader` |
| S3 keys read | — | The PDF itself. QP PDF `<main>/teachers/QP_Rubric/<aid>__<qp>/QP_pdfs/<qp>.pdf` (text via fitz). Schema `<main>/teachers/QP_Rubric/<aid>__<qp>/QP_schemas/QP_schemas_answers/<qp>_schema_answer.txt`. `<main>` is **hard-coded `Data_warehouse`** in `extract_assignment_info` | `head` `<main>/teachers/QPRubricWithAnswers/<aid>__<qp>/solution_pdfs/<aid>_<qp>_answer.pdf` decides the base (`QPRubricWithAnswers` if it exists, else `QP_Rubric`). Then `<base>/QP_schemas/QP_schemas_json/<qp>_schema.json` and `<base>/rubric_jsons/<qp>_rubric.json`. `<main>` = the key's first segment |
| S3 keys written | Presigned PUT. Student: `Data_warehouse/students/assignments/<aid>[__<courseId>]/original_pdf/<safe name>`. Teacher: `Data_warehouse/teachers/QP_Rubric/<aid>__<courseId>/QP_pdfs/<safe name>`. Answer: `Data_warehouse/teachers/QPRubricWithAnswers/<aid>__<courseId>/solution_pdfs/<safe name>`. Also folder-marker objects `<path>/` | `<folder>/Parsed_pdfs/text/<safe>.txt`, `…/JSON_Seg/<name>.txt`, `…/JSON_Text/<name>.txt`, and `…/JSON/<name>.json` **written twice** (by the jsonifier, then copied by the executor to the same key). `<folder>` = key minus its last 2 segments | Only `run_pipeline` writes `<main>/students/assignments/<aid>__<qp>/results/total_marks.json`, and **the handler never calls `run_pipeline`**. So the live Lambda writes no S3 objects |
| DynamoDB | — | — | `assignment_grading` (env `DYNAMODB_TABLE`), via `batch_writer` (= **BatchWriteItem**). One row per QA pair: `evaluation_id` (uuid4), `assignment_id`, `qp_id`, `student_id` (= file base name), `question`, `student_answer`, `rubric`, `max_marks`, `question_number`, `subpart`, `answer_with`, [`standard_answer`], `marks`, `reasoning`, `feedback`, `timestamp`. Then one summary row: `type="summary"`, `evaluation_id`, `assignment_id`, `qp_id`, `student_id`, `total_marks`, `timestamp`, `answer_with` |
| LLM | — | Gemini `gemini-3-flash-preview` (constant in both files). Method 2 = `genai.upload_file` (the PDF goes to Google's file API). Method 1 = fitz text + prompt. The final fallback is raw fitz text. Then a segregation prompt (QP text + transcript) and a schema prompt (with up to 3 parse attempts) | Gemini `gemini_model` env (default `gemini-2.5-flash`), temperature 0.6. 2 calls per QA pair (grade, then feedback). Back-off on 429/quota |
| Env vars | — | `GEMINI_API_KEY`, `S3_BUCKET_NAME`, `region_name`, `aws_access_key_id`, `aws_secret_access_key` (+ `load_dotenv`) | `GEMINI_API_KEY`, `gemini_model`, `S3_BUCKET_NAME`, `DYNAMODB_TABLE`, `region_name` (default **us-east-1**), `aws_access_key_id`, `aws_secret_access_key` (+ `load_dotenv`) |
| Response | `{urls, fileIds, uploadId, message}` (+ `answerType`), CORS `*` | `{statusCode: 200 or 207, body}` | `{statusCode: 200 or 207, headers, body}` |
| Libraries | boto3 | boto3, google-generativeai, PyMuPDF (`fitz`), tqdm, python-dotenv | boto3, google-generativeai, python-dotenv |

### 1.2 Defects (each checked against the code), and what the port does

| # | Defect | Verified? | Port decision |
|---|---|---|---|
| D1 | SQS partial failure returns 207, but Lambda treats any return as success, so failed messages are **deleted** | Yes (both workers) | Return `{"batchItemFailures":[{"itemIdentifier": messageId}]}`. Set `FunctionResponseTypes: [ReportBatchItemFailures]` on the event source mapping. Set batch size 1 for the grader |
| D2 | `evaluation_id = uuid4()`, so a re-run (SQS retry, the double JSON write in D8, a re-upload) adds a second set of rows and a second summary | Yes | Deterministic `uuid5(NS, "aid|qp|sid|qnum|subpart")` and `uuid5(NS, "aid|qp|sid|summary")`. Before writing, delete this student's other rows for the same aid/qp (old uuid4 rows) |
| D3 | `int(exc.response["Error"]["Code"])` | **Partly.** The grader's `_object_exists` is dead code. The jsonifier's `object_exists` is live, and `head_object` returns `"404"`/`"403"`, so it only crashes on non-numeric codes (`SlowDown`, `RequestTimeout`, …) | One helper `s3io.exists()` that compares string codes (`404`, `NoSuchKey`, `NotFound`) and re-raises anything else |
| D4 | A Gemini failure turns into `"ERROR: …"` text, which parses to `marks_awarded: 0` (not `None`), so the retry loop never retries and the student silently gets 0. The feedback text can also be an error string | Yes | `LLMError` is raised after the retries run out. Per-question parsing is strict: JSON with a numeric `marks_awarded`, clamped to `[0, max_marks]`. A bad parse → retry, then `GradingError`. **Nothing is written for that student.** The message fails (D1), and after N tries it goes to the DLQ |
| D5 | `asyncio.gather` over blocking code = sequential | Yes | `ThreadPoolExecutor(max_workers=GRADER_MAX_WORKERS, default 4)` across the QA pairs of one submission |
| D6 | Static AWS keys from env, plus `load_dotenv` | Yes (both) | The Lambda role only. The Gemini key comes from Secrets Manager through `gradeant_common.secrets` (`GEMINI_API_KEY_ARN`). No dotenv, and no default bucket or model value that is secret |
| D7 | `student_id` = file base name chosen by whoever uploads | Yes | In the pipeline upload layout the server names the file: student uploads are stored as `<studentId>.pdf`, and a student's own `studentId` is forced from the token. The grader checks the stem against `^[A-Za-z0-9._-]{1,200}$` |
| D8 | The jsonifier writes `Parsed_pdfs/JSON/<name>.json` twice (the processor, then the executor's "copy" to the same key). That means 2 S3 events and 2 grader runs | Yes (same key whenever the name has no spaces) | One write |
| D9 | The record handler ignores `process_submission`'s result, so a missing QP or 0 QA pairs still returns 200 "Saved to DynamoDB" | Yes | A missing QP, schema or rubric, or 0 pairs, raises `PipelineInputError`. The message fails (it is retried, then goes to the DLQ) |
| D10 | The grader ignores the event's bucket and has hard-coded `us-east-1` / `Data_warehouse` defaults | Yes | The bucket comes from the record and must equal `PIPELINE_BUCKET` (anything else is rejected and logged). The root comes from the key. The region is `config.data_region()`, or `PIPELINE_BUCKET_REGION` |
| D11 | The key is decoded with `replace('+', ' ')` only, so `%xx` escapes are not decoded | Yes | `urllib.parse.unquote_plus` |
| D12 | `upload_pdf`: no auth. `handle_answer_upload` hits a `NameError` (`base_path`) when `courseId` is missing. The QP must be named exactly `<qp>.pdf` but keeps the user's filename. `handle_confirmation` is unreachable. Folder markers create extra S3 events | Yes | Covered by the merge (Group C) |
| D13 | `google-generativeai` has been deprecated by Google since 2025 (past end of support) | Upstream fact | Use the `google-genai` SDK behind the LLM interface (decision O1) |

### 1.3 Two entry shapes for the grader (important)

`tests/test_grader_contract.py` pins what `workflow._trigger_grading` sends, as an async `Event` invoke with this payload:
`{"assignment_id","qp_id","student_id","method","persona","partial_credit","rubric":[{question_number,subpart,title,max_marks,criteria}]}`.
The original grader returns 400 for this payload (it has no `Records`). So "Start grading" has never actually graded anything through the original. Also, `workflow.start_grading` only triggers students whose detail rows are **all `status="corrected"` and who have no summary row**. The original SQS grader writes a summary right away, so those students show as `ai_graded` and are never triggered.

The port handles both shapes in `functions/grader.py`:

| Event shape | Mode | What it does |
|---|---|---|
| `{"Records":[…]}` with `eventSource == "aws:sqs"` | **upload** | Parses the S3 key → loads the QP, rubric and student JSON → builds the QA pairs. If `GRADE_ON_UPLOAD=true` (default, the same as the original): grade, write the question rows and the summary. If `false`: write the question rows **without marks and without a summary** (→ `ocr_done`), and wait for the teacher's "Start grading" |
| Dict with `assignment_id`, `qp_id`, `student_id` (no `Records`) | **regrade** (from workflow) | Reads the student's existing non-summary rows (`student_id-index`, filtered by aid/qp). Uses `student_answer` (the student-corrected text) and `question` from the row. For each row, takes `max_marks` and `criteria` from the payload `rubric` matched on `(str(question_number), subpart or None)`, and falls back to the row's `rubric` / `max_marks`. Uses `persona` (balanced/strict/lenient). Grades → `update_item` on each row (`marks`, `reasoning`, `feedback`, `graded_at`, `status="graded"`) → writes the deterministic summary row **last**. An error raises before any write, so the rows stay `corrected` and the teacher can press "Start grading" again |
| Anything else | — | `ValueError("unsupported event")`. For async invokes, Lambda retries it and then sends it to the on-failure destination |

`method` and `partial_credit` go into the prompt (`partial_credit=false` → "award either full or zero marks per criterion"). `method` is stored on the summary row. See O5.

---

## 2. Target layout (new = N, changed = C)

```
python-backend/
  functions/grader.py                      N  SQS + invoke entry; thin, all logic in gradeant_common.pipeline
  functions/jsonifier.py                   N  SQS entry: PDF → text → segregated → JSON
  functions/upload.py                      C  + UPLOAD_LAYOUT=pipeline, metadata unwrap, ownership, fileIds/uploadId
  gradeant_common/secrets.py               C  + gemini_api_key()  (GEMINI_API_KEY / GEMINI_API_KEY_ARN)
  gradeant_common/pipeline/__init__.py     N
  gradeant_common/pipeline/paths.py        N  the one S3 layout (QP, rubric, schema, parsed, original_pdf, solution), key parsing
  gradeant_common/pipeline/s3io.py         N  exists/read_json/read_text/read_bytes/write_*/list_keys (paginated); bucket allow-check
  gradeant_common/pipeline/sqs.py          N  iter_s3_records(event) (skips s3:TestEvent), batch_response(failures)
  gradeant_common/pipeline/llm.py          N  LLMClient protocol, LLMError, GeminiClient (google-genai, lazy import), get_client()/set_client()
  gradeant_common/pipeline/pdf.py          N  fitz text extraction (lazy import)
  gradeant_common/pipeline/prompts.py      N  persona/instruction/feedback/segregation/schema prompts, copied VERBATIM from the originals
  gradeant_common/pipeline/grading.py      N  extract_qa_pairs, parse_marks, evaluate, grade_pairs (thread pool), row ids, write_rows
  gradeant_common/pipeline/jsonify.py      N  transcribe → segregate → to_json (port of JsonifierProcessor.process_file)
  pipeline-requirements.txt                N  PyYAML>=6, pymupdf>=1.24, google-genai>=1.0   (pin exact versions in step A4)
  requirements-dev.txt                     C  + -r pipeline-requirements.txt (pymupdf only needed by one test; see T-J)
  lambdas.py                               C  + 'grader', 'jsonifier' entries; new optional keys (see §5)
  build.py                                 C  layers per function; pipeline-deps layer; verify workers with an empty SQS event
  deploy/generate_template.py              C  workers: no API permission, per-function env, layers, ESMs, async config, DLQ, BatchWriteItem
  deploy/template.yaml                     C  regenerated
  check_permissions.py                     C  + 'sqs' in SERVICE map (only if code ever calls boto3 sqs; today it won't)
  tests/test_pipeline_grader.py            N
  tests/test_pipeline_jsonifier.py         N
  tests/test_upload_pipeline.py            N
  tests/test_build_registry.py             N  lambdas.py ↔ template consistency for worker functions
  README.md, DEPLOYMENT.md                 C  new functions, params, cut-over runbook (Group D)
```

Rule for everything in `gradeant_common/pipeline/`: **`fitz` and `google.genai` are imported only inside the functions that use them.** This matters for three reasons. The common layer is shared with the API functions, which don't have those packages. `build.py --verify` runs Linux-arm64 wheels on Windows, so native imports at module level would fail there. And the tests stub the LLM.

---

## 3. Steps

### Group A — foundations (no behaviour change for existing functions)

**A1. Gemini secret accessor.** `gradeant_common/secrets.py`: add `def gemini_api_key(): return _resolve('GEMINI_API_KEY')`, plus a docstring line. The secret may be a plain string or `{"GEMINI_API_KEY": "..."}` (existing `_fetch` rule). There is no default.
Done-check: new test in `tests/test_pipeline_grader.py::test_gemini_key_from_secrets_manager` (moto secretsmanager: the value is read from the ARN; '' when neither is set).

**A2. S3 layout + I/O.**
- `pipeline/paths.py`: `pipeline_bucket()` (env `PIPELINE_BUCKET`, required, no default), `pipeline_root()` (env `PIPELINE_ROOT`, default `Data_warehouse`, which is a folder name and not a secret), and `@dataclass Layout(root, assignment_id, qp_id)` with properties `folder` (`<aid>__<qp>`), `student_base`, `original_pdf_prefix`, `parsed_text(name)`, `parsed_seg(name)`, `parsed_json_text(name)`, `parsed_json(name)`, `parsed_json_prefix`, `qp_pdf`, `qp_schema_answer_txt`, `solution_pdf` (`…/QPRubricWithAnswers/<folder>/solution_pdfs/<aid>_<qp>_answer.pdf`), `rubric_base(has_solution)`, `qp_schema_json(has_solution)`, `rubric_json(has_solution)`. Also `parse_student_key(key) -> (Layout, kind, stem)`. `kind` is one of `original_pdf` or `parsed_json`, and anything else raises `PipelineInputError`. It requires `parts[1:3] == ['students','assignments']` and splits `parts[3]` on the first `__`. The assignment_id handling matches the original: `split('__')[0]`.
- `pipeline/s3io.py`: an `s3()` client cached with `region_name=env('PIPELINE_BUCKET_REGION') or config.data_region()`. `exists(bucket,key)` returns False on codes `404/NoSuchKey/NotFound`, and also on `403`. That 403 case is kept from the original: without `s3:ListBucket`, S3 answers 403 for a missing key. We grant ListBucket, so a 403 should not happen, but it is logged as a warning. Also `read_json`, `read_text`, `read_bytes`, `write_text`, `write_json`, `list_keys` (paginator), and `check_bucket(bucket)`, which raises unless `bucket == pipeline_bucket()`.

Done-check: `tests/test_pipeline_grader.py::test_paths_match_original_layout` builds the keys for `A1`/`QP1` with and without a solution PDF, and compares them with string literals copied from `grader_lambda.py` L1870–1886 and `jsonifier.py` L92–102. `::test_exists_handles_string_codes` covers moto missing key → False, and a stubbed `ClientError('SlowDown')` → raises.

**A3. LLM client interface.** `pipeline/llm.py`:
```python
class LLMError(Exception): ...
class LLMClient(Protocol):
    def generate(self, prompt: str, *, temperature: float, max_output_tokens: int | None = None) -> str: ...
    def transcribe_pdf(self, pdf_bytes: bytes, prompt: str, *, temperature: float, max_output_tokens: int) -> str: ...
class GeminiClient:   # google-genai: genai.Client(api_key=secrets.gemini_api_key())
    # generate → client.models.generate_content(model, contents=prompt, config=GenerateContentConfig(temperature=…))
    # transcribe_pdf → contents=[types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'), prompt]
    #   (inline bytes, no Files API upload/cleanup; PDFs ≤ UPLOAD_MAX_BYTES 10 MB are under the 20 MB inline limit)
    # retries: PIPELINE_LLM_RETRIES (default 3); 429/RESOURCE_EXHAUSTED/5xx → back-off min(60, 2*2**n) or the server's retry delay
    #   (port _extract_retry_delay); then raise LLMError. Empty response text → LLMError.
def get_client(purpose: str) -> LLMClient   # purpose 'grade' → env GRADER_MODEL (default gemini-2.5-flash),
                                            # 'parse' → env JSONIFIER_MODEL (default gemini-3-flash-preview); env PIPELINE_LLM=gemini (only value for now)
def set_client(client | None)               # tests only
```
A missing key raises `LLMError('Gemini API key not configured')` on first use, not at import.
Done-check: `::test_llm_retries_then_raises` uses a fake `google.genai` module inserted in `sys.modules` that raises 429 twice and then succeeds (the call succeeds); when it always fails, `LLMError` is raised. Patch `time.sleep`.

**A4. Registry + build + template support for worker functions (no new functions yet).**
- `lambdas.py`: document the new optional keys: `kind` (`'api'` default | `'worker'`), `layers` (default `['deps','common']`), `env` (dict of extra variables, values are template expressions), `queue_param` (template parameter holding the source SQS queue ARN), `ddb_batch` (bool → add `dynamodb:BatchWriteItem`), `s3_bucket_param`, `async_invoke` (bool → EventInvokeConfig + failure queue). Workers have `routes: []`.
- `pipeline-requirements.txt`: PyYAML, pymupdf, google-genai, with exact versions pinned after one `pip download --platform manylinux2014_aarch64 --python-version 3.13 --only-binary=:all:` succeeds.
- `build.py`: build a third layer `layer-gradeant-pipeline-deps.zip` from `pipeline-requirements.txt` (same pip flags, `dist/stage/pipeline`), but only if some function lists `'pipeline-deps'`. The per-function size check uses that function's own layers. `manifest.functions[n].layers` comes from the spec. `verify()` handles workers: for `kind=='worker'` invoke `{"Records": []}` and expect `{"batchItemFailures": []}`. Generalise `VERIFY_SCRIPT`'s status read (it currently expects an HTTP `statusCode`), e.g. emit `{"status": ..., "body": ...}` and compare per kind.
- `deploy/generate_template.py`, changes:
  - `function()`: per-function `Layers:` when `layers` differs from the default (it overrides Globals), per-function `Environment.Variables` from `env`, and `Description` that handles `routes == []`.
  - `statements()`: S3 statement for `s3_bucket_param`: `s3:GetObject, s3:PutObject` on `arn:aws:s3:::${PipelineBucketName}/${PipelineRoot}/*`, plus `s3:ListBucket` on the bucket with a `s3:prefix` condition `${PipelineRoot}/*`. `BatchWriteItem` when `ddb_batch` is set. A `gemini` secret → `!Ref GeminiSecretArn`.
  - ApiInvoke permissions only for `kind=='api'`.
  - New params: `PipelineBucketName` (''), `PipelineRoot` (`Data_warehouse`), `GeminiSecretArn` (''), `GraderModel` (`gemini-2.5-flash`), `JsonifierModel` (`gemini-3-flash-preview`), `GradeOnUpload` (`'true'`), `JsonifierQueueArn` (''), `GraderQueueArn` (''), `PipelineQueuesEnabled` (`'false'`).
  - Conditions `HasPipeline` (bucket and secret both set), `HasJsonifierQueue`, `HasGraderQueue`.
  - Per worker with `queue_param`: `AWS::Lambda::EventSourceMapping` (Condition, `Enabled: !Equals [!Ref PipelineQueuesEnabled, 'true']`, `BatchSize: 1`, `FunctionResponseTypes: [ReportBatchItemFailures]`, `ScalingConfig.MaximumConcurrency: 5`) plus a conditional IAM policy with `sqs:ReceiveMessage, sqs:DeleteMessage, sqs:GetQueueAttributes, sqs:ChangeMessageVisibility` on that ARN.
  - For `async_invoke`: an in-stack `AWS::SQS::Queue` `gradeant-grader-failures` (SSE on, 14-day retention), `AWS::Lambda::EventInvokeConfig` (`MaximumRetryAttempts: 1`, `MaximumEventAgeInSeconds: 3600`, `DestinationConfig.OnFailure` → that queue), and `sqs:SendMessage` on it in the grader role.
  - Log groups, alarm metrics and Outputs are unchanged but now include the workers.

Done-check: `tests/test_build_registry.py` imports `generate_template` and renders it in memory. Refactor `main()` into `render() -> str` plus a write (the only non-test change needed for this). It asserts the existing 9 API functions render **byte-identically to the current `deploy/template.yaml`** before the workers are added (run the check once at the end of A4, with no worker entries yet). Also `python build.py --verify` still passes for the 9 functions (venv `%TEMP%\g313`).

> **Check-in A:** report the new files, the test count, and that the API template is unchanged.

### Group B — grader

**B1. Prompts + core.**
- `pipeline/prompts.py`: copy `_default_persona`, `_default_instructions`, `_std_answer_instructions`, `_build_feedback_prompt(_with_std_solution)` **verbatim** from `grader_lambda.py` L1972–2060. Add a `PERSONAS` dict: `balanced` = the original default persona text; `strict` / `lenient` = the same text plus one sentence each (see O5; the wording is the user's call, and a placeholder sentence is marked TODO in a comment).
- `pipeline/grading.py`:
  - `extract_qa_pairs(qp, sol, std, include_std_answer)`: port L2082–2152 unchanged, but raise `PipelineInputError` on `KeyError`/`TypeError` instead of returning `[]`.
  - `build_eval_prompt(pair, persona, partial_credit=True)`: port L2221–2245. Keep the output text identical when the persona is the default and `partial_credit` is True, including the literal `{{`/`}}` in the format block. That is the original's behaviour, and changing it would change model output.
  - `parse_marks(text, max_marks) -> (marks, reasoning)`: JSON slice as in the original. `marks_awarded` must be int/float or a numeric string, otherwise `ValueError`. Clamp to `[0, max_marks]`.
  - `evaluate(pair, llm, persona, partial_credit)`: up to 3 attempts on `ValueError`. Then the feedback call, whose failure is non-fatal (`feedback = "Feedback generation encountered an error."`, the original's text). The grade call failing raises `GradingError`.
  - `grade_pairs(pairs, llm, **kw)`: `ThreadPoolExecutor(min(len(pairs), GRADER_MAX_WORKERS))`. Results come back in input order, and the first exception propagates.
  - `row_id(aid, qp, sid, qnum, subpart)` / `summary_id(aid, qp, sid)`: `uuid5(NAMESPACE_URL, 'gradeant:' + '|'.join(...))`.
  - `write_student(aid, qp, sid, rows, summary)`: query `student_id-index` (paginated, filter aid/qp). Delete the rows whose id is not in the new set (old uuid4 rows, plus a stale summary). `batch_writer` the detail rows. `put_item` the summary **last**. Values go through `dynamo.to_dynamo`. Note: the original `_to_dynamodb` keeps `None`, and `to_dynamo` drops `None` map values. That is acceptable: readers use `.get`. It is stated here so the reviewer knows.
- Row fields written = exactly the original set (§1.1), plus `graded_at` (ISO) and `method`/`persona` on the summary. `timestamp` stays epoch seconds. `answer_with` is unchanged.

Done-check: `tests/test_pipeline_grader.py` B-tests below pass.

**B2. `functions/grader.py`.**
```python
def lambda_handler(event, context):
    if isinstance(event, dict) and 'Records' in event:
        return _sqs(event)            # → {'batchItemFailures': [...]}
    if isinstance(event, dict) and {'assignment_id', 'qp_id', 'student_id'} <= event.keys():
        return _regrade(event)        # → {'ok': True, 'student_id': ..., 'total_marks': ...}; raises on failure
    raise ValueError('unsupported event shape')
```
- `_sqs`: for each `(message_id, bucket, key)` from `sqs.iter_s3_records` (a malformed body counts as a failure of that message):
  - `check_bucket`.
  - Key not `.json` or not `parsed_json` kind → log + **ack**. Skipping is correct, not a failure; the original returned 400 and deleted it too.
  - Load the QP (required), the rubric (optional) and the student JSON.
  - `GRADE_ON_UPLOAD`: grade + `write_student`; otherwise `write_student` with ungraded rows (no `marks`/`reasoning`/`feedback`, no summary).
  - Any exception → append `message_id` to the failures and log with `http.log('error', …)`.
- `_regrade`: validate the ids against the D7 regex. Rows come from `grading.student_rows(sid, assignment_id=, qp_id=, type='notSummary')` (existing helper in `gradeant_common/grading.py`). No rows → `PipelineInputError`. Map the rubric, grade, then `update_item` each row (by `evaluation_id`, keeping the existing ids) and put the summary.
- Logging: structured `http.log` with `aws_request_id`. No answer text in logs (privacy).

Done-check: B-tests + `pytest tests/test_grader_contract.py` (unchanged, still green) + new `::test_workflow_payload_is_accepted`.

**B3. Registry entry** (`lambdas.py`):
```python
'grader': {'kind': 'worker', 'routes': [], 'description': 'grading pipeline: grades parsed answer sheets (SQS) and workflow regrade invokes',
           'aws': ['dynamodb', 'secrets', 's3'], 'tables': ['assignment_grading'], 'secrets': ['gemini'],
           'layers': ['pipeline-deps', 'common'], 'memory': 1024, 'timeout': 900,
           'env': {'PIPELINE_BUCKET': '!Ref PipelineBucketName', 'PIPELINE_ROOT': '!Ref PipelineRoot',
                   'GEMINI_API_KEY_ARN': '!Ref GeminiSecretArn', 'GRADER_MODEL': '!Ref GraderModel',
                   'GRADE_ON_UPLOAD': '!Ref GradeOnUpload', 'GRADER_MAX_WORKERS': '4'},
           's3_bucket_param': 'PipelineBucketName', 'queue_param': 'GraderQueueArn', 'ddb_batch': True, 'async_invoke': True},
```
Regenerate `deploy/template.yaml`. Done-check: `python deploy/generate_template.py` → `test_build_registry.py` (worker has no ApiInvoke, has an ESM with ReportBatchItemFailures, BatchWriteItem, EventInvokeConfig, and the gemini secret only) → `python build.py --verify` (10 functions; the grader answers `{"batchItemFailures": []}` with the pipeline layer).

> **Check-in B:** show a sample row and summary from the moto test, the template diff for the grader, and the layer size.

### Group C — jsonifier + upload merge

**C1. `pipeline/pdf.py` + `pipeline/jsonify.py`.**
- `pdf.text_from_bytes(data) -> str`: fitz, `--- Page n ---` separators, the same as the original `extract_text_from_pdf_fallback`.
- `jsonify.transcribe(pdf_bytes, llm)`: the original order is method 2 (file → model) first, then method 1 (text → model) if the result is < 50 chars, then raw text if < 10 chars. Prompt constants come verbatim from `excute_jsonifier.py` (`PROMPT`, `GENERATION_CONFIG` max 5120 tokens, temperature 0.6).
- `jsonify.segregate(transcript, qp_text, llm)` and `jsonify.to_json(seg, schema_text, llm)`: prompts verbatim from `jsonifier.py` `get_prompt_segregation` / `get_prompt_schema`, max 8192 tokens. `to_json` makes 3 attempts using the original `output_to_json` parsing (port L289–325). Failure → `PipelineInputError`.
- Writes, in this order: `Parsed_pdfs/text/<stem>.txt`, `JSON_Seg/<stem>.txt`, `JSON_Text/<stem>.txt`, then **once** `JSON/<stem>.json`. `<stem>` = `safe_file_name(basename without .pdf)` (port: spaces/`+` → `_`).
- QP text comes from `layout.qp_pdf` and the schema from `layout.qp_schema_answer_txt`, using the **key's root**, not hard-coded `Data_warehouse`. Missing → `PipelineInputError` (the message fails and is retried; the teacher may still be uploading the QP).

**C2. `functions/jsonifier.py`.** The SQS loop works the same as the grader's:
- `check_bucket`.
- Not `.pdf`, or not `original_pdf` kind → ack and skip. This also stops the teacher QP/solution PDFs from being parsed as student answers if the bucket notification is broad.
- Idempotency: if `parsed_json(stem)` already exists → ack.
- Read the PDF bytes (no temp file), run C1, and return failures.

Registry entry like the grader's, but: `'tables': []`, `'aws': ['secrets','s3']`, memory 2048, timeout 900, env `JSONIFIER_MODEL: !Ref JsonifierModel` instead of the grader vars, `queue_param: 'JsonifierQueueArn'`, no `async_invoke`/`ddb_batch`.
Done-check: J-tests + `build.py --verify` (11 functions).

**C3. Merge `upload_pdf.py` behaviour into `functions/upload.py`.** Keep everything that is there, and add:
1. `parse_body` result: if it has a dict `metadata`, use that (the original accepts `{"metadata": {...}}`; the web app and the old static site send it unwrapped, as checked in `web/src/components/UploadPage.tsx` and `web/src/lib/upload.ts`).
2. `qpId` = `body.qpId or body.courseId` (the web app sends `courseId` for every type and `qpId` only for answers).
3. Env `UPLOAD_LAYOUT` = `template` (default, current behaviour) | `pipeline`. In `pipeline` mode keys come from `pipeline.paths.Layout(pipeline_root(), aid, qp)`, and `UPLOAD_BUCKET` must equal `PIPELINE_BUCKET`:
   - student → `original_pdf/<studentId>.pdf`. `studentId` = the caller for students. Teachers must send `studentId`. Exactly 1 file, otherwise 400 (one key per student).
   - teacher → `QP_pdfs/<qp>.pdf` (the name the jsonifier reads), exactly 1 file.
   - answer → `solution_pdfs/<aid>_<qp>_answer.pdf` (the name the grader checks), exactly 1 file.
   - `assignmentId` and `qpId` are required, with the D7 charset. For **students**, `qpId` is taken from the assignment record (`GradeAnt_Assignments.qp_id`, already loaded for the class check), and a client value that differs → 400.
4. Teacher/answer uploads (both layouts): the caller must own the assignment, `ownership.can_manage(caller, assignment_id)` (admins pass). If the assignment doesn't exist in `GradeAnt_Assignments`, allow it only in `template` layout (legacy flows create folders before the assignment exists; see O6).
5. Response `{urls, fileIds, uploadId}`: `fileIds` = the object keys' basenames, `uploadId` = `upload-<uuid4>`. The web app reads only `urls`, so this is additive.
6. **Not ported, by decision:** folder-marker objects (they trigger S3 events and S3 has no folders), `handle_confirmation` (unreachable in the original), and CORS `*` (the config-driven CORS stays).
7. Add an `upload` entry for `lambdas.py` `env`: `UPLOAD_LAYOUT`, `PIPELINE_ROOT`, `PIPELINE_BUCKET` (template params `UploadLayout` default `template`). Keep `UploadPolicy` and narrow it to `${UploadBucketName}/*`; it is unchanged.

Done-check: U-tests + the existing `tests/test_security*.py` / `test_new_endpoints.py` upload tests are still green.
Node twin: `backend/src/services/upload.service.js` claims the same behaviour. **Not changed in this plan** (O7).

> **Check-in C:** demo with moto: upload presign key → put a PDF → jsonifier (stub LLM) → grader (stub LLM) → rows in the table → workflow `start_grading` with `GRADE_ON_UPLOAD=false` → regrade invoke → summary.

### Group D — deploy wiring, permissions check, docs

**D1.** `python deploy/generate_template.py`; review the full diff of `deploy/template.yaml`.
**D2.** `check_permissions.py`: no code change expected. The S3/DynamoDB/secrets clients in `gradeant_common/pipeline/*` are found by the existing regex once the graph is refreshed. **Ask the user to run `graphify update .`** (do not run it yourself), then run `python check_permissions.py`. Expected: `ok gradeant-grader … reaches: dynamodb, s3, secrets`, the same for the jsonifier minus dynamodb, and upload unchanged. If `STALE` shows, the graph wasn't refreshed.
**D3.** Docs: `README.md` (function table: 2 workers; env vars `PIPELINE_*`, `GEMINI_API_KEY_ARN`, `GRADER_MODEL`, `JSONIFIER_MODEL`, `GRADE_ON_UPLOAD`, `UPLOAD_LAYOUT`; test count) and `DEPLOYMENT.md` (new Part B step "Grading pipeline", the cut-over runbook below, the DLQ check). No new .md files besides this plan.
**D4.** Final: full `pytest tests` in `%TEMP%\g313` **and** `%TEMP%\g310`. The worker code must not use 3.11+ syntax/stdlib (`tomllib`, `except*`), and `typing` must stay 3.10-safe. Also `python -m pyflakes functions gradeant_common`, `python build.py --verify`, and `python check_permissions.py`.

**Cut-over runbook (for DEPLOYMENT.md):**
1. Create the secret `gradeant/gemini` (value = the key currently on the legacy Lambdas, copied by the user; never pasted in chat or the repo).
2. `aws s3api get-bucket-location --bucket assignmentgrader` and `aws s3api get-bucket-notification-configuration --bucket assignmentgrader`: record the queue ARNs and prefixes/suffixes.
3. Deploy with `PipelineQueuesEnabled=false`, `GradeOnUpload=true`, and `GradingFunctionName` still set to its current value.
4. Smoke test: invoke `gradeant-grader` with a synthetic SQS event that points at an existing parsed JSON of a test assignment, and compare the rows with the legacy rows.
5. Disable the legacy Lambdas' SQS event source mappings, then redeploy with `PipelineQueuesEnabled=true`. **Never run both on one queue**, because they would compete for the messages.
6. Set the queues' visibility timeout to ≥ 6 × 900 s = 5400 s (AWS guidance for SQS → Lambda), and give each source queue a redrive policy to a DLQ (maxReceiveCount 3) if it lacks one.
7. Set `GradingFunctionName=gradeant-grader`, which turns on the workflow → grader path.
8. Later, and only as a separate decision: `UPLOAD_LAYOUT=pipeline` + `UploadBucketName=assignmentgrader`, and switch the `/uploadpdf` route (existing DEPLOYMENT step B12).

---

## 4. Test plan (pytest + moto, LLM always stubbed, no network)

Shared fixture in each new test file: env as in `test_new_endpoints.py` (`AWS_DEFAULT_REGION=eu-north-1`, fake keys, `JWT_SECRET`, `PIPELINE_BUCKET=test-bucket`, `PIPELINE_ROOT=Data_warehouse`). `mock_aws()` creates the bucket and the `assignment_grading` table with `student_id-index` and `type-index` (copy `_tables` from `test_workflow.py`). Call `dynamo.reset()`. `llm.set_client(FakeLLM(script))`, where `FakeLLM` returns queued strings per call and records the prompts. Also a `sqs_event(bucket, key, message_id)` helper that wraps a real-looking S3 event JSON.

**Grader (`tests/test_pipeline_grader.py`):**
| Test | Asserts |
|---|---|
| `test_paths_match_original_layout` | §A2 |
| `test_exists_handles_string_codes` | D3 |
| `test_gemini_key_from_secrets_manager` | A1 |
| `test_llm_retries_then_raises` | A3 / D4 |
| `test_prompt_matches_original` | `build_eval_prompt` for a pair (with and without `standard_answer`) equals the string produced by the **original** `GradingPipeline._build_evaluation_prompt` (load `grader_lambda.py` with `importlib`, stub `google.generativeai` + `dotenv` in `sys.modules`, bypass `__init__` with `object.__new__`) |
| `test_extract_qa_pairs_matches_original` | Same QP, solution and rubric fixture (with subparts) → identical pair lists from the original and the port |
| `test_sqs_grades_and_writes_rows` | 2 questions (one with subparts a/b) → 3 detail rows + 1 summary. The fields equal the original set. `total_marks` = sum. `student_id` = stem. Response `{'batchItemFailures': []}` |
| `test_sqs_rerun_is_idempotent` | Same event twice → still 3+1 rows, same ids. A pre-seeded legacy uuid4 row for that student/aid/qp is deleted |
| `test_llm_failure_reports_batch_item_failure_and_writes_nothing` | FakeLLM raises `LLMError` → `batchItemFailures == [{'itemIdentifier': 'm1'}]`, table empty |
| `test_unparseable_marks_retry_then_fail` | 3 non-JSON replies → failure, no rows (D4) |
| `test_marks_are_clamped` | Reply 99 for max 5 → 5 |
| `test_missing_qp_fails_message` | D9 |
| `test_other_bucket_rejected` / `test_non_json_key_acked` | D10; a skip is not a failure |
| `test_s3_test_event_acked` | An `s3:TestEvent` body → no failure |
| `test_grade_on_upload_false_writes_ungraded_rows` | No `marks`, no summary. Then workflow `_derived_status` on those rows → `ocr_done` |
| `test_workflow_payload_is_accepted` | Build the payload with `workflow._trigger_grading` + `FakeLambda` (reuse from `test_grader_contract.py`) and feed `json.loads(Payload)` to `grader.lambda_handler`. Rows seeded with `status='corrected'` → updated in place (same ids), the rubric criteria from the payload appear in the prompt, the summary is written, and the workflow `_derived_status` → `ai_graded` |
| `test_regrade_failure_leaves_rows_corrected` | LLM error → raises, rows still `corrected`, no summary |
| `test_unsupported_event_raises` | `{}` → `ValueError` |
| `test_questions_graded_concurrently` | FakeLLM with a `threading.Barrier(2)` → completes (proves ≥ 2 in flight) |

**Existing contract test:** `tests/test_grader_contract.py` stays unchanged and must stay green. It pins the sender, and `test_workflow_payload_is_accepted` pins the receiver.

**Jsonifier (`tests/test_pipeline_jsonifier.py`):** `test_pdf_to_json_single_write` (a moto bucket with the QP PDF + schema txt + a student PDF made with fitz in the test; `pytest.importorskip('fitz')`; FakeLLM returns transcript → segregation → JSON). It asserts the 4 output keys, that `JSON/<stem>.json` was put exactly once (count PUTs via a `put_object` spy), and the parsed content. Also `test_existing_json_skips`, `test_teacher_pdf_ignored`, `test_bad_json_three_attempts_then_failure`, `test_missing_schema_fails_message`, and `test_root_from_key_not_hardcoded` (root `Other_root`).

**Upload (`tests/test_upload_pipeline.py`):** In pipeline layout:
- A student's key is `…/<aid>__<qp from assignment>/original_pdf/<caller id>.pdf`, and the `qpId` mismatch → 400.
- 2 files → 400.
- Teacher key `QP_pdfs/<qp>.pdf`, answer key `solution_pdfs/<aid>_<qp>_answer.pdf`.
- A teacher who doesn't own the assignment → 403, admin → 200.
- The `{"metadata": {...}}` wrapper works, the response has `fileIds`/`uploadId`, and template layout is byte-for-byte as before (existing tests).

**Registry (`tests/test_build_registry.py`):** the API part of the template is unchanged. The workers have no ApiInvoke and do have the ESM, BatchWriteItem (grader only), EventInvokeConfig (grader only), and the gemini secret only. Every `FUNCTIONS` entry has a `functions/<name>.py` with `lambda_handler`.

---

## 5. Deploy / IAM summary

| Function | Trigger | Layers | Memory/timeout | IAM (generated) | Tables |
|---|---|---|---|---|---|
| `gradeant-grader` | SQS ESM on `GraderQueueArn` (batch 1, ReportBatchItemFailures, max concurrency 5) + async invoke from `gradeant-workflow` | pipeline-deps, common | 1024 MB / 900 s | DDB Get/Put/Update/Delete/Query/Scan/**BatchWriteItem** on `assignment_grading` (+ `/index/*`); S3 Get/Put + ListBucket (prefix `${PipelineRoot}/`); secrets `GeminiSecretArn`; SQS poll on the source queue; `sqs:SendMessage` on `gradeant-grader-failures` | `assignment_grading` |
| `gradeant-jsonifier` | SQS ESM on `JsonifierQueueArn` (batch 1, ReportBatchItemFailures, max concurrency 5) | pipeline-deps, common | 2048 MB / 900 s | S3 Get/Put + ListBucket (prefix); secrets `GeminiSecretArn`; SQS poll | — |
| `gradeant-upload` (changed) | `POST /uploadpdf` | deps, common | unchanged | unchanged (`UploadPolicy` on `${UploadBucketName}/*`) | `Users`, `GradeAnt_Assignments` |
| `gradeant-workflow` (unchanged code) | — | — | — | `GradingPolicy` unchanged: invoke `${GradingFunctionName}` | — |

- The ESMs and their SQS policies are separate conditional resources, so an empty queue parameter means no mapping. `PipelineQueuesEnabled` lets you deploy first and switch later.
- No secret values appear in the template, only `GeminiSecretArn`. The region is `DataRegion` (eu-north-1). The bucket region comes from `PIPELINE_BUCKET_REGION` only if step 2 of the runbook shows it isn't eu-north-1.
- `check_permissions.py`: run after the user refreshes the graph (D2). Its SERVICE map doesn't need `sqs` (no boto3 sqs calls in code; the polling permission comes from the template).

---

## 6. Risks and rollback

| Risk | Mitigation |
|---|---|
| Gemini model ids retired (`gemini-2.5-flash` is old by now; `gemini-3-flash-preview` is a preview) | Both are template params (`GraderModel`, `JsonifierModel`), so changing one needs no code change. The smoke test in runbook step 4 catches a dead model before cut-over |
| The grading output changes vs. legacy (new SDK, clamp, strict parse) | Prompts are verbatim and pinned by `test_prompt_matches_original`. Run runbook step 4 side by side on one assignment |
| Both the legacy and the new consumers on the same queue | Runbook step 5 order. `PipelineQueuesEnabled=false` by default |
| Layer > 250 MB or a missing aarch64 wheel (PyMuPDF/grpc) | `build.py` enforces the limit. A4 pins the versions only after a successful `pip download` for manylinux2014_aarch64/cp313. Fallback: container image (O3) |
| `build.py --verify` can't import Linux-native wheels on Windows | Lazy imports. The verify event (`{"Records": []}`) never touches fitz/genai. Optional: `sam local invoke` in Docker for a real import test |
| Student PDFs go to Google (Gemini) | Already true today. It is called out in O1 for the user to confirm |
| Regrade of a student whose rows came from the legacy grader (no `status`) | `startGrading` only triggers rows that are all `corrected`, so legacy rows are untouched until a student corrects them |
| `to_dynamo` drops `None` values (e.g. `subpart: None`) | Readers use `.get('subpart')`. The `row_id` key uses `'-'` for None, so the ids stay stable |

**Rollback** (no code revert needed; `lambda_python_files/` is untouched):
1. Set `PipelineQueuesEnabled=false`, then re-enable the legacy Lambdas' SQS mappings.
2. Set `GradingFunctionName` back to its previous value, or '' to turn it off.
3. Keep `UPLOAD_LAYOUT=template`, or keep the `/uploadpdf` route on the legacy Lambda.

Rows written by the new grader stay readable by every reader, because they have the same fields. Deterministic ids don't conflict with the legacy uuid4 rows, but a legacy re-run would add duplicates again. That is the legacy behaviour.

---

## 7. Open decisions (need the user)

| # | Question | Recommendation |
|---|---|---|
| **O1** | Gemini or Bedrock (Claude) for the ported pipeline? | **Keep Gemini now**, for behaviour parity, behind `LLMClient`. Store the key in Secrets Manager (`gradeant/gemini`). Move to the `google-genai` SDK, because `google-generativeai` is past end of support. A `BedrockClient` (the same Anthropic Bedrock client that `workflow.py` uses) can be added later as `PIPELINE_LLM=bedrock` without touching the grading code. Please confirm that it's acceptable for student PDFs to keep going to Google |
| **O2** | Name of the grading Lambda, and how `GRADING_FUNCTION_NAME` is wired | Names **`gradeant-grader`** and **`gradeant-jsonifier`**. Following `gradeant-<name>` keeps IAM, logs and alarms generated, and the contract test already uses `gradeant-grader`. Keep the existing `GradingFunctionName` parameter as an **explicit** switch. Set it to `gradeant-grader` at runbook step 7, and roll back by setting the old name. Alternative: default it to the in-stack function. That's one less manual step, but it would turn grading on at the first deploy |
| **O3** | Packaging PyMuPDF + Gemini SDK | **A zip layer `gradeant-pipeline-deps`** (PyYAML, pymupdf, google-genai), used only by the 2 workers. The estimate is well under 250 MB unzipped, and `build.py` enforces the limit. That keeps one build/deploy flow (`build.py` → SAM). A container image is the fallback only if the size check fails |
| **O4** | `GRADE_ON_UPLOAD`: grade as soon as answers are parsed (legacy), or write ungraded rows and wait for the student's corrections and the teacher's "Start grading" (the new workflow's intended chain) | **`true` at cut-over** (parity, while the legacy UI is still used). Switch to `false` once the new teacher/student workflow screens are live. With `true`, "Start grading" has nothing to trigger (the students are already `ai_graded`) |
| **O5** | Persona texts for `strict` / `lenient`, and what `method` (`ai` / `standard`) should change in grading | Treat `balanced` as the original "QRee" persona. Please supply one or two sentences for strict and lenient. For now, store `method` on the summary only. Ask what `standard` should mean (use the solution PDF? that's already automatic) |
| **O6** | Teacher/answer uploads for an assignment that isn't in `GradeAnt_Assignments` yet | In `pipeline` layout, reject (require the assignment, owned by the caller). The `template` layout stays permissive, as today |
| **O7** | The Node twin (`backend/src/services/upload.service.js`) has the same upload behaviour | Leave Node unchanged in this port. The Python `template` layout stays identical to it. Mirror `UPLOAD_LAYOUT=pipeline` in Node only if Node is still going to be deployed |
| **O8** | Producers of `QP_schemas_json/<qp>_schema.json`, `rubric_jsons/<qp>_rubric.json` and `QP_schemas_answers/<qp>_schema_answer.txt` are **not in this repo** (searched all `.py/.js/.ts/.md`) | Please confirm where these come from (another Lambda, or a manual or teacher step). The port only reads them. If a Lambda produces them, it is a follow-up port |
| **O9** | `results/total_marks.json` is only written by `run_pipeline`, which the live handler never calls, and nothing in the repo reads it | **Don't port it** (dead path). Say if an external consumer needs it |
| **O10** | Region of bucket `assignmentgrader` and the existing S3→SQS notification and queue settings (outside the repo) | The user runs the 2 `aws s3api` commands in runbook step 2 before Group D. They set `PIPELINE_BUCKET_REGION` and the queue ARNs |
