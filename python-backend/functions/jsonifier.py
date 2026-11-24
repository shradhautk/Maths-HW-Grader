"""Lambda gradeant-jsonifier (handler jsonifier.lambda_handler). Not behind API Gateway.

Port of lambda_python_files/excute_jsonifier.py + jsonifier.py. Fed by SQS with S3 events for
<root>/students/assignments/<aid>__<qp>/original_pdf/<student>.pdf: transcribes the answer
sheet, segregates it by question, converts it to the question paper's JSON schema and writes
Parsed_pdfs/JSON/<student>.json once, which triggers gradeant-grader.

Returns {"batchItemFailures": [...]}: only failed messages go back to the queue. Anything that
isn't a student's uploaded PDF (question papers, solutions, parsed files) is acknowledged and
skipped. An answer sheet that was already parsed is skipped too (idempotent).

Settings: PIPELINE_BUCKET, PIPELINE_ROOT, GEMINI_API_KEY(_ARN), GEMINI_MODEL. No answer text is logged.
"""
from gradeant_common.http import log
from gradeant_common.pipeline import jsonify, llm, paths, s3io, sqs


def parse_upload(bucket, key):
    """Returns a short outcome string; raises on anything that should be retried."""
    s3io.check_bucket(bucket)
    if '/original_pdf/' not in key or not key.lower().endswith('.pdf'):
        return 'skipped (not a student answer sheet)'
    layout, kind, student_id = paths.parse_student_key(key)
    if kind != 'original_pdf':
        return 'skipped (not a student answer sheet)'
    if s3io.exists(bucket, layout.parsed_json(student_id)):
        return 'skipped (already parsed)'
    out = jsonify.process(bucket, layout, student_id, s3io.read_bytes(bucket, key), llm.get_client('parse'))
    log('info', 'jsonifier: answer sheet parsed', assignment_id=layout.assignment_id, qp_id=layout.qp_id,
        student_id=student_id, output=out)
    return 'parsed'


def lambda_handler(event, context):
    request_id = getattr(context, 'aws_request_id', 'local')
    if not isinstance(event, dict) or 'Records' not in event:
        raise ValueError('unsupported event shape')
    failures = []
    for message_id, record in sqs.messages(event):
        try:
            for bucket, key in sqs.s3_objects(record):
                outcome = parse_upload(bucket, key)
                if outcome.startswith('skipped'):
                    log('info', 'jsonifier: ' + outcome, key=key, messageId=message_id, requestId=request_id)
        except Exception as exc:  # noqa: BLE001 — every failure goes back to the queue, then the DLQ
            log('error', 'jsonifier: message failed', messageId=message_id, requestId=request_id,
                error=type(exc).__name__, detail=str(exc)[:500])
            failures.append(message_id)
    return sqs.batch_response(failures)
