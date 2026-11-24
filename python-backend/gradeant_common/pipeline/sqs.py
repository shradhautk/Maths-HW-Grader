"""SQS → S3-event plumbing for the pipeline workers.

Each SQS message body is an S3 event notification. Failed messages are reported through
batchItemFailures (the event source mapping needs ReportBatchItemFailures), so only those
return to the queue; the originals answered 207 and every message was deleted.
"""
import json
from urllib.parse import unquote_plus


def messages(event):
    """(messageId, record) for every SQS record."""
    for record in (event or {}).get('Records') or []:
        yield record.get('messageId') or '', record


def s3_objects(record):
    """[(bucket, key)] in one message. S3's s3:TestEvent → []. A malformed body raises ValueError."""
    try:
        body = json.loads(record['body'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'SQS message body is not JSON: {exc}') from exc
    if not isinstance(body, dict):
        raise ValueError('SQS message body is not an object')
    if body.get('Event') == 's3:TestEvent':
        return []
    try:
        return [(r['s3']['bucket']['name'], unquote_plus(r['s3']['object']['key'])) for r in body['Records']]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Not an S3 event: {exc}') from exc


REGRADE_KEYS = {'assignment_id', 'qp_id', 'student_id'}


def regrade_job(record):
    """The "Start grading" payload when the message is one (grading: sqs), else None.

    Same body as the asynchronous invoke (pinned by tests/test_grader_contract.py). An S3 event
    has a top-level "Records" list instead, so the two shapes can't be confused.
    """
    try:
        body = json.loads(record['body'])
    except (KeyError, TypeError, ValueError):
        return None  # s3_objects() reports the malformed body
    if isinstance(body, dict) and 'Records' not in body and REGRADE_KEYS <= body.keys():
        return body
    return None


def batch_response(failed_message_ids):
    return {'batchItemFailures': [{'itemIdentifier': m} for m in failed_message_ids]}
