"""S3 reads and writes for the pipeline bucket. Credentials come from the Lambda role only."""
import json
import logging

import boto3
from botocore.exceptions import ClientError

from ..config import data_region, env
from . import PipelineInputError
from .paths import pipeline_bucket

log = logging.getLogger(__name__)

_MISSING = {'404', 'NoSuchKey', 'NotFound'}


_client = None


def s3():
    global _client
    if _client is None:
        _client = boto3.client('s3', region_name=env('PIPELINE_BUCKET_REGION') or data_region())
    return _client


def reset():
    """Tests only: drop the cached client."""
    global _client
    _client = None


def check_bucket(bucket):
    """Events must come from the configured pipeline bucket; anything else is refused."""
    if bucket != pipeline_bucket():
        raise PipelineInputError(f'Event from unexpected bucket {bucket!r}')


def exists(bucket, key):
    """True if the object exists. Missing → False; any other error is raised (the original crashed on non-numeric codes)."""
    try:
        s3().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = str(exc.response.get('Error', {}).get('Code', ''))
        if code in _MISSING:
            return False
        if code == '403':
            # Without s3:ListBucket S3 answers 403 for a missing key. We grant ListBucket, so this is unexpected.
            log.warning('403 checking s3://%s/%s; treating as missing', bucket, key)
            return False
        raise


def read_bytes(bucket, key):
    return s3().get_object(Bucket=bucket, Key=key)['Body'].read()


def read_text(bucket, key):
    return read_bytes(bucket, key).decode('utf-8')


def read_json(bucket, key):
    return json.loads(read_text(bucket, key))


def write_text(bucket, key, text, content_type='text/plain; charset=utf-8'):
    s3().put_object(Bucket=bucket, Key=key, Body=text.encode('utf-8'), ContentType=content_type)


def write_json(bucket, key, data):
    write_text(bucket, key, json.dumps(data), 'application/json')


def list_keys(bucket, prefix):
    """Every object key under prefix (paginated), excluding folder markers."""
    keys = []
    for page in s3().get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o['Key'] for o in page.get('Contents', []) if not o['Key'].endswith('/'))
    return keys
