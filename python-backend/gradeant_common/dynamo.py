"""DynamoDB access shared by every function (backend/src/aws/dynamodb.js).

table('users') etc. resolve names from config.TABLE_ENV. query_all/scan_all
follow LastEvaluatedKey, so results over DynamoDB's 1 MB page are complete.
"""
import functools
from decimal import Decimal

import boto3

from . import config


@functools.lru_cache(maxsize=None)
def _resource():
    return boto3.resource('dynamodb', region_name=config.data_region())


def table(key):
    """A boto3 Table by logical name ('users', 'grading', 'assignments', ...)."""
    if config.database() == 'supabase':
        # Anything not converted to the Supabase path fails loudly instead of writing to DynamoDB.
        raise RuntimeError('DynamoDB is not used when database: supabase (config/app.yaml / DATABASE_PROVIDER)')
    return _resource().Table(config.table_name(key))


def reset():
    """Tests only: drop the cached client (e.g. after starting a new moto mock)."""
    _resource.cache_clear()


def _all(call, kwargs):
    items, start = [], None
    while True:
        args = dict(kwargs)
        if start:
            args['ExclusiveStartKey'] = start
        page = call(**args)
        items.extend(page.get('Items', []))
        start = page.get('LastEvaluatedKey')
        if not start:
            return items


def query_all(tbl, **kwargs):
    return _all(tbl.query, kwargs)


def scan_all(tbl, **kwargs):
    return _all(tbl.scan, kwargs)


def to_dynamo(value):
    """floats → Decimal, recursively; None map values dropped (DynamoDB rejects float)."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: to_dynamo(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [to_dynamo(v) for v in value]
    return value


def from_dynamo(value):
    """Decimal → int/float, recursively (for JSON and arithmetic)."""
    if isinstance(value, list):
        return [from_dynamo(v) for v in value]
    if isinstance(value, dict):
        return {k: from_dynamo(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    return value


def is_conditional_check_failed(err):
    return getattr(err, 'response', {}).get('Error', {}).get('Code') == 'ConditionalCheckFailedException'


def table_named(name):
    """A boto3 Table by its actual name (for code that already resolved the name)."""
    return _resource().Table(name)
