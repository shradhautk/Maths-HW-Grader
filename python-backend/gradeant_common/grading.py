"""assignment_grading queries (backend/src/repositories/grading.repository.js).

One row per question per student plus one type="summary" row per student, keyed by
evaluation_id. GSIs: student_id-index, type-index and (optional) assignment_id-index.
Every query follows pagination.

config.database() picks the store (config/app.yaml deployment, or DATABASE_PROVIDER):
  dynamodb  boto3, as always
  supabase  PostgREST (pg_rest) on gradeant.assignment_grading: reads by the generated
            columns, writes only through the two SQL functions (backend/sql/002_pipeline_api.sql)
The grading pipeline uses student_rows / patch_row / replace_rows, which work on both.
"""
from decimal import Decimal

from botocore.exceptions import ClientError

from . import config, dynamo, pg_rest

PG_TABLE = 'assignment_grading'


def _supabase():
    return config.database() == 'supabase'


def _table():
    return dynamo.table('grading')


def clean_none(value):
    """Drops None map values recursively (what to_dynamo does), keeping floats as floats for JSON."""
    if isinstance(value, dict):
        return {k: clean_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [clean_none(v) for v in value]
    return value


def _same(a, b):
    """DynamoDB equality: numbers compare by value (1 == 1.0, never == "1"); everything else exactly."""
    numeric = (int, float, Decimal)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, numeric) and isinstance(b, numeric):
        return a == b
    if isinstance(a, numeric) or isinstance(b, numeric):
        return False
    return a == b


def get_row(evaluation_id):
    return _table().get_item(Key={'evaluation_id': evaluation_id}).get('Item')


def update_row(evaluation_id, **params):
    return _table().update_item(Key={'evaluation_id': evaluation_id}, **params)


def student_rows(student_id, *, assignment_id=None, qp_id=None, question_number=None, type=None):  # noqa: A002
    """Rows for one student; filters are ANDed. type: 'summary' | 'notSummary'."""
    if _supabase():
        return _pg_student_rows(student_id, assignment_id, qp_id, question_number, type)
    names, values, conditions = {}, {':sid': student_id}, []
    if assignment_id is not None:
        names['#aid'] = 'assignment_id'
        values[':aid'] = assignment_id
        conditions.append('#aid = :aid')
    if qp_id is not None:
        names['#qp'] = 'qp_id'
        values[':qp'] = qp_id
        conditions.append('#qp = :qp')
    if question_number is not None:
        names['#qn'] = 'question_number'
        values[':qn'] = question_number
        conditions.append('#qn = :qn')
    if type:
        names['#type'] = 'type'
        values[':summary'] = 'summary'
        conditions.append('#type = :summary' if type == 'summary' else '#type <> :summary')
    args = {'IndexName': 'student_id-index', 'KeyConditionExpression': 'student_id = :sid', 'ExpressionAttributeValues': values}
    if conditions:
        args['FilterExpression'] = ' AND '.join(conditions)
    if names:
        args['ExpressionAttributeNames'] = names
    return dynamo.query_all(_table(), **args)


def _pg_student_rows(student_id, assignment_id, qp_id, question_number, type):  # noqa: A002
    """Same rows and semantics as the DynamoDB query: `type <> summary` keeps rows without a type."""
    filters = {'student_id': student_id}
    if assignment_id is not None:
        filters['assignment_id'] = assignment_id
    if qp_id is not None:
        filters['qp_id'] = qp_id
    rows = [r['item'] for r in pg_rest.select(PG_TABLE, filters)]
    if question_number is not None:
        rows = [r for r in rows if 'question_number' in r and _same(r['question_number'], question_number)]
    if type == 'summary':
        rows = [r for r in rows if r.get('type') == 'summary']
    elif type:
        rows = [r for r in rows if r.get('type') != 'summary']
    return rows


def patch_row(evaluation_id, set_fields, remove=()):
    """UpdateItem "SET k = v, … REMOVE r, …" without a condition (creates the row if missing)."""
    if _supabase():
        return pg_rest.rpc('grading_patch', {'p_id': evaluation_id, 'p_set': clean_none(set_fields), 'p_remove': list(remove)})
    names, values, sets = {}, {}, []
    for i, (key, value) in enumerate(set_fields.items()):
        names[f'#k{i}'] = key
        values[f':v{i}'] = value
        sets.append(f'#k{i} = :v{i}')
    expression = 'SET ' + ', '.join(sets)
    if remove:
        for i, key in enumerate(remove):
            names[f'#r{i}'] = key
        expression += ' REMOVE ' + ', '.join(f'#r{i}' for i in range(len(remove)))
    return update_row(evaluation_id, UpdateExpression=expression, ExpressionAttributeNames=names,
                      ExpressionAttributeValues=dynamo.to_dynamo(values))


def replace_rows(rows, delete_ids=()):
    """Deletes `delete_ids` and writes `rows` (whole items). One transaction on Supabase; a batch on DynamoDB."""
    if _supabase():
        return pg_rest.rpc('grading_replace_rows', {'p_rows': [clean_none(r) for r in rows], 'p_delete': list(delete_ids)})
    with _table().batch_writer() as batch:
        for evaluation_id in delete_ids:
            batch.delete_item(Key={'evaluation_id': evaluation_id})
        for row in rows:
            batch.put_item(Item=dynamo.to_dynamo(row))
    return len(rows)


def summary_rows(assignment_id, qp_id):
    """Every student's summary row for one assignment / question paper (type-index)."""
    if _supabase():
        raise NotImplementedError('summary_rows is not used by the grading pipeline on Supabase (the Node API reads it)')
    return dynamo.query_all(
        _table(),
        IndexName='type-index',
        KeyConditionExpression='#type = :summary',
        FilterExpression='#aid = :aid AND #qp = :qp',
        ExpressionAttributeNames={'#type': 'type', '#aid': 'assignment_id', '#qp': 'qp_id'},
        ExpressionAttributeValues={':summary': 'summary', ':aid': assignment_id, ':qp': qp_id},
    )


def assignment_rows(assignment_id, qp_id):
    """All rows via the assignment_id GSI, or None when that index doesn't exist."""
    if _supabase():
        raise NotImplementedError('assignment_rows is not used by the grading pipeline on Supabase (the Node API reads it)')
    try:
        rows = dynamo.query_all(
            _table(),
            IndexName=config.grading_assignment_index(),
            KeyConditionExpression='#aid = :aid',
            FilterExpression='#qp = :qp',
            ExpressionAttributeNames={'#aid': 'assignment_id', '#qp': 'qp_id'},
            ExpressionAttributeValues={':aid': assignment_id, ':qp': qp_id},
        )
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in ('ValidationException', 'ResourceNotFoundException'):
            return None
        raise
    return [r for r in rows if r.get('student_id')]
