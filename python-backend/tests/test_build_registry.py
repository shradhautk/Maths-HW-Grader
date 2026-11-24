"""lambdas.py ↔ deploy/template.yaml: the committed template is exactly what the generator
renders, and worker functions get the pipeline pieces (and API functions don't change)."""
import copy
import os
import sys

import yaml

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [os.path.join(ROOT, 'deploy'), ROOT]

import generate_template  # noqa: E402
from lambdas import FUNCTIONS  # noqa: E402


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):  # CloudFormation short tags (!Ref, !Sub, !If, ...)
    if isinstance(node, yaml.ScalarNode):
        return {suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {suffix: loader.construct_sequence(node, deep=True)}
    return {suffix: loader.construct_mapping(node, deep=True)}


_Loader.add_multi_constructor('!', _tag)


def _load(text):
    return yaml.load(text, Loader=_Loader)


def test_committed_template_is_generated():
    with open(os.path.join(ROOT, 'deploy', 'template.yaml'), encoding='utf-8') as fh:
        assert generate_template.render() == fh.read(), 'run: python deploy/generate_template.py'
    with open(os.path.join(ROOT, 'deploy', 'pipeline-template.yaml'), encoding='utf-8') as fh:
        assert generate_template.render(pipeline_only=True) == fh.read(), 'run: python deploy/generate_template.py'


def test_pipeline_only_template_is_slim():
    """Hybrid deployment: only the workers; nothing that costs money or belongs to the API stack."""
    tpl = _load(generate_template.render(pipeline_only=True))
    res, params = tpl['Resources'], tpl['Parameters']
    assert sorted(k for k in res if k.endswith('Function')) == ['GraderFunction', 'JsonifierFunction']
    for absent in ('SessionSecret', 'ProfilePhotoBucket', 'DepsLayer', 'UploadPolicy', 'EmailPolicy', 'GradingPolicy'):
        assert absent not in res, absent
    assert not any(k.endswith('ApiInvoke') for k in res)
    assert 'JwtSecretArn' not in params and 'RestApiId' not in params
    for p in ('DatabaseProvider', 'SupabaseUrl', 'SupabaseServiceRoleKey', 'GeminiApiKey', 'GraderQueueArn', 'JsonifierQueueArn'):
        assert p in params, p
    assert params['SupabaseServiceRoleKey']['NoEcho'] is True and params['GeminiApiKey']['NoEcho'] is True
    assert res['ErrorsAlarm']['Properties']['AlarmName'] == 'gradeant-pipeline-errors'
    assert set(tpl['Outputs']) == {'GraderFunctionArn', 'JsonifierFunctionArn', 'CommonLayerArn', 'PipelineDepsLayerArn', 'GraderFailureQueueUrl'}


WORKER = {
    'kind': 'worker', 'routes': [], 'description': 'grades answer sheets',
    'aws': ['dynamodb', 'secrets', 's3'], 'tables': ['assignment_grading'], 'secrets': ['gemini'],
    'layers': ['pipeline-deps', 'common'], 'memory': 1024, 'timeout': 900,
    'env': {'PIPELINE_BUCKET': '!Ref PipelineBucketName', 'GRADE_ON_UPLOAD': '!Ref GradeOnUpload'},
    'queue_param': 'GraderQueueArn', 'ddb_batch': True, 's3_bucket_param': True, 'async_invoke': True,
}


def test_worker_rendering():
    functions = copy.deepcopy(FUNCTIONS)
    functions['grader'] = WORKER
    with_worker = generate_template.render(functions)
    tpl = _load(with_worker)
    res, params, conds = tpl['Resources'], tpl['Parameters'], tpl['Conditions']

    fn = res['GraderFunction']['Properties']
    assert fn['Description'] == 'worker (grades answer sheets)'
    assert fn['Layers'] == [{'Ref': 'PipelineDepsLayer'}, {'Ref': 'CommonLayer'}]  # NOT stacked on DepsLayer
    assert 'Layers' not in tpl['Globals']['Function']
    assert fn['Environment']['Variables'] == {'PIPELINE_BUCKET': {'Ref': 'PipelineBucketName'},
                                              'GRADE_ON_UPLOAD': {'Ref': 'GradeOnUpload'}}
    assert fn['EventInvokeConfig']['DestinationConfig']['OnFailure']['Destination'] == {'GetAtt': 'GraderFailureQueue.Arn'}
    actions = fn['Policies'][0]['Statement'][0]['Action']
    assert 'dynamodb:BatchWriteItem' in actions

    assert 'GraderApiInvoke' not in res                      # workers aren't behind API Gateway
    assert res['GraderQueueMapping']['Condition'] == 'HasGraderQueueArn'
    assert res['GraderQueueMapping']['Properties']['FunctionResponseTypes'] == ['ReportBatchItemFailures']
    assert res['GraderQueueMapping']['Properties']['BatchSize'] == 1
    assert res['PipelinePolicy']['Condition'] == 'HasPipeline'
    assert res['GraderFailureQueue']['Properties']['SqsManagedSseEnabled'] is True
    assert res['PipelineDepsLayer']['Properties']['LayerName'] == 'gradeant-pipeline-deps'
    for p in ('PipelineBucketName', 'PipelineRoot', 'GeminiSecretArn', 'GradeOnUpload', 'PipelineQueuesEnabled', 'GraderQueueArn'):
        assert p in params
    assert {'HasPipeline', 'PipelineQueuesOn', 'HasGraderQueueArn'} <= set(conds)

    # The API functions render exactly as without any worker.
    api_only = {n: s for n, s in FUNCTIONS.items() if s.get('kind', 'api') == 'api'}
    plain = _load(generate_template.render(api_only))['Resources']
    for name in api_only:
        L = generate_template.logical(name)
        assert res[f'{L}Function'] == plain[f'{L}Function']


def test_api_part_unchanged_by_workers():
    """Adding workers must not change a single line of an API function, its invoke permission or log group."""
    api_only = {n: s for n, s in FUNCTIONS.items() if s.get('kind', 'api') == 'api'}
    full, plain = _load(generate_template.render())['Resources'], _load(generate_template.render(api_only))['Resources']
    for key, value in plain.items():
        if key != 'ErrorsAlarm':  # the alarm sums every function's errors, workers included
            assert full[key] == value, key


def test_grader_in_committed_template():
    tpl = _load(generate_template.render())
    res = tpl['Resources']
    fn = res['GraderFunction']['Properties']
    assert fn['Handler'] == 'grader.lambda_handler' and fn['Timeout'] == 900
    assert 'GraderApiInvoke' not in res
    stmts = fn['Policies'][0]['Statement']
    # The DynamoDB statement exists only while the grading table is on DynamoDB.
    cond, dynamo_stmt, otherwise = stmts[0]['If']
    assert cond == 'UsesDynamoDB' and otherwise == {'Ref': 'AWS::NoValue'}
    assert 'dynamodb:BatchWriteItem' in dynamo_stmt['Action']
    assert tpl['Conditions']['UsesDynamoDB'] == {'Not': [{'Equals': [{'Ref': 'DatabaseProvider'}, 'supabase']}]}
    assert not any('secretsmanager' in str(s.get('Action')) for s in stmts[1:])  # Gemini secret only via PipelineSecretPolicy
    secret = res['PipelineSecretPolicy']
    assert secret['Condition'] == 'HasGeminiSecret'
    assert secret['Properties']['PolicyDocument']['Statement'][0]['Resource'] == {'Ref': 'GeminiSecretArn'}
    env = fn['Environment']['Variables']
    assert env['SUPABASE_SERVICE_ROLE_KEY'] == {'Ref': 'SupabaseServiceRoleKey'} and env['DATABASE_PROVIDER'] == {'Ref': 'DatabaseProvider'}
    assert res['GraderQueueMapping']['Properties']['FunctionResponseTypes'] == ['ReportBatchItemFailures']
    assert fn['EventInvokeConfig']['MaximumRetryAttempts'] == 1


def test_every_function_has_a_handler():
    for name in FUNCTIONS:
        with open(os.path.join(ROOT, 'functions', f'{name}.py'), encoding='utf-8') as fh:
            assert 'def lambda_handler(event, context)' in fh.read(), name


def test_layers_must_include_common():
    import pytest
    bad = dict(WORKER, layers=['pipeline-deps'])  # no common layer
    with pytest.raises(SystemExit):
        generate_template.render({'grader': bad})
