"""Writes deploy/template.yaml (AWS SAM) from lambdas.py, so the functions, their IAM
roles and the code can't drift apart. Run after changing lambdas.py:

    python deploy/generate_template.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from lambdas import ARCHITECTURE, FUNCTIONS, RUNTIME  # noqa: E402

D, A = '${DataRegion}', '${AWS::AccountId}'
DDB_ACTIONS = '[dynamodb:GetItem, dynamodb:PutItem, dynamodb:UpdateItem, dynamodb:DeleteItem, dynamodb:Query, dynamodb:Scan]'
DDB_BATCH_ACTIONS = DDB_ACTIONS[:-1] + ', dynamodb:BatchWriteItem]'
DEFAULT_LAYERS = ['deps', 'common']
LAYER_REFS = {'deps': 'DepsLayer', 'common': 'CommonLayer', 'pipeline-deps': 'PipelineDepsLayer'}


def logical(name):
    return ''.join(part[0].upper() + part[1:] for part in name.split('-'))


def kind(spec):
    return spec.get('kind', 'api')


def layers_of(spec):
    """The function's own layers. Set per function (not in Globals): SAM appends a function's Layers
    to the Globals ones, and the workers must NOT get gradeant-python-deps on top of their own
    self-contained package layer (two separately resolved copies of httpx/pydantic in one /opt)."""
    layers = spec.get('layers', DEFAULT_LAYERS)
    unknown = [x for x in layers if x not in LAYER_REFS]
    if unknown or 'common' not in layers:
        raise SystemExit(f'layers must be from {sorted(LAYER_REFS)} and include common: {layers}')
    return layers


def table_arns(tables):
    arns = []
    for t in tables:
        arns.append(f'arn:aws:dynamodb:{D}:{A}:table/{t}')
        if t != 'Users' and t != 'GradeAnt_Announcements':  # tables with GSIs (or a wildcard that may have them)
            arns.append(f'arn:aws:dynamodb:{D}:{A}:table/{t}/index/*')
    return arns


def statements(name, spec):
    out = []
    if 'dynamodb' in spec['aws'] and spec['tables']:
        actions = '[dynamodb:GetItem]' if name == 'health' else (DDB_BATCH_ACTIONS if spec.get('ddb_batch') else DDB_ACTIONS)
        tables = spec['tables'] if name != 'health' else ['Users']
        arns = [f'arn:aws:dynamodb:{D}:{A}:table/Users'] if name == 'health' else table_arns(tables)
        stmt = ('            - Effect: Allow\n'
                f'              Action: {actions}\n'
                '              Resource:\n' + ''.join(f'                - !Sub {a}\n' for a in arns))
        if spec.get('database_switch'):
            # Granted only while the grading table is on DynamoDB (DatabaseProvider is not supabase).
            stmt = ('            - !If\n              - UsesDynamoDB\n'
                    + ''.join('  ' + line + '\n' for line in stmt.rstrip('\n').split('\n'))
                    + '              - !Ref AWS::NoValue\n')
        out.append(stmt)
    secret_refs = []
    if 'jwt' in spec['secrets']:
        secret_refs.append('!Ref JwtSecretArn')
    if 'session' in spec['secrets']:
        secret_refs.append('!If [CreateSessionSecret, !Ref SessionSecret, !Ref SessionSecretArn]')
    if secret_refs:
        out.append('            - Effect: Allow\n'
                   '              Action: secretsmanager:GetSecretValue\n'
                   '              Resource:\n' + ''.join(f'                - {r}\n' for r in secret_refs))
    if name == 'auth':  # profile photos (the upload bucket is UploadPolicy, below)
        out.append('            - Effect: Allow\n'
                   '              Action: [s3:GetObject, s3:PutObject, s3:DeleteObject]\n'
                   '              Resource: !Sub\n'
                   '                - arn:aws:s3:::${Bucket}/profile-photos/*\n'
                   '                - Bucket: !If [CreatePhotoBucket, !Ref ProfilePhotoBucket, !Ref ProfilePhotoBucketName]\n')
    if 'gemini' in spec['secrets'] and kind(spec) == 'api':
        out.append('            - !If  # the Gemini key secret, only when GeminiSecretArn is set\n'
                   '              - HasGeminiSecret\n'
                   '              - Effect: Allow\n'
                   '                Action: secretsmanager:GetSecretValue\n'
                   '                Resource: !Ref GeminiSecretArn\n'
                   '              - !Ref AWS::NoValue\n')
    if 'bedrock' in spec['aws']:
        out.append("            - Effect: Allow\n              Action: bedrock-mantle:CreateInference\n              Resource: '*'\n")
    if spec.get('self_invoke'):
        out.append('            - Effect: Allow  # asynchronous self-invocation (AI rubric drafting)\n'
                   '              Action: lambda:InvokeFunction\n'
                   f'              Resource: !Sub arn:aws:lambda:${{AWS::Region}}:{A}:function:gradeant-{name}\n')
    if spec.get('async_invoke'):
        out.append('            - Effect: Allow  # failed asynchronous invocations go to the failure queue\n'
                   '              Action: sqs:SendMessage\n'
                   f'              Resource: !GetAtt {logical(name)}FailureQueue.Arn\n')
    return ''.join(out)


def function(name, spec):
    L = logical(name)
    stmts = statements(name, spec)
    policies = ("      Policies:\n        - Version: '2012-10-17'\n          Statement:\n" + stmts) if stmts else ''
    description = (f"{', '.join(spec['routes'])} ({spec['description']})" if spec['routes']
                   else f"worker ({spec['description']})")
    if ': ' in description or ' #' in description:  # not a valid plain YAML scalar → single-quote it
        description = "'" + description.replace("'", "''") + "'"
    layer_block = '      Layers:\n' + ''.join(f'        - !Ref {LAYER_REFS[x]}\n' for x in layers_of(spec))
    env_block = ('      Environment:\n        Variables:\n'
                 + ''.join(f'          {k}: {v}\n' for k, v in spec['env'].items())) if spec.get('env') else ''
    async_block = ('      EventInvokeConfig:\n'
                   '        MaximumRetryAttempts: 1\n'
                   '        MaximumEventAgeInSeconds: 3600\n'
                   '        DestinationConfig:\n'
                   '          OnFailure:\n'
                   '            Type: SQS\n'
                   f'            Destination: !GetAtt {L}FailureQueue.Arn\n') if spec.get('async_invoke') else ''
    return (f'  {L}Function:\n'
            '    Type: AWS::Serverless::Function\n'
            '    Properties:\n'
            f'      FunctionName: gradeant-{name}\n'
            f'      Description: {description}\n'
            f'      CodeUri: ../dist/gradeant-{name}.zip\n'
            f'      Handler: {name}.lambda_handler\n'
            f"      MemorySize: {spec['memory']}\n"
            f"      Timeout: {spec['timeout']}\n"
            + layer_block + env_block + async_block +
            '      LoggingConfig:\n'
            f'        LogGroup: !Ref {L}Logs\n'
            + policies + '\n')


def optional_policy(logical_id, condition, policy, roles, statement):
    if not roles:
        return ''
    return (f'  {logical_id}:\n'
            '    Type: AWS::IAM::Policy\n'
            f'    Condition: {condition}\n'
            '    Properties:\n'
            f'      PolicyName: {policy}\n'
            f"      Roles: [{', '.join(f'!Ref {logical(r)}FunctionRole' for r in roles)}]\n"
            '      PolicyDocument:\n'
            "        Version: '2012-10-17'\n"
            '        Statement:\n' + statement + '\n')


HEADER = f"""# GENERATED by deploy/generate_template.py from lambdas.py — edit those, then regenerate.
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: >-
  GradeAnt API as Python 3.13 Lambdas, one per API path, sharing two layers:
  gradeant-python-deps (PyJWT, PyYAML, boto3) and gradeant-python-common (shared code
  and config/app.yaml). Each function has its own IAM role, limited to what its code
  reaches. Plug the functions into the existing API Gateway (see python-backend/README.md).
  Build first: python build.py --verify.

Parameters:
  DataRegion:
    Type: String
    Default: eu-north-1
    Description: Region of the DynamoDB tables, SNS topic and grading Lambda (deploy this stack there).
  RestApiId:
    Type: String
    Default: ''
    Description: ID of the existing API Gateway REST API (e.g. rus9nultj9). When set, it may invoke every function.
  JwtSecretArn:
    Type: String
    Description: Secrets Manager secret with the JWT key. Must be the SAME value the current Lambdas use.
  SessionSecretArn:
    Type: String
    Default: ''
    Description: Admin session-token secret (the admin Lambda's SESSION_TOKEN_SECRET). Empty = create a new random one.
  WebOrigins:
    Type: String
    Default: https://app.gradeant.io,https://grade-ant.com,https://www.grade-ant.com
    Description: Comma-separated browser origins allowed to call the API (CORS) and upload profile photos.
  ProfilePhotoBucketName:
    Type: String
    Default: ''
    Description: Existing profile-photo bucket. Empty = create a private, encrypted one.
  UploadBucketName:
    Type: String
    Default: ''
    Description: Bucket the grading pipeline reads uploads from. Empty = /uploadpdf stays off (keep the legacy upload Lambda).
  UploadKeyTemplate:
    Type: String
    Default: uploads/{{uploadType}}/{{assignmentId}}/{{fileName}}
  SesSenderEmail:
    Type: String
    Default: ''
    Description: SES-verified From address. Empty = no email is sent.
  ContactInbox:
    Type: String
    Default: contact@gradeant.com
  SnsTopicArn:
    Type: String
    Default: ''
    Description: SNS topic for "grades published". Empty = off.
  GradingFunctionName:
    Type: String
    Default: ''
    Description: Existing grading-pipeline Lambda, invoked asynchronously by "Start grading". Empty = off.
  GeminiModel:
    Type: String
    Default: gemini-3.5-flash-preview
    Description: The one Gemini model for every AI call (grading, reading answer sheets, rubric drafting, assistant).
  GeminiSecretArn:
    Type: String
    Default: ''
    Description: 'Secrets Manager secret holding the Gemini API key (a plain string, or JSON with a GEMINI_API_KEY field). Or use GeminiApiKey.'
  GeminiApiKey:
    Type: String
    Default: ''
    NoEcho: true
    Description: 'Free-tier alternative to GeminiSecretArn: the key as an encrypted Lambda environment variable (no Secrets Manager cost).'
  LogRetentionDays:
    Type: Number
    Default: 30
    AllowedValues: [7, 14, 30, 60, 90, 180, 365]
  AlarmTopicArn:
    Type: String
    Default: ''
    Description: Optional SNS topic notified when the functions error.

Conditions:
  HasRestApi: !Not [!Equals [!Ref RestApiId, '']]
  CreateSessionSecret: !Equals [!Ref SessionSecretArn, '']
  CreatePhotoBucket: !Equals [!Ref ProfilePhotoBucketName, '']
  HasUploadBucket: !Not [!Equals [!Ref UploadBucketName, '']]
  HasSes: !Not [!Equals [!Ref SesSenderEmail, '']]
  HasSnsTopic: !Not [!Equals [!Ref SnsTopicArn, '']]
  HasGradingFunction: !Not [!Equals [!Ref GradingFunctionName, '']]
  HasAlarmTopic: !Not [!Equals [!Ref AlarmTopicArn, '']]
  HasGeminiSecret: !Not [!Equals [!Ref GeminiSecretArn, '']]

Globals:
  Function:
    Runtime: {RUNTIME}
    Architectures: [{ARCHITECTURE}]
    # Layers are set per function (the pipeline workers use their own package layer).
    Environment:
      Variables:
        # Same settings for every function (no secret values: only ARNs).
        APP_ENV: production
        AWS_DATA_REGION: !Ref DataRegion
        CORS_ALLOWED_ORIGINS: !Ref WebOrigins
        JWT_SECRET_ARN: !Ref JwtSecretArn
        SESSION_TOKEN_SECRET_ARN: !If [CreateSessionSecret, !Ref SessionSecret, !Ref SessionSecretArn]
        PROFILE_PHOTO_BUCKET: !If [CreatePhotoBucket, !Ref ProfilePhotoBucket, !Ref ProfilePhotoBucketName]
        UPLOAD_BUCKET: !Ref UploadBucketName
        UPLOAD_KEY_TEMPLATE: !Ref UploadKeyTemplate
        SES_SENDER_EMAIL: !Ref SesSenderEmail
        CONTACT_INBOX: !Ref ContactInbox
        SNS_TOPIC_ARN: !Ref SnsTopicArn
        GRADING_FUNCTION_NAME: !Ref GradingFunctionName
        GEMINI_MODEL: !Ref GeminiModel

Resources:
  # ---- Layers ----------------------------------------------------------------
  DepsLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: gradeant-python-deps
      Description: GradeAnt API Python packages (PyJWT, PyYAML, boto3) for {RUNTIME} {ARCHITECTURE}
      ContentUri: ../dist/layer-gradeant-python-deps.zip
      CompatibleRuntimes: [{RUNTIME}]
      CompatibleArchitectures: [{ARCHITECTURE}]
      RetentionPolicy: Retain

  CommonLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: gradeant-python-common
      Description: GradeAnt shared code (gradeant_common) and config/app.yaml
      ContentUri: ../dist/layer-gradeant-python-common.zip
      CompatibleRuntimes: [{RUNTIME}]
      CompatibleArchitectures: [{ARCHITECTURE}]
      RetentionPolicy: Retain

  # ---- Shared resources --------------------------------------------------------
  SessionSecret:
    Type: AWS::SecretsManager::Secret
    Condition: CreateSessionSecret
    Properties:
      Name: gradeant/session
      Description: GradeAnt admin first-login session-token signing key
      GenerateSecretString:
        PasswordLength: 64
        ExcludePunctuation: true

  ProfilePhotoBucket:
    Type: AWS::S3::Bucket
    Condition: CreatePhotoBucket
    DeletionPolicy: Retain
    UpdateReplacePolicy: Retain
    Properties:
      BucketName: !Sub gradeant-profile-photos-${{AWS::AccountId}}-${{AWS::Region}}
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      OwnershipControls:
        Rules:
          - ObjectOwnership: BucketOwnerEnforced
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: expire-replaced-photos
            Status: Enabled
            NoncurrentVersionExpiration:
              NoncurrentDays: 30
      CorsConfiguration:
        CorsRules:
          - AllowedMethods: [PUT, GET]
            AllowedOrigins: !Split [',', !Ref WebOrigins]
            AllowedHeaders: ['*']
            MaxAge: 3000

  ProfilePhotoBucketTls:
    Type: AWS::S3::BucketPolicy
    Condition: CreatePhotoBucket
    Properties:
      Bucket: !Ref ProfilePhotoBucket
      PolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Sid: DenyInsecureTransport
            Effect: Deny
            Principal: '*'
            Action: 's3:*'
            Resource:
              - !GetAtt ProfilePhotoBucket.Arn
              - !Sub '${{ProfilePhotoBucket.Arn}}/*'
            Condition:
              Bool:
                aws:SecureTransport: 'false'

  # ---- Functions (one per API path) -------------------------------------------
"""


PIPELINE_PARAMS = """  PipelineBucketName:
    Type: String
    Default: ''
    Description: Bucket the grading pipeline workers read and write (the original pipeline's bucket). Empty = workers off.
  PipelineRoot:
    Type: String
    Default: Data_warehouse
    Description: Top-level folder of the pipeline layout in that bucket.
  DatabaseProvider:
    Type: String
    Default: ''
    AllowedValues: ['', dynamodb, supabase]
    Description: 'Grading table for the grader: dynamodb or supabase. Empty = follow config/app.yaml (deployment: aws | hybrid).'
  SupabaseUrl:
    Type: String
    Default: ''
    Description: 'https://<project>.supabase.co (database: supabase).'
  SupabaseServiceRoleKey:
    Type: String
    Default: ''
    NoEcho: true
    Description: 'Supabase service-role key (database: supabase); its grants are limited by backend/sql/003_supabase_access.sql.'
  GradeOnUpload:
    Type: String
    Default: 'true'
    AllowedValues: ['true', 'false']
    Description: true = grade as soon as a parsed answer sheet lands (as today); false = wait for "Start grading".
  UploadLayout:
    Type: String
    Default: template
    AllowedValues: [template, pipeline]
    Description: '/uploadpdf object keys: template = UploadKeyTemplate (as today); pipeline = the grading pipeline layout (needs UploadBucketName = PipelineBucketName).'
  PipelineQueuesEnabled:
    Type: String
    Default: 'false'
    AllowedValues: ['true', 'false']
    Description: Turns the SQS → worker mappings on. Keep 'false' until the old pipeline Lambdas are detached from the queues.
"""


PIPELINE_HEADER = f"""# GENERATED by deploy/generate_template.py from lambdas.py — edit those, then regenerate.
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: >-
  GradeAnt grading pipeline only (free-tier hybrid: web on Netlify, Node API on Render,
  Supabase database). The two workers gradeant-jsonifier and gradeant-grader, their layers,
  SQS mappings and failure queue. No API functions. Deploy this OR template.yaml, not both
  (same function names). Build first: python build.py --verify.

Parameters:
  DataRegion:
    Type: String
    Default: eu-north-1
    Description: Region of the pipeline bucket and queues (deploy this stack there).
  LogRetentionDays:
    Type: Number
    Default: 14
    AllowedValues: [7, 14, 30, 60, 90, 180, 365]
  AlarmTopicArn:
    Type: String
    Default: ''
    Description: Optional SNS topic notified when the workers error.
  GeminiModel:
    Type: String
    Default: gemini-3.5-flash-preview
    Description: The one Gemini model for every AI call (grading, reading answer sheets, rubric drafting, assistant).
  GeminiSecretArn:
    Type: String
    Default: ''
    Description: 'Secrets Manager secret holding the Gemini API key (a plain string, or JSON with a GEMINI_API_KEY field). Or use GeminiApiKey.'
  GeminiApiKey:
    Type: String
    Default: ''
    NoEcho: true
    Description: 'Free-tier alternative to GeminiSecretArn: the key as an encrypted Lambda environment variable (no Secrets Manager cost).'

Conditions:
  HasAlarmTopic: !Not [!Equals [!Ref AlarmTopicArn, '']]
  HasGeminiSecret: !Not [!Equals [!Ref GeminiSecretArn, '']]

Globals:
  Function:
    Runtime: {RUNTIME}
    Architectures: [{ARCHITECTURE}]
    Environment:
      Variables:
        APP_ENV: production
        AWS_DATA_REGION: !Ref DataRegion
        GEMINI_MODEL: !Ref GeminiModel

Resources:
  # ---- Layers ----------------------------------------------------------------
  CommonLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: gradeant-python-common
      Description: GradeAnt shared code (gradeant_common) and config/app.yaml
      ContentUri: ../dist/layer-gradeant-python-common.zip
      CompatibleRuntimes: [{RUNTIME}]
      CompatibleArchitectures: [{ARCHITECTURE}]
      RetentionPolicy: Retain

  # ---- Shared resources --------------------------------------------------------

  # ---- Functions (pipeline workers) ----------------------------------------------
"""


def queue_params(functions):
    return [s['queue_param'] for s in functions.values() if s.get('queue_param')]


def render(functions=None, pipeline_only=False):
    """The whole template as a string. Pipeline sections appear only when lambdas.py has worker functions,
    so the API-only template is unchanged. pipeline_only: just the workers, on a slim header (no API
    parameters, session secret or photo bucket), for the hybrid deployment."""
    functions = FUNCTIONS if functions is None else functions
    if pipeline_only:
        functions = {n: s for n, s in functions.items() if kind(s) == 'worker'}
    names = list(functions)
    workers = {n: s for n, s in functions.items() if kind(s) == 'worker'}
    header = PIPELINE_HEADER if pipeline_only else HEADER
    if workers:
        params = PIPELINE_PARAMS + ''.join(
            f"  {p}:\n    Type: String\n    Default: ''\n    Description: SQS queue (ARN) of S3 events feeding this worker. Empty = no mapping.\n"
            for p in queue_params(functions))
        conditions = ("  HasPipeline: !Not [!Equals [!Ref PipelineBucketName, '']]\n"
                      "  UsesDynamoDB: !Not [!Equals [!Ref DatabaseProvider, supabase]]\n"
                      "  PipelineQueuesOn: !Equals [!Ref PipelineQueuesEnabled, 'true']\n"
                      + ''.join(f"  Has{p}: !Not [!Equals [!Ref {p}, '']]\n" for p in queue_params(functions)))
        header = header.replace('\nConditions:\n', '\n' + params + '\nConditions:\n', 1)
        header = header.replace('\nGlobals:\n', conditions + '\nGlobals:\n', 1)
    if any('pipeline-deps' in s.get('layers', []) for s in functions.values()):
        header = header.replace(
            '  # ---- Shared resources', (
                '  PipelineDepsLayer:\n'
                '    Type: AWS::Serverless::LayerVersion\n'
                '    Properties:\n'
                '      LayerName: gradeant-pipeline-deps\n'
                f'      Description: Grading pipeline packages (PyYAML, PyMuPDF, google-genai) for {RUNTIME} {ARCHITECTURE}\n'
                '      ContentUri: ../dist/layer-gradeant-pipeline-deps.zip\n'
                f'      CompatibleRuntimes: [{RUNTIME}]\n'
                f'      CompatibleArchitectures: [{ARCHITECTURE}]\n'
                '      RetentionPolicy: Retain\n\n'
                '  # ---- Shared resources'), 1)
    out = header + ''.join(function(n, s) for n, s in functions.items())

    out += '  # ---- Optional permissions: each exists only when its setting is filled in ----\n'
    out += optional_policy('EmailPolicy', 'HasSes', 'email', [n for n, s in functions.items() if 'ses' in s['aws']],
                           '          - Effect: Allow\n            Action: ses:SendEmail\n            Resource: \'*\'\n'
                           '            Condition:\n              StringEquals:\n                ses:FromAddress: !Ref SesSenderEmail\n')
    out += optional_policy('NotifyPolicy', 'HasSnsTopic', 'notify', [n for n, s in functions.items() if 'sns' in s['aws']],
                           '          - Effect: Allow\n            Action: sns:Publish\n            Resource: !Ref SnsTopicArn\n')
    out += optional_policy('GradingPolicy', 'HasGradingFunction', 'grading', [n for n, s in functions.items() if 'lambda' in s['aws']],
                           '          - Effect: Allow\n            Action: lambda:InvokeFunction\n'
                           f'            Resource: !Sub arn:aws:lambda:{D}:{A}:function:${{GradingFunctionName}}\n')
    out += optional_policy('UploadPolicy', 'HasUploadBucket', 'uploads', [n for n in ['upload'] if n in functions],
                           '          - Effect: Allow\n            Action: [s3:PutObject, s3:GetObject]\n'
                           '            Resource: !Sub arn:aws:s3:::${UploadBucketName}/*\n')
    if workers:
        out += optional_policy(
            'PipelinePolicy', 'HasPipeline', 'pipeline', [n for n, s in workers.items() if s.get('s3_bucket_param')],
            '          - Effect: Allow\n            Action: [s3:GetObject, s3:PutObject]\n'
            '            Resource: !Sub arn:aws:s3:::${PipelineBucketName}/${PipelineRoot}/*\n'
            '          - Effect: Allow  # lets a missing key answer 404 instead of 403\n'
            '            Action: s3:ListBucket\n'
            '            Resource: !Sub arn:aws:s3:::${PipelineBucketName}\n'
            '            Condition:\n              StringLike:\n                s3:prefix: !Sub ${PipelineRoot}/*\n')
        out += optional_policy(
            'PipelineSecretPolicy', 'HasGeminiSecret', 'pipeline-gemini-secret', [n for n, s in workers.items() if s.get('s3_bucket_param')],
            '          - Effect: Allow\n            Action: secretsmanager:GetSecretValue\n'
            '            Resource: !Ref GeminiSecretArn\n')

        out += '  # ---- Pipeline workers: SQS feeds and failure queues ----------------------------\n'
        for n, s in workers.items():
            L, p = logical(n), s.get('queue_param')
            if p:
                out += optional_policy(
                    f'{L}QueuePolicy', f'Has{p}', f'{n}-queue', [n],
                    '          - Effect: Allow\n'
                    '            Action: [sqs:ReceiveMessage, sqs:DeleteMessage, sqs:GetQueueAttributes, sqs:ChangeMessageVisibility]\n'
                    f'            Resource: !Ref {p}\n')
                out += (f'  {L}QueueMapping:\n    Type: AWS::Lambda::EventSourceMapping\n    Condition: Has{p}\n'
                        f'    DependsOn: {L}QueuePolicy\n    Properties:\n'
                        f'      FunctionName: !Ref {L}Function\n      EventSourceArn: !Ref {p}\n'
                        '      Enabled: !If [PipelineQueuesOn, true, false]\n'
                        '      BatchSize: 1\n      FunctionResponseTypes: [ReportBatchItemFailures]\n'
                        '      ScalingConfig:\n        MaximumConcurrency: 5\n\n')
            if s.get('async_invoke'):
                out += (f'  {L}FailureQueue:\n    Type: AWS::SQS::Queue\n    Properties:\n'
                        f'      QueueName: gradeant-{n}-failures\n      SqsManagedSseEnabled: true\n'
                        '      MessageRetentionPeriod: 1209600\n\n')

    out += '  # ---- The existing REST API may invoke every function ------------------------\n'
    for n in [n for n in names if kind(functions[n]) == 'api']:
        out += (f'  {logical(n)}ApiInvoke:\n    Type: AWS::Lambda::Permission\n    Condition: HasRestApi\n    Properties:\n'
                f'      FunctionName: !Ref {logical(n)}Function\n      Action: lambda:InvokeFunction\n'
                '      Principal: apigateway.amazonaws.com\n'
                '      SourceArn: !Sub arn:aws:execute-api:${AWS::Region}:${AWS::AccountId}:${RestApiId}/*\n\n')

    out += '  # ---- Logs (one group per function, with retention) ----------------------------\n'
    for n in names:
        out += (f'  {logical(n)}Logs:\n    Type: AWS::Logs::LogGroup\n    Properties:\n'
                f'      LogGroupName: /aws/lambda/gradeant-{n}\n      RetentionInDays: !Ref LogRetentionDays\n\n')

    metrics = ''.join(
        f'        - Id: e{i}\n          ReturnData: false\n          MetricStat:\n            Metric:\n'
        '              Namespace: AWS/Lambda\n              MetricName: Errors\n              Dimensions:\n'
        f'                - Name: FunctionName\n                  Value: !Ref {logical(n)}Function\n'
        '            Period: 300\n            Stat: Sum\n' for i, n in enumerate(names))
    out += ('  # ---- Alarms -------------------------------------------------------------------\n'
            '  ErrorsAlarm:\n    Type: AWS::CloudWatch::Alarm\n    Properties:\n'
            f"      AlarmName: gradeant-{'pipeline' if pipeline_only else 'python'}-errors\n"
            '      AlarmDescription: GradeAnt functions crashed or timed out 5 or more times in 5 minutes (all functions together).\n'
            '      Metrics:\n        - Id: total\n          Label: Errors (all functions)\n          Expression: SUM(METRICS())\n          ReturnData: true\n'
            + metrics +
            '      EvaluationPeriods: 1\n      Threshold: 5\n      ComparisonOperator: GreaterThanOrEqualToThreshold\n'
            '      TreatMissingData: notBreaching\n      AlarmActions: !If [HasAlarmTopic, [!Ref AlarmTopicArn], !Ref AWS::NoValue]\n\n')

    out += 'Outputs:\n'
    for n in names:
        out += f'  {logical(n)}FunctionArn:\n    Value: !GetAtt {logical(n)}Function.Arn\n'
    if pipeline_only:
        out += '  CommonLayerArn:\n    Value: !Ref CommonLayer\n'
    else:
        out += ('  DepsLayerArn:\n    Value: !Ref DepsLayer\n  CommonLayerArn:\n    Value: !Ref CommonLayer\n'
                '  ProfilePhotoBucket:\n    Value: !If [CreatePhotoBucket, !Ref ProfilePhotoBucket, !Ref ProfilePhotoBucketName]\n'
                '  SessionSecretArn:\n    Value: !If [CreateSessionSecret, !Ref SessionSecret, !Ref SessionSecretArn]\n')
    if any('pipeline-deps' in s.get('layers', []) for s in functions.values()):
        out += '  PipelineDepsLayerArn:\n    Value: !Ref PipelineDepsLayer\n'
    for n, s in workers.items():
        if s.get('async_invoke'):
            out += f'  {logical(n)}FailureQueueUrl:\n    Value: !Ref {logical(n)}FailureQueue\n'
    return out


def main():
    """Writes both templates, so they can't drift apart:
    template.yaml           everything on AWS (API functions + pipeline workers)
    pipeline-template.yaml  pipeline workers only (hybrid: Node API on Render, Supabase database)"""
    for file, kwargs in (('template.yaml', {}), ('pipeline-template.yaml', {'pipeline_only': True})):
        out = render(**kwargs)
        path = os.path.join(HERE, file)
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(out)
        count = sum(1 for s in FUNCTIONS.values() if not kwargs or kind(s) == 'worker')
        print(f'Wrote {os.path.relpath(path)} ({count} functions)')


if __name__ == '__main__':
    main()
