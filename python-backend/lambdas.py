"""The Python Lambdas: one per API path, all sharing the gradeant-python-common layer.

Single source for build.py (zips), deploy/generate_template.py (SAM template, IAM)
and check_permissions.py (each role vs. what the code reaches in the graphify graph).

    file     functions/<file>.py, handler <file>.lambda_handler
    routes   API Gateway routes it serves (existing REST API, Lambda proxy)
    aws      AWS services its code reaches → its IAM role
    tables   DynamoDB tables it reads/writes → its IAM role

Optional keys (grading-pipeline workers; API functions leave them out):
    kind             'api' (default) or 'worker' (no API routes: routes = [])
    layers           default ['deps', 'common']; workers use ['pipeline-deps', 'common'] (pipeline-requirements.txt,
                     self-contained: never stacked on 'deps')
    env              extra environment variables, values are template expressions (e.g. '!Ref PipelineBucketName')
    queue_param      template parameter holding the SQS queue ARN that feeds the worker
    ddb_batch        True → also dynamodb:BatchWriteItem on its tables
    s3_bucket_param  True → read/write the pipeline bucket (PipelineBucketName/PipelineRoot) + the Gemini secret
    async_invoke     True → asynchronous invokes get 1 retry, then go to a gradeant-<name>-failures queue
    database_switch  True → its DynamoDB permission only while the DatabaseProvider parameter is not supabase
"""

WORKFLOW_TABLES = ['Users', 'assignment_grading', 'GradeAnt_Assignments', 'GradeAnt_Submissions', 'GradeAnt_Announcements']
ADMIN_TABLES = ['Users', 'GradeAnt_*']  # GradeAnt_Users, _ResetRequests, _Content, _Tickets, _AuditLogs, _Settings, _Invites, _Analytics

FUNCTIONS = {
    'auth': {
        'routes': ['POST /auth/*'],
        'description': 'login, signup, logout, forgot/change password, me, profile photo',
        'aws': ['dynamodb', 'secrets', 's3'],
        'tables': ADMIN_TABLES,  # also files password-reset requests
        'secrets': ['jwt'],
        'memory': 1024,  # PBKDF2 (100k rounds) on every login is CPU-bound
        'timeout': 29,
    },
    'grades': {
        'routes': ['POST /getgrades'],
        'description': 'one student’s result, class roster, total-mark edits',
        'aws': ['dynamodb', 'secrets'],
        'tables': WORKFLOW_TABLES,
        'secrets': ['jwt'],
        'memory': 512,
        'timeout': 29,
    },
    'corrections': {
        'routes': ['POST /correctanswer'],
        'description': 'answer and question corrections',
        'aws': ['dynamodb', 'secrets'],
        'tables': WORKFLOW_TABLES,
        'secrets': ['jwt'],
        'memory': 512,
        'timeout': 29,
    },
    'marks': {
        'routes': ['POST /updatedb'],
        'description': 'teacher mark edits',
        'aws': ['dynamodb', 'secrets'],
        'tables': WORKFLOW_TABLES,
        'secrets': ['jwt'],
        'memory': 512,
        'timeout': 29,
    },
    'workflow': {
        'routes': ['POST /workflow'],
        'description': 'assignments, rubric, grading runs, publishing, assistant, student screens',
        'aws': ['dynamodb', 'secrets', 'lambda', 'ses', 'sns'],
        'tables': WORKFLOW_TABLES,
        'secrets': ['jwt', 'gemini'],
        # Gemini (google-genai) comes from the pipeline layer; deps is pinned to the same boto3/PyYAML.
        'layers': ['deps', 'pipeline-deps', 'common'],
        'memory': 1024,
        # API Gateway still stops browser requests at 29 s; the higher limit is for the
        # asynchronous self-invocation that drafts rubrics with Gemini.
        'timeout': 300,
        'self_invoke': True,
        'env': {
            'GEMINI_API_KEY_ARN': '!Ref GeminiSecretArn',
            'GEMINI_API_KEY': '!Ref GeminiApiKey',
        },
    },
    'admin': {
        'routes': ['POST /admin'],
        'description': 'admin console and invite links',
        'aws': ['dynamodb', 'secrets', 'ses'],
        'tables': ADMIN_TABLES,
        'secrets': ['jwt', 'session'],
        'memory': 1024,  # invites and password resets hash passwords
        'timeout': 29,
    },
    'contact': {
        'routes': ['POST /contact'],
        'description': 'public contact and request-a-pilot forms',
        'aws': ['ses'],
        'tables': [],
        'secrets': [],
        'memory': 256,
        'timeout': 15,
    },
    'upload': {
        'routes': ['POST /uploadpdf'],
        'description': 'presigned upload URLs (off until UPLOAD_BUCKET is set)',
        'aws': ['dynamodb', 'secrets', 's3'],
        'tables': ['Users', 'GradeAnt_Assignments'],  # a student uploads only for their own classes' assignments
        'secrets': ['jwt'],
        'memory': 512,
        'timeout': 15,
        'env': {  # UPLOAD_LAYOUT=pipeline writes straight into the grading pipeline's layout
            'UPLOAD_LAYOUT': '!Ref UploadLayout',
            'PIPELINE_BUCKET': '!Ref PipelineBucketName',
            'PIPELINE_ROOT': '!Ref PipelineRoot',
        },
    },
    'health': {
        'routes': ['GET /health', 'GET /health/ready'],
        'description': 'liveness and readiness checks',
        'aws': ['dynamodb', 'secrets'],
        'tables': ['Users'],
        'secrets': ['jwt'],
        'memory': 256,
        'timeout': 10,
    },
    # ---- Grading-pipeline workers (no API routes; fed by SQS / asynchronous invokes) ----
    'grader': {
        'kind': 'worker',
        'routes': [],
        'description': 'grading pipeline: grades parsed answer sheets and Start-grading jobs (SQS) and workflow regrade invokes',
        'aws': ['dynamodb', 'secrets', 's3'],
        'tables': ['assignment_grading'],
        'secrets': ['gemini'],
        'layers': ['pipeline-deps', 'common'],
        'memory': 1024,
        'timeout': 900,
        'env': {
            'PIPELINE_BUCKET': '!Ref PipelineBucketName',
            'PIPELINE_ROOT': '!Ref PipelineRoot',
            'GEMINI_API_KEY_ARN': '!Ref GeminiSecretArn',
            'GEMINI_API_KEY': '!Ref GeminiApiKey',
            'GRADE_ON_UPLOAD': '!Ref GradeOnUpload',
            'GRADER_MAX_WORKERS': "'4'",
            # database: supabase → PostgREST (gradeant_common/pg_rest.py); empty = config/app.yaml
            'DATABASE_PROVIDER': '!Ref DatabaseProvider',
            'SUPABASE_URL': '!Ref SupabaseUrl',
            'SUPABASE_SERVICE_ROLE_KEY': '!Ref SupabaseServiceRoleKey',
        },
        'database_switch': True,  # DynamoDB permission only while DatabaseProvider is not supabase
        's3_bucket_param': True,
        'queue_param': 'GraderQueueArn',
        'ddb_batch': True,
        'async_invoke': True,
    },
    'jsonifier': {
        'kind': 'worker',
        'routes': [],
        'description': 'grading pipeline: answer-sheet PDF to JSON in the question paper schema (SQS)',
        'aws': ['secrets', 's3'],
        'tables': [],
        'secrets': ['gemini'],
        'layers': ['pipeline-deps', 'common'],
        'memory': 2048,  # PyMuPDF + whole PDFs in memory
        'timeout': 900,
        'env': {
            'PIPELINE_BUCKET': '!Ref PipelineBucketName',
            'PIPELINE_ROOT': '!Ref PipelineRoot',
            'GEMINI_API_KEY_ARN': '!Ref GeminiSecretArn',
            'GEMINI_API_KEY': '!Ref GeminiApiKey',
        },
        's3_bucket_param': True,
        'queue_param': 'JsonifierQueueArn',
    },
}

RUNTIME = 'python3.13'
ARCHITECTURE = 'arm64'
