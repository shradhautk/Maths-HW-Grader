"""Shared code for the GradeAnt Python Lambdas (the `gradeant-common` layer).

Python 3.10+ (deployed on python3.13). Mirrors the Node backend in backend/src:

    config       settings from the environment and config/app.yaml   (src/config)
    secrets      JWT/session keys from Secrets Manager or env          (src/aws/secrets.js)
    http         request wrapper: JSON, CORS allowlist, error shapes    (src/middleware, controllers/respond.js)
    auth         bearer-token authentication                            (src/middleware/auth.js)
    permissions  role checks from config/app.yaml                       (src/lib/permissions.js)
    dynamo       tables, paginated query/scan, Decimal conversion       (src/aws/dynamodb.js)
    ownership    teacher-to-teacher isolation                           (src/services/ownership.js)
"""

VERSION = "1.0.0"  # same as backend/package.json
