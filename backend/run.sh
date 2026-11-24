#!/bin/sh
# Lambda entry point with the AWS Lambda Web Adapter (handler: run.sh,
# AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap). The adapter forwards each Function URL
# request to this Express server on 127.0.0.1:$PORT. Must keep LF line endings
# and mode 755 (scripts/package.mjs enforces both, on any OS).
exec node index.js
