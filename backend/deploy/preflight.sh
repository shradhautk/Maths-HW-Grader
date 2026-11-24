#!/usr/bin/env bash
# Read-only checks before the first deploy (DEPLOY.md §3). Changes nothing.
# Uses the AWS CLI with whatever credentials/profile are active.
#
#   bash deploy/preflight.sh                  # Region eu-north-1
#   REGION=eu-west-1 bash deploy/preflight.sh
#   JWT_SECRET_ARN=... SES_SENDER=... GRADING_FUNCTION=... bash deploy/preflight.sh
set -u
REGION="${REGION:-eu-north-1}"
aws() { command aws --region "$REGION" --output text "$@"; }
pass=0; fail=0; warn=0
ok()   { echo "  [ok]   $*"; pass=$((pass + 1)); }
bad()  { echo "  [FAIL] $*"; fail=$((fail + 1)); }
note() { echo "  [warn] $*"; warn=$((warn + 1)); }

echo "== Identity ($REGION)"
if id=$(aws sts get-caller-identity --query '[Account,Arn]' 2>&1); then ok "signed in as $id"; else bad "AWS CLI not signed in: $id"; exit 1; fi

echo "== DynamoDB tables and indexes the API uses"
check_table() { # name [index ...]
  local t="$1"; shift
  if ! idx=$(aws dynamodb describe-table --table-name "$t" --query 'Table.GlobalSecondaryIndexes[].IndexName' 2>/dev/null); then
    bad "table $t is missing"; return
  fi
  ok "table $t"
  for i in "$@"; do
    if grep -qw -- "$i" <<<"$idx"; then ok "  index $t/$i"; else bad "  index $t/$i is missing"; fi
  done
}
check_table Users
check_table assignment_grading student_id-index type-index
# Optional: without it the API falls back to a slower scan (same as the Python Lambdas).
if aws dynamodb describe-table --table-name assignment_grading --query 'Table.GlobalSecondaryIndexes[].IndexName' 2>/dev/null | grep -qw assignment_id-index; then
  ok "  index assignment_grading/assignment_id-index"
else
  note "  index assignment_grading/assignment_id-index not found (optional; class lists use a scan without it)"
fi
check_table GradeAnt_Assignments teacher_id-index
check_table GradeAnt_Submissions student_id-index
for t in GradeAnt_Announcements GradeAnt_Users GradeAnt_ResetRequests GradeAnt_Content GradeAnt_Tickets \
         GradeAnt_AuditLogs GradeAnt_Settings GradeAnt_Invites GradeAnt_Analytics; do
  check_table "$t"
done

echo "== JWT secret"
if [ -n "${JWT_SECRET_ARN:-}" ]; then
  if aws secretsmanager describe-secret --secret-id "$JWT_SECRET_ARN" --query Name >/dev/null 2>&1; then ok "secret $JWT_SECRET_ARN exists"; else bad "secret $JWT_SECRET_ARN not found"; fi
else
  note "JWT_SECRET_ARN not set; skipped (see DEPLOY.md §3.2)"
fi

echo "== Email (SES)"
if [ -n "${SES_SENDER:-}" ]; then
  status=$(aws ses get-identity-verification-attributes --identities "$SES_SENDER" "${SES_SENDER#*@}" \
    --query 'VerificationAttributes.*.VerificationStatus' 2>/dev/null)
  if grep -q Success <<<"$status"; then ok "$SES_SENDER (or its domain) is verified"; else bad "$SES_SENDER is not verified in SES ($REGION)"; fi
  max=$(aws ses get-send-quota --query Max24HourSend 2>/dev/null)
  if [ "${max%.*}" -le 200 ] 2>/dev/null; then note "SES is in sandbox (max $max/day): mail only reaches verified addresses until you request production access"; else ok "SES production access (max $max/day)"; fi
else
  note "SES_SENDER not set; skipped (no email will be sent)"
fi

echo "== Grading pipeline Lambda"
if [ -n "${GRADING_FUNCTION:-}" ]; then
  if aws lambda get-function --function-name "$GRADING_FUNCTION" --query Configuration.FunctionName >/dev/null 2>&1; then ok "function $GRADING_FUNCTION exists"; else bad "function $GRADING_FUNCTION not found"; fi
else
  note "GRADING_FUNCTION not set; skipped (Start grading will not reach the pipeline)"
fi

echo "== Lambda Web Adapter layer"
layer="arn:aws:lambda:$REGION:753240598075:layer:LambdaAdapterLayerArm64:${LWA_VERSION:-30}"
if aws lambda get-layer-version-by-arn --arn "$layer" --query LayerVersionArn >/dev/null 2>&1; then ok "$layer"; else bad "$layer not reachable (check the version in the Web Adapter docs)"; fi

echo
echo "Result: $pass ok, $warn warnings, $fail failed"
[ "$fail" -eq 0 ]
