#!/usr/bin/env bash
set -euo pipefail

# Loads .env.lambda for Lambda training instances and performs quick checks.
# Usage:
#   bash scripts/lambda_prepare_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.lambda"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}"
  echo "Create it from ${ROOT_DIR}/.env.lambda.example"
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

required_vars=(
  AWS_DEFAULT_REGION
  AWS_S3_BUCKET
  WANDB_API_KEY
  WANDB_PROJECT
)

for v in "${required_vars[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "Missing required env var: ${v}"
    exit 1
  fi
done

if [[ -z "${AWS_PROFILE:-}" && ( -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ) ]]; then
  echo "Set either AWS_PROFILE or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in ${ENV_FILE}"
  exit 1
fi

echo "Env loaded from ${ENV_FILE}"
echo "Validating AWS credentials..."
AWS_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_ARGS=(--profile "${AWS_PROFILE}")
fi

aws "${AWS_ARGS[@]}" sts get-caller-identity >/dev/null
echo "AWS credentials OK"

echo "Checking S3 access..."
S3_PREFIX="${AWS_S3_PREFIX:-dataraw_v2}"
aws "${AWS_ARGS[@]}" s3 ls "s3://${AWS_S3_BUCKET}/${S3_PREFIX}/" >/dev/null
echo "S3 access OK: s3://${AWS_S3_BUCKET}/${S3_PREFIX}/"

if command -v wandb >/dev/null 2>&1; then
  wandb login "${WANDB_API_KEY}" --relogin >/dev/null
  echo "W&B login OK"
else
  echo "wandb CLI not found. Install with: pip install wandb"
fi

echo "Ready for training."
