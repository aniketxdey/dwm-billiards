#!/usr/bin/env bash
set -euo pipefail

# Upload a world-model run directory to S3.

if [[ $# -lt 1 ]]; then
  echo "Usage: bash world_model_training/scripts/sync_run_to_s3.sh <run_id>"
  exit 1
fi

RUN_ID="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-${REPO_ROOT}/world_model_training/runs/${RUN_ID}}"
S3_DEST_ROOT="${S3_DEST_ROOT:-s3://videogen-pool-v2-237586137680/world_model_v1}"

AWS_PROFILE_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_PROFILE_ARGS=(--profile "${AWS_PROFILE}")
fi

if [[ ! -d "${LOCAL_RUN_DIR}" ]]; then
  echo "Run dir not found: ${LOCAL_RUN_DIR}"
  exit 1
fi

echo "Syncing ${LOCAL_RUN_DIR} -> ${S3_DEST_ROOT}/${RUN_ID}/"
aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "${LOCAL_RUN_DIR}/" \
  "${S3_DEST_ROOT}/${RUN_ID}/" \
  --only-show-errors

echo "Sync complete"
