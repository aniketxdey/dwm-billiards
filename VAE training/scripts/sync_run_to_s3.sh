#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id>"
  exit 1
fi

RUN_ID="$1"
S3_PREFIX="${S3_PREFIX:-s3://videogen-pool-v2-237586137680/vae_v1}"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs}"

AWS_PROFILE_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_PROFILE_ARGS=(--profile "${AWS_PROFILE}")
fi

SRC="${LOCAL_ROOT}/${RUN_ID}"
DST="${S3_PREFIX}/${RUN_ID}"

if [[ ! -d "${SRC}" ]]; then
  echo "Run directory not found: ${SRC}"
  exit 1
fi

echo "Syncing ${SRC} -> ${DST}"
aws "${AWS_PROFILE_ARGS[@]}" s3 sync "${SRC}" "${DST}" --only-show-errors

echo "Done. Uploaded to ${DST}"
