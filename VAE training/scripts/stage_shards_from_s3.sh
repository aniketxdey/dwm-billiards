#!/usr/bin/env bash
set -euo pipefail

# Download canonical full dataset shards + metadata to local SSD.

S3_ROOT="${S3_ROOT:-s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/ubuntu/neural-pool/full_20260220_112101}"

AWS_PROFILE_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_PROFILE_ARGS=(--profile "${AWS_PROFILE}")
fi

mkdir -p "${LOCAL_ROOT}/raw/shards"
mkdir -p "${LOCAL_ROOT}/meta"

echo "Syncing shards from ${S3_ROOT}/raw/shards/ to ${LOCAL_ROOT}/raw/shards/"
aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "${S3_ROOT}/raw/shards/" \
  "${LOCAL_ROOT}/raw/shards/" \
  --only-show-errors

echo "Copying metadata.json"
aws "${AWS_PROFILE_ARGS[@]}" s3 cp \
  "${S3_ROOT}/meta/metadata.json" \
  "${LOCAL_ROOT}/meta/metadata.json" \
  --only-show-errors

echo "Done. Local dataset root: ${LOCAL_ROOT}"
