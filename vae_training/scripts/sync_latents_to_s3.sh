#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <latent_export_id>"
  exit 1
fi

EXPORT_ID="$1"
LOCAL_ROOT="${LOCAL_ROOT:-/home/ubuntu/neural-pool/latents_v1}"
S3_PREFIX="${S3_PREFIX:-s3://videogen-pool-v2-237586137680/latents_v1}"

AWS_PROFILE_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_PROFILE_ARGS=(--profile "${AWS_PROFILE}")
fi

SRC="${LOCAL_ROOT}/${EXPORT_ID}"
DST="${S3_PREFIX}/${EXPORT_ID}"

if [[ ! -d "${SRC}" ]]; then
  echo "Latent export directory not found: ${SRC}"
  exit 1
fi

echo "Syncing ${SRC} -> ${DST}"
aws "${AWS_PROFILE_ARGS[@]}" s3 sync "${SRC}" "${DST}" --only-show-errors
echo "Done. Uploaded to ${DST}"

