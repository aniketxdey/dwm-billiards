#!/usr/bin/env bash
set -euo pipefail

# Download canonical latent shards to local SSD.

S3_ROOT="${S3_ROOT:-s3://videogen-pool-v2-237586137680/latents_v1/latents_60m_from_vae_60m_20260220_204310_run01}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/ubuntu/neural-pool/latents_v1/latents_60m_from_vae_60m_20260220_204310_run01}"

AWS_PROFILE_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_PROFILE_ARGS=(--profile "${AWS_PROFILE}")
fi

mkdir -p "${LOCAL_ROOT}/shards"
mkdir -p "${LOCAL_ROOT}/logs"

echo "Syncing latent shards from ${S3_ROOT}/shards/ -> ${LOCAL_ROOT}/shards/"
aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "${S3_ROOT}/shards/" \
  "${LOCAL_ROOT}/shards/" \
  --only-show-errors

echo "Copying summary/log metadata"
aws "${AWS_PROFILE_ARGS[@]}" s3 cp \
  "${S3_ROOT}/summary.json" \
  "${LOCAL_ROOT}/summary.json" \
  --only-show-errors || true
aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "${S3_ROOT}/logs/" \
  "${LOCAL_ROOT}/logs/" \
  --only-show-errors || true

echo "Done. Local latent root: ${LOCAL_ROOT}"
