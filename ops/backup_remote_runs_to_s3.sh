#!/usr/bin/env bash
set -euo pipefail

# Run this on the remote training host.
# It performs non-destructive syncs of remote run artifacts to S3.

STAMP="$(date -u +%Y%m%d_%H%M%S)"
HOST_TAG="${HOST_TAG:-$(hostname | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')}"
BACKUP_TAG="${BACKUP_TAG:-remote_${HOST_TAG}_${STAMP}}"
S3_ROOT="${S3_ROOT:-s3://videogen-pool-v2-237586137680/project_backups/maat/${BACKUP_TAG}}"
AWS_PROFILE="${AWS_PROFILE:-codex}"

AWS_PROFILE_ARGS=(--profile "${AWS_PROFILE}")
OPS_DIR="${OPS_DIR:-/home/ubuntu/maat/ops}"
mkdir -p "${OPS_DIR}"

MANIFEST_LOCAL="${OPS_DIR}/backup_manifest_${BACKUP_TAG}.txt"
{
  echo "host=$(hostname)"
  echo "timestamp_utc=${STAMP}"
  echo "s3_root=${S3_ROOT}"
  echo "aws_profile=${AWS_PROFILE}"
  echo "paths:"
  echo " - /home/ubuntu/maat/world_model_training/runs"
  echo " - /home/ubuntu/maat/world_model_training/evals"
  echo " - /home/ubuntu/maat/VAE training/runs"
  echo " - /home/ubuntu/maat/world_model_inference/runs"
} > "${MANIFEST_LOCAL}"

aws "${AWS_PROFILE_ARGS[@]}" s3 cp "${MANIFEST_LOCAL}" "${S3_ROOT}/backup_manifest.txt" --only-show-errors

aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "/home/ubuntu/maat/world_model_training/runs" \
  "${S3_ROOT}/remote/world_model_training/runs" \
  --only-show-errors

aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "/home/ubuntu/maat/world_model_training/evals" \
  "${S3_ROOT}/remote/world_model_training/evals" \
  --only-show-errors

aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "/home/ubuntu/maat/VAE training/runs" \
  "${S3_ROOT}/remote/VAE training/runs" \
  --only-show-errors

aws "${AWS_PROFILE_ARGS[@]}" s3 sync \
  "/home/ubuntu/maat/world_model_inference/runs" \
  "${S3_ROOT}/remote/world_model_inference/runs" \
  --only-show-errors

echo "backup_complete ${S3_ROOT}"
