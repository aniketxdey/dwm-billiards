#!/usr/bin/env bash
set -euo pipefail

# Full project backup to S3 (code, docs, configs, eval samples, ops history).
# Requires valid AWS CLI auth (aws login) and s3 permissions.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
S3_ROOT="${S3_ROOT:-s3://videogen-pool-v2-237586137680/project_backups/maat/${STAMP}}"

AWS_PROFILE_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_PROFILE_ARGS=(--profile "${AWS_PROFILE}")
fi

echo "[backup] destination: ${S3_ROOT}"

# Upload full reproducibility snapshot archive if present.
LATEST_ARCHIVE="$(ls -1t "${REPO_ROOT}/ops/backups"/maat_full_snapshot_*.tar.gz 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_ARCHIVE}" && -f "${LATEST_ARCHIVE}" ]]; then
  echo "[backup] uploading snapshot archive: ${LATEST_ARCHIVE}"
  aws "${AWS_PROFILE_ARGS[@]}" s3 cp "${LATEST_ARCHIVE}" "${S3_ROOT}/archives/$(basename "${LATEST_ARCHIVE}")" --only-show-errors
fi

SYNC_DIRS=(
  "README.md"
  "docs"
  "data_generation_package"
  "data_generation_package_v2"
  "data_generation_package_v3"
  "VAE training"
  "world_model_training"
  "world_model_inference"
  "samples"
  "ops/runs"
)

for rel in "${SYNC_DIRS[@]}"; do
  src="${REPO_ROOT}/${rel}"
  if [[ -e "${src}" ]]; then
    if [[ -f "${src}" ]]; then
      echo "[backup] copying file ${rel}"
      aws "${AWS_PROFILE_ARGS[@]}" s3 cp "${src}" "${S3_ROOT}/repo/${rel}" --only-show-errors
    else
      echo "[backup] syncing ${rel}"
      aws "${AWS_PROFILE_ARGS[@]}" s3 sync "${src}" "${S3_ROOT}/repo/${rel}" \
        --only-show-errors \
        --exclude "*.pyc" \
        --exclude "__pycache__/*"
    fi
  fi
done

MANIFEST_PATH="${REPO_ROOT}/ops/backups/backup_manifest_${STAMP}.txt"
{
  echo "timestamp_utc=${STAMP}"
  echo "s3_root=${S3_ROOT}"
  echo "repo_root=${REPO_ROOT}"
  echo "dirs_synced=${#SYNC_DIRS[@]}"
  for rel in "${SYNC_DIRS[@]}"; do
    echo " - ${rel}"
  done
} > "${MANIFEST_PATH}"

aws "${AWS_PROFILE_ARGS[@]}" s3 cp "${MANIFEST_PATH}" "${S3_ROOT}/backup_manifest_${STAMP}.txt" --only-show-errors

echo "[backup] complete"
echo "[backup] manifest: ${MANIFEST_PATH}"
echo "[backup] s3 root: ${S3_ROOT}"
