#!/usr/bin/env bash
set -euo pipefail

# Sync local repo to Lambda instance (code only, no run artifacts).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_HOST="${REMOTE_HOST:-129.146.183.88}"
REMOTE_PORT="${REMOTE_PORT:-22}"
SSH_KEY_PATH="${SSH_KEY_PATH:-${ROOT_DIR}/bill-diff.pem}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/maat}"

if [[ ! -f "${SSH_KEY_PATH}" ]]; then
  echo "SSH key not found: ${SSH_KEY_PATH}"
  exit 1
fi

chmod 600 "${SSH_KEY_PATH}"

echo "Syncing ${ROOT_DIR} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'VAE training/runs/' \
  --exclude 'world_model_training/runs/' \
  --exclude 'world_model_training/launcher_logs/' \
  --exclude 'wandb/' \
  -e "ssh -i ${SSH_KEY_PATH} -p ${REMOTE_PORT} -o StrictHostKeyChecking=accept-new" \
  "${ROOT_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Sync complete"
