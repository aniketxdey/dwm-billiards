#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/vae_1m_1xa100.yaml}"
RUN_ID="${RUN_ID:-}"
RUN_NOTES="${RUN_NOTES:-}"
RESUME_CKPT="${RESUME_CKPT:-}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if [[ -z "${RUN_ID}" ]]; then
  echo "Missing RUN_ID."
  echo "Example:"
  echo "  RUN_ID=vae_1m_a100_run01 bash \"VAE training/scripts/run_train_1xa100.sh\""
  exit 1
fi

CMD=(python3 -m vae_training.train --config "${CONFIG_PATH}" --run-id "${RUN_ID}")

if [[ -n "${RUN_NOTES}" ]]; then
  CMD+=(--notes "${RUN_NOTES}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume "${RESUME_CKPT}")
fi

"${CMD[@]}"
