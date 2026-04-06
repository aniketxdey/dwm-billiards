#!/usr/bin/env bash
set -euo pipefail

WM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${WM_ROOT}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${WM_ROOT}/configs/dit_5m_1xa100.yaml}"
RUN_ID="${RUN_ID:-}"
RUN_NOTES="${RUN_NOTES:-}"
RESUME_CKPT="${RESUME_CKPT:-}"

if [[ -z "${RUN_ID}" ]]; then
  echo "Missing RUN_ID."
  echo "Example:"
  echo "  RUN_ID=dit_5m_1xa100_20260221_000000_run01 bash \"world_model_training/scripts/run_train_1xa100.sh\""
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${WM_ROOT}/src:${PYTHONPATH:-}"

CMD=(python3 -m world_model_training.train --config "${CONFIG_PATH}" --run-id "${RUN_ID}")

if [[ -n "${RUN_NOTES}" ]]; then
  CMD+=(--notes "${RUN_NOTES}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume "${RESUME_CKPT}")
fi

"${CMD[@]}"
