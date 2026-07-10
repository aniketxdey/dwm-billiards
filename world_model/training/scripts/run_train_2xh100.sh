#!/usr/bin/env bash
set -euo pipefail

WM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${WM_ROOT}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${WM_ROOT}/configs/dit_60m_2xh100_ctx8.yaml}"
RUN_ID="${RUN_ID:-}"
RUN_NOTES="${RUN_NOTES:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29511}"

if [[ -z "${RUN_ID}" ]]; then
  echo "Missing RUN_ID."
  echo "Example:"
  echo "  RUN_ID=dit_60m_ctx8_2xh100_20260222_000000_run01 bash \"world_model_training/scripts/run_train_2xh100.sh\""
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${WM_ROOT}/src:${PYTHONPATH:-}"

CMD=(
  torchrun
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  -m world_model_training.train
  --config "${CONFIG_PATH}"
  --run-id "${RUN_ID}"
)

if [[ -n "${RUN_NOTES}" ]]; then
  CMD+=(--notes "${RUN_NOTES}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume "${RESUME_CKPT}")
fi

"${CMD[@]}"
