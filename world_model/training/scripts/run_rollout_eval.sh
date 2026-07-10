#!/usr/bin/env bash
set -euo pipefail

WM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${WM_ROOT}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${WM_ROOT}/configs/rollout_eval_60m_ctx8_vs_ctx12.yaml}"
EVAL_ID="${EVAL_ID:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}"
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/world_model_training/src:${REPO_ROOT}/VAE training/src:${PYTHONPATH:-}"

CMD=(python3 -m world_model_training.eval_rollout --config "${CONFIG_PATH}")
if [[ -n "${EVAL_ID}" ]]; then
  CMD+=(--eval-id "${EVAL_ID}")
fi

"${CMD[@]}"
