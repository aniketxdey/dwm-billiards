#!/usr/bin/env bash
#
# Golf live-play server (drag-to-putt, websocket canvas).
#
# Wires the interactive UI to the trained golf checkpoints produced by
# local_run_golf/run_golf_pipeline.sh. The "Start" button in the browser sends
# an empty request, so the model/VAE/data paths come from the env defaults set
# below. Override any of them inline, e.g.:
#
#   WM_INF_PORT=7900 DIT_CKPT=.../ckpt_080000000.pt \
#     bash world_model_inference/scripts/run_live_play_golf.sh
#
# IMPORTANT: golf was trained with a VAE of base_channels=48 (vs. 64 for the
# original pool model). WM_INF_VAE_BASE_CHANNELS makes the loader match.
set -euo pipefail

INF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${INF_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/world_model_inference/src:${REPO_ROOT}/world_model_training/src:${REPO_ROOT}/VAE training/src:${PYTHONPATH:-}"

# --- golf checkpoint/config/data defaults (overridable) ----------------------
DIT_CFG="${DIT_CFG:-${REPO_ROOT}/local_run_golf/dit_golf_df.yaml}"
DIT_CKPT="${DIT_CKPT:-${REPO_ROOT}/local_run_golf/wm_runs/dit_golf_df_run01/checkpoints/ckpt_120000000.pt}"
VAE_CKPT="${VAE_CKPT:-${REPO_ROOT}/local_run_golf/vae_runs/vae_golf_run01/checkpoints/ckpt_4000000.pt}"
LATENTS_DIR="${LATENTS_DIR:-${REPO_ROOT}/local_run_golf/latents/golf_main/shards}"

export WM_INF_DEFAULT_TRAIN_CONFIG="${WM_INF_DEFAULT_TRAIN_CONFIG:-${DIT_CFG}}"
export WM_INF_DEFAULT_CHECKPOINT="${WM_INF_DEFAULT_CHECKPOINT:-${DIT_CKPT}}"
export WM_INF_DEFAULT_VAE_CHECKPOINT="${WM_INF_DEFAULT_VAE_CHECKPOINT:-${VAE_CKPT}}"
export WM_INF_DEFAULT_DATA_PATH="${WM_INF_DEFAULT_DATA_PATH:-${LATENTS_DIR}}"
export WM_INF_VAE_BASE_CHANNELS="${WM_INF_VAE_BASE_CHANNELS:-48}"

HOST="${WM_LIVE_HOST:-127.0.0.1}"
PORT="${WM_LIVE_PORT:-7863}"

echo "=================================================="
echo " GOLF LIVE PLAY"
echo "   train_config : ${WM_INF_DEFAULT_TRAIN_CONFIG}"
echo "   dit_ckpt     : ${WM_INF_DEFAULT_CHECKPOINT}"
echo "   vae_ckpt     : ${WM_INF_DEFAULT_VAE_CHECKPOINT}  (base_channels=${WM_INF_VAE_BASE_CHANNELS})"
echo "   latents_dir  : ${WM_INF_DEFAULT_DATA_PATH}"
echo "   serving      : http://${HOST}:${PORT}"
echo "=================================================="

for p in "${WM_INF_DEFAULT_CHECKPOINT}" "${WM_INF_DEFAULT_VAE_CHECKPOINT}"; do
  if [ ! -f "${p}" ]; then
    echo "WARNING: checkpoint not found yet: ${p}" >&2
    echo "         (run local_run_golf/run_golf_pipeline.sh first, or override the path)" >&2
  fi
done

exec python3 -m world_model_inference.live_play --host "${HOST}" --port "${PORT}"
