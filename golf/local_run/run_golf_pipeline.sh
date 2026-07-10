#!/usr/bin/env bash
#
# Turnkey golf world-model pipeline (single GPU, e.g. RTX 4090).
#
#   data-gen (CPU) -> VAE -> latent export -> DiT baseline -> Diffusion Forcing -> preview
#
# Run-ids below MUST match the checkpoint paths referenced inside the YAML configs.
# Estimated spend on a ~$0.5/hr 4090: ~$15-25. Set START_STAGE to resume midway.
#
# Usage:
#   bash golf/local_run/run_golf_pipeline.sh
#   EPISODES=10000 START_STAGE=3 bash golf/local_run/run_golf_pipeline.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/world_model/inference/src:${REPO_ROOT}/world_model/training/src:${REPO_ROOT}/vae_training/src:${PYTHONPATH:-}"

PY="${PY:-python3}"
CFG="golf/local_run"

# Data parameters
EPISODES="${EPISODES:-8000}"
SHARD_SIZE="${SHARD_SIZE:-100}"
FRAMES="${FRAMES:-600}"
WORKERS="${WORKERS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
DATA_DIR="${REPO_ROOT}/golf/local_run/raw_shards_main"

# Deterministic run-ids (referenced by the YAML configs)
VAE_RUN_ID="vae_golf_run01"
DIT_BASE_RUN_ID="dit_golf_base_run01"
DIT_DF_RUN_ID="dit_golf_df_run01"
BASE_CKPT="${REPO_ROOT}/golf/local_run/wm_runs/${DIT_BASE_RUN_ID}/checkpoints/ckpt_080000000.pt"

START_STAGE="${START_STAGE:-1}"

echo "=================================================="
echo " GOLF PIPELINE  (start stage: ${START_STAGE})"
echo " episodes=${EPISODES} frames=${FRAMES} workers=${WORKERS}"
echo "=================================================="

# ---- Stage 1: data generation (CPU) ----------------------------------------
if [ "${START_STAGE}" -le 1 ]; then
  echo "[1/6] Generating golf dataset -> ${DATA_DIR}"
  "${PY}" golf/rl_data_gen/generate_golf_shards.py \
    --episodes "${EPISODES}" --shard-size "${SHARD_SIZE}" --frames "${FRAMES}" \
    --workers "${WORKERS}" --output-dir "${DATA_DIR}" --dry-run
fi

# ---- Stage 2: VAE training (the accuracy gate) -----------------------------
if [ "${START_STAGE}" -le 2 ]; then
  echo "[2/6] Training VAE (${VAE_RUN_ID})"
  "${PY}" -m vae_training.train --config "${CFG}/vae_golf.yaml" --run-id "${VAE_RUN_ID}"
fi

# ---- Stage 3: latent export ------------------------------------------------
if [ "${START_STAGE}" -le 3 ]; then
  echo "[3/6] Exporting latents"
  "${PY}" -m vae_training.export_latents --config "${CFG}/latent_export_golf.yaml"
fi

# ---- Stage 4: DiT baseline (single-step) -----------------------------------
if [ "${START_STAGE}" -le 4 ]; then
  echo "[4/6] Training DiT baseline (${DIT_BASE_RUN_ID})"
  "${PY}" -m world_model_training.train --config "${CFG}/dit_golf.yaml" --run-id "${DIT_BASE_RUN_ID}"
fi

# ---- Stage 5: Diffusion Forcing (resume from baseline) ---------------------
if [ "${START_STAGE}" -le 5 ]; then
  echo "[5/6] Training Diffusion Forcing (${DIT_DF_RUN_ID}) resume from ${BASE_CKPT}"
  if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: baseline checkpoint not found: ${BASE_CKPT}" >&2
    exit 1
  fi
  "${PY}" -m world_model_training.train --config "${CFG}/dit_golf_df.yaml" \
    --run-id "${DIT_DF_RUN_ID}" --resume "${BASE_CKPT}"
fi

# ---- Stage 6: rollout preview ----------------------------------------------
if [ "${START_STAGE}" -le 6 ]; then
  echo "[6/6] Rendering rollout preview"
  "${PY}" -m world_model_inference.preview_checkpoint --config "${CFG}/preview_golf.yaml"
fi

echo "=================================================="
echo " DONE. Outputs under golf/local_run/{vae_runs,latents,wm_runs,inference_runs}"
echo "=================================================="
