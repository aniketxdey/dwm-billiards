#!/usr/bin/env bash
set -euo pipefail

# Run a controlled DDIM-step comparison from one checkpoint/config prompt.
# Outputs one preview run per DDIM step under world_model_inference/runs/ddim_matrix/.

INF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${INF_ROOT}/.." && pwd)"

BASE_CONFIG="${BASE_CONFIG:-${INF_ROOT}/configs/preview_latest_resume360_dataset.yaml}"
STEPS_CSV="${STEPS_CSV:-4,6,8,12,20,30}"
HORIZON="${HORIZON:-64}"
NUM_CLIPS="${NUM_CLIPS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./world_model_inference/runs/ddim_matrix}"
PREFIX="${PREFIX:-ddim_matrix}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/world_model_inference/src:${REPO_ROOT}/world_model_training/src:${REPO_ROOT}/VAE training/src:${PYTHONPATH:-}"

mkdir -p "${INF_ROOT}/configs/ddim_matrix"

IFS=',' read -r -a STEPS <<< "${STEPS_CSV}"

for STEP in "${STEPS[@]}"; do
  STEP="$(echo "${STEP}" | xargs)"
  CFG_PATH="${INF_ROOT}/configs/ddim_matrix/${PREFIX}_ddim${STEP}.yaml"

  python3 - <<PY
from pathlib import Path
import yaml
base = Path("${BASE_CONFIG}").expanduser().resolve()
cfg = yaml.safe_load(base.read_text())
cfg["run"]["preview_id"] = "${PREFIX}_ddim${STEP}"
cfg["run"]["ddim_steps"] = int(${STEP})
cfg["run"]["horizon"] = int(${HORIZON})
cfg["run"]["num_clips"] = int(${NUM_CLIPS})
cfg["run"]["output_root"] = "${OUTPUT_ROOT}"
out = Path("${CFG_PATH}").expanduser().resolve()
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY

  echo "[benchmark] running ddim=${STEP}"
  CONFIG_PATH="${CFG_PATH}" bash "${INF_ROOT}/scripts/run_preview.sh"
done

echo "[benchmark] done. output_root=${OUTPUT_ROOT}"
