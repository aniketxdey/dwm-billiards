#!/usr/bin/env bash
set -euo pipefail

# Action-strength sweep from one base preset config.
# Generates one preview run per (seed, force, direction) tuple.

INF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${INF_ROOT}/.." && pwd)"

BASE_CONFIG="${BASE_CONFIG:-${INF_ROOT}/configs/preview_latest_resume360_preset_mouseish.yaml}"
FORCES_CSV="${FORCES_CSV:-20,40,80,120,180,240}"
SEEDS_CSV="${SEEDS_CSV:-42,1337}"
DIRECTIONS_CSV="${DIRECTIONS_CSV:-diag_up_right,diag_up_left}"
DDIM_STEPS="${DDIM_STEPS:-12}"
HORIZON="${HORIZON:-96}"
NUM_CLIPS="${NUM_CLIPS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./world_model_inference/runs/action_scale_sweep}"
PREFIX="${PREFIX:-action_scale}"
SHOT_FRAME="${SHOT_FRAME:-6}"
Y_RATIO="${Y_RATIO:-0.7}" # force_y = sign_y * force * Y_RATIO

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/world_model_inference/src:${REPO_ROOT}/world_model_training/src:${REPO_ROOT}/VAE training/src:${PYTHONPATH:-}"

mkdir -p "${INF_ROOT}/configs/action_scale_sweep"

IFS=',' read -r -a FORCES <<< "${FORCES_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a DIRECTIONS <<< "${DIRECTIONS_CSV}"

run_one() {
  local force="$1"
  local seed="$2"
  local direction="$3"

  local sx="+"
  local sy="-"
  case "${direction}" in
    diag_up_right) sx="+"; sy="-" ;;
    diag_up_left)  sx="-"; sy="-" ;;
    diag_dn_right) sx="+"; sy="+" ;;
    diag_dn_left)  sx="-"; sy="+" ;;
    *) echo "[action-sweep] unknown direction: ${direction}" >&2; return 1 ;;
  esac

  local cfg_path="${INF_ROOT}/configs/action_scale_sweep/${PREFIX}_s${seed}_f${force}_${direction}.yaml"
  local preview_id="${PREFIX}_s${seed}_f${force}_${direction}"

  python3 - <<PY
from pathlib import Path
import yaml

base = Path("${BASE_CONFIG}").expanduser().resolve()
cfg = yaml.safe_load(base.read_text())

force = float(${force})
y_ratio = float(${Y_RATIO})
force_x = force if "${sx}" == "+" else -force
force_y = (force * y_ratio) if "${sy}" == "+" else -(force * y_ratio)

cfg["run"]["preview_id"] = "${preview_id}"
cfg["run"]["ddim_steps"] = int(${DDIM_STEPS})
cfg["run"]["horizon"] = int(${HORIZON})
cfg["run"]["num_clips"] = int(${NUM_CLIPS})
cfg["run"]["seed"] = int(${seed})
cfg["run"]["output_root"] = "${OUTPUT_ROOT}"

cfg["actions"]["source"] = "preset"
cfg["actions"]["preset"]["name"] = "single_shot"
cfg["actions"]["preset"]["horizon"] = int(${HORIZON})
cfg["actions"]["preset"]["shot_frame"] = int(${SHOT_FRAME})
cfg["actions"]["preset"]["force_x"] = float(force_x)
cfg["actions"]["preset"]["force_y"] = float(force_y)
cfg["actions"]["preset"]["seed"] = int(${seed})

out = Path("${cfg_path}").expanduser().resolve()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY

  echo "[action-sweep] running ${preview_id}"
  CONFIG_PATH="${cfg_path}" bash "${INF_ROOT}/scripts/run_preview.sh"
}

for seed in "${SEEDS[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  for force in "${FORCES[@]}"; do
    force="$(echo "${force}" | xargs)"
    for direction in "${DIRECTIONS[@]}"; do
      direction="$(echo "${direction}" | xargs)"
      run_one "${force}" "${seed}" "${direction}"
    done
  done
done

echo "[action-sweep] done. output_root=${OUTPUT_ROOT}"
