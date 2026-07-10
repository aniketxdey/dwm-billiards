#!/usr/bin/env bash
set -euo pipefail

INF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${INF_ROOT}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/world_model/inference/src:${REPO_ROOT}/world_model/training/src:${REPO_ROOT}/vae_training/src:${PYTHONPATH:-}"
python3 -m world_model_inference.ui_app
