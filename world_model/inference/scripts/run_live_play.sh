#!/usr/bin/env bash
set -euo pipefail

INF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${INF_ROOT}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/world_model_inference/src:${REPO_ROOT}/world_model_training/src:${REPO_ROOT}/VAE training/src:${PYTHONPATH:-}"
python3 -m world_model_inference.live_play --host 127.0.0.1 --port 7863
