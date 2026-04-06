#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/latent_export_60m.yaml}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python3 -m vae_training.export_latents --config "${CONFIG_PATH}"

