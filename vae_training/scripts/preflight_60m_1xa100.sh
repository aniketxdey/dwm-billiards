#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/vae_60m_1xa100.yaml}"

bash "${ROOT_DIR}/scripts/preflight_1xa100.sh"

