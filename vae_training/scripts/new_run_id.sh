#!/usr/bin/env bash
set -euo pipefail

# Generate a consistent run id. Optional suffix argument.
# Example: bash scripts/new_run_id.sh baseline

SUFFIX="${1:-run01}"
TS="$(date -u +%Y%m%d_%H%M%S)"
PREFIX="${RUN_PREFIX:-vae_5m_1xa100}"

echo "${PREFIX}_${TS}_${SUFFIX}"
