#!/usr/bin/env bash
set -euo pipefail

# Generate a consistent world-model run id.
# Example: RUN_PREFIX=dit_5m_1xa100 bash world_model/training/scripts/new_run_id.sh run01

SUFFIX="${1:-run01}"
TS="$(date -u +%Y%m%d_%H%M%S)"
PREFIX="${RUN_PREFIX:-dit_5m_1xa100}"

echo "${PREFIX}_${TS}_${SUFFIX}"
