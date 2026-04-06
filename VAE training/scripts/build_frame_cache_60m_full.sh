#!/usr/bin/env bash
set -euo pipefail

# Build full 60M-frame cache (all shards, no replacement).
# Output size is ~1.66 TB (uint8 frames) plus index metadata.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARDS_DIR="${SHARDS_DIR:-/home/ubuntu/neural-pool/full_20260220_112101/raw/shards}"
OUT_DIR="${OUT_DIR:-/home/ubuntu/neural-pool/frame_cache}"
TARGET_FRAMES="${TARGET_FRAMES:-60000000}"
MAX_SHARDS="${MAX_SHARDS:-0}" # 0 means all shards
SEED="${SEED:-42}"

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
python3 -m vae_training.prepare_frame_cache \
  --shards-dir "${SHARDS_DIR}" \
  --output-dir "${OUT_DIR}" \
  --target-frames "${TARGET_FRAMES}" \
  --max-shards "${MAX_SHARDS}" \
  --seed "${SEED}"

