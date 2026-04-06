#!/usr/bin/env bash
set -euo pipefail

# Quick preflight before launching VAE training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/vae_1m_1xa100.yaml}"
RUN_ID="${RUN_ID:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}"
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "Warning: RUN_ID not set. Set RUN_ID before launch for strict run tracking."
else
  echo "RUN_ID=${RUN_ID}"
fi

echo "[1/5] GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found"
  exit 1
fi

echo "[2/5] Python + torch check"
PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" python3 - <<'PY'
import torch
print('python_ok=True')
print('torch_version=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('cuda_device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('cuda_device_name=', torch.cuda.get_device_name(0))
PY

echo "[3/5] Required paths check"
python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
source = str(cfg.get('data', {}).get('source', 'frame_cache')).lower()
if source == 'shards':
    shards_dir = Path(cfg['data']['shards_dir'])
    if not shards_dir.exists():
        raise SystemExit(f'missing shards dir: {shards_dir}')
    shard_count = len(list(shards_dir.glob('shard_*.npz')))
    if shard_count == 0:
        raise SystemExit(f'no shard_*.npz found in: {shards_dir}')
    print('data_source=shards')
    print('shards_dir_ok=', shards_dir)
    print('shard_count=', shard_count)
else:
    frame_cache = Path(cfg['data']['frame_cache_path'])
    if not frame_cache.exists():
        raise SystemExit(f'missing frame cache: {frame_cache}')
    print('data_source=frame_cache')
    print('frame_cache_ok=', frame_cache)
PY

echo "[4/5] Disk check"
DATA_ROOT="$(python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg.get('run', {}).get('dataset_local_root', '/'))
PY
)"
df -h "${DATA_ROOT}" || df -h /

echo "[5/5] Config summary"
python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print('target_frames=', cfg['run']['target_frames'])
print('batch_size=', cfg['data']['batch_size'])
print('checkpoint_every_frames=', cfg['run']['checkpoint_every_frames'])
print('preview_every_frames=', cfg['run']['preview_every_frames'])
print('mixed_precision=', cfg['training']['mixed_precision'])
PY

echo "Preflight OK"
