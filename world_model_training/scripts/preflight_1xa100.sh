#!/usr/bin/env bash
set -euo pipefail

# Quick preflight before launching world-model (DiT) training.

WM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${WM_ROOT}/configs/dit_5m_1xa100.yaml}"
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

echo "[1/6] GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,utilization.gpu,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found"
  exit 1
fi

echo "[2/6] Python + torch check"
PYTHONPATH="${WM_ROOT}/src:${PYTHONPATH:-}" python3 - <<'PY'
import torch
print('python_ok=True')
print('torch_version=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('cuda_device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('cuda_device_name=', torch.cuda.get_device_name(0))
PY

echo "[3/6] Latent shard contract check"
python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import numpy as np
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
shards_dir = Path(cfg['data']['shards_dir'])
if not shards_dir.exists():
    raise SystemExit(f'missing shards dir: {shards_dir}')

shards = sorted(shards_dir.glob('latent_shard_*.npz'))
if not shards:
    raise SystemExit(f'no latent_shard_*.npz files in {shards_dir}')

sample = shards[0]
with np.load(sample, allow_pickle=False) as d:
    required = ['latents', 'actions', 'lengths']
    missing = [k for k in required if k not in d]
    if missing:
        raise SystemExit(f'missing keys in {sample}: {missing}')

    latents = d['latents']
    actions = d['actions']
    lengths = d['lengths']

    exp_c = int(cfg['model']['latent_channels'])
    exp_h = int(cfg['model']['latent_h'])
    exp_w = int(cfg['model']['latent_w'])
    exp_a = int(cfg['model']['action_dim'])
    if latents.shape[-3:] != (exp_c, exp_h, exp_w):
        raise SystemExit(
            f'latent shape mismatch: got {latents.shape[-3:]}, expected {(exp_c, exp_h, exp_w)}'
        )
    if actions.shape[-1] != exp_a:
        raise SystemExit(f'action dim mismatch: got {actions.shape[-1]}, expected {exp_a}')

print('shards_dir_ok=', shards_dir)
print('shard_count=', len(shards))
print('sample_shard=', sample)
print('latents_shape=', latents.shape, 'dtype=', latents.dtype)
print('actions_shape=', actions.shape, 'dtype=', actions.dtype)
print('lengths_shape=', lengths.shape, 'dtype=', lengths.dtype)
PY

echo "[4/6] Disk check"
DATA_ROOT="$(python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg['data']['shards_dir'])
PY
)"
df -h "${DATA_ROOT}" || df -h /

echo "[5/6] W&B env check"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is set"
else
  echo "WANDB_API_KEY not set (only required when wandb.enabled=true)"
fi

echo "[6/6] Config summary"
python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print('target_samples=', cfg['run']['target_samples'])
print('checkpoint_every_samples=', cfg['run']['checkpoint_every_samples'])
print('batch_size=', cfg['data']['batch_size'])
print('num_workers=', cfg['data']['num_workers'])
print('context_len=', cfg['data']['context_len'])
print('mixed_precision=', cfg['training']['mixed_precision'])
print('compile_model=', cfg['training'].get('compile_model', False))
print('wandb_enabled=', cfg['wandb']['enabled'])
PY

echo "Preflight OK"
