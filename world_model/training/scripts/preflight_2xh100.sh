#!/usr/bin/env bash
set -euo pipefail

# Preflight before launching world-model training on 2x H100 (DDP).

WM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${WM_ROOT}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${WM_ROOT}/configs/dit_60m_2xh100_ctx8.yaml}"
RUN_ID="${RUN_ID:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29511}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}"
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "Warning: RUN_ID not set. Set RUN_ID before launch for strict run tracking."
else
  echo "RUN_ID=${RUN_ID}"
fi

echo "[1/7] GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found"
  exit 1
fi

echo "[2/7] Python + torch check"
PYTHONPATH="${WM_ROOT}/src:${PYTHONPATH:-}" python3 - <<'PY' "${NPROC_PER_NODE}"
import sys
import torch
need = int(sys.argv[1])
print('python_ok=True')
print('torch_version=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('cuda_device_count=', torch.cuda.device_count())
if torch.cuda.device_count() < need:
    raise SystemExit(f'Need at least {need} CUDA devices, found {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'cuda_device[{i}]=', torch.cuda.get_device_name(i))
PY

echo "[3/7] Distributed all-reduce smoke test"
cd "${REPO_ROOT}"
export PYTHONPATH="${WM_ROOT}/src:${PYTHONPATH:-}"
torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m world_model_training.dist_smoke

echo "[4/7] Latent shard contract check"
python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import numpy as np
import yaml
from world_model_training.data import resolve_train_val_shards_from_data_cfg

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())

train_shards, val_shards = resolve_train_val_shards_from_data_cfg(cfg['data'])
shards = train_shards + val_shards
if not shards:
    raise SystemExit('no latent shards resolved from config')

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

print('resolved_train_shards=', len(train_shards))
print('resolved_val_shards=', len(val_shards))
print('shard_count=', len(shards))
print('sample_shard=', sample)
print('latents_shape=', latents.shape, 'dtype=', latents.dtype)
print('actions_shape=', actions.shape, 'dtype=', actions.dtype)
print('lengths_shape=', lengths.shape, 'dtype=', lengths.dtype)
PY

echo "[5/7] Disk check"
DATA_ROOT="$(python3 - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import yaml
from world_model_training.data import resolve_train_val_shards_from_data_cfg
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
shards_dir = str(cfg['data'].get('shards_dir', '') or '').strip()
if shards_dir:
    print(shards_dir)
else:
    train_shards, val_shards = resolve_train_val_shards_from_data_cfg(cfg['data'])
    probe = train_shards[0] if train_shards else val_shards[0]
    print(str(probe.parent))
PY
)"
df -h "${DATA_ROOT}" || df -h /

echo "[6/7] W&B env check"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is set"
else
  echo "WANDB_API_KEY not set (only required when wandb.enabled=true)"
fi

echo "[7/7] Config summary"
python3 - <<'PY' "${CONFIG_PATH}" "${NPROC_PER_NODE}"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
nproc = int(sys.argv[2])
print('target_samples=', cfg['run']['target_samples'])
print('checkpoint_every_samples=', cfg['run']['checkpoint_every_samples'])
print('context_len=', cfg['data']['context_len'])
print('batch_size_per_rank=', cfg['data']['batch_size'])
print('global_batch_size=', int(cfg['data']['batch_size']) * nproc)
print('num_workers_per_rank=', cfg['data']['num_workers'])
print('train_shards_manifest=', cfg['data'].get('train_shards_manifest', ''))
print('val_shards_manifest=', cfg['data'].get('val_shards_manifest', ''))
print('mixed_precision=', cfg['training']['mixed_precision'])
print('compile_model=', cfg['training'].get('compile_model', False))
print('wandb_enabled=', cfg['wandb']['enabled'])
PY

echo "Preflight OK"
