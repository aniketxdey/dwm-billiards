from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "id_prefix": "vae",
        "seed": 42,
        "output_root": "./VAE training/runs",
        "dataset_s3_root": "",
        "dataset_local_root": "",
        "target_frames": 1_000_000,
        "milestone_frames": [],
        "checkpoint_every_frames": 100_000,
        "preview_every_frames": 10_000,
        "log_every_steps": 50,
        "save_optimizer_state": True,
    },
    "data": {
        "source": "frame_cache",
        "frame_cache_path": "",
        "frame_index_path": "",
        "shards_dir": "",
        "shuffle_shards": True,
        "shuffle_frames_within_shard": True,
        "batch_size": 256,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    },
    "model": {
        "base_channels": 64,
        "latent_channels": 4,
    },
    "loss": {
        "recon_l1_weight": 1.0,
        "lpips_weight": 0.0,
        "kl_weight": 1.0,
        "kl_beta_start": 1e-6,
        "kl_beta_end": 5e-4,
        "kl_warmup_frames": 300_000,
    },
    "optimizer": {
        "lr": 2e-4,
        "weight_decay": 1e-4,
    },
    "training": {
        "mixed_precision": "bf16",
        "max_grad_norm": 1.0,
        "compile_model": False,
    },
    "preview": {
        "fixed_preview_images": 16,
        "fixed_preview_video_frames": 64,
    },
    "wandb": {
        "enabled": False,
        "project": "neural-pool-vae",
        "entity": "",
    },
}


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    _deep_update(cfg, loaded)
    return cfg
