from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "id_prefix": "dit_base",
        "seed": 42,
        "output_root": "./world_model/training/runs",
        "source_latent_s3_root": "",
        "source_latent_local_root": "",
        "target_samples": 1_000_000,
        "checkpoint_every_samples": 200_000,
        "log_every_steps": 50,
        "eval_every_steps": 200,
        "save_optimizer_state": True,
    },
    "data": {
        "shards_dir": "",
        "shards_manifest": "",
        "train_shards_manifest": "",
        "val_shards_manifest": "",
        "eval_shards_manifest": "",
        "context_len": 8,
        "val_shards": 50,
        "batch_size": 256,
        "drop_last": True,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "shuffle_shards": True,
        "shuffle_within_episode": True,
    },
    "model": {
        "latent_channels": 4,
        "latent_h": 9,
        "latent_w": 16,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "action_dim": 3,
    },
    "diffusion": {
        "timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 2e-2,
    },
    "diffusion_forcing": {
        "enabled": False,
        "rollout_steps": 2,
        "teacher_forcing_prob_start": 1.0,
        "teacher_forcing_prob_end": 0.25,
        "teacher_forcing_decay_samples": 20_000_000,
        "detach_predicted_context": True,
    },
    # GameNGen-style context corruption: each context latent is independently
    # noised to a random diffusion level in [0, tau_max] during training.
    "context_noise": {
        "enabled": False,
        "tau_max": 150,
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
    "eval": {
        "val_batches": 20,
    },
    "wandb": {
        "enabled": False,
        "project": "video_generation_project202",
        "entity": "",
        "group": "dit_baseline",
        "tags": ["world-model", "dit", "baseline"],
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
