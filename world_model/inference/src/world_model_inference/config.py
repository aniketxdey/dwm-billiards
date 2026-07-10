from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_PREVIEW_CONFIG: Dict[str, Any] = {
    "run": {
        "preview_id": "",
        "seed": 42,
        "output_root": "./world_model/inference/runs",
        "device": "cuda",
        "ddim_steps": 20,
        "horizon": 32,
        "video_fps": 12,
        "decode_batch_size": 128,
        "num_clips": 1,
    },
    "model": {
        "name": "world_model",
        "checkpoint_path": "",
        "train_config_path": "",
    },
    "data": {
        "shards_dir": "",
        "shards_manifest": "",
        "train_shards_manifest": "",
        "val_shards_manifest": "",
        "eval_shards_manifest": "",
        "val_shards": 50,
        "sample_from": "eval",  # eval|train
    },
    "vae": {
        "enabled": True,
        "checkpoint_path": "",
        "base_channels": 64,
        "latent_channels": 4,
    },
    "actions": {
        "source": "dataset",  # dataset|preset
        "preset": {
            "name": "single_shot",
            "horizon": 32,
            "shot_frame": 0,
            "force_x": 0.0,
            "force_y": -8.0,
            "seed": 42,
            "num_shots": 1,
            "max_force": 12.0,
            "min_gap": 6,
        },
    },
    "viz": {
        "write_video": True,
        "write_action_timeline": True,
        "include_gt_if_available": True,
    },
}


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_structured(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_preview_config(path: str | Path) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_PREVIEW_CONFIG)
    loaded = _load_structured(Path(path))
    _deep_update(cfg, loaded)
    return cfg
