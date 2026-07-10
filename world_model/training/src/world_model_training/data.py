from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    yaml = None


SHARD_RE = re.compile(r"latent_shard_(\d+)\.npz$")


def parse_shard_id(path: Path) -> int:
    m = SHARD_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected latent shard filename: {path.name}")
    return int(m.group(1))


def discover_latent_shards(shards_dir: str | Path) -> List[Path]:
    root = Path(shards_dir)
    if not root.exists():
        raise FileNotFoundError(f"Latent shards directory not found: {root}")
    shards = sorted(root.glob("latent_shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No latent_shard_*.npz files found in {root}")
    return shards


def _resolve_manifest_path_entry(entry: str, manifest_path: Path) -> Path:
    p = Path(entry).expanduser()
    if not p.is_absolute():
        p = (manifest_path.parent / p).resolve()
    return p


def _normalize_manifest_entries(obj: object, manifest_path: Path) -> List[Path]:
    if isinstance(obj, dict):
        for key in ("shards", "paths", "items"):
            if key in obj:
                return _normalize_manifest_entries(obj[key], manifest_path)
        raise ValueError(
            f"Unsupported manifest dict structure in {manifest_path}. "
            "Expected one of keys: shards, paths, items."
        )
    if isinstance(obj, list):
        out: List[Path] = []
        for item in obj:
            if isinstance(item, str):
                out.append(_resolve_manifest_path_entry(item, manifest_path))
                continue
            if isinstance(item, dict):
                path_str = item.get("path") or item.get("shard") or item.get("file")
                if isinstance(path_str, str):
                    out.append(_resolve_manifest_path_entry(path_str, manifest_path))
                    continue
            raise ValueError(f"Unsupported manifest list item in {manifest_path}: {item!r}")
        return out
    raise ValueError(f"Unsupported manifest contents in {manifest_path}: {type(obj).__name__}")


def load_latent_shard_manifest(manifest_path: str | Path) -> List[Path]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Shard manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = json.loads(text)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML shard manifests.")
            payload = yaml.safe_load(text) or []
        shards = _normalize_manifest_entries(payload, path)
    else:
        # Plain text manifest: one path per line. Supports comments.
        shards = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            shards.append(_resolve_manifest_path_entry(line, path))

    if not shards:
        raise ValueError(f"Shard manifest is empty: {path}")

    missing = [str(p) for p in shards if not p.exists()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix_txt = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} shard paths from manifest do not exist: {preview}{suffix_txt}"
        )
    return shards


def split_train_val_shards(shards: Sequence[Path], val_shards: int) -> Tuple[List[Path], List[Path]]:
    if val_shards <= 0:
        return list(shards), []
    if val_shards >= len(shards):
        raise ValueError("val_shards must be smaller than total shard count")
    train = list(shards[:-val_shards])
    val = list(shards[-val_shards:])
    return train, val


def resolve_train_val_shards_from_data_cfg(data_cfg: dict) -> Tuple[List[Path], List[Path]]:
    train_manifest = str(data_cfg.get("train_shards_manifest", "") or "").strip()
    val_manifest = str(data_cfg.get("val_shards_manifest", "") or "").strip()
    all_manifest = str(data_cfg.get("shards_manifest", "") or "").strip()
    shards_dir = str(data_cfg.get("shards_dir", "") or "").strip()

    if train_manifest or val_manifest:
        if not train_manifest:
            raise ValueError("data.train_shards_manifest is required when data.val_shards_manifest is set")
        train_shards = load_latent_shard_manifest(train_manifest)
        val_shards = load_latent_shard_manifest(val_manifest) if val_manifest else []
        return train_shards, val_shards

    if all_manifest:
        shards = load_latent_shard_manifest(all_manifest)
    else:
        if not shards_dir:
            raise ValueError(
                "No latent shard source configured. Set data.shards_dir or data.(train|val)_shards_manifest."
            )
        shards = discover_latent_shards(shards_dir)
    return split_train_val_shards(shards, int(data_cfg["val_shards"]))


def resolve_eval_shards_from_data_cfg(data_cfg: dict) -> List[Path]:
    eval_manifest = str(data_cfg.get("eval_shards_manifest", "") or "").strip()
    val_manifest = str(data_cfg.get("val_shards_manifest", "") or "").strip()
    train_manifest = str(data_cfg.get("train_shards_manifest", "") or "").strip()
    all_manifest = str(data_cfg.get("shards_manifest", "") or "").strip()

    if eval_manifest:
        return load_latent_shard_manifest(eval_manifest)
    if val_manifest:
        return load_latent_shard_manifest(val_manifest)
    if train_manifest:
        return load_latent_shard_manifest(train_manifest)
    if all_manifest:
        shards = load_latent_shard_manifest(all_manifest)
        _, val_shards = split_train_val_shards(shards, int(data_cfg["val_shards"]))
        return val_shards if val_shards else shards

    shards_dir = str(data_cfg.get("shards_dir", "") or "").strip()
    if not shards_dir:
        raise ValueError(
            "No latent shard source configured. Set data.shards_dir or data.eval_shards_manifest."
        )
    shards = discover_latent_shards(shards_dir)
    _, val_shards = split_train_val_shards(shards, int(data_cfg["val_shards"]))
    return val_shards if val_shards else shards


class LatentActionIterableDataset(IterableDataset):
    """
    Streams samples of (context_latents, action_t, target_latent_{t+1}) from latent shards.

    context: [L, C, H, W]
    action: [A]
    target: [C, H, W]
    """

    def __init__(
        self,
        shard_paths: Sequence[Path],
        context_len: int,
        seed: int,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
        repeat: bool = True,
        shuffle_shards: bool = True,
        shuffle_within_episode: bool = True,
    ) -> None:
        super().__init__()
        if context_len < 1:
            raise ValueError("context_len must be >= 1")
        if not shard_paths:
            raise ValueError("shard_paths must not be empty")

        self.shard_paths = list(shard_paths)
        self.context_len = int(context_len)
        self.seed = int(seed)
        self.distributed_rank = int(distributed_rank)
        self.distributed_world_size = int(distributed_world_size)
        if self.distributed_rank < 0:
            raise ValueError("distributed_rank must be >= 0")
        if self.distributed_world_size < 1:
            raise ValueError("distributed_world_size must be >= 1")
        if self.distributed_rank >= self.distributed_world_size:
            raise ValueError("distributed_rank must be < distributed_world_size")
        self.repeat = bool(repeat)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_episode = bool(shuffle_within_episode)

    def _worker_shards(self) -> List[Path]:
        info = get_worker_info()
        local_worker_id = 0 if info is None else int(info.id)
        local_num_workers = 1 if info is None else int(info.num_workers)
        global_worker_id = self.distributed_rank * local_num_workers + local_worker_id
        global_num_workers = self.distributed_world_size * local_num_workers
        return self.shard_paths[global_worker_id::global_num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        rng = np.random.default_rng(self.seed + worker_id)
        worker_shards = self._worker_shards()

        if not worker_shards:
            return

        while True:
            shard_indices = np.arange(len(worker_shards))
            if self.shuffle_shards:
                rng.shuffle(shard_indices)

            for si in shard_indices:
                shard_path = worker_shards[int(si)]
                with np.load(shard_path, allow_pickle=False) as d:
                    latents = d["latents"]  # [E, T, C, H, W]
                    actions = d["actions"]  # [E, T, A]
                    lengths = d["lengths"]  # [E]

                    n_eps = int(latents.shape[0])
                    ep_indices = np.arange(n_eps)
                    rng.shuffle(ep_indices)

                    for ep in ep_indices:
                        ep_len = int(lengths[int(ep)])
                        min_t = self.context_len - 1
                        max_t = ep_len - 2
                        if max_t < min_t:
                            continue

                        t_vals = np.arange(min_t, max_t + 1)
                        if self.shuffle_within_episode:
                            rng.shuffle(t_vals)

                        for t in t_vals:
                            t = int(t)
                            ctx_np = np.array(
                                latents[int(ep), t - self.context_len + 1 : t + 1],
                                dtype=np.float32,
                                copy=True,
                            )
                            act_np = np.array(actions[int(ep), t], dtype=np.float32, copy=True)
                            tgt_np = np.array(latents[int(ep), t + 1], dtype=np.float32, copy=True)

                            yield (
                                torch.from_numpy(ctx_np),
                                torch.from_numpy(act_np),
                                torch.from_numpy(tgt_np),
                            )

            if not self.repeat:
                break


class LatentActionRolloutIterableDataset(IterableDataset):
    """
    Streams multi-step samples for diffusion-forcing style training.

    context: [L, C, H, W]
    actions_seq: [K, A] where actions_seq[j] is action at time t+j
    targets_seq: [K, C, H, W] where targets_seq[j] is latent at time t+j+1
    """

    def __init__(
        self,
        shard_paths: Sequence[Path],
        context_len: int,
        rollout_steps: int,
        seed: int,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
        repeat: bool = True,
        shuffle_shards: bool = True,
        shuffle_within_episode: bool = True,
    ) -> None:
        super().__init__()
        if context_len < 1:
            raise ValueError("context_len must be >= 1")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be >= 1")
        if not shard_paths:
            raise ValueError("shard_paths must not be empty")

        self.shard_paths = list(shard_paths)
        self.context_len = int(context_len)
        self.rollout_steps = int(rollout_steps)
        self.seed = int(seed)
        self.distributed_rank = int(distributed_rank)
        self.distributed_world_size = int(distributed_world_size)
        if self.distributed_rank < 0:
            raise ValueError("distributed_rank must be >= 0")
        if self.distributed_world_size < 1:
            raise ValueError("distributed_world_size must be >= 1")
        if self.distributed_rank >= self.distributed_world_size:
            raise ValueError("distributed_rank must be < distributed_world_size")
        self.repeat = bool(repeat)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_episode = bool(shuffle_within_episode)

    def _worker_shards(self) -> List[Path]:
        info = get_worker_info()
        local_worker_id = 0 if info is None else int(info.id)
        local_num_workers = 1 if info is None else int(info.num_workers)
        global_worker_id = self.distributed_rank * local_num_workers + local_worker_id
        global_num_workers = self.distributed_world_size * local_num_workers
        return self.shard_paths[global_worker_id::global_num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        rng = np.random.default_rng(self.seed + worker_id)
        worker_shards = self._worker_shards()

        if not worker_shards:
            return

        while True:
            shard_indices = np.arange(len(worker_shards))
            if self.shuffle_shards:
                rng.shuffle(shard_indices)

            for si in shard_indices:
                shard_path = worker_shards[int(si)]
                with np.load(shard_path, allow_pickle=False) as d:
                    latents = d["latents"]  # [E, T, C, H, W]
                    actions = d["actions"]  # [E, T, A]
                    lengths = d["lengths"]  # [E]

                    n_eps = int(latents.shape[0])
                    ep_indices = np.arange(n_eps)
                    rng.shuffle(ep_indices)

                    for ep in ep_indices:
                        ep_len = int(lengths[int(ep)])
                        min_t = self.context_len - 1
                        max_t = ep_len - 1 - self.rollout_steps
                        if max_t < min_t:
                            continue

                        t_vals = np.arange(min_t, max_t + 1)
                        if self.shuffle_within_episode:
                            rng.shuffle(t_vals)

                        for t in t_vals:
                            t = int(t)
                            ctx_np = np.array(
                                latents[int(ep), t - self.context_len + 1 : t + 1],
                                dtype=np.float32,
                                copy=True,
                            )
                            act_seq_np = np.array(
                                actions[int(ep), t : t + self.rollout_steps],
                                dtype=np.float32,
                                copy=True,
                            )
                            tgt_seq_np = np.array(
                                latents[int(ep), t + 1 : t + 1 + self.rollout_steps],
                                dtype=np.float32,
                                copy=True,
                            )

                            yield (
                                torch.from_numpy(ctx_np),
                                torch.from_numpy(act_seq_np),
                                torch.from_numpy(tgt_seq_np),
                            )

            if not self.repeat:
                break
