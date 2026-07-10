from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class FrameCacheDataset(Dataset):
    """
    Dataset over a prebuilt frame cache of shape [N, H, W, 3] uint8.
    Returns tensors normalized to [-1, 1] in CHW format.
    """

    def __init__(self, frame_cache_path: str, frame_index_path: str | None = None) -> None:
        self.frame_cache_path = str(frame_cache_path)
        self.frame_index_path = str(frame_index_path) if frame_index_path else None

        if not Path(self.frame_cache_path).exists():
            raise FileNotFoundError(f"Frame cache not found: {self.frame_cache_path}")

        self.frames = np.load(self.frame_cache_path, mmap_mode="r")
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError(f"Expected frame cache shape [N,H,W,3], got {self.frames.shape}")

        self.index = None
        if self.frame_index_path and Path(self.frame_index_path).exists():
            self.index = np.load(self.frame_index_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        x = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float()
        x = x.div(127.5).sub(1.0)
        return x


def discover_shard_paths(shards_dir: str | Path) -> List[Path]:
    root = Path(shards_dir)
    if not root.exists():
        raise FileNotFoundError(f"Shards directory not found: {root}")
    shards = sorted(root.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz files found in {root}")
    return shards


class ShardStreamDataset(IterableDataset):
    """
    Stream frames directly from NPZ shards.
    Suitable when full expanded frame cache would be too large for local disk.
    """

    def __init__(
        self,
        shards_dir: str,
        seed: int = 42,
        shuffle_shards: bool = True,
        shuffle_frames_within_shard: bool = True,
        repeat: bool = True,
    ) -> None:
        self.shards = discover_shard_paths(shards_dir)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_frames_within_shard = bool(shuffle_frames_within_shard)
        self.repeat = bool(repeat)

    def _worker_shards(self) -> List[Path]:
        info = get_worker_info()
        if info is None:
            return self.shards
        return self.shards[info.id :: info.num_workers]

    def __iter__(self) -> Iterator[torch.Tensor]:
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

            for shard_i in shard_indices:
                shard_path = worker_shards[int(shard_i)]
                with np.load(shard_path, allow_pickle=False) as d:
                    frames = d["frames"]  # [E, T, H, W, 3]
                    flat = frames.reshape(-1, *frames.shape[2:])
                    frame_order = np.arange(flat.shape[0])
                    if self.shuffle_frames_within_shard:
                        rng.shuffle(frame_order)

                    for frame_i in frame_order:
                        frame = flat[int(frame_i)]
                        x = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float()
                        x = x.div(127.5).sub(1.0)
                        yield x

            if not self.repeat:
                break


def sample_preview_frames_from_shards(
    shards_dir: str | Path,
    count: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Deterministically sample preview frames from one random shard.
    Returns [count, C, H, W] normalized to [-1, 1].
    """
    if count <= 0:
        raise ValueError("count must be > 0")

    shards = discover_shard_paths(shards_dir)
    rng = np.random.default_rng(seed)
    shard_path = shards[int(rng.integers(0, len(shards)))]

    with np.load(shard_path, allow_pickle=False) as d:
        frames = d["frames"]  # [E, T, H, W, 3]
        n_flat = int(frames.shape[0] * frames.shape[1])
        flat_idx = rng.choice(n_flat, size=count, replace=count > n_flat)
        n_t = int(frames.shape[1])
        ep_idx = flat_idx // n_t
        t_idx = flat_idx % n_t
        sampled = frames[ep_idx, t_idx]

    tensors = []
    for frame in sampled:
        x = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float()
        x = x.div(127.5).sub(1.0)
        tensors.append(x)
    return torch.stack(tensors, dim=0)
