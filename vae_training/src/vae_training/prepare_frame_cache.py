from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


SHARD_RE = re.compile(r"shard_(\d+)\.npz$")


def parse_shard_id(path: Path) -> int:
    m = SHARD_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected shard filename: {path.name}")
    return int(m.group(1))


def discover_shards(shards_dir: Path) -> List[Path]:
    shards = sorted(shards_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz files found in {shards_dir}")
    return shards


def choose_shards(shards: List[Path], max_shards: int, seed: int) -> List[Path]:
    if max_shards <= 0 or max_shards >= len(shards):
        return shards
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(shards), size=max_shards, replace=False))
    return [shards[int(i)] for i in idx]


def frame_shape_from_shard(shard_path: Path) -> Tuple[int, int]:
    with np.load(shard_path, allow_pickle=False) as d:
        frames = d["frames"]
        return int(frames.shape[2]), int(frames.shape[3])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build randomized frame cache from NPZ shards")
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-frames", type=int, default=1_000_000)
    parser.add_argument("--max-shards", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help="Allow sampling with replacement within each shard. "
        "By default sampling is without replacement to preserve frame uniqueness.",
    )
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = discover_shards(shards_dir)
    selected = choose_shards(shards, max_shards=args.max_shards, seed=args.seed)

    h, w = frame_shape_from_shard(selected[0])

    target = int(args.target_frames)
    per_shard = target // len(selected)
    remainder = target % len(selected)

    if target % 1_000_000 == 0:
        tag = f"{target // 1_000_000}m"
    else:
        tag = f"{target}f"

    frame_cache_path = out_dir / f"frame_cache_{tag}.npy"
    frame_index_path = out_dir / f"frame_index_{tag}.npy"
    manifest_path = out_dir / f"frame_cache_manifest_{tag}.json"

    frame_cache = np.lib.format.open_memmap(
        frame_cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=(target, h, w, 3),
    )
    frame_index = np.zeros((target, 3), dtype=np.int32)

    rng = np.random.default_rng(args.seed)
    cursor = 0

    print(f"Selected shards: {len(selected)} / {len(shards)}")
    print(f"Writing frame cache: {frame_cache_path}")

    for i, shard_path in enumerate(selected):
        take = per_shard + (1 if i < remainder else 0)
        shard_id = parse_shard_id(shard_path)

        with np.load(shard_path, allow_pickle=False) as d:
            frames = d["frames"]
            n_eps = int(frames.shape[0])
            n_t = int(frames.shape[1])

            n_flat = n_eps * n_t
            if args.allow_replacement:
                flat_idx = rng.integers(0, n_flat, size=take)
            else:
                if take > n_flat:
                    raise ValueError(
                        f"Requested {take} unique frames from shard {shard_path.name}, "
                        f"but only {n_flat} are available. "
                        "Reduce target-frames/max-shards or enable --allow-replacement."
                    )
                flat_idx = (
                    rng.permutation(n_flat)
                    if take == n_flat
                    else rng.choice(n_flat, size=take, replace=False)
                )

            ep_idx = flat_idx // n_t
            t_idx = flat_idx % n_t
            sampled = frames[ep_idx, t_idx]

            frame_cache[cursor : cursor + take] = sampled
            frame_index[cursor : cursor + take, 0] = shard_id
            frame_index[cursor : cursor + take, 1] = ep_idx
            frame_index[cursor : cursor + take, 2] = t_idx

        cursor += take
        if (i + 1) % 10 == 0 or i == len(selected) - 1:
            print(f"Processed shards: {i + 1}/{len(selected)} | frames: {cursor}/{target}")

        gc.collect()

    np.save(frame_index_path, frame_index)

    manifest = {
        "target_frames": target,
        "selected_shards": len(selected),
        "total_shards_available": len(shards),
        "seed": args.seed,
        "allow_replacement": bool(args.allow_replacement),
        "frame_cache_path": str(frame_cache_path),
        "frame_index_path": str(frame_index_path),
        "frame_shape": [h, w, 3],
        "shard_ids": [parse_shard_id(p) for p in selected],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Frame cache build complete.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
