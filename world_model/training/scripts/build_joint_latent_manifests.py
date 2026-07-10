#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence


def discover_latent_shards(shards_dir: Path) -> List[Path]:
    if not shards_dir.exists():
        raise FileNotFoundError(f"Latent shards directory not found: {shards_dir}")
    shards = sorted(shards_dir.glob("latent_shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No latent_shard_*.npz files found in {shards_dir}")
    return shards


def split_tail(shards: Sequence[Path], n_val: int) -> tuple[List[Path], List[Path]]:
    if n_val <= 0:
        return list(shards), []
    if n_val >= len(shards):
        raise ValueError(f"n_val={n_val} must be smaller than shard count ({len(shards)})")
    return list(shards[:-n_val]), list(shards[-n_val:])


def interleave_lists(a: Sequence[Path], b: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    n = max(len(a), len(b))
    for i in range(n):
        if i < len(a):
            out.append(Path(a[i]))
        if i < len(b):
            out.append(Path(b[i]))
    return out


def write_manifest_txt(paths: Iterable[Path], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(Path(p).resolve()) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build joint latent shard manifests for v1+v2(+v3) training")
    parser.add_argument("--v1-shards-dir", required=True, help="Path to latents_v1/.../shards")
    parser.add_argument("--v2-shards-dir", required=True, help="Path to latents_v2/.../shards")
    parser.add_argument("--v3-shards-dir", default="", help="Optional path to latents_v3/.../shards")
    parser.add_argument("--out-dir", required=True, help="Output manifest directory")
    parser.add_argument("--val-shards-v1", type=int, default=50, help="Tail shards from v1 reserved for validation")
    parser.add_argument("--val-shards-v2", type=int, default=50, help="Tail shards from v2 reserved for validation")
    parser.add_argument("--val-shards-v3", type=int, default=100, help="Tail shards from v3 reserved for validation")
    parser.add_argument(
        "--train-order",
        choices=["interleave", "concat_v1_v2", "concat_v2_v1", "concat_v1_v2_v3", "concat_v3_v1_v2"],
        default="interleave",
        help="Ordering of train shards in manifest (training still shuffles by default)",
    )
    parser.add_argument(
        "--write-train-all-manifest",
        action="store_true",
        help="Also write train_all_shards.txt containing all shards from v1+v2 (no holdout removed)",
    )
    args = parser.parse_args()

    v1_dir = Path(args.v1_shards_dir).expanduser().resolve()
    v2_dir = Path(args.v2_shards_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_cfgs = [
        ("v1", Path(args.v1_shards_dir).expanduser().resolve(), int(args.val_shards_v1)),
        ("v2", Path(args.v2_shards_dir).expanduser().resolve(), int(args.val_shards_v2)),
    ]
    if args.v3_shards_dir:
        source_cfgs.append(("v3", Path(args.v3_shards_dir).expanduser().resolve(), int(args.val_shards_v3)))

    src_all = {}
    src_train = {}
    src_val = {}
    for name, shards_dir, n_val in source_cfgs:
        shards_all = discover_latent_shards(shards_dir)
        shards_train, shards_val = split_tail(shards_all, n_val)
        src_all[name] = shards_all
        src_train[name] = shards_train
        src_val[name] = shards_val

    names = [name for name, _, _ in source_cfgs]
    if args.train_order == "interleave":
        train_joint = []
        max_len = max(len(src_train[n]) for n in names)
        for i in range(max_len):
            for n in names:
                if i < len(src_train[n]):
                    train_joint.append(Path(src_train[n][i]))
    elif args.train_order == "concat_v2_v1":
        train_joint = list(src_train["v2"]) + list(src_train["v1"])
    elif args.train_order == "concat_v1_v2":
        train_joint = list(src_train["v1"]) + list(src_train["v2"])
    elif args.train_order == "concat_v3_v1_v2":
        if "v3" not in src_train:
            raise ValueError("concat_v3_v1_v2 requires --v3-shards-dir")
        train_joint = list(src_train["v3"]) + list(src_train["v1"]) + list(src_train["v2"])
    else:  # concat_v1_v2_v3
        train_joint = list(src_train["v1"]) + list(src_train["v2"])
        if "v3" in src_train:
            train_joint += list(src_train["v3"])

    val_joint = []
    max_val_len = max(len(src_val[n]) for n in names)
    for i in range(max_val_len):
        for n in names:
            if i < len(src_val[n]):
                val_joint.append(Path(src_val[n][i]))

    all_joint = []
    max_all_len = max(len(src_all[n]) for n in names)
    for i in range(max_all_len):
        for n in names:
            if i < len(src_all[n]):
                all_joint.append(Path(src_all[n][i]))

    train_count = write_manifest_txt(train_joint, out_dir / "train_shards.txt")
    val_count = write_manifest_txt(val_joint, out_dir / "val_shards.txt")
    all_count = write_manifest_txt(all_joint, out_dir / "all_shards.txt")
    eval_count = write_manifest_txt(val_joint, out_dir / "eval_shards.txt")
    train_all_count = 0
    if args.write_train_all_manifest:
        train_all_count = write_manifest_txt(all_joint, out_dir / "train_all_shards.txt")

    summary = {
        "version": 1,
        "kind": "joint_latent_manifest",
        "sources": {
            name: {
                "shards_dir": str(next(p for n, p, _ in source_cfgs if n == name)),
                "total_shards": len(src_all[name]),
                "train_shards": len(src_train[name]),
                "val_shards": len(src_val[name]),
            }
            for name in names
        },
        "train_order": args.train_order,
        "outputs": {
            "train_shards_txt": str((out_dir / "train_shards.txt").resolve()),
            "val_shards_txt": str((out_dir / "val_shards.txt").resolve()),
            "eval_shards_txt": str((out_dir / "eval_shards.txt").resolve()),
            "all_shards_txt": str((out_dir / "all_shards.txt").resolve()),
            "train_all_shards_txt": str((out_dir / "train_all_shards.txt").resolve())
            if args.write_train_all_manifest
            else "",
        },
        "counts": {
            "train_joint_shards": train_count,
            "val_joint_shards": val_count,
            "eval_joint_shards": eval_count,
            "all_joint_shards": all_count,
            "train_all_joint_shards": train_all_count,
        },
        "notes": {
            "episodes_per_shard_assumed": 100,
            "joint_total_episodes_estimate": all_count * 100,
            "joint_train_episodes_estimate": train_count * 100,
            "joint_val_episodes_estimate": val_count * 100,
            "sources_included": names,
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
