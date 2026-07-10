#!/usr/bin/env python3
"""
Create a small, inspectable sample bundle for Neural Pool data.

What this script does:
1) Generates one local simulation episode from the current generator.
2) Exports:
   - video (mp4)
   - frames (png files)
   - combined episode data (.npz)
   - actions (.csv)
   - summary (.json)
3) Optionally downloads one shard from S3 and exports one episode in the same format.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_pool_shards import generate_episode, FPS


DEFAULT_S3_URI = "s3://videogen-dataset-dp/dataraw/raw/shards/shard_00000.npz"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_video(frames_rgb: np.ndarray, out_path: Path, fps: int) -> None:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected frames shape (T,H,W,3), got {frames_rgb.shape}")

    height, width = frames_rgb.shape[1], frames_rgb.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    try:
        for frame in frames_rgb:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
    finally:
        writer.release()


def write_frames(frames_rgb: np.ndarray, out_dir: Path, frame_limit: int) -> int:
    ensure_dir(out_dir)
    total = frames_rgb.shape[0]
    if frame_limit < 0:
        limit = total
    else:
        limit = min(total, frame_limit)

    for i in range(limit):
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), frames_rgb[i][:, :, ::-1])
    return limit


def write_actions_csv(actions: np.ndarray, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "force_x", "force_y", "trigger"])
        for idx, row in enumerate(actions):
            writer.writerow([idx, float(row[0]), float(row[1]), float(row[2])])


def save_episode_npz(
    out_path: Path,
    frames: np.ndarray,
    actions: np.ndarray,
    sim_state: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    np.savez_compressed(
        out_path,
        frames=frames.astype(np.uint8),
        actions=actions.astype(np.float32),
        sim_state=sim_state.astype(np.float32),
        meta=np.array(meta, dtype=object),
    )


def save_summary_json(
    out_path: Path,
    source: str,
    frames: np.ndarray,
    actions: np.ndarray,
    sim_state: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    shot_frames = np.where(actions[:, 2] > 0.5)[0]
    active_balls = int(np.sum(np.linalg.norm(sim_state[0, :, :2], axis=1) != 0))
    summary = {
        "source": source,
        "num_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "channels": int(frames.shape[3]),
        "shots_from_actions": int(shot_frames.shape[0]),
        "shot_frames": shot_frames.tolist(),
        "active_balls_frame0": active_balls,
        "meta": meta,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


def export_episode(
    prefix: str,
    frames: np.ndarray,
    actions: np.ndarray,
    sim_state: np.ndarray,
    meta: Dict[str, Any],
    out_root: Path,
    frame_limit: int,
) -> None:
    video_dir = out_root / "video"
    frames_dir = out_root / "frames" / prefix
    data_dir = out_root / "data"
    ensure_dir(video_dir)
    ensure_dir(data_dir)

    video_path = video_dir / f"{prefix}.mp4"
    npz_path = data_dir / f"{prefix}.npz"
    actions_path = data_dir / f"{prefix}_actions.csv"
    summary_path = data_dir / f"{prefix}_summary.json"

    write_video(frames, video_path, FPS)
    written_frames = write_frames(frames, frames_dir, frame_limit)
    write_actions_csv(actions, actions_path)
    save_episode_npz(npz_path, frames, actions, sim_state, meta)
    save_summary_json(summary_path, prefix, frames, actions, sim_state, meta)

    print(f"[{prefix}] video:   {video_path}")
    print(f"[{prefix}] frames:  {frames_dir} ({written_frames} files)")
    print(f"[{prefix}] data:    {npz_path}")
    print(f"[{prefix}] actions: {actions_path}")
    print(f"[{prefix}] summary: {summary_path}")


def download_s3_shard(s3_uri: str, profile: str | None, dest_path: Path) -> None:
    cmd = ["aws"]
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(["s3", "cp", s3_uri, str(dest_path)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed downloading {s3_uri}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def load_episode_from_shard(shard_path: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(shard_path, allow_pickle=True) as data:
        if "frames" not in data or "actions" not in data:
            raise ValueError(
                "Shard does not contain raw frames/actions. "
                "Use a raw shard path like s3://.../dataraw/raw/shards/shard_00000.npz"
            )
        if "sim_state" not in data:
            raise ValueError("Shard missing sim_state key.")

        n_episodes = data["frames"].shape[0]
        if episode_index < 0 or episode_index >= n_episodes:
            raise IndexError(f"episode_index {episode_index} out of range 0..{n_episodes-1}")

        if "lengths" in data:
            length = int(data["lengths"][episode_index])
        else:
            length = int(data["frames"].shape[1])

        frames = data["frames"][episode_index, :length]
        actions = data["actions"][episode_index, :length]
        sim_state = data["sim_state"][episode_index, :length]

        meta: Dict[str, Any]
        if "episode_meta" in data:
            raw_meta = data["episode_meta"][episode_index]
            meta = raw_meta if isinstance(raw_meta, dict) else {"episode_meta": str(raw_meta)}
        else:
            meta = {"episode_index": episode_index}

        return frames, actions, sim_state, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and export a sample bundle.")
    parser.add_argument("--output-dir", default="sample_bundle", help="Output directory for sample bundle.")
    parser.add_argument("--episode-id", type=int, default=0, help="Episode id for locally generated sample.")
    parser.add_argument("--frames", type=int, default=600, help="Frames for locally generated sample.")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=120,
        help="Number of frames to export as PNG per sample. Use -1 for all frames.",
    )
    parser.add_argument("--skip-s3", action="store_true", help="Skip S3 sample download/export.")
    parser.add_argument("--s3-uri", default=DEFAULT_S3_URI, help="Raw shard S3 URI to download.")
    parser.add_argument("--s3-episode-index", type=int, default=0, help="Episode index inside downloaded shard.")
    parser.add_argument("--aws-profile", default="codex-admin", help="AWS profile for s3 download.")
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    ensure_dir(out_root)
    ensure_dir(out_root / "video")
    ensure_dir(out_root / "frames")
    ensure_dir(out_root / "data")

    print("Generating local simulation sample...")
    generated = generate_episode(episode_id=args.episode_id, num_frames=args.frames)
    export_episode(
        prefix="generated_episode",
        frames=generated["frames"],
        actions=generated["actions"],
        sim_state=generated["sim_state"],
        meta=generated["episode_meta"],
        out_root=out_root,
        frame_limit=args.frame_limit,
    )

    if args.skip_s3:
        print("Skipping S3 sample export (--skip-s3).")
        print(f"Done. Sample bundle at: {out_root}")
        return

    print(f"Downloading shard from S3: {args.s3_uri}")
    with tempfile.TemporaryDirectory(prefix="neural_pool_sample_") as tmp_dir:
        shard_local = Path(tmp_dir) / "sample_shard.npz"
        download_s3_shard(args.s3_uri, args.aws_profile, shard_local)
        frames, actions, sim_state, meta = load_episode_from_shard(shard_local, args.s3_episode_index)
        meta = dict(meta)
        meta["source_shard"] = args.s3_uri
        meta["source_episode_index"] = args.s3_episode_index
        export_episode(
            prefix="s3_episode",
            frames=frames,
            actions=actions,
            sim_state=sim_state,
            meta=meta,
            out_root=out_root,
            frame_limit=args.frame_limit,
        )

    print(f"Done. Sample bundle at: {out_root}")


if __name__ == "__main__":
    main()

