#!/usr/bin/env python3
"""
Create a small, inspectable sample bundle for Neural Golf data.

Generates one local mini-golf episode and exports:
  - video (mp4)
  - frames (png)
  - combined episode data (.npz)
  - actions (.csv)
  - summary (.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_golf_shards import FPS, generate_episode


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_video(frames_rgb: np.ndarray, out_path: Path, fps: int) -> None:
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
    limit = total if frame_limit < 0 else min(total, frame_limit)
    for i in range(limit):
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), frames_rgb[i][:, :, ::-1])
    return limit


def write_actions_csv(actions: np.ndarray, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "force_x", "force_y", "trigger"])
        for idx, row in enumerate(actions):
            writer.writerow([idx, float(row[0]), float(row[1]), float(row[2])])


def save_summary_json(out_path: Path, frames, actions, sim_state, meta: Dict[str, Any]) -> None:
    shot_frames = np.where(actions[:, 2] > 0.5)[0]
    summary = {
        "source": "generated_golf_episode",
        "num_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "channels": int(frames.shape[3]),
        "shots_from_actions": int(shot_frames.shape[0]),
        "shot_frames": shot_frames.tolist(),
        "meta": meta,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and export a golf sample bundle.")
    parser.add_argument("--output-dir", default="sample_bundle")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--frame-limit", type=int, default=120,
                        help="PNG frames to export (-1 = all).")
    args = parser.parse_args()

    out_root = Path(args.output_dir).resolve()
    ensure_dir(out_root / "video")
    ensure_dir(out_root / "frames")
    ensure_dir(out_root / "data")

    print("Generating local golf simulation sample...")
    ep = generate_episode(episode_id=args.episode_id, num_frames=args.frames)
    frames, actions, sim_state, meta = ep["frames"], ep["actions"], ep["sim_state"], ep["episode_meta"]

    prefix = "generated_golf_episode"
    video_path = out_root / "video" / f"{prefix}.mp4"
    frames_dir = out_root / "frames" / prefix
    npz_path = out_root / "data" / f"{prefix}.npz"
    actions_path = out_root / "data" / f"{prefix}_actions.csv"
    summary_path = out_root / "data" / f"{prefix}_summary.json"

    write_video(frames, video_path, FPS)
    written = write_frames(frames, frames_dir, args.frame_limit)
    write_actions_csv(actions, actions_path)
    np.savez_compressed(npz_path, frames=frames.astype(np.uint8),
                        actions=actions.astype(np.float32),
                        sim_state=sim_state.astype(np.float32),
                        meta=np.array(meta, dtype=object))
    save_summary_json(summary_path, frames, actions, sim_state, meta)

    print(f"[{prefix}] video:   {video_path}")
    print(f"[{prefix}] frames:  {frames_dir} ({written} files)")
    print(f"[{prefix}] data:    {npz_path}")
    print(f"[{prefix}] actions: {actions_path}")
    print(f"[{prefix}] summary: {summary_path}")
    print(f"Done. Sample bundle at: {out_root}")


if __name__ == "__main__":
    main()
