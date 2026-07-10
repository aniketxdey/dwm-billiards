#!/usr/bin/env python3
"""
Neural Golf - Data Generation Script
====================================

Generates 2D mini-golf gameplay episodes in the SAME shard schema as the
billiards package, so the existing VAE / world-model / inference pipeline
consumes it with no changes.

Shard (.npz) contents:
- frames:      [N, T, 72, 128, 3] uint8
- actions:     [N, T, 3] float32  -> [force_x, force_y, trigger]
- sim_state:   [N, T, 16, 4] float32 -> [pos_x, pos_y, vel_x, vel_y]
- lengths:     [N] int32
- episode_meta:[N] object

Usage:
    python generate_golf_shards.py --episodes 10 --shard-size 5 --output-dir ./data --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from record_golf import GolfBot, GolfWorld

# =============================================================================
# CONFIGURATION
# =============================================================================
EPISODE_FRAMES = 600
MAX_BALLS = 16  # sim_state slots: ball + hole + bumpers, padded
FPS = 30
FRAME_WIDTH = config.SCREEN_WIDTH
FRAME_HEIGHT = config.SCREEN_HEIGHT
RENDER_SCALE = 1


# =============================================================================
# EPISODE GENERATION
# =============================================================================
def generate_episode(episode_id: int, num_frames: int) -> Dict[str, np.ndarray]:
    """Generate one mini-golf episode.

    actions[t] = [force_x, force_y, 1.0] on the frame the bot putts, else zeros.
    sim_state slot 0 = ball, slot 1 = hole (static), slots 2.. = bumpers (static).
    """
    world = GolfWorld()
    bot = GolfBot(world)
    dt = 1.0 / FPS

    frames_list = []
    actions_list = []
    sim_state_list = []
    shot_count = 0

    for _frame_idx in range(num_frames):
        bot.update(dt)

        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if bot.did_shoot:
            action = np.array([bot.shot_force[0], bot.shot_force[1], 1.0], dtype=np.float32)
            shot_count += 1

        world.step(dt)

        frame = world.render(RENDER_SCALE)[:, :, ::-1].copy()  # BGR -> RGB

        sim_state = np.zeros((MAX_BALLS, 4), dtype=np.float32)
        if world.ball is not None:
            sim_state[0, 0] = world.ball.position.x
            sim_state[0, 1] = world.ball.position.y
            sim_state[0, 2] = world.ball.velocity.x
            sim_state[0, 3] = world.ball.velocity.y
        sim_state[1, 0] = world.hole[0]
        sim_state[1, 1] = world.hole[1]
        for i, (bx, by, _r) in enumerate(world.bumpers):
            slot = 2 + i
            if slot >= MAX_BALLS:
                break
            sim_state[slot, 0] = bx
            sim_state[slot, 1] = by

        frames_list.append(frame)
        actions_list.append(action)
        sim_state_list.append(sim_state)

    frames = np.stack(frames_list, axis=0).astype(np.uint8)
    actions = np.stack(actions_list, axis=0).astype(np.float32)
    sim_state = np.stack(sim_state_list, axis=0).astype(np.float32)

    episode_meta = {
        "episode_id": episode_id,
        "num_frames": len(frames_list),
        "shots_fired": shot_count,
        "cups_sunk": world.sunk,
        "num_bumpers": len(world.bumpers),
        "bot_personality": {
            "accuracy": float(bot.accuracy),
            "power_mult": float(bot.power_mult),
            "miss_prob": float(bot.miss_prob),
        },
    }

    return {
        "frames": frames,
        "actions": actions,
        "sim_state": sim_state,
        "episode_meta": episode_meta,
    }


def generate_episode_wrapper(args: Tuple[int, int, int]) -> Dict[str, Any]:
    episode_id, seed_offset, num_frames = args
    np.random.seed(episode_id + seed_offset)
    import random
    random.seed(episode_id + seed_offset)
    return generate_episode(episode_id, num_frames)


# =============================================================================
# SHARD CREATION
# =============================================================================
def create_shard(episodes: List[Dict], shard_id: int, output_dir: str) -> str:
    num_episodes = len(episodes)
    max_T = max(ep["frames"].shape[0] for ep in episodes)
    H, W = episodes[0]["frames"].shape[1:3]
    N = episodes[0]["sim_state"].shape[1]

    frames_all = np.zeros((num_episodes, max_T, H, W, 3), dtype=np.uint8)
    actions_all = np.zeros((num_episodes, max_T, 3), dtype=np.float32)
    sim_state_all = np.zeros((num_episodes, max_T, N, 4), dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)

    episode_metas = []
    for i, ep in enumerate(episodes):
        T = ep["frames"].shape[0]
        lengths[i] = T
        frames_all[i, :T] = ep["frames"]
        actions_all[i, :T] = ep["actions"]
        sim_state_all[i, :T] = ep["sim_state"]
        episode_metas.append(ep["episode_meta"])

    shard_path = os.path.join(output_dir, f"shard_{shard_id:05d}.npz")
    np.savez_compressed(
        shard_path,
        frames=frames_all,
        actions=actions_all,
        sim_state=sim_state_all,
        lengths=lengths,
        episode_meta=np.array(episode_metas, dtype=object),
    )
    return shard_path


# =============================================================================
# S3 UPLOAD
# =============================================================================
def upload_to_s3(local_path: str, s3_uri: str, delete_local: bool = True,
                 aws_profile: Optional[str] = None) -> bool:
    try:
        cmd = ["aws"]
        if aws_profile:
            cmd.extend(["--profile", aws_profile])
        cmd.extend(["s3", "cp", local_path, s3_uri])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"ERROR uploading {local_path}: {result.stderr}")
            return False
        if delete_local:
            os.remove(local_path)
        return True
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT uploading {local_path}")
        return False
    except FileNotFoundError:
        print("ERROR: aws CLI not found.")
        return False


def normalize_s3_prefix(prefix: str) -> str:
    return prefix.rstrip("/")


# =============================================================================
# METADATA
# =============================================================================
def create_metadata(output_dir: str, total_episodes: int, shard_size: int) -> str:
    import json
    metadata = {
        "dataset_name": "Neural Golf Dataset",
        "version": "1.0",
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_episodes": total_episodes,
        "episodes_per_shard": shard_size,
        "num_shards": (total_episodes + shard_size - 1) // shard_size,
        "resolution": [FRAME_HEIGHT, FRAME_WIDTH],
        "fps": FPS,
        "frames_per_episode": EPISODE_FRAMES,
        "color_format": "RGB",
        "physics": {
            "gravity": config.GRAVITY,
            "friction": config.FRICTION,
            "restitution": config.RESTITUTION,
            "damping": config.DAMPING,
            "ball_radius": config.BALL_RADIUS,
            "ball_mass": config.BALL_MASS,
            "hole_radius": config.HOLE_RADIUS,
        },
        "data_format": {
            "frames": {"shape": "[num_episodes, T, H, W, 3]", "dtype": "uint8"},
            "actions": {"shape": "[num_episodes, T, 3]", "dtype": "float32",
                        "columns": ["force_x", "force_y", "trigger"]},
            "sim_state": {"shape": "[num_episodes, T, N, 4]", "dtype": "float32",
                          "columns": ["pos_x", "pos_y", "vel_x", "vel_y"],
                          "slots": "0=ball, 1=hole, 2+=bumpers"},
            "lengths": {"shape": "[num_episodes]", "dtype": "int32"},
        },
        "max_balls": MAX_BALLS,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    global EPISODE_FRAMES

    parser = argparse.ArgumentParser(description="Generate Neural Golf dataset")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--s3-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--aws-profile", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=EPISODE_FRAMES)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--start-shard-id", type=int, default=0)
    args = parser.parse_args()

    target_prefix = args.s3_prefix
    if target_prefix is not None:
        target_prefix = normalize_s3_prefix(target_prefix)
        if not target_prefix.startswith("s3://"):
            raise ValueError(f"Expected S3 prefix starting with s3://, got: {target_prefix}")

    EPISODE_FRAMES = args.frames
    if args.workers is None:
        args.workers = max(1, cpu_count() - 2)

    print("=" * 60)
    print("NEURAL GOLF - DATA GENERATION")
    print("=" * 60)
    print(f"Episodes:        {args.episodes}")
    print(f"Shard size:      {args.shard_size}")
    print(f"Workers:         {args.workers}")
    print(f"Resolution:      {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Frames/episode:  {EPISODE_FRAMES}")
    print(f"S3 prefix:       {target_prefix or 'None (local only)'}")
    print(f"Dry run:         {args.dry_run}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = args.output_dir

    metadata_path = create_metadata(args.output_dir, args.episodes, args.shard_size)
    print(f"Created metadata: {metadata_path}")

    num_shards = (args.episodes + args.shard_size - 1) // args.shard_size
    print(f"Total shards:    {num_shards}\n")

    start_time = time.time()
    total_generated = 0

    for local_shard_idx in range(num_shards):
        shard_start_time = time.time()
        shard_id = args.start_shard_id + local_shard_idx
        ep_start = args.start_episode + local_shard_idx * args.shard_size
        ep_end = min(ep_start + args.shard_size, args.start_episode + args.episodes)

        print(f"[Shard {local_shard_idx + 1}/{num_shards}] Generating episodes {ep_start}-{ep_end-1}...")
        worker_args = [(ep_id, args.seed, args.frames) for ep_id in range(ep_start, ep_end)]

        if args.workers > 1:
            with Pool(args.workers) as pool:
                episodes = pool.map(generate_episode_wrapper, worker_args)
        else:
            episodes = [generate_episode_wrapper(arg) for arg in worker_args]

        shard_path = create_shard(episodes, shard_id, temp_dir)
        shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        shard_elapsed = time.time() - shard_start_time
        total_generated += (ep_end - ep_start)

        print(f"  Created: {os.path.basename(shard_path)} ({shard_size_mb:.1f} MB) in {shard_elapsed:.1f}s")

        if target_prefix and not args.dry_run:
            shard_uri = f"{target_prefix}/raw/shards/{os.path.basename(shard_path)}"
            print(f"  Uploading to {shard_uri}...")
            ok = upload_to_s3(shard_path, shard_uri, delete_local=True, aws_profile=args.aws_profile)
            print("  Uploaded." if ok else "  WARNING: upload failed, keeping local file.")

        elapsed = time.time() - start_time
        eps_per_sec = total_generated / elapsed if elapsed > 0 else 0
        print(f"  Progress: {total_generated}/{args.episodes} | {eps_per_sec:.2f} ep/s\n")

    if temp_dir != args.output_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print("GENERATION COMPLETE")
    print(f"Total episodes:  {total_generated}")
    print(f"Total shards:    {num_shards}")
    print(f"Total time:      {total_elapsed/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
