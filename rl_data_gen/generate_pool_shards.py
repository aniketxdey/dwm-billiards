#!/usr/bin/env python3
"""
Neural Pool - High-Performance Data Generation Script
======================================================

Generates large-scale billiard gameplay dataset for AI training.

Output Format:
- Compressed NumPy archives (.npz) with 100 episodes per shard.
- Each shard contains: frames, actions, sim_state, lengths, episode_meta.

Hardware Target: Large multi-core CPU machine (EC2 C-family recommended)
Storage: Amazon S3 (s3://bucket/prefix)

Usage:
    # Dry run (10 episodes locally)
    python generate_pool_shards.py --dry-run --episodes 10
    
    # Full production run (100,000 episodes)
    python generate_pool_shards.py --episodes 100000 --s3-prefix s3://my-bucket/dataraw_v3
    
    # Custom settings
    python generate_pool_shards.py --episodes 1000 --workers 8 --shard-size 100
"""

import os
import sys
import json
import argparse
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from record_billiard import BilliardWorld, PoolBot

# =============================================================================
# CONFIGURATION
# =============================================================================

# Episode settings
EPISODE_FRAMES = 600  # 20 seconds at 30 FPS
MAX_BALLS = 16        # Pad sim_state to this many balls
FPS = 30

# Resolution (keeping original for realism)
FRAME_WIDTH = config.SCREEN_WIDTH   # 128
FRAME_HEIGHT = config.SCREEN_HEIGHT  # 72

# Rendering scale (1 = native resolution for training)
RENDER_SCALE = 1

# =============================================================================
# ACTIONS FORMAT DOCUMENTATION
# =============================================================================
"""
ACTIONS ARRAY: [T, 3] float32

Each frame records the control signal applied by the bot.

Column 0: force_x (float32)
    - The X component of the flick/shot velocity
    - Range: approximately -200 to +200
    - 0.0 when no shot is happening
    
Column 1: force_y (float32)  
    - The Y component of the flick/shot velocity
    - Range: approximately -200 to +200
    - 0.0 when no shot is happening
    
Column 2: trigger (float32)
    - 0.0 = No action this frame
    - 1.0 = SHOOT! (ball is being released with force)
    
Timing:
    - Most frames will be [0, 0, 0] (bot moving, watching, or picking)
    - On the EXACT frame the bot releases the ball, we record:
      [force_x, force_y, 1.0]
    - The force values are the velocity imparted to the ball
    
Example sequence:
    Frame 100: [0, 0, 0]      # Bot approaching ball
    Frame 101: [0, 0, 0]      # Bot still approaching
    Frame 102: [0, 0, 0]      # Bot grabbed ball, dragging
    Frame 103: [-85.2, 42.1, 1.0]  # SHOOT! Ball released with this velocity
    Frame 104: [0, 0, 0]      # Watching ball roll
    Frame 105: [0, 0, 0]      # Still watching
    ...
"""

# =============================================================================
# EPISODE GENERATION
# =============================================================================

def _layout_contact_proxy(sim_state_t0: np.ndarray) -> Dict[str, float]:
    """Cheap proxy for collision likelihood from the initial arrangement."""
    coords = sim_state_t0[:, :2]
    valid = ~np.all(coords == 0.0, axis=1)
    pts = coords[valid]
    if len(pts) < 2:
        return {
            "initial_min_pair_distance": 0.0,
            "initial_mean_nearest_distance": 0.0,
            "initial_dense_pairs_r7": 0,
            "initial_dense_pairs_r10": 0,
        }

    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    np.fill_diagonal(dists, np.inf)
    tri = np.triu_indices(len(pts), k=1)
    pair_dists = dists[tri]
    nn = np.min(dists, axis=1)
    finite_nn = nn[np.isfinite(nn)]
    return {
        "initial_min_pair_distance": float(pair_dists.min()) if pair_dists.size else 0.0,
        "initial_mean_nearest_distance": float(finite_nn.mean()) if finite_nn.size else 0.0,
        "initial_dense_pairs_r7": int(np.sum(pair_dists < 7.0)),
        "initial_dense_pairs_r10": int(np.sum(pair_dists < 10.0)),
    }


def _action_energy_proxy(actions: np.ndarray) -> Dict[str, float]:
    triggers = actions[:, 2] > 0.5
    if not np.any(triggers):
        return {
            "shots_fired_proxy": 0,
            "shot_power_mean_proxy": 0.0,
            "shot_power_max_proxy": 0.0,
            "high_power_shots_proxy": 0,
        }
    shot_vecs = actions[triggers, :2]
    shot_powers = np.linalg.norm(shot_vecs, axis=1)
    return {
        "shots_fired_proxy": int(len(shot_powers)),
        "shot_power_mean_proxy": float(np.mean(shot_powers)),
        "shot_power_max_proxy": float(np.max(shot_powers)),
        "high_power_shots_proxy": int(np.sum(shot_powers >= 140.0)),
    }


def generate_episode(episode_id: int, num_frames: int) -> Dict[str, np.ndarray]:
    """
    Generate a single episode of billiard gameplay.
    
    Returns:
        dict with keys:
            - 'frames': [T, H, W, 3] uint8
            - 'actions': [T, 3] float32
            - 'sim_state': [T, N, 4] float32
            - 'episode_meta': dict with bot personality, score, etc.
    """
    # Create world and bot
    world = BilliardWorld()
    bot = PoolBot(world)
    
    dt = 1.0 / FPS
    
    # Storage for this episode
    frames_list = []
    actions_list = []
    sim_state_list = []
    
    # Track shots for metadata
    shot_count = 0
    last_bot_state = bot.state
    pending_action = None  # Store action to record on flick frame
    
    for frame_idx in range(num_frames):
        # Check if bot is about to shoot (transition from shoot to watch)
        prev_state = bot.state
        prev_grabbed = world.grabbed
        
        # Update bot
        bot.update(dt)
        
        # Detect shot: bot was in "shoot" state with grabbed ball, now released
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        if prev_state == "shoot" and bot.state == "watch" and prev_grabbed is not None:
            # Shot happened! Record the flick velocity
            # The flick was: (-shoot_dir[0] * shot_power, -shoot_dir[1] * shot_power)
            force_x = -bot.shoot_dir[0] * bot.shot_power
            force_y = -bot.shoot_dir[1] * bot.shot_power
            action = np.array([force_x, force_y, 1.0], dtype=np.float32)
            shot_count += 1
        
        # Step physics
        world.step(dt)
        
        # Render frame (no cursor, just the table and balls)
        frame = world.render(RENDER_SCALE)
        # Convert BGR to RGB
        frame = frame[:, :, ::-1].copy()
        
        # Record sim_state: [N, 4] for all balls (pos_x, pos_y, vel_x, vel_y)
        sim_state = np.zeros((MAX_BALLS, 4), dtype=np.float32)
        for i, (body, radius, color) in enumerate(world.balls):
            if i >= MAX_BALLS:
                break
            sim_state[i, 0] = body.position.x
            sim_state[i, 1] = body.position.y
            sim_state[i, 2] = body.velocity.x
            sim_state[i, 3] = body.velocity.y
        
        frames_list.append(frame)
        actions_list.append(action)
        sim_state_list.append(sim_state)
    
    # Stack arrays
    frames = np.stack(frames_list, axis=0).astype(np.uint8)      # [T, H, W, 3]
    actions = np.stack(actions_list, axis=0).astype(np.float32)  # [T, 3]
    sim_state = np.stack(sim_state_list, axis=0).astype(np.float32)  # [T, N, 4]
    layout_proxy = _layout_contact_proxy(sim_state[0])
    action_proxy = _action_energy_proxy(actions)
    
    # Episode metadata
    episode_meta = {
        'episode_id': episode_id,
        'num_frames': len(frames_list),
        'shots_fired': shot_count,
        'balls_pocketed': world.pocketed,
        'balls_remaining': len(world.balls),
        'bot_personality': {
            'style': getattr(bot, 'style', 'unknown'),
            'w_random': float(bot.w_random),
            'w_ball': float(bot.w_ball),
            'w_pocket': float(bot.w_pocket),
            'w_bank': float(bot.w_bank),
            'w_tap': float(bot.w_tap),
            'speed_mult': float(bot.speed_mult),
            'accuracy': float(bot.accuracy),
            'power_mult': float(getattr(bot, 'power_mult', 1.0)),
            'watch_mult': float(getattr(bot, 'watch_mult', 1.0)),
        }
    }
    episode_meta['world_profile'] = {
        'spawn_mode': getattr(world, 'spawn_mode', 'uniform'),
        'event_profile': getattr(world, 'event_profile', 'mixed_random'),
        'collision_bias_strength': float(getattr(world, 'collision_bias_strength', 0.0)),
        'table_profile': getattr(world, 'table_profile', 'standard'),
        'damping': float(getattr(world, 'damping', config.DAMPING)),
        'restitution': float(getattr(world, 'restitution', config.RESTITUTION)),
        'ball_friction': float(getattr(world, 'ball_friction', config.FRICTION)),
    }
    episode_meta['collision_proxies'] = {
        **layout_proxy,
        **action_proxy,
        'collision_focus_target': bool(getattr(world, 'event_profile', '') == 'collision_heavy'),
        'bot_contact_oriented': bool(getattr(bot, 'style', '') in ('breaker', 'collider', 'chaos', 'banker')),
    }
    
    return {
        'frames': frames,
        'actions': actions,
        'sim_state': sim_state,
        'episode_meta': episode_meta,
    }


def generate_episode_wrapper(args: Tuple[int, int, int]) -> Dict[str, Any]:
    """Wrapper for multiprocessing (unpacks arguments)."""
    episode_id, seed_offset, num_frames = args
    # Set unique random seed for reproducibility
    np.random.seed(episode_id + seed_offset)
    import random
    random.seed(episode_id + seed_offset)
    
    return generate_episode(episode_id, num_frames)


# =============================================================================
# SHARD CREATION
# =============================================================================

def create_shard(episodes: List[Dict], shard_id: int, output_dir: str) -> str:
    """
    Combine multiple episodes into a single shard file.
    
    Variable-length handling:
        - Pad all episodes to max T in this shard
        - Store 'lengths' array to know actual length of each episode
    
    Returns:
        Path to the created shard file
    """
    num_episodes = len(episodes)
    
    # Find max length in this shard
    max_T = max(ep['frames'].shape[0] for ep in episodes)
    
    # Get dimensions
    H, W = episodes[0]['frames'].shape[1:3]
    N = episodes[0]['sim_state'].shape[1]
    
    # Allocate padded arrays
    frames_all = np.zeros((num_episodes, max_T, H, W, 3), dtype=np.uint8)
    actions_all = np.zeros((num_episodes, max_T, 3), dtype=np.float32)
    sim_state_all = np.zeros((num_episodes, max_T, N, 4), dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)
    
    # Copy episode data (with padding)
    episode_metas = []
    for i, ep in enumerate(episodes):
        T = ep['frames'].shape[0]
        lengths[i] = T
        frames_all[i, :T] = ep['frames']
        actions_all[i, :T] = ep['actions']
        sim_state_all[i, :T] = ep['sim_state']
        episode_metas.append(ep['episode_meta'])
    
    # Save shard
    shard_filename = f"shard_{shard_id:05d}.npz"
    shard_path = os.path.join(output_dir, shard_filename)
    
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
# GCS UPLOAD
# =============================================================================

def upload_to_s3(
    local_path: str,
    s3_uri: str,
    delete_local: bool = True,
    aws_profile: Optional[str] = None,
) -> bool:
    """
    Upload file to Amazon S3 and optionally delete local copy.
    
    Args:
        local_path: Path to local file
        s3_uri: Full S3 URI (e.g., s3://bucket-name/path/to/file)
        delete_local: Whether to delete local file after upload
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ['aws']
        if aws_profile:
            cmd.extend(['--profile', aws_profile])
        cmd.extend(['s3', 'cp', local_path, s3_uri])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
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
    """Normalize s3 prefix by removing trailing slash."""
    return prefix.rstrip("/")


# =============================================================================
# METADATA
# =============================================================================

def create_metadata(output_dir: str, total_episodes: int, shard_size: int) -> str:
    """Create metadata.json file."""
    metadata = {
        'dataset_name': 'Neural Pool Dataset (Diverse v2)',
        'version': '2.0',
        'created': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        
        # Episode info
        'total_episodes': total_episodes,
        'episodes_per_shard': shard_size,
        'num_shards': (total_episodes + shard_size - 1) // shard_size,
        
        # Frame info
        'resolution': [FRAME_HEIGHT, FRAME_WIDTH],
        'fps': FPS,
        'frames_per_episode': EPISODE_FRAMES,
        'color_format': 'RGB',
        
        # Physics parameters
        'physics': {
            'gravity': config.GRAVITY,
            'friction': config.FRICTION,
            'restitution': config.RESTITUTION,
            'damping': config.DAMPING,
            'ball_radius': config.BALL_RADIUS,
            'ball_mass': config.BALL_MASS,
        },
        
        # Data structure
        'data_format': {
            'frames': {
                'shape': '[num_episodes, T, H, W, 3]',
                'dtype': 'uint8',
                'description': 'RGB frames of table (no cursor)',
            },
            'actions': {
                'shape': '[num_episodes, T, 3]',
                'dtype': 'float32',
                'columns': ['force_x', 'force_y', 'trigger'],
                'description': 'Control signals. trigger=1.0 on shot frame.',
            },
            'sim_state': {
                'shape': '[num_episodes, T, N, 4]',
                'dtype': 'float32',
                'columns': ['pos_x', 'pos_y', 'vel_x', 'vel_y'],
                'description': 'Physics ground truth for all balls.',
            },
            'lengths': {
                'shape': '[num_episodes]',
                'dtype': 'int32',
                'description': 'Actual length of each episode (for padding).',
            },
        },
        
        # Ball count
        'max_balls': MAX_BALLS,
        'ball_count_range': [config.BALL_COUNT_MIN, config.BALL_COUNT_MAX],
        'generation_notes': {
            'policy': 'collision_heavy_v3',
            'description': 'Collision-biased layouts and bot strategies, mixed with regular gameplay for robustness.',
        },
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

def main():
    global EPISODE_FRAMES
    
    parser = argparse.ArgumentParser(description='Generate Neural Pool dataset')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Total number of episodes to generate')
    parser.add_argument('--shard-size', type=int, default=100,
                        help='Episodes per shard file')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--s3-prefix', type=str, default=None,
                        help='S3 prefix (e.g., s3://my-bucket/dataraw_v2)')
    parser.add_argument('--bucket', type=str, default=None,
                        help='Deprecated alias for --s3-prefix')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Local output directory')
    parser.add_argument('--aws-profile', type=str, default=None,
                        help='Optional AWS CLI profile for S3 upload')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run mode (no S3 upload, keep local files)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed offset for reproducibility')
    parser.add_argument('--frames', type=int, default=EPISODE_FRAMES,
                        help='Number of frames per episode')
    parser.add_argument('--start-episode', type=int, default=0,
                        help='Global starting episode index (for resume runs)')
    parser.add_argument('--start-shard-id', type=int, default=0,
                        help='Starting shard id to write (for resume runs)')
    parser.add_argument('--no-upload-metadata', action='store_true',
                        help='Skip metadata upload even when S3 prefix is set')
    
    args = parser.parse_args()
    target_prefix = args.s3_prefix or args.bucket
    if target_prefix is not None:
        target_prefix = normalize_s3_prefix(target_prefix)
        if not target_prefix.startswith("s3://"):
            raise ValueError(
                f"Expected S3 prefix starting with s3://, got: {target_prefix}"
            )
    
    EPISODE_FRAMES = args.frames
    
    # Auto-detect workers
    if args.workers is None:
        args.workers = max(1, cpu_count() - 2)
    
    print("=" * 60)
    print("NEURAL POOL - DATA GENERATION")
    print("=" * 60)
    print(f"Episodes:        {args.episodes}")
    print(f"Shard size:      {args.shard_size}")
    print(f"Workers:         {args.workers}")
    print(f"Resolution:      {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"FPS:             {FPS}")
    print(f"Frames/episode:  {EPISODE_FRAMES}")
    print(f"Max balls:       {MAX_BALLS}")
    print(f"Start episode:   {args.start_episode}")
    print(f"Start shard id:  {args.start_shard_id}")
    print(f"S3 prefix:       {target_prefix or 'None (local only)'}")
    print(f"Dry run:         {args.dry_run}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use /dev/shm for temp files only if uploading to S3 (RAM disk speeds up upload)
    # Otherwise write directly to output_dir
    if os.path.exists('/dev/shm') and target_prefix and not args.dry_run:
        temp_dir = tempfile.mkdtemp(prefix='neural_pool_', dir='/dev/shm')
    else:
        temp_dir = args.output_dir
    
    print(f"Temp directory:  {temp_dir}")
    
    # Create metadata
    metadata_path = create_metadata(args.output_dir, args.episodes, args.shard_size)
    print(f"Created metadata: {metadata_path}")
    
    # Calculate shards
    num_shards = (args.episodes + args.shard_size - 1) // args.shard_size
    print(f"Total shards:    {num_shards}")
    print()
    
    # Upload metadata to S3 first (skip on resume unless explicitly desired)
    should_upload_metadata = (
        target_prefix
        and not args.dry_run
        and not args.no_upload_metadata
        and args.start_episode == 0
        and args.start_shard_id == 0
    )
    if should_upload_metadata:
        metadata_uri = f"{target_prefix}/meta/metadata.json"
        print(f"Uploading metadata to S3: {metadata_uri}")
        upload_to_s3(
            metadata_path,
            metadata_uri,
            delete_local=False,
            aws_profile=args.aws_profile,
        )
    elif target_prefix and not args.dry_run:
        print("Skipping metadata upload (resume or --no-upload-metadata).")
    
    # Generate episodes in batches
    start_time = time.time()
    total_generated = 0
    
    for local_shard_idx in range(num_shards):
        shard_start_time = time.time()
        shard_id = args.start_shard_id + local_shard_idx
        
        # Calculate episode range for this shard
        ep_start = args.start_episode + local_shard_idx * args.shard_size
        ep_end = min(ep_start + args.shard_size, args.start_episode + args.episodes)
        num_in_shard = ep_end - ep_start
        
        print(f"[Shard {local_shard_idx + 1}/{num_shards}] Generating episodes {ep_start}-{ep_end-1}...")
        
        # Prepare arguments for workers
        worker_args = [(ep_id, args.seed, args.frames) for ep_id in range(ep_start, ep_end)]
        
        # Generate episodes in parallel
        if args.workers > 1:
            with Pool(args.workers) as pool:
                episodes = pool.map(generate_episode_wrapper, worker_args)
        else:
            episodes = [generate_episode_wrapper(arg) for arg in worker_args]
        
        # Create shard
        shard_path = create_shard(episodes, shard_id, temp_dir)
        shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        
        shard_elapsed = time.time() - shard_start_time
        total_generated += num_in_shard
        
        print(f"  Created: {os.path.basename(shard_path)} ({shard_size_mb:.1f} MB) in {shard_elapsed:.1f}s")
        
        # Upload to S3 if configured
        if target_prefix and not args.dry_run:
            shard_uri = f"{target_prefix}/raw/shards/{os.path.basename(shard_path)}"
            print(f"  Uploading to {shard_uri}...")
            success = upload_to_s3(
                shard_path, 
                shard_uri,
                delete_local=True,
                aws_profile=args.aws_profile,
            )
            if success:
                print(f"  Uploaded and deleted local file.")
            else:
                print(f"  WARNING: Upload failed, keeping local file.")
        
        # Progress
        elapsed = time.time() - start_time
        eps_per_sec = total_generated / elapsed
        remaining = (args.episodes - total_generated) / eps_per_sec if eps_per_sec > 0 else 0
        
        print(f"  Progress: {total_generated}/{args.episodes} ({100*total_generated/args.episodes:.1f}%)")
        print(f"  Speed: {eps_per_sec:.2f} ep/s | ETA: {remaining/60:.1f} min")
        print()
    
    # Cleanup temp directory
    if temp_dir != args.output_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    total_elapsed = time.time() - start_time
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total episodes:  {total_generated}")
    print(f"Total shards:    {num_shards}")
    print(f"Total time:      {total_elapsed/60:.1f} minutes")
    print(f"Average speed:   {total_generated/total_elapsed:.2f} episodes/second")
    print("=" * 60)


if __name__ == "__main__":
    main()
