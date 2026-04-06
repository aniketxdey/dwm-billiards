# Implementation Map

## Core code paths
- `src/vae_training/prepare_frame_cache.py`
  - Builds local random frame cache from NPZ shards.
- `src/vae_training/data.py`
  - Loads cached frame tensor dataset from `.npy` memmap.
- `src/vae_training/model.py`
  - Conv VAE encoder/decoder with latent bottleneck.
- `src/vae_training/losses.py`
  - L1 + optional LPIPS + KL with beta warmup.
- `src/vae_training/train.py`
  - Main training loop, checkpoints, previews, resume support.
- `src/vae_training/preview.py`
  - Saves reconstruction grid and side-by-side preview video.

## Runtime scripts
- `scripts/stage_shards_from_s3.sh`
  - Syncs full dataset to local SSD.
- `scripts/build_frame_cache_1m.sh`
  - Builds 1M-frame cache (`MAX_SHARDS` tunable).
- `scripts/preflight_1xa100.sh`
  - Validates GPU, torch/cuda, disk, config, and cache path.
- `scripts/new_run_id.sh`
  - Generates consistent run ids for traceable launches.
- `scripts/sync_repo_to_lambda.sh`
  - Rsyncs local repo to Lambda instance with key-based SSH.
- `scripts/run_train_1xa100.sh`
  - Launches training with explicit `RUN_ID`, optional notes, optional resume checkpoint.
- `scripts/sync_run_to_s3.sh`
  - Uploads local run folder to S3.

## Artifact outputs (local)
- `runs/<run_id>/checkpoints/`
- `runs/<run_id>/previews/`
- `runs/<run_id>/metrics/`
- `runs/<run_id>/config/resolved_config.json`
- `runs/<run_id>/manifest.json`
- `runs/<run_id>/summary.json`
- `runs/run_registry.jsonl`
