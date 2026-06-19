# Neural Golf - Minimal Data Generation

Top-down 2D mini-golf data generator. It is a drop-in sibling of
`data_generation_package/` (billiards): same resolution, same `.npz` shard
schema, and the same 3-dim action vector, so the existing VAE / world-model /
inference pipeline consumes it **without any changes**.

## Game
- One white golf ball on a striped felt course with a cup (hole + flag).
- Border cushions + a few static bumpers ("rocks") for richer trajectories.
- Strong damping makes the ball roll and settle like a putt.
- An autoplay bot aims at the cup and putts (with randomized accuracy/power).
- When the ball is captured by the cup it re-tees with a new cup; `cups_sunk`
  is tracked as the score.

## Files
- `generate_golf_shards.py`: main shard generator (`frames`, `actions`, `sim_state`, `lengths`, `episode_meta`)
- `record_golf.py`: golf world + autoplay bot
- `config.py`: physics and resolution configuration
- `run_sample_bundle.py`: one-command sample export (video + frames + data)
- `requirements.txt`: minimal dependencies

## Install
```bash
pip install -r requirements.txt
```

## Generate Shards
```bash
python generate_golf_shards.py --episodes 100 --shard-size 10 --output-dir ./data --dry-run
```

Directly to S3 (same convention as billiards):
```bash
python generate_golf_shards.py \
  --episodes 1000 --shard-size 100 --workers 32 \
  --s3-prefix s3://<your-bucket>/dataraw_golf --aws-profile <profile>
```

## Create Sample Bundle
```bash
python run_sample_bundle.py --output-dir ./sample_bundle
```

## Shard Format
Each shard (`.npz`) contains:
- `frames`: `(N, T, 72, 128, 3)` `uint8`
- `actions`: `(N, T, 3)` `float32` as `[force_x, force_y, trigger]` (trigger=1.0 on the putt frame)
- `sim_state`: `(N, T, 16, 4)` `float32` as `[pos_x, pos_y, vel_x, vel_y]`
  - slot 0 = ball, slot 1 = hole (static), slots 2+ = bumpers (static)
- `lengths`: `(N,)` `int32`
- `episode_meta`: `(N,)` `object`
