# Neural Pool - Collision-Heavy Data Generation (v3)

Minimal package focused on generating pool simulation data for training.
This v3 variant biases generation toward collision-heavy layouts and contact-rich shots
while keeping the shard schema compatible with existing VAE/world-model pipelines.

## Files
- `generate_pool_shards.py`: main shard generator (`frames`, `actions`, `sim_state`, `lengths`, `episode_meta`)
- `record_billiard.py`: pool world + bot behavior
- `config.py`: physics and resolution configuration
- `run_sample_bundle.py`: one-command sample export (video + frames + combined episode data)
- `requirements.txt`: minimal dependencies

## Install
```bash
pip install -r requirements.txt
```

## Lambda Env Setup (AWS + W&B)
```bash
cd ..
cp .env.lambda.example .env.lambda
# edit .env.lambda (AWS_PROFILE or keys, plus W&B key)
bash scripts/lambda_prepare_env.sh
```

## Generate Shards
```bash
python generate_pool_shards.py --episodes 100 --shard-size 10 --output-dir ./data
```

Generate directly to S3:
```bash
python generate_pool_shards.py \
  --episodes 1000 \
  --shard-size 100 \
  --workers 32 \
  --s3-prefix s3://videogen-pool-v2-237586137680/dataraw_v3 \
  --aws-profile codex-admin
```

## Create Sample Bundle
This creates:
- `video/` mp4 files
- `frames/` png frames
- `data/` combined `.npz`, `actions.csv`, and summary `.json`

```bash
python run_sample_bundle.py --output-dir ./sample_bundle
```

Local-only sample (skip S3 download):
```bash
python run_sample_bundle.py --output-dir ./sample_bundle --skip-s3
```

Custom S3 sample source:
```bash
python run_sample_bundle.py \
  --s3-uri s3://videogen-dataset-dp/dataraw/raw/shards/shard_00000.npz \
  --aws-profile codex-admin \
  --s3-episode-index 0
```

## Shard Format
Each shard (`.npz`) contains:
- `frames`: `(N, T, 72, 128, 3)` `uint8`
- `actions`: `(N, T, 3)` `float32` as `[force_x, force_y, trigger]`
- `sim_state`: `(N, T, 16, 4)` `float32` as `[pos_x, pos_y, vel_x, vel_y]`
- `lengths`: `(N,)` `int32`
- `episode_meta`: `(N,)` `object`
  - includes `world_profile.event_profile` and `collision_proxies` tags in v3
