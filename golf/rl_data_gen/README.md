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

## Full Pipeline (scaled, single GPU)
A turnkey driver runs data-gen -> VAE -> latent export -> DiT baseline ->
Diffusion Forcing -> rollout preview using the configs in `local_run_golf/`:

```bash
# optional: better small-ball VAE detail
pip install lpips
# launch everything (sized for a single RTX 4090, ~$15-25 of GPU time)
bash local_run_golf/run_golf_pipeline.sh
# resume from a later stage (1=data .. 6=preview)
START_STAGE=4 bash local_run_golf/run_golf_pipeline.sh
```

Configs: `vae_golf.yaml` (Phase 1 accuracy gate, LPIPS on), `latent_export_golf.yaml`,
`dit_golf.yaml` (baseline), `dit_golf_df.yaml` (Diffusion Forcing, resumes baseline),
`preview_golf.yaml`. The smaller `*_smoke.yaml` configs are the CPU end-to-end test.

## Live Play (Phase 4)
Once the checkpoints exist, drive the model interactively (drag from the ball to
aim/power a putt; the world model imagines the roll):

```bash
bash world_model_inference/scripts/run_live_play_golf.sh
# then open http://127.0.0.1:7863
```

The launcher points the inference server at the golf DiT + VAE checkpoints and
sets `WM_INF_VAE_BASE_CHANNELS=48` (golf's VAE is narrower than the pool model's
64). Override any path inline, e.g. to preview before Diffusion Forcing finishes:

```bash
DIT_CKPT=./local_run_golf/wm_runs/dit_golf_base_run01/checkpoints/ckpt_080000000.pt \
  bash world_model_inference/scripts/run_live_play_golf.sh
```

## Shard Format
Each shard (`.npz`) contains:
- `frames`: `(N, T, 72, 128, 3)` `uint8`
- `actions`: `(N, T, 3)` `float32` as `[force_x, force_y, trigger]` (trigger=1.0 on the putt frame)
- `sim_state`: `(N, T, 16, 4)` `float32` as `[pos_x, pos_y, vel_x, vel_y]`
  - slot 0 = ball, slot 1 = hole (static), slots 2+ = bumpers (static)
- `lengths`: `(N,)` `int32`
- `episode_meta`: `(N,)` `object`
