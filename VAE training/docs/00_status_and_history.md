# Status and History

## Project status
- Data generation is complete.
- VAE training is complete for full `60M` frames using shard streaming.

## Completed data-generation runs

### Pilot
- Run id: `pilot_20260220_104323`
- Date (UTC): February 20, 2026
- Episodes: `1,000`
- Shards: `10`
- Total shard bytes: `355,581,698`
- Prefix: `s3://videogen-pool-v2-237586137680/dataraw_v2/pilot_20260220_104323/`

### Full dataset (canonical)
- Run id: `full_20260220_112101`
- Date (UTC): February 20, 2026
- Episodes: `100,000`
- Frames per episode: `600`
- Total frames: `60,000,000`
- Shards: `1,000` (`shard_00000.npz` to `shard_00999.npz`)
- Total shard bytes: `35,558,550,764` (~35.56 GB)
- Metadata: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/meta/metadata.json`
- Shards: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/raw/shards/`

## Dataset schema (per shard)
- `frames`: `(N, 600, 72, 128, 3)` `uint8`
- `actions`: `(N, 600, 3)` `float32` as `[force_x, force_y, trigger]`
- `sim_state`: `(N, 600, 16, 4)` `float32` as `[pos_x, pos_y, vel_x, vel_y]`
- `lengths`: `(N,)` `int32`
- `episode_meta`: `(N,)` object

## Current step
- Freeze final checkpoint and hand off to latent-dataset export stage.

## Active VAE run (current)
- Date started (UTC): February 20, 2026
- Run id: `vae_60m_1xa100_20260220_204310_run01`
- Config: `VAE training/configs/vae_60m_1xa100.yaml`
- Hardware: Lambda `gpu_1x_a100_sxm4` (1x A100 40GB)
- Data mode: `source=shards` (streaming from NPZ shards)
- Target frames: `60,000,000`
- Milestone frame: `1,000,000`
- Checkpoint cadence: every `1,000,000` frames
- Preview cadence: every `1,000,000` frames
- W&B run: `https://wandb.ai/moin-a-mattar/video_generation_project202/runs/hqzimuvd`
- Verified status:
  - MP4 preview encoding works (ffmpeg backend installed)
  - Full-cache attempt was blocked by disk (`~409GB` free vs `~1.66TB` required)
  - Streaming path launched successfully and reached steady-state GPU utilization near `100%`
  - Completed at `60,000,000 / 60,000,000` frames
  - Final checkpoint: `checkpoints/ckpt_60000000.pt`
  - Finished at: `2026-02-21 05:17:01 UTC`
  - Average throughput: `1946.86 fps`

## Freeze note
- Freeze marker written:
  `VAE training/runs/vae_60m_1xa100_20260220_204310_run01/FROZEN_HANDOFF.json`
- S3 sync is pending AWS reauthentication on the Lambda host.
