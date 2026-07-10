# VAE training

This folder is the dedicated workspace for Neural Pool VAE work.
It is organized to keep implementation, planning, and operational history in one place.

## Current objective
Train a custom VAE on the full `60,000,000` frame dataset:
- Frame budget: `60,000,000`
- Explicit review milestone: `1,000,000`
- Hardware profile: `1x A100 40GB`
- Checkpoints: every `1,000,000` frames
- Preview renders: every `1,000,000` frames

## Folder layout
- `configs/`: training configs (starting with 1x A100 profile)
- `scripts/`: operational scripts (stage data, build frame cache, run training)
- `src/vae_training/`: implementation code
- `runs/`: local run outputs (checkpoints, previews, metrics)

## Quick start (when GPU instance is ready)
1. Stage dataset shards from S3 to local SSD.
2. Launch full-dataset streaming training (no huge frame cache required).

Commands:

```bash
RUN_ID="$(bash vae_training/scripts/new_run_id.sh run01)"
bash vae_training/scripts/sync_repo_to_lambda.sh
bash vae_training/scripts/stage_shards_from_s3.sh
RUN_ID="$RUN_ID" bash vae_training/scripts/preflight_60m_1xa100.sh
RUN_ID="$RUN_ID" RUN_NOTES="full_60m_streaming" bash vae_training/scripts/run_train_60m_1xa100.sh
```

## Post-VAE latent export
After selecting/finalizing a VAE checkpoint, export latents for world-model training:

```bash
bash vae_training/scripts/export_latents_60m.sh
AWS_PROFILE=codex-admin-web bash vae_training/scripts/sync_latents_to_s3.sh "latents_60m_from_vae_60m_20260220_204310_run01"
```

## Streaming latent export (overlap with data generation)
When raw shards are still being generated/uploaded, you can start latent export early.
This overlaps CPU data generation and GPU VAE encoding.

1. Sync raw shards incrementally from S3 to local SSD (loop).
2. Run the watcher exporter to encode any completed `shard_*.npz` into `latent_shard_*.npz`.
3. Optionally sync latent outputs back to S3 in another loop.

Watcher command:
```bash
CONFIG_PATH="vae_training/configs/latent_export_watch_template.yaml" \
bash vae_training/scripts/export_latents_watch.sh
```

Notes:
- Output format matches the regular exporter, so downstream world-model training is unchanged.
- `skip_existing: true` makes it resume-safe and shard-incremental.
- Set `watch.expected_shards` to stop automatically when the full dataset is present.

## Quick Latent/Recon Comparison Utility
Build a side-by-side panel video (`raw | recon | latent-vis`) for spot checks:
```bash
PYTHONPATH=vae_training/src python3 vae_training/scripts/make_latent_comparison.py \
  --checkpoint "/home/ubuntu/maat/vae_training/runs/<vae_run>/checkpoints/ckpt_60000000.pt" \
  --raw-shard "/home/ubuntu/neural-pool/full_20260220_112101/raw/shards/shard_00000.npz" \
  --latent-shard "/home/ubuntu/neural-pool/latents_v1/latents_60m_from_vae_60m_20260220_204310_run01/shards/shard_00000.npz" \
  --episode-index 0 \
  --num-frames 120 \
  --out-video "samples/vae_latent_compare_ep0.mp4"
```

## Notes
- Keep secrets in `.env.lambda` only.
- This folder assumes dataset source is:
  `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/`
- Full uncompressed cache is ~1.66 TB; use shard streaming unless local disk is sized for it.
- All run artifacts are structured for easy sync/upload.
- Run launches require explicit `RUN_ID` for strict tracking.
