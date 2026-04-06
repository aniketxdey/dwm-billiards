# S3 Contract

## Dataset inputs (read)
- Canonical data root:
  `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/`
- Shards:
  `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/raw/shards/`
- Metadata:
  `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/meta/metadata.json`

## VAE outputs (write)
Use one run id per training launch:

`vae_v1/<run_id>/`

Recommended layout:
- `vae_v1/<run_id>/checkpoints/`
- `vae_v1/<run_id>/previews/`
- `vae_v1/<run_id>/metrics/`
- `vae_v1/<run_id>/config/`
- `vae_v1/<run_id>/manifest.json`

## Local-to-S3 sync policy
- Write first to local disk during training for stability.
- Sync checkpoints and previews on interval or at run end.
- Always upload config + manifest for reproducibility.

## Latent dataset outputs (write)
Latent exports are versioned under:

`latents_v1/<latent_export_id>/`

Recommended layout:
- `latents_v1/<latent_export_id>/shards/latent_shard_00000.npz`
- `latents_v1/<latent_export_id>/manifest.json`
- `latents_v1/<latent_export_id>/summary.json`
- `latents_v1/<latent_export_id>/logs/export_progress.jsonl`
