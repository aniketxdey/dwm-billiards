# Current Focus

## What we are doing now
- VAE 60M run is complete and frozen.
- Preparing latent-dataset export (post-VAE stage).

## Next immediate actions
1. Reauthenticate AWS on Lambda and sync frozen run artifacts to S3.
2. Implement latent export pipeline from frozen VAE checkpoint.
3. Export `z_t` shards aligned with `actions`, `lengths`, and episode metadata.
4. Begin DiT baseline training on latent dataset.
