# Full Run: `full_v3collision_20260223_094313`

Collision-heavy v3 raw data generation run.

- Dataset prefix: `s3://videogen-pool-v2-237586137680/dataraw_v3/full_v3collision_20260223_094313/`
- Shards path: `s3://videogen-pool-v2-237586137680/dataraw_v3/full_v3collision_20260223_094313/raw/shards/`
- Metadata path: `s3://videogen-pool-v2-237586137680/dataraw_v3/full_v3collision_20260223_094313/meta/metadata.json`
- Run logs prefix: `s3://videogen-pool-v2-237586137680/ops/runs/full_v3collision_20260223_094313/`
- Artifact URI: `s3://videogen-pool-v2-237586137680/ops/artifacts/full_v3collision_20260223_094313/data_generation_package_v3.tar.gz`
- Generator variant: `data_generation_package_v3_collision_heavy`
- Target: `200000` episodes, `600` frames/episode, shard size `100`

Launch uses EC2 + SSM with `remote_full_run.primary.sh` (fallback script available).
