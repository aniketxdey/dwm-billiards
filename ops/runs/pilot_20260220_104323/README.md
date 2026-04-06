# Pilot Run: `pilot_20260220_104323`

## Goal
Validate the end-to-end AWS pipeline for clean v2 dataset generation:
- launch EC2 runner
- execute generator on instance
- upload shards + metadata to S3
- capture logs and run summary

## Final Status
- Result: `SUCCESS`
- AWS region: `us-east-1`
- Instance: `i-010b1d74113e693d5` (`c7i.24xlarge`)
- Instance state after pilot: `stopped` (cost-safe)

## Output Locations
- Dataset prefix: `s3://videogen-pool-v2-237586137680/dataraw_v2/pilot_20260220_104323/`
- Shards: `s3://videogen-pool-v2-237586137680/dataraw_v2/pilot_20260220_104323/raw/shards/`
- Metadata: `s3://videogen-pool-v2-237586137680/dataraw_v2/pilot_20260220_104323/meta/metadata.json`
- Run logs: `s3://videogen-pool-v2-237586137680/ops/runs/pilot_20260220_104323/logs/`
- Run summary: `s3://videogen-pool-v2-237586137680/ops/runs/pilot_20260220_104323/summary.json`

## Pilot Config
- Episodes: `1000`
- Frames per episode: `600`
- Shard size: `100` (10 shards total)
- Workers: `48`
- Seed: `42`

## Metrics
- Uploaded shards: `10`
- Uploaded shard bytes: `355,581,698` (`339.11 MiB`)
- Mean shard size: `33.91 MiB`
- Run wall time (including environment setup): `146s` (`2.43 min`)
- Generator-reported speed: `7.25 episodes/s`
- Generator-reported total generation time: `2.3 min`

## Estimated Full Run (100k Episodes)
Using pilot throughput:
- At `7.25 episodes/s`: about `3.83 hours`
- Including setup and overhead buffer: plan for `~4.0-4.5 hours`
- Expected raw shard size total: about `35.6 GB` (same order as prior raw dataset)

## Worker Sweep (Post-Pilot Optimization)
Benchmark command: SSM `bbef2b65-10a6-42dc-abcd-1f88396bae4f`

- workers `32`: `8.01 ep/s` (best)
- workers `48`: `7.92 ep/s`
- workers `64`: `7.75 ep/s`
- workers `80`: `7.66 ep/s`
- workers `96`: `7.52 ep/s`

Recommendation for full run on `c7i.24xlarge`: use `--workers 32`.

## Issues Hit During Pilot and Fixes
1. Tar extraction warning from macOS extended attributes caused non-zero exit on EC2.
   - Fix: rebuilt artifact with `COPYFILE_DISABLE=1 tar --no-xattrs`.
2. System pip install failed under managed Python environment.
   - Fix: remote run switched to dedicated virtualenv.
3. `opencv-python` required missing system X11 library (`libxcb.so.1`) on EC2.
   - Fix: switched dependency to `opencv-python-headless` in `data_generation_package/requirements.txt`.

## Command History
- Initial run command: `2a615d87-cb4d-443d-a8f7-5aa57ad91b25` (failed: tar xattr handling)
- Rerun command: `15a18632-9157-4f23-be98-1abb250e7f19` (failed early; superseded by debug)
- Debug trace: `1b039d3f-ce58-48cf-9a49-1a514ede8a28` (identified pip/system-env issue)
- Venv rerun: `378d0f7b-18d2-4d1b-b80f-7334ecae9604` (failed: `libxcb.so.1`)
- Final successful run: `942bf66b-b2cd-44e2-a250-452c2e6fe965`

## Validation Snapshot
Sample shard inspected: `shard_00000.npz`
- Keys: `frames`, `actions`, `sim_state`, `lengths`, `episode_meta`
- Shapes:
  - `frames`: `(100, 600, 72, 128, 3)` `uint8`
  - `actions`: `(100, 600, 3)` `float32`
  - `sim_state`: `(100, 600, 16, 4)` `float32`
  - `lengths`: `(100,)` `int32`
- Quick quality:
  - `mean_shots_per_ep`: `8.24`
  - lengths fixed at `600`
