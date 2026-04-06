# Full Run: `full_v2diverse_20260222_154743`

## Status
- Current status: `PREPARING`

## Live Execution
- EC2 instance: `i-0157fad8542f41df2` (`c7i.48xlarge`)
- SSM command: `e735d971-9ebb-4e1f-bd4e-76db70c211e2`
- Remote script: `remote_full_run.primary.sh` (`workers=64`)

## Target Output
- Dataset prefix: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_v2diverse_20260222_154743/`
- Shards path: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_v2diverse_20260222_154743/raw/shards/`
- Metadata path: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_v2diverse_20260222_154743/meta/metadata.json`

## Run Logs
- Run logs prefix: `s3://videogen-pool-v2-237586137680/ops/runs/full_v2diverse_20260222_154743/`
- Live partial log: `s3://videogen-pool-v2-237586137680/ops/runs/full_v2diverse_20260222_154743/logs/generate.partial.log`
- Final summary: `s3://videogen-pool-v2-237586137680/ops/runs/full_v2diverse_20260222_154743/summary.json`

## Config
- Episodes: `100000`
- Frames per episode: `600`
- Shard size: `100`
- Seed: `4242`
- Workers (primary/fallback): `64/32`

## Notes
- Generator variant: `data_generation_package_v2_diverse`
- Changes: diversified bot archetypes, spawn layouts, and mild per-episode physics variation.
