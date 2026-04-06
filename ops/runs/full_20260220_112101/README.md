# Full Run: `full_20260220_112101`

## Status
- Current status: `RUNNING`
- SSM command: `95058219-d35e-4314-9c0c-027d9b8098c1`
- EC2 instance: `i-010b1d74113e693d5` (`c7i.24xlarge`)
- Worker setting: `32` (from pilot worker sweep optimization)
- Resume SSM command: `30a78cf0-326a-4a70-8605-711bf4f3af13`

## Target Output
- Dataset prefix: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/`
- Shards path: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/raw/shards/`
- Metadata path: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/meta/metadata.json`

## Run Logs
- Run logs prefix: `s3://videogen-pool-v2-237586137680/ops/runs/full_20260220_112101/`
- Live partial log: `s3://videogen-pool-v2-237586137680/ops/runs/full_20260220_112101/logs/generate.partial.log`
- Final summary (on completion): `s3://videogen-pool-v2-237586137680/ops/runs/full_20260220_112101/summary.json`

## Config
- Episodes: `100000`
- Frames per episode: `600`
- Shard size: `100` (expected `1000` shards)
- Seed: `42`

## Incident + Resume
- Initial full command timed out at ~`26.5%` because SSM plugin execution timeout was hit.
- Resume support was added to generator:
  - `--start-episode`
  - `--start-shard-id`
  - `--no-upload-metadata`
- Resume started from:
  - shard id `265`
  - episode `26500`
  - remaining episodes `73500`
- Resume command includes explicit `executionTimeout=43200` to prevent 1-hour plugin timeout.

## Monitor Commands
```bash
aws ssm list-command-invocations \
  --region us-east-1 \
  --profile codex-admin \
  --command-id 95058219-d35e-4314-9c0c-027d9b8098c1 \
  --details \
  --query 'CommandInvocations[0].{Status:Status,ResponseCode:CommandPlugins[0].ResponseCode}' \
  --output json

aws s3 ls s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/raw/shards/ \
  --profile codex-admin | wc -l

aws s3 cp s3://videogen-pool-v2-237586137680/ops/runs/full_20260220_112101/logs/generate.partial.log - \
  --profile codex-admin
```
