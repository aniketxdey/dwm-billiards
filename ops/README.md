# Neural Pool AWS Ops

This folder tracks reproducible AWS data-generation operations.

## Active AWS Resources
- Bucket (v2 target): `s3://videogen-pool-v2-237586137680`
- EC2 runner role: `NeuralPoolEc2RunnerRole`
- EC2 instance profile: `NeuralPoolEc2RunnerProfile`
- Runner security group: `sg-0cd6b54e100af5aad` (`neural-pool-runner-sg`, no ingress)

## Run Records
- Pilot run: `ops/runs/pilot_20260220_104323/README.md`
- Full run: `ops/runs/full_20260220_112101/README.md`

## Data Layout (v2)
- Dataset shards: `s3://videogen-pool-v2-237586137680/dataraw_v2/<run_id>/raw/shards/`
- Dataset metadata: `s3://videogen-pool-v2-237586137680/dataraw_v2/<run_id>/meta/metadata.json`
- Run logs and summary: `s3://videogen-pool-v2-237586137680/ops/runs/<run_id>/`
