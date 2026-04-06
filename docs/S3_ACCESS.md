# S3 Access Guide

This project keeps canonical data in S3. GitHub stores code/docs only.

## 1. Authenticate AWS CLI

Use one of these methods.

### Option A: Web auth (recommended)
```bash
aws login --profile codex
export AWS_PROFILE=codex
```

### Option B: Access keys
```bash
aws configure --profile codex-admin
export AWS_PROFILE=codex-admin
```

## 2. Verify Access
```bash
aws sts get-caller-identity
aws s3 ls s3://videogen-pool-v2-237586137680/
```

## 3. Canonical Dataset Paths
- Root: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/`
- Shards: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/raw/shards/`
- Metadata: `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/meta/metadata.json`

## 4. Pull Data

Download one shard:
```bash
aws s3 cp \
  s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/raw/shards/shard_00000.npz \
  ./shard_00000.npz
```

Stage full dataset to Lambda/local SSD:
```bash
bash "VAE training/scripts/stage_shards_from_s3.sh"
```

Override source/destination:
```bash
S3_ROOT=s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101 \
LOCAL_ROOT=/home/ubuntu/neural-pool/full_20260220_112101 \
bash "VAE training/scripts/stage_shards_from_s3.sh"
```

## 5. Push Training Artifacts Back to S3
```bash
bash "VAE training/scripts/sync_run_to_s3.sh" "<run_id>"
```

Default destination prefix:
- `s3://videogen-pool-v2-237586137680/vae_v1/<run_id>/`

## 6. Access Scope Recommendation

For day-to-day work, IAM should allow:
- `s3:ListBucket` on `videogen-pool-v2-237586137680`
- `s3:GetObject` on `dataraw_v2/*`
- `s3:PutObject` on `vae_v1/*`

Avoid root credentials for routine usage.
