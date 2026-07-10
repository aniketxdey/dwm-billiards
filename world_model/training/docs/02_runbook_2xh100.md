# Runbook (2x H100 SXM5, DDP)

## 1. Optional: sync code to Lambda
```bash
bash "vae_training/scripts/sync_repo_to_lambda.sh"
```

## 2. Verify latent shards are present
```bash
ls /home/ubuntu/neural-pool/latents_v1/latents_60m_from_vae_60m_20260220_204310_run01/shards | head
```

If missing, stage from S3:
```bash
AWS_PROFILE=codex-admin-web \
bash world_model/training/scripts/stage_latent_shards_from_s3.sh
```

## 3. Preflight (NCCL + shard contract + env)
```bash
RUN_ID="$(RUN_PREFIX=dit_60m_ctx8_2xh100 bash world_model/training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash world_model/training/scripts/preflight_2xh100.sh
```

## 4. Launch DDP training
```bash
RUN_ID="$RUN_ID" RUN_NOTES="ctx8_60m_2xh100" bash world_model/training/scripts/run_train_2xh100.sh
```

Optional overrides:
```bash
CONFIG_PATH=world_model/training/configs/dit_60m_2xh100_ctx8.yaml \
NPROC_PER_NODE=2 \
MASTER_PORT=29511 \
RUN_ID="$RUN_ID" \
bash world_model/training/scripts/run_train_2xh100.sh
```

## 5. Monitor
- stdout logs include:
  - `step`
  - `processed_samples/target_samples`
  - `samples_per_sec`
  - `train_loss`
  - `val_loss`
  - `ckpt=true/false`
  - `world_size`
- Run outputs:
  - `world_model/training/runs/<run_id>/checkpoints/`
  - `world_model/training/runs/<run_id>/metrics/train_metrics.jsonl`
  - `world_model/training/runs/<run_id>/summary.json`

## 6. Upload to S3
```bash
AWS_PROFILE=codex-admin-web \
bash world_model/training/scripts/sync_run_to_s3.sh "$RUN_ID"
```
