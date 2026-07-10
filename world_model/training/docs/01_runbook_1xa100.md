# Runbook (1x A100)

For `2x H100` DDP launch flow, use:
- `world_model_training/docs/02_runbook_2xh100.md`

## 1. Optional: sync code to Lambda
```bash
bash "VAE training/scripts/sync_repo_to_lambda.sh"
```

## 2. Verify latent shards are present
```bash
ls /home/ubuntu/neural-pool/latents_v1/latents_60m_from_vae_60m_20260220_204310_run01/shards | head
```

If missing, stage from S3:
```bash
AWS_PROFILE=codex-admin-web \
bash world_model_training/scripts/stage_latent_shards_from_s3.sh
```

## 3. Preflight
```bash
RUN_ID="$(RUN_PREFIX=dit_5m_1xa100 bash world_model_training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash world_model_training/scripts/preflight_1xa100.sh
```

## 4. Launch
```bash
RUN_ID="$RUN_ID" RUN_NOTES="dit_baseline_5m" bash world_model_training/scripts/run_train_1xa100.sh
```

## 5. Monitor
- stdout logs include:
  - `step`
  - `processed_samples/target_samples`
  - `samples_per_sec`
  - `train_loss`
  - `val_loss`
  - `ckpt=true/false`
- Run outputs:
  - `world_model_training/runs/<run_id>/checkpoints/`
  - `world_model_training/runs/<run_id>/metrics/train_metrics.jsonl`
  - `world_model_training/runs/<run_id>/summary.json`

## 6. Upload to S3
```bash
AWS_PROFILE=codex-admin-web \
bash world_model_training/scripts/sync_run_to_s3.sh "$RUN_ID"
```

## 7. Rollout Evaluation (After Two Candidate Runs Finish)
```bash
CONFIG_PATH=world_model_training/configs/rollout_eval_60m_ctx8_vs_ctx12.yaml \
bash world_model_training/scripts/run_rollout_eval.sh
```

Key outputs:
- `world_model_training/evals/<eval_id>/summary.json`
- `world_model_training/evals/<eval_id>/videos/clip_*.mp4`
