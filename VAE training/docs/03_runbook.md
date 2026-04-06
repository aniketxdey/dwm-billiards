# Runbook (Lambda 1x A100)

## Preconditions
- Lambda instance is ready (`gpu_1x_a100_sxm4`).
- `.env.lambda` has valid AWS and W&B values.
- Full shards are staged locally (`~35.6 GB` compressed).
- Local SSD free space should be `>120 GB` (streaming mode does not need 1.66 TB cache).

## Step 0: sync code to Lambda (optional but recommended)
```bash
bash "VAE training/scripts/sync_repo_to_lambda.sh"
```

## Step 1: install dependencies
```bash
pip install -r "VAE training/requirements.txt"
```

## Step 2: stage dataset from S3
```bash
bash "VAE training/scripts/stage_shards_from_s3.sh"
```

## Step 3: preflight checks (full 60M streaming)
```bash
RUN_ID="$(RUN_PREFIX=vae_60m_1xa100 bash "VAE training/scripts/new_run_id.sh" run01)"
RUN_ID="$RUN_ID" bash "VAE training/scripts/preflight_60m_1xa100.sh"
```

## Step 4: launch training
```bash
RUN_ID="$RUN_ID" RUN_NOTES="full_60m_streaming" bash "VAE training/scripts/run_train_60m_1xa100.sh"
```

## Milestone review at 1M
- A milestone event is emitted at `1,000,000` frames.
- Checkpoint `ckpt_1000000.pt` is expected (checkpoint interval is `1,000,000`).
- Use this point for qualitative/quantitative review before continuing deeper into the run.

## Step 5: monitor
- Check console logs for throughput and losses.
- Inspect `runs/<run_id>/previews/` every preview interval.
- Confirm checkpoints appear every `1M` frames.

## Step 6: sync outputs to S3
```bash
bash "VAE training/scripts/sync_run_to_s3.sh" "$RUN_ID"
```

## Step 7: export full latent dataset from frozen VAE
```bash
bash "VAE training/scripts/export_latents_60m.sh"
```

Sync latent export to S3:
```bash
AWS_PROFILE=codex-admin-web \
bash "VAE training/scripts/sync_latents_to_s3.sh" "latents_60m_from_vae_60m_20260220_204310_run01"
```

## Failure handling
- If interrupted, relaunch with same run id and `RESUME_CKPT`:
```bash
RUN_ID="$RUN_ID" RESUME_CKPT="/path/to/ckpt.pt" bash "VAE training/scripts/run_train_60m_1xa100.sh"
```
- If dataloader is slow, reduce workers or increase prefetch in config.
