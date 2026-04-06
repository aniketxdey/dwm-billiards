# Run Governance

## Non-negotiables
- All implementation changes stay in this repo (local-first workflow).
- Every training run must have an explicit `RUN_ID`.
- No ad-hoc output paths outside `VAE training/runs/<run_id>/`.
- Every run must produce `manifest.json`, `summary.json`, and `metrics/train_metrics.jsonl`.

## Run id policy
- Format: letters, numbers, `_`, `-`, `.` only.
- Recommended generation:
```bash
RUN_PREFIX=vae_60m_1xa100 bash "VAE training/scripts/new_run_id.sh" run01
```

## Required run artifacts
Under `VAE training/runs/<run_id>/`:
- `config/resolved_config.json`
- `manifest.json`
- `summary.json`
- `metrics/train_metrics.jsonl`
- `checkpoints/`
- `previews/`

Global registry:
- `VAE training/runs/run_registry.jsonl`

## Logging policy
- Keep logs concise and useful.
- Log training rows at:
  - periodic step interval (`log_every_steps`)
  - preview events
  - checkpoint events
  - run end

## Sync policy
- Train locally first.
- Upload only organized run directories:
```bash
bash "VAE training/scripts/sync_run_to_s3.sh" "$RUN_ID"
```
