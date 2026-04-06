# Neural Pool Workspace

Repo for pool-game data generation and VAE/world-model training.


https://github.com/user-attachments/assets/1c44df17-dda8-4185-af50-38ea8b0315d0


## Structure
- `data_generation_package/`: dataset generation (`frames`, `actions`, `sim_state`).
- `VAE training/`: VAE code, configs, scripts, and run docs.
- `world_model_training/`: DiT world-model code, configs, scripts, and run docs.
- `docs/`: repo-level operational docs (S3 access, push checklist).
- `ops/`: run manifests and operational notes.
- `scripts/`: workspace utility scripts.
- `samples/`: local sample outputs only (not canonical dataset storage).
- `Research refrences/`: local reading material (not tracked in git).

## Canonical Data Location
Raw dataset is stored in S3, not in git:
- `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/`

Use `VAE training/docs/02_s3_contract.md` and `docs/S3_ACCESS.md` for exact paths and commands.
Project timeline summary: `docs/PROJECT_JOURNEY.md`.
World-model baseline decision summary: `docs/WORLD_MODEL_DECISION_2026-02-21.md`.
Current detailed status snapshot (code + experiments + demos + docs): `docs/PROJECT_STATUS_2026-03-04.md`.

## Quick Start
1. Configure AWS + W&B env
```bash
cp .env.lambda.example .env.lambda
# fill .env.lambda
bash scripts/lambda_prepare_env.sh
```

2. Sync code to Lambda and stage shards
```bash
bash "VAE training/scripts/sync_repo_to_lambda.sh"
bash "VAE training/scripts/stage_shards_from_s3.sh"
```

3. Launch VAE training (full 60M streaming mode)
```bash
RUN_ID="$(RUN_PREFIX=vae_60m_1xa100 bash "VAE training/scripts/new_run_id.sh" run01)"
RUN_ID="$RUN_ID" bash "VAE training/scripts/preflight_60m_1xa100.sh"
RUN_ID="$RUN_ID" RUN_NOTES="full_60m_streaming" bash "VAE training/scripts/run_train_60m_1xa100.sh"
```

4. Launch world-model baseline (DiT, 5M pilot)
```bash
RUN_ID="$(RUN_PREFIX=dit_5m_1xa100 bash world_model_training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash world_model_training/scripts/preflight_1xa100.sh
RUN_ID="$RUN_ID" RUN_NOTES="dit_baseline_5m" bash world_model_training/scripts/run_train_1xa100.sh
```

## Git Policy
- Do not commit secrets (`.env`, keys, PEM files).
- Do not commit raw dataset shards or large binary artifacts.
- Keep canonical data in S3 and reference it by URI in docs/manifests.

See `docs/GITHUB_PUSH_CHECKLIST.md` before pushing.
