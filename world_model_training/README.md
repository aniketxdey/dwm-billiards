# World Model Training (DiT Baseline)

This folder contains the action-conditioned diffusion world model stage trained on VAE latent shards.

## Objective
Train a baseline DiT that predicts latent noise for the next latent frame:
- Input: context latents `z_{t-L+1:t}` + action `a_t`
- Target: noise for `z_{t+1}` under Gaussian diffusion
- Output: trained baseline checkpoint for rollout/eval and comparison against DiT+Diffusion-Forcing.

## Layout
- `configs/`: launch-ready configs (`smoke`, `1m`, `5m`, `60m`)
- `scripts/`: preflight, run launch, data staging, S3 sync, run-id helper
- `src/world_model_training/`: implementation
- `docs/`: status, runbook, and data contract
- `runs/`: local run outputs (excluded from git)

## Current Configs
- `configs/dit_smoke_200k.yaml`: quick pipeline validation
- `configs/dit_1m_1xa100.yaml`: 1M-sample baseline with 10 checkpoints
- `configs/dit_5m_1xa100.yaml`: main pilot baseline
- `configs/dit_60m_1xa100.yaml`: full pass baseline
- `configs/dit_60m_context12_1xa100.yaml`: context-length ablation variant
- `configs/dit_60m_2xh100_ctx8.yaml`: DDP launch profile for `2x H100`
- `configs/dit_df_joint_v1v2_ctx8_2xh100_686m_smoke_200k.yaml`: joint-dataset DF smoke (`~686.5M`)
- `configs/dit_df_joint_v1v2_ctx8_2xh100_686m_120m.yaml`: primary joint DF config (`~686.5M`)
- `configs/dit_df_joint_v1v2_ctx8_2xh100_573m_120m_fallback.yaml`: fallback joint DF config (`~573.2M`)

## Latest Validated Baseline (2026-02-21)
- Winner: `ctx8` (`dit_60m_1xa100_20260221_185520_resume5m`)
- Compared against: `ctx12` (`dit_60m_ctx12_1xa100_20260221_192300_run01`)
- Rollout eval: `world_model_training/evals/rollout_eval_ctx8_vs_ctx12_60m_rerun_20260221_210642/summary.json`
- Key long-horizon metrics:
  - `latent_mse@h16`: `0.979` (ctx8) vs `3.186` (ctx12)
  - `latent_mse@h32`: `3.817` (ctx8) vs `149.161` (ctx12)
  - `frame_psnr@h32`: `18.05` (ctx8) vs `14.56` (ctx12)
- Decision: use `ctx8` checkpoint as baseline for next stage (Diffusion Forcing branch).


## Current Flagship DF Lineage (v1+v2+v3, 2x H100)
- Main completed resume run:
  - `dit_df_joint_v1v2v3_ctx8_2xh100_1521m_resume480m_20260301_080158_run01`
  - final checkpoint: `.../checkpoints/ckpt_480000000.pt`
- Full rollout eval (360M vs 480M):
  - `world_model_training/evals/rollout_eval_joint_1521m_360m_vs_480m_20260302/summary.json`
- Continuation run in same lineage:
  - config: `configs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_555m_resume_lr1e5.yaml`
  - target samples: `555,000,000`

## Distillation Status
- Distillation/student model training is **not implemented yet** in this repo.
- Current artifacts are teacher-style DiT+DF checkpoints and inference tooling.

## Distillation Scaffold (Experimental)
- Distillation trainer entrypoint:
  - `src/world_model_training/train_distill.py`
- Starter config (teacher `480M` -> student `~573M`):
  - `configs/dit_distill_joint_v1v2v3_ctx8_2xh100_573m_from480m_60m.yaml`
- Notes:
  - This path is newly added and currently intended for pilot runs.
  - Throughput tuning is still required before long production distillation jobs.

## Quick Start (1x A100)
```bash
RUN_ID="$(RUN_PREFIX=dit_5m_1xa100 bash world_model_training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash world_model_training/scripts/preflight_1xa100.sh
RUN_ID="$RUN_ID" RUN_NOTES="dit_baseline_5m" bash world_model_training/scripts/run_train_1xa100.sh
```

## Quick Start (2x H100, DDP)
```bash
RUN_ID="$(RUN_PREFIX=dit_60m_ctx8_2xh100 bash world_model_training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash world_model_training/scripts/preflight_2xh100.sh
RUN_ID="$RUN_ID" RUN_NOTES="ctx8_60m_2xh100" bash world_model_training/scripts/run_train_2xh100.sh
```

## Joint Dataset Manifests (v1 + v2)
Build manifest files that mix `latents_v1` and `latents_v2` without copying shards:
```bash
python3 world_model_training/scripts/build_joint_latent_manifests.py \
  --v1-shards-dir /home/ubuntu/neural-pool/latents_v1/latents_60m_from_vae_60m_20260220_204310_run01/shards \
  --v2-shards-dir /home/ubuntu/neural-pool/latents_v2/latents_v2diverse_20260222_154743_watch01/shards \
  --out-dir /home/ubuntu/maat/world_model_training/manifests/joint_v1v2_full_200k \
  --val-shards-v1 50 \
  --val-shards-v2 50 \
  --train-order interleave \
  --write-train-all-manifest
```

Outputs:
- `world_model_training/manifests/joint_v1v2_full_200k/train_shards.txt`
- `world_model_training/manifests/joint_v1v2_full_200k/val_shards.txt`
- `world_model_training/manifests/joint_v1v2_full_200k/eval_shards.txt`
- `world_model_training/manifests/joint_v1v2_full_200k/all_shards.txt`
- `world_model_training/manifests/joint_v1v2_full_200k/manifest.json`

## Big-Model Joint DF Smoke (2x H100)
```bash
RUN_ID="$(RUN_PREFIX=dit_df_joint_686m_smoke bash world_model_training/scripts/new_run_id.sh run01)"
CONFIG_PATH=world_model_training/configs/dit_df_joint_v1v2_ctx8_2xh100_686m_smoke_200k.yaml \
RUN_ID="$RUN_ID" \
bash world_model_training/scripts/preflight_2xh100.sh

CONFIG_PATH=world_model_training/configs/dit_df_joint_v1v2_ctx8_2xh100_686m_smoke_200k.yaml \
RUN_ID="$RUN_ID" RUN_NOTES="joint_v1v2_686m_df_smoke" \
bash world_model_training/scripts/run_train_2xh100.sh
```

## Resume Training
```bash
RUN_ID="$(RUN_PREFIX=dit_5m_1xa100 bash world_model_training/scripts/new_run_id.sh resume01)"
RESUME_CKPT="/home/ubuntu/maat/world_model_training/runs/<old_run>/checkpoints/ckpt_005000000.pt" \
RUN_ID="$RUN_ID" \
RUN_NOTES="resume_from_5m" \
bash world_model_training/scripts/run_train_1xa100.sh
```

## Push Run Artifacts to S3
```bash
AWS_PROFILE=codex-admin-web \
bash world_model_training/scripts/sync_run_to_s3.sh "<run_id>"
```

S3 destination prefix:
- `s3://videogen-pool-v2-237586137680/world_model_v1/<run_id>/`

## Notes
- Canonical latent source used by configs:
  `/home/ubuntu/neural-pool/latents_v1/latents_60m_from_vae_60m_20260220_204310_run01/shards`
- Keep secrets out of git; use `.env.lambda` for runtime env vars.

## Rollout Evaluation
Run fixed-horizon rollout evaluation and generate comparison videos:
```bash
CONFIG_PATH=world_model_training/configs/rollout_eval_60m_ctx8_vs_ctx12.yaml \
bash world_model_training/scripts/run_rollout_eval.sh
```

Outputs are written to:
- `world_model_training/evals/<eval_id>/summary.json`
- `world_model_training/evals/<eval_id>/videos/` (GT | model1 | model2)

## Implementation Scope Notes
- Trainer supports single-GPU and `torchrun` DDP multi-GPU.
- Diffusion Forcing v0 is implemented (rollout training with scheduled teacher forcing).
- Manifest-based train/val/eval shard lists are supported for joint datasets.
