# World Model Status And Plan

## Snapshot (as of 2026-03-03)
- World-model training mode is **DiT + Diffusion Forcing (DF)**, not baseline-only.
- Joint latent dataset (`v1+v2+v3`) pipeline is active through manifest-based train/val shards.
- Flagship model family is `~1.521B` parameters (`d_model=2048`, `n_layers=30`, `n_heads=32`).
- Distillation is **not implemented yet** in this repository.

## Latest Completed Major Run
### 1521M DF resume to 480M samples
- Run ID: `dit_df_joint_v1v2v3_ctx8_2xh100_1521m_resume480m_20260301_080158_run01`
- Config: `world_model_training/configs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_480m_resume_lr1e5.yaml`
- Training mode: `dit_df_v0` (`rollout_steps=4`)
- Final checkpoint:
  - `/home/ubuntu/maat/world_model_training/runs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_resume480m_20260301_080158_run01/checkpoints/ckpt_480000000.pt`
- Summary:
  - `processed_samples=480,000,000`
  - `avg_samples_per_sec=1382.63`

## Rollout Eval (360M vs 480M)
- Eval ID: `rollout_eval_joint_1521m_360m_vs_480m_20260302`
- Summary path:
  - `/home/ubuntu/maat/world_model_training/evals/rollout_eval_joint_1521m_360m_vs_480m_20260302/summary.json`
- Key result:
  - `480M` wins most horizons/metrics (not uniformly every metric at every horizon).
- Decision:
  - Promote `ckpt_480000000.pt` as best checkpoint for inference branch.

## Overnight Continuation (In Progress)
### 1521M DF resume to 555M samples
- Run ID: `dit_df_joint_v1v2v3_ctx8_2xh100_1521m_resume555m_20260303_045021_run01`
- Config: `world_model_training/configs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_555m_resume_lr1e5.yaml`
- Resume checkpoint:
  - `.../resume480m.../checkpoints/ckpt_480000000.pt`
- Target samples:
  - `555,000,000`

## Inference Track Status
- `world_model_inference/` now contains:
  - preview pipeline + CLI
  - Gradio sandbox route
  - low-latency websocket live-play route
  - DDIM/action benchmark scripts
- These are inference/runtime tools, not model distillation.

## Distillation Status
- Student/teacher distillation training loop: **not present**.
- No distilled checkpoint artifact has been produced yet.

## Next Steps (ordered)
1. Let `555M` run finish and evaluate `480M` vs `555M` on same rollout protocol.
2. If `555M` wins consistently, promote it as new `best_ckpt_latest`.
3. Start explicit distillation branch only after selecting final teacher checkpoint.
