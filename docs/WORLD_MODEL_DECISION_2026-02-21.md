# World Model Decision (2026-02-21)

## Objective
Choose the baseline checkpoint for the next stage by comparing context lengths on the same `60M` latent/action dataset.

## Compared Runs
- `ctx8`: `dit_60m_1xa100_20260221_185520_resume5m`
- `ctx12`: `dit_60m_ctx12_1xa100_20260221_192300_run01`

Both were evaluated with the same rollout harness, held-out shard split, and sampling settings.

## Evaluation Artifact
- `world_model_training/evals/rollout_eval_ctx8_vs_ctx12_60m_rerun_20260221_210642/summary.json`

## Key Metrics
- `latent_mse@h16`: `0.979` (ctx8) vs `3.186` (ctx12)
- `latent_mse@h32`: `3.817` (ctx8) vs `149.161` (ctx12)
- `frame_psnr@h32`: `18.05` (ctx8) vs `14.56` (ctx12)
- `frame_l1@h32`: `0.0336` (ctx8) vs `0.0707` (ctx12)

## Decision
- Baseline checkpoint is `ctx8`:
  `/home/ubuntu/maat/world_model_training/runs/dit_60m_1xa100_20260221_185520_resume5m/checkpoints/ckpt_060000000.pt`

## Next Steps
1. Implement Diffusion Forcing training branch.
2. Add DDP/multi-GPU support before scaling training.
3. Evaluate `ctx8 baseline` vs `ctx8 + DF` with the same rollout harness.
