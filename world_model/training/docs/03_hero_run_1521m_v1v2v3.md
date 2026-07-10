# 1.52B Hero Run (Joint v1+v2+v3)

## Goal

Long-run DF training on the full joint latent dataset (`v1 + v2 + v3`) with a larger world model and higher throughput on `2x H100 80GB`.

## Run Configuration

- Config: `world_model_training/configs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_240m_hero.yaml`
- Model size: `1,520,566,848` params (`d_model=2048`, `n_layers=30`, `n_heads=32`)
- Context: `ctx8`
- Training method: Diffusion Forcing (`rollout_steps=4`)
- Target budget: `240,000,000` processed samples
- Checkpoints: every `24,000,000` samples (10 checkpoints total)
- Batch size: `512` per GPU (`global_batch_size=1024`)

## Dataset

- Train manifest: `world_model_training/manifests/joint_v1v2v3_full_400k/train_shards.txt`
- Val manifest: `world_model_training/manifests/joint_v1v2v3_full_400k/val_shards.txt`
- Source datasets:
  - `latents_v1` (1000 shards)
  - `latents_v2` (1000 shards)
  - `latents_v3` collision-heavy (2000 shards)

## Operational Notes

- Run previews should be generated from checkpoints to avoid interfering with training on the only active `2x H100` box.
- `val_loss` appears as `NaN` between eval steps by design (`eval_every_steps=5000`) to reduce eval overhead.
- W&B state can temporarily show old terminated runs as "Running" until heartbeat timeout; verify actual GPU jobs via `ps`/`nvidia-smi`.
