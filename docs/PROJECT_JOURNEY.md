# Project Journey (What We Have Done So Far)

This is the single chronological summary of this project up to now.

## 1. Workspace Organization
- Created clear module boundaries:
  - `data_generation_package/` for generation code.
  - `VAE training/` for VAE implementation + run operations.
  - `world_model_training/` for DiT/DF training and rollout eval.
  - `world_model_inference/` for preview/UI/live-play runtime.
  - `docs/` for operational + decision documentation.

## 2. Data Generation Completed
- Canonical full dataset run:
  - Run ID: `full_20260220_112101`
  - Episodes: `100,000`
  - Frames per episode: `600`
  - Total frames: `60,000,000`
  - Shards: `1,000` NPZ files
- Additional diversity/collision-focused generation lineages were added later (`v2`, `v3`) and merged through latent manifests.

## 3. Canonical Storage Contract Defined
- Canonical dataset root:
  - `s3://videogen-pool-v2-237586137680/dataraw_v2/full_20260220_112101/`
- Schema documented and fixed:
  - `frames`, `actions`, `sim_state`, `lengths`, `episode_meta`
- VAE artifact destination documented:
  - `s3://videogen-pool-v2-237586137680/vae_v1/<run_id>/`

## 4. VAE Pipeline Implemented And Trained
- Implemented VAE stack under `VAE training/src/vae_training/`:
  - config/model/losses/train loop/checkpointing/previews
- Full 60M-frame VAE training completed and used as latent encoder/decoder for world-model stages.

## 5. World Model (DiT) Training Progression
- Baseline DiT line validated first (single-step).
- Diffusion Forcing (DF) training path implemented in trainer.
- Scaled to 2x H100 DDP for high-throughput runs.
- Major model lineage: `~1.521B` parameters on joint `v1+v2+v3` latent manifests.

## 6. Rollout Evaluation System
- Added fixed-horizon rollout eval with DDIM rollout + optional VAE decode.
- Produces:
  - `summary.json` metrics (`latent_mse`, `latent_mae`, `frame_psnr`, `frame_l1`)
  - side-by-side videos (`GT | model1 | model2`)
- Latest major comparison (360M vs 480M checkpoints) favored `480M` overall.

## 7. Inference Runtime Track Added
- Added `world_model_inference/`:
  - checkpoint preview CLI
  - Gradio interactive sandbox
  - live websocket play route (click-drag-release actions)
  - DDIM/action benchmark scripts
- This is runtime/inference engineering and does **not** imply model distillation.

## 8. Current State (as of 2026-03-03)
- Best completed checkpoint in active use:
  - `.../ckpt_480000000.pt`
- Overnight continuation launched to `555M` samples on same data/settings.
- Distillation branch is **not started yet** (no distilled model artifact exists).

## 9. Immediate Next Steps
1. Finish `555M` continuation run and run same rollout eval protocol.
2. Promote winner checkpoint (`480M` vs `555M`) as teacher/reference.
3. Start explicit distillation only after teacher is frozen.
