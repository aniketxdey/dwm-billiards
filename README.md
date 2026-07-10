# Diffusion World Modelling for Billiards Physics

A world model utilizing diffusion forcing & autoregressive DDIM rollouts in latent space to predict future frames of a 2D billiards game in real-time with accurate physics.

https://github.com/user-attachments/assets/7fb60037-203e-499e-9533-c7397b9cf1ee

## Abstract

This project studies whether diffusion transformers can preserve physics in an interactive,
action-conditioned billiards simulator, and whether diffusion forcing improves long-horizon roll-outs.
We build a full pipeline: simulation data, VAE latent compression, DiT world-model training,
rollout evaluation, and interactive inference tools. The core question is whether multi-step
diffusion forcing preserves game logic better than single-step diffusion training. Across multiple
runs and model scales, diffusion forcing strongly improves long-horizon stability. Remaining
failure modes are mostly identity consistency (color drift and occasional object swallowing),
which motivates targeted follow-up losses.


## Methodology

<img width="1434" height="542" alt="pipeline" src="https://github.com/user-attachments/assets/216e570d-925a-4891-a053-b5b54fcae488" />

### 1) Data generation

- Pymunk-based 2D physics simulator renders billiards episodes at 128×72 resolution, 30 FPS
- Automated bot (RL) executes shots with varied strategies (scoring, banking, chaos, breaking)
- Three dataset versions increase in diversity:
  - v1 (100k episodes, fixed physics)
  - v2 (100k episodes, randomized table physics and spawn layouts)
  - v3 (200k episodes, collision-heavy biasing with denser ball configurations)
    - Each episode produces 600 RGB frames, per-frame actions `[force_x, force_y, trigger]`, and ball state vectors, stored as compressed `.npz` shards on S3.

### 2) VAE

A convolutional VAE with spatial latents (4 channels, 9×16 spatial) is trained on raw frames with an L1 + LPIPS + KL loss. The trained encoder converts the full dataset into latent shards for downstream training.

### 3) World Model

An action-conditioned DiT predicts noise in the VAE latent space (DDPM formulation). Given 8 context latent frames and an action, it denoises a target latent for the next frame. Two training regimes are used: 

1) **Baseline** single-step prediction
2) **Diffusion forcing**, which unrolls multi-step predictions during training with a decaying teacher-forcing schedule for improved long-horizon consistency. A **distillation** pipeline compresses a large teacher into a smaller student model.

### 4) Inference

Autoregressive DDIM rollouts in latent space, decoded back to pixels by the VAE. Served through a CLI preview tool, a Gradio UI with live aiming, or a FastAPI WebSocket endpoint for real-time play.

## Repository Structure

```
rl_data_gen/                   # Billiards data generator (pymunk simulation + shard export,
                               # v3: collision-heavy biasing, breaker bots)
golf/
  rl_data_gen/                 # Mini-golf variant of the data generator
  local_run/                   # Local end-to-end golf pipeline (configs + run artifacts)
vae_training/                  # ConvVAE model, training loop, latent export pipeline
world_model/
  training/                    # DiT model, diffusion, DF training, distillation, rollout eval
  inference/                   # Inference pipeline, Gradio UI, FastAPI live play
  local_run/                   # Local smoke-run artifacts (shards, latents, checkpoints)
milestones/                    # Curated metrics, eval summaries, media references
ops/                           # AWS run manifests, remote execution scripts
scripts/                       # Workspace setup utilities
```

## Setup

**Requirements:** Python 3.10+, PyTorch ≥ 2.2, CUDA GPU. See per-package `requirements.txt` for full dependencies.

**1. Environment configuration**

```bash
cp .env.lambda.example .env.lambda   # fill in AWS credentials and W&B key
bash scripts/lambda_prepare_env.sh
```

**2. Stage data from S3**

```bash
bash vae_training/scripts/stage_shards_from_s3.sh
```

**3. Train VAE**

```bash
RUN_ID="$(RUN_PREFIX=vae_60m_1xa100 bash vae_training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash vae_training/scripts/run_train_60m_1xa100.sh
```

**4. Export latents**

```bash
bash vae_training/scripts/export_latents_60m.sh
```

**5. Train world model**

```bash
RUN_ID="$(RUN_PREFIX=dit_5m_1xa100 bash world_model/training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash world_model/training/scripts/run_train_1xa100.sh
```

**6. Run inference**

```bash
bash world_model/inference/scripts/run_preview.sh       # CLI preview
bash world_model/inference/scripts/run_ui.sh            # Gradio UI
bash world_model/inference/scripts/run_live_play.sh     # Real-time WebSocket play
```

## Data

Raw datasets are stored on S3 (not in git):

- **v1:** `s3://videogen-pool-v2-237586137680/dataraw/full_20260220_112101/`
- **v2:** `s3://videogen-pool-v2-237586137680/dataraw_v2/full_v2diverse_20260222_154743/`
- **v3:** `s3://videogen-pool-v2-237586137680/dataraw_v3/full_v3collision_20260223_094313/`

Shard format (`.npz`): `frames [N,T,72,128,3] uint8`, `actions [N,T,3] float32`, `sim_state [N,T,16,4] float32`, `lengths [N,] int32`, `episode_meta [N,] object`.

See `vae_training/README.md` and `world_model/training/docs/02_data_contract.md` for staging instructions and the data contract.

## Evaluation

Rollout evaluation compares models via autoregressive DDIM generation over increasing horizons (1–256 frames). Metrics include **latent MSE/MAE** and **frame-level PSNR/L1** (after VAE decode). Side-by-side comparison videos are generated automatically. See `milestones/` for curated metric snapshots and `world_model/training/docs/00_status_and_plan.md` for the latest results summary.

## Documentation

| Document                                                | Description                                 |
| ------------------------------------------------------- | ------------------------------------------- |
| `world_model/training/docs/00_status_and_plan.md`     | Latest experiment status and next steps     |
| `world_model/training/docs/01_runbook_1xa100.md`      | Single-GPU training runbook                 |
| `world_model/training/docs/02_runbook_2xh100.md`      | Multi-GPU training runbook                  |
| `world_model/training/docs/02_data_contract.md`       | Latent shard data contract                  |
| `world_model/training/docs/03_hero_run_1521m_v1v2v3.md` | Hero run notes (1.5B frames, v1+v2+v3)    |
| `milestones/README.md`                                | Curated results and media index             |

## Git Policy

Do not commit secrets, raw dataset shards, or model checkpoints. All large artifacts live on S3 and are referenced by URI.
