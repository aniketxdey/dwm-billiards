# Billiards Physics Diffusion World Model

A diffusion-based world model to predicts future frames of a 2D billiards game from actions alone, with accurate physics. COmplete with playable simulator.

https://github.com/user-attachments/assets/f10ec87d-ae13-4e87-a8cb-d7a821f4ac3a


## Methodology

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
data_generation_package/       # v1 data generator (pymunk simulation + shard export)
data_generation_package_v2/    # v2: diverse physics, spawn layouts, bot archetypes
data_generation_package_v3/    # v3: collision-heavy biasing, breaker bots
VAE training/                  # ConvVAE model, training loop, latent export pipeline
world_model_training/          # DiT model, diffusion, DF training, distillation, rollout eval
world_model_inference/         # Inference pipeline, Gradio UI, FastAPI live play
docs/                          # Project docs, LaTeX essay, Beamer slides
milestones_preview/            # Curated metrics, eval summaries, media references
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
bash "VAE training/scripts/stage_shards_from_s3.sh"
```

**3. Train VAE**

```bash
RUN_ID="$(RUN_PREFIX=vae_60m_1xa100 bash "VAE training/scripts/new_run_id.sh" run01)"
RUN_ID="$RUN_ID" bash "VAE training/scripts/run_train_60m_1xa100.sh"
```

**4. Export latents**

```bash
bash "VAE training/scripts/export_latents_60m.sh"
```

**5. Train world model**

```bash
RUN_ID="$(RUN_PREFIX=dit_5m_1xa100 bash world_model_training/scripts/new_run_id.sh run01)"
RUN_ID="$RUN_ID" bash world_model_training/scripts/run_train_1xa100.sh
```

**6. Run inference**

```bash
bash world_model_inference/scripts/run_preview.sh       # CLI preview
bash world_model_inference/scripts/run_ui.sh            # Gradio UI
bash world_model_inference/scripts/run_live_play.sh     # Real-time WebSocket play
```

## Data

Raw datasets are stored on S3 (not in git):

- **v1:** `s3://videogen-pool-v2-237586137680/dataraw/full_20260220_112101/`
- **v2:** `s3://videogen-pool-v2-237586137680/dataraw_v2/full_v2diverse_20260222_154743/`
- **v3:** `s3://videogen-pool-v2-237586137680/dataraw_v3/full_v3collision_20260223_094313/`

Shard format (`.npz`): `frames [N,T,72,128,3] uint8`, `actions [N,T,3] float32`, `sim_state [N,T,16,4] float32`, `lengths [N,] int32`, `episode_meta [N,] object`.

See `docs/S3_ACCESS.md` for credentials and staging instructions.

## Evaluation

Rollout evaluation compares models via autoregressive DDIM generation over increasing horizons (1–256 frames). Metrics include **latent MSE/MAE** and **frame-level PSNR/L1** (after VAE decode). Side-by-side comparison videos are generated automatically. See `milestones_preview/` for curated metric snapshots and `docs/PROJECT_STATUS_2026-03-04.md` for the latest results summary.

## Documentation

| Document                                    | Description                                 |
| ------------------------------------------- | ------------------------------------------- |
| `docs/PROJECT_JOURNEY.md`                 | Chronological project narrative             |
| `docs/PROJECT_STATUS_2026-03-04.md`       | Latest experiment status and next steps     |
| `docs/WORLD_MODEL_DECISION_2026-02-21.md` | Baseline context-length ablation decision   |
| `docs/pipeline_math_mini.tex`             | Tensor shapes, loss functions, and DDP math |
| `docs/final_project_essay/`               | Full academic write-up                      |
| `docs/final_project_slides/`              | Beamer presentation decks                   |

## Git Policy

Do not commit secrets, raw dataset shards, or model checkpoints. All large artifacts live on S3 and are referenced by URI. See `docs/GITHUB_PUSH_CHECKLIST.md`.
