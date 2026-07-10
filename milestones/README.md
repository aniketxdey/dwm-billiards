# Milestones Preview

This folder is a presentation-ready walkthrough of the project pipeline and results.

## Structure
- `data_collection/`: raw-data generation examples and dataset scale stats.
- `vae/`: VAE reconstruction quality samples and latent comparison.
- `world_model/`: DiT baseline vs DF comparisons and checkpoint progression clips.
  - Includes `ddim_steps_comparison/` for inference speed/quality tradeoff demos.
- `charts/`: extracted metric charts for quick explanation in slides/demo.
- `run_registry.json`: canonical run names/IDs used during milestones.

## Suggested demo order
1. Data collection scale + sample episodes.
2. VAE recon quality (`real | reconstruction`) and latent comparison clip.
3. World model: baseline vs DF side-by-side.
4. Checkpoint progression (686M -> 1.52B -> 1.52B resume360).
5. Long-horizon rollout (`h128` clip) and metric charts.
