# World Model Milestone

Folders:
- `baseline_vs_df/`: side-by-side baseline vs diffusion-forcing comparison.
- `checkpoint_progression/`: qualitative progression from 686M to 1.52B and resumed run.
- `long_rollout/`: long-horizon stability clips and summaries.
- `ddim_steps_comparison/`: same checkpoint/prompt rendered with different DDIM steps.

Key point for presentation:
- DF materially improved long-horizon stability versus baseline in our early comparisons.
- Larger/resumed checkpoints improved short/mid horizon while long-horizon artifacts still exist.
