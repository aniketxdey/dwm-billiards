# DDIM Steps Comparison (Same Checkpoint, Same Prompt)

This folder isolates the effect of DDIM step count on inference quality/speed.

## What was held constant
- Same world-model checkpoint: `ckpt_360000000.pt` (resume360 1.521B run).
- Same VAE checkpoint: `ckpt_060000000.pt`.
- Same sampled clip (`seed=42`, same shard/episode/anchor).
- Same horizon: `64` frames.

Only `ddim_steps` changed.

## Files
- `ddim08_8s.mp4`
- `ddim12_8s.mp4`
- `ddim20_8s.mp4`
- `ddim30_8s.mp4`
- `ddim_steps_2x2_8s.mp4` (quick visual compare in one file)
- `ddim_steps_1x4_8s.mp4` (left-to-right compare)
- `actions_timeline.png` (action sequence used for this clip)
- `summary_ddim08.json`, `summary_ddim12.json`, `summary_ddim20.json`, `summary_ddim30.json`
- `ddim_metrics_snapshot.json`, `ddim_metrics_snapshot.csv`

## Reading order for demo
1. Play `ddim_steps_1x4_8s.mp4` to compare behavior at a glance.
2. If needed, open each per-step clip individually.
3. Use `ddim_metrics_snapshot.csv` to discuss speed/quality tradeoff numerically.

## Quick speed note (from summaries)
- DDIM 8: ~4.07s rollout
- DDIM 12: ~6.06s rollout
- DDIM 20: ~10.00s rollout
- DDIM 30: ~14.81s rollout

Inference cost scales up with DDIM steps; quality gains are scenario-dependent and not always monotonic for every single clip.
