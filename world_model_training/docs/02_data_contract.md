# Latent Data Contract For World Model

Each shard file:
- `latent_shard_XXXXX.npz`

Required arrays:
- `latents`: shape `[E, T, C, H, W]`, dtype `float16`
- `actions`: shape `[E, T, A]`, dtype `float32`
- `lengths`: shape `[E]`, dtype `int32`

Optional arrays (ignored by current baseline trainer):
- `sim_state`: shape `[E, T, 16, 4]`, dtype `float32`

## Sample Dimensions In Current Dataset
- `E=100` episodes per shard
- `T=600` frames per episode
- `C=4, H=9, W=16` latent shape
- `A=3` action dimensions

## Training Sample Definition
For each valid time index `t`:
- Context: `latents[t-L+1:t+1]`
- Action: `actions[t]`
- Target latent: `latents[t+1]`

With `L=8`, valid range is:
- `t in [7, length-2]`

## Shard Split Policy
- Train shards: all but final `val_shards`
- Val shards: final `val_shards` by shard index

This split is deterministic and controlled by config.
