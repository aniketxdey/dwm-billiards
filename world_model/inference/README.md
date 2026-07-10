# World Model Inference (Preparation)

This folder prepares the inference/runtime side while training continues.

It follows the same high-level structure used in Oasis-style generation:
1. load world model + VAE
2. build prompt context latents
3. build/align action stream
4. autoregressive DDIM rollout in latent space
5. decode latents to frames/video

## Why this exists
- Keep inference code separate from training/eval code paths.
- Reuse the validated rollout/DDIM implementation from `world_model_training.eval_rollout` so previews and inference stay consistent.
- Prepare a visual/interactive sandbox before the realtime game loop is built.

## Layout
- `src/world_model_inference/pipeline.py`: reusable inference wrapper (loads model/VAE, samples prompt, rolls out, decodes)
- `src/world_model_inference/preview_checkpoint.py`: CLI to generate rollout preview videos from a checkpoint
- `src/world_model_inference/ui_app.py`: interactive Gradio sandbox (optional dependency)
- `src/world_model_inference/live_play.py`: low-latency websocket live-play route (canvas drag/release)
- `src/world_model_inference/action_presets.py`: action sequence presets for `[force_x, force_y, trigger]`
- `src/world_model_inference/viz.py`: action timeline renderer (PNG)
- `configs/preview_checkpoint_template.yaml`: preview config template
- `scripts/run_preview.sh`: shell wrapper for CLI preview
- `scripts/run_ui.sh`: shell wrapper for interactive UI
- `scripts/run_live_play.sh`: shell wrapper for websocket live-play app
- `scripts/benchmark_ddim_matrix.sh`: batch runner for DDIM step comparisons from one base config

## Action Schema (current project)
The world model uses the same action schema as the data generator and latent shards:
- `action[0] = force_x`
- `action[1] = force_y`
- `action[2] = trigger` (`1.0` on shot frame, otherwise `0.0`)

## Quick Preview (offline)
Generate a checkpoint preview (single clip by default):

```bash
CONFIG_PATH=world_model/inference/configs/preview_checkpoint_template.yaml \
  bash world_model/inference/scripts/run_preview.sh
```

Outputs go to:
- `world_model/inference/runs/<preview_id>/summary.json`
- `world_model/inference/runs/<preview_id>/videos/`
- `world_model/inference/runs/<preview_id>/actions/`

## Interactive Sandbox UI (optional)
Install optional UI deps and run:

```bash
pip install -r world_model/inference/requirements.txt
bash world_model/inference/scripts/run_ui.sh
```

The UI lets you:
- point to a checkpoint + train config + VAE checkpoint
- sample a prompt from eval/train shards
- override dataset actions with presets (`single_shot`, `random_shots`, `bank_left`, `bank_right`, `chaos_burst`)
- generate a rollout preview video and action timeline
- run a stateful **Live Session**:
  - `Start Live Session` to lock prompt/context
  - `Step Live` to generate the next N frames from user action
  - `Live Action Mode = manual` for shot controls (`force_x`, `force_y`, trigger)
  - `Live Action Mode = dataset` to replay recorded actions
  - click **Aim Canvas** to set shot vector from center cue ball
  - `Run 10s Live` to auto-step for a timed interactive clip and auto-export video
  - `Export Live Video` to save the accumulated session clip with logs

## Remote UI On Lambda (recommended for play)
From your local machine, launch the UI on the GPU box and tunnel it locally:

```bash
bash world_model/inference/scripts/run_remote_ui_tunnel.sh
```

Then open:
- `http://127.0.0.1:7860`

Defaults are pre-filled for the latest resume360 checkpoint, VAE checkpoint, and eval manifest.
Override if needed:

```bash
REMOTE_SSH="ubuntu@192.222.52.102" \
REMOTE_KEY_PATH="./bill-diff.pem" \
LOCAL_PORT=7861 \
REMOTE_PORT=7860 \
WM_INF_DEFAULT_CHECKPOINT="/home/ubuntu/maat/world_model/training/runs/<run_id>/checkpoints/ckpt_360000000.pt" \
bash world_model/inference/scripts/run_remote_ui_tunnel.sh
```

## Minimal Game Route (start button + rolling canvas + click actions)
For the stripped interface you asked for:

```bash
bash world_model/inference/scripts/run_remote_game_tunnel.sh
```

Then open:
- `http://127.0.0.1:7862`

Flow:
1. `Start Game`
2. Generation rolls frame-by-frame automatically
3. Click canvas once to **pause + set drag start**
4. Click second point within ~2s to **commit action + resume**
5. Max 2 interactions per session
6. `Export` (writes video + action logs json)

## Live Streaming Play (click-drag-release on canvas)
For lower-latency interaction than Gradio image updates, use the websocket route:

```bash
bash world_model/inference/scripts/run_remote_live_play_tunnel.sh
```

Then open:
- `http://127.0.0.1:7863`

Behavior:
1. `Start` loads prompt context and begins continuous generation loop
2. `Click -> drag -> release` on canvas queues one shot action (`force_x`, `force_y`, trigger=1)
3. New frames stream over websocket as JPEGs
4. Tune `Mode`, `DDIM`, `FPS`, `MaxForce`, `JPEG`, `Scale` from the top bar
5. Optional toggles:
   - `compile`: tries `torch.compile` for the model at start (falls back safely if unsupported)
   - `bench`: writes per-frame timing logs and a session summary

Notes:
- First start still has model warmup cost (checkpoint load to GPU).
- For best responsiveness: DDIM `5-8`, FPS `10-20`.
- For visible action impact, use `MaxForce` in roughly `80-220` (dataset shots are much larger than `~30`).
- `Mode` presets:
  - `fast`: lower DDIM, higher FPS, lower JPEG quality
  - `balanced`: use exactly your requested values
  - `quality`: higher DDIM, lower FPS, higher JPEG quality

Benchmark outputs (when `bench` is enabled):
- default root: `world_model/inference/runs/live_play_bench/`
- per-session files:
  - `manifest.json`
  - `timings.jsonl`
  - `actions.jsonl` (queued + consumed action telemetry with `fx`, `fy`, `mag`)
  - `summary.json`
- override root with `WM_LIVE_BENCH_ROOT=/path/to/bench_root`

## DDIM Matrix Benchmark (offline previews)
Run one preview per DDIM step from a single base config:

```bash
BASE_CONFIG=world_model/inference/configs/preview_latest_resume360_dataset.yaml \
STEPS_CSV=4,6,8,12,20,30 \
HORIZON=64 \
NUM_CLIPS=1 \
PREFIX=ddim_matrix \
bash world_model/inference/scripts/benchmark_ddim_matrix.sh
```

Outputs:
- generated configs: `world_model/inference/configs/ddim_matrix/`
- preview runs: `world_model/inference/runs/ddim_matrix/`

## Notes / Constraints
- On a single machine with both GPUs saturated by training, preview generation should run at checkpoint boundaries (or on another instance) to avoid OOM/interference.
- This package currently reuses internal functions from `world_model_training.eval_rollout`; if we stabilize the API later, we can promote them into shared public inference utilities.
