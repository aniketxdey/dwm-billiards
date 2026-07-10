from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from world_model_training.video_io import write_video

from .config import DEFAULT_PREVIEW_CONFIG
from .pipeline import InferenceEngine, build_prompt_from_config, load_engine, run_preview_from_config
from world_model_training import eval_rollout as wm_eval
try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None


_ENGINE_CACHE: Dict[Tuple[str, str, str], InferenceEngine] = {}


def _resolve_data_cfg(data_manifest_or_dir: str, sample_source: str) -> Dict[str, Any]:
    data_cfg = {
        "shards_dir": "",
        "shards_manifest": "",
        "train_shards_manifest": "",
        "val_shards_manifest": "",
        "eval_shards_manifest": "",
        "val_shards": 50,
        "sample_from": str(sample_source).lower().strip() or "eval",
    }
    p = str(data_manifest_or_dir).strip()
    if not p:
        return data_cfg
    if p.endswith(".txt") or p.endswith(".json") or p.endswith(".yaml") or p.endswith(".yml"):
        data_cfg["eval_shards_manifest"] = p
    else:
        data_cfg["shards_dir"] = p
    return data_cfg


def _make_aim_canvas(width: int = 640, height: int = 360) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = np.array([26, 92, 50], dtype=np.uint8)  # felt
    rail = np.array([61, 43, 28], dtype=np.uint8)
    img[:12, :] = rail
    img[-12:, :] = rail
    img[:, :12] = rail
    img[:, -12:] = rail

    cx, cy = width // 2, height // 2
    img[max(cy - 2, 0) : cy + 3, :] = np.array([225, 225, 225], dtype=np.uint8)
    img[:, max(cx - 2, 0) : cx + 3] = np.array([225, 225, 225], dtype=np.uint8)

    # cue-ball marker
    rr = 10
    yy, xx = np.ogrid[:height, :width]
    ball = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
    img[ball] = np.array([245, 245, 245], dtype=np.uint8)
    return img


def _get_engine(train_config_path: str, checkpoint_path: str, vae_checkpoint_path: str, device: str) -> InferenceEngine:
    key = (
        str(Path(train_config_path).resolve()),
        str(Path(checkpoint_path).resolve()),
        str(Path(vae_checkpoint_path).resolve()),
        str(device).lower(),
    )
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = load_engine(
            model_name="ui_preview_model",
            checkpoint_path=checkpoint_path,
            train_config_path=train_config_path,
            device=device,
            vae_cfg={
                "enabled": True,
                "checkpoint_path": vae_checkpoint_path,
                "base_channels": int(os.environ.get("WM_INF_VAE_BASE_CHANNELS", "64")),
                "latent_channels": 4,
            },
        )
    return _ENGINE_CACHE[key]


def _build_cfg(
    *,
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    device: str,
    horizon: int,
    ddim_steps: int,
    fps: int,
    preview_id: str,
    sample_source: str,
    data_manifest_or_dir: str,
    use_preset_actions: bool,
    preset_name: str,
    shot_frame: int,
    force_x: float,
    force_y: float,
    num_shots: int,
    max_force: float,
    seed: int,
) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_PREVIEW_CONFIG)
    cfg["run"].update(
        {
            "preview_id": preview_id,
            "seed": int(seed),
            "device": device,
            "horizon": int(horizon),
            "ddim_steps": int(ddim_steps),
            "video_fps": int(fps),
            "num_clips": 1,
            "output_root": "./world_model/inference/runs/ui_previews",
        }
    )
    cfg["model"].update(
        {
            "name": "ui_preview_model",
            "checkpoint_path": checkpoint_path,
            "train_config_path": train_config_path,
        }
    )
    cfg["vae"].update({"enabled": True, "checkpoint_path": vae_checkpoint_path})

    cfg["data"].update(_resolve_data_cfg(data_manifest_or_dir, sample_source))

    cfg["actions"]["source"] = "preset" if use_preset_actions else "dataset"
    cfg["actions"]["preset"] = {
        "name": preset_name,
        "horizon": int(horizon),
        "shot_frame": int(shot_frame),
        "force_x": float(force_x),
        "force_y": float(force_y),
        "seed": int(seed),
        "num_shots": int(num_shots),
        "max_force": float(max_force),
        "min_gap": 6,
    }
    return cfg


def _infer(
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    data_manifest_or_dir: str,
    device: str,
    sample_source: str,
    horizon: int,
    ddim_steps: int,
    fps: int,
    use_preset_actions: bool,
    preset_name: str,
    shot_frame: int,
    force_x: float,
    force_y: float,
    num_shots: int,
    max_force: float,
    seed: int,
    preview_id: str,
):
    engine = _get_engine(train_config_path, checkpoint_path, vae_checkpoint_path, device)
    cfg = _build_cfg(
        train_config_path=train_config_path,
        checkpoint_path=checkpoint_path,
        vae_checkpoint_path=vae_checkpoint_path,
        device=device,
        horizon=horizon,
        ddim_steps=ddim_steps,
        fps=fps,
        preview_id=preview_id,
        sample_source=sample_source,
        data_manifest_or_dir=data_manifest_or_dir,
        use_preset_actions=use_preset_actions,
        preset_name=preset_name,
        shot_frame=shot_frame,
        force_x=force_x,
        force_y=force_y,
        num_shots=num_shots,
        max_force=max_force,
        seed=seed,
    )
    artifacts = run_preview_from_config(cfg, engine)

    summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
    timeline_img = None
    if artifacts.action_timeline_paths:
        try:
            from PIL import Image

            timeline_img = np.asarray(Image.open(artifacts.action_timeline_paths[0]).convert("RGB"))
        except Exception:
            timeline_img = None
    video_path = str(artifacts.video_paths[0]) if artifacts.video_paths else None
    return video_path, summary, timeline_img, str(artifacts.output_dir)


def _session_init(
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    data_manifest_or_dir: str,
    device: str,
    sample_source: str,
    seed: int,
):
    t0 = time.time()
    engine = _get_engine(train_config_path, checkpoint_path, vae_checkpoint_path, device)
    data_cfg = _resolve_data_cfg(data_manifest_or_dir, sample_source)

    cfg = deepcopy(DEFAULT_PREVIEW_CONFIG)
    cfg["run"]["horizon"] = 8
    cfg["run"]["seed"] = int(seed)
    cfg["data"].update(data_cfg)
    cfg["actions"]["source"] = "dataset"

    prompt = build_prompt_from_config(cfg, engine)
    context = np.array(
        prompt.context_max[:1, -engine.bundle.context_len :],
        dtype=np.float32,
        copy=True,
    )
    dataset_actions = np.array(prompt.actions[:1], dtype=np.float32, copy=True)

    ctx_t = torch.from_numpy(context[:, -1:, ...]).float()
    prompt_frame_t = wm_eval._decode_latents(
        engine.vae,
        ctx_t,
        device=engine.device,
        batch_size=1,
    )[0, 0]
    prompt_frame = (
        torch.clamp(prompt_frame_t, 0.0, 1.0).permute(1, 2, 0).mul(255.0).byte().numpy()
    )

    session_id = f"live_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}"
    output_dir = Path("./world_model/inference/runs/live_sessions").resolve() / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "session_id": session_id,
        "output_dir": str(output_dir),
        "context": context,
        "dataset_actions": dataset_actions,
        "dataset_cursor": 0,
        "frames": [prompt_frame],
        "steps_generated": 0,
        "prompt_meta": prompt.meta[0] if prompt.meta else {},
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    log = {
        "event": "session_started",
        "session_id": session_id,
        "context_len": int(engine.bundle.context_len),
        "prompt_source": prompt.source,
        "prompt_meta": state["prompt_meta"],
        "startup_sec": round(time.time() - t0, 3),
        "output_dir": str(output_dir),
    }
    return state, prompt_frame, log, str(output_dir)


def _session_step(
    session_state: Dict[str, Any] | None,
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    device: str,
    ddim_steps: int,
    live_chunk: int,
    live_action_mode: str,
    live_trigger: bool,
    live_force_x: float,
    live_force_y: float,
):
    if not session_state:
        return session_state, None, {"error": "Start a live session first."}, ""

    t0 = time.time()
    engine = _get_engine(train_config_path, checkpoint_path, vae_checkpoint_path, device)
    chunk = int(max(1, live_chunk))

    context = np.array(session_state["context"], dtype=np.float32, copy=True)
    latent_shape = tuple(int(x) for x in context.shape[2:])
    actions = np.zeros((1, chunk, 3), dtype=np.float32)

    mode = str(live_action_mode).strip().lower()
    if mode == "dataset":
        ds_actions = np.array(session_state.get("dataset_actions", np.zeros((1, 1, 3), dtype=np.float32)))
        cursor = int(session_state.get("dataset_cursor", 0))
        max_idx = max(ds_actions.shape[1] - 1, 0)
        for i in range(chunk):
            idx = min(cursor + i, max_idx)
            actions[0, i] = ds_actions[0, idx]
        session_state["dataset_cursor"] = min(cursor + chunk, max_idx)
        action_logged = actions[0, 0].tolist()
    else:
        if bool(live_trigger):
            actions[0, 0, 0] = float(live_force_x)
            actions[0, 0, 1] = float(live_force_y)
            actions[0, 0, 2] = 1.0
        action_logged = actions[0, 0].tolist()

    clips = wm_eval.EvalClips(
        context_max=context,
        actions=actions,
        gt_future=np.zeros((1, chunk, *latent_shape), dtype=np.float32),
        meta=[dict(session_state.get("prompt_meta", {}))],
    )
    preds = wm_eval._rollout_predictions(
        bundle=engine.bundle,
        clips=clips,
        max_context_len=int(context.shape[1]),
        max_horizon=chunk,
        ddim_steps=int(ddim_steps),
        device=engine.device,
        batch_size=1,
    )
    pred_np = preds.numpy()
    new_context = np.concatenate([context[0], pred_np[0]], axis=0)[-engine.bundle.context_len :]
    session_state["context"] = new_context[None, ...].astype(np.float32, copy=False)

    decoded = wm_eval._decode_latents(
        engine.vae,
        preds,
        device=engine.device,
        batch_size=min(128, max(1, chunk)),
    )
    frame_batch = (
        torch.clamp(decoded[0], 0.0, 1.0).permute(0, 2, 3, 1).mul(255.0).byte().numpy()
    )

    frames_list = list(session_state.get("frames", []))
    for i in range(frame_batch.shape[0]):
        frames_list.append(frame_batch[i])
    session_state["frames"] = frames_list[-720:]  # cap memory to ~60s at 12 fps
    session_state["steps_generated"] = int(session_state.get("steps_generated", 0)) + chunk

    latest_frame = frame_batch[-1]
    out_dir = str(session_state.get("output_dir", ""))
    log = {
        "event": "step",
        "session_id": session_state.get("session_id", ""),
        "action_mode": mode,
        "action_t0": action_logged,
        "chunk": chunk,
        "steps_generated": int(session_state["steps_generated"]),
        "latency_sec": round(time.time() - t0, 3),
    }
    if torch.cuda.is_available() and str(device).lower() == "cuda":
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            log["gpu_mem_used_gb"] = round((total_b - free_b) / (1024**3), 2)
            log["gpu_mem_total_gb"] = round(total_b / (1024**3), 2)
        except Exception:
            pass
    return session_state, latest_frame, log, out_dir


def _aim_click_to_action(
    canvas_img: np.ndarray | None,
    live_max_force: float,
    evt: "gr.SelectData",
):
    if canvas_img is None:
        return 0.0, -8.0, True, {"event": "aim_set", "warning": "canvas image missing"}

    try:
        px = float(evt.index[0])  # x
        py = float(evt.index[1])  # y
    except Exception:
        return 0.0, -8.0, True, {"event": "aim_set", "warning": "could not parse click position"}

    h, w = int(canvas_img.shape[0]), int(canvas_img.shape[1])
    cx, cy = w * 0.5, h * 0.5
    nx = (px - cx) / max(cx, 1.0)
    ny = (py - cy) / max(cy, 1.0)
    mag = float(np.sqrt(nx * nx + ny * ny))
    if mag > 1.0:
        nx /= mag
        ny /= mag

    force_scale = float(max(0.0, live_max_force))
    fx = float(nx * force_scale)
    fy = float(ny * force_scale)
    log = {
        "event": "aim_set",
        "pixel_xy": [px, py],
        "force_xy": [round(fx, 3), round(fy, 3)],
        "max_force": float(force_scale),
    }
    return fx, fy, True, log


def _session_run_for_duration(
    session_state: Dict[str, Any] | None,
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    device: str,
    ddim_steps: int,
    live_chunk: int,
    live_action_mode: str,
    live_trigger: bool,
    live_force_x: float,
    live_force_y: float,
    live_seconds: float,
    live_fps_target: int,
):
    if not session_state:
        yield session_state, None, {"error": "Start a live session first."}, "", None
        return

    target_frames = int(max(1, round(float(live_seconds) * float(max(1, live_fps_target)))))
    chunk = int(max(1, live_chunk))
    generated = 0
    trigger_for_next = bool(live_trigger)

    while generated < target_frames:
        use_chunk = min(chunk, target_frames - generated)
        t0 = time.time()
        session_state, latest_frame, log, out_dir = _session_step(
            session_state=session_state,
            train_config_path=train_config_path,
            checkpoint_path=checkpoint_path,
            vae_checkpoint_path=vae_checkpoint_path,
            device=device,
            ddim_steps=int(ddim_steps),
            live_chunk=use_chunk,
            live_action_mode=live_action_mode,
            live_trigger=trigger_for_next,
            live_force_x=float(live_force_x),
            live_force_y=float(live_force_y),
        )
        if isinstance(log, dict):
            log["event"] = "run_10s_step"
            log["target_frames"] = int(target_frames)
            log["generated_frames"] = int(generated + use_chunk)

        generated += use_chunk
        trigger_for_next = False

        # best-effort pacing toward target fps (if generation is faster than playback)
        step_elapsed = time.time() - t0
        ideal = use_chunk / float(max(1, live_fps_target))
        sleep_s = max(0.0, ideal - step_elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)

        yield session_state, latest_frame, log, out_dir, None

    video_path, export_log = _session_export_video(session_state, live_fps_target)
    if isinstance(export_log, dict):
        export_log["event"] = "run_10s_done"
        export_log["target_seconds"] = float(live_seconds)
        export_log["target_frames"] = int(target_frames)
    yield session_state, latest_frame, export_log, str(session_state.get("output_dir", "")), video_path


def _session_export_video(session_state: Dict[str, Any] | None, fps: int):
    if not session_state:
        return None, {"error": "No live session found. Start session first."}

    frames = list(session_state.get("frames", []))
    if not frames:
        return None, {"error": "No frames in session buffer yet."}

    output_dir = Path(str(session_state.get("output_dir", "./world_model/inference/runs/live_sessions")))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "live_session.mp4"

    video = torch.from_numpy(np.stack(frames, axis=0))
    write_video(str(out_path), video, fps=int(max(1, fps)))
    log = {
        "event": "export_video",
        "session_id": session_state.get("session_id", ""),
        "frames": int(video.shape[0]),
        "fps": int(max(1, fps)),
        "video_path": str(out_path),
    }
    return str(out_path), log


def build_demo() -> "gr.Blocks":
    if gr is None:
        raise RuntimeError("gradio is not installed. Install with: pip install -r world_model/inference/requirements.txt")

    default_train_config = os.environ.get("WM_INF_DEFAULT_TRAIN_CONFIG", "").strip()
    default_checkpoint = os.environ.get("WM_INF_DEFAULT_CHECKPOINT", "").strip()
    default_vae_ckpt = os.environ.get("WM_INF_DEFAULT_VAE_CHECKPOINT", "").strip()
    default_data_path = os.environ.get("WM_INF_DEFAULT_DATA_PATH", "").strip()

    css = """
    .hero {background: linear-gradient(135deg, #0b1624 0%, #13253d 55%, #1b3b2f 100%); padding: 18px; border-radius: 16px; color: white;}
    .hint {opacity: .85; font-size: 13px;}
    """

    with gr.Blocks(css=css, title="Pool World-Model Inference Sandbox") as demo:
        gr.Markdown(
            """
            <div class='hero'>
              <h2 style='margin:0'>Pool World-Model Inference Sandbox</h2>
              <div class='hint'>Oasis-style inference flow: prompt context -> action stream -> DDIM rollout -> VAE decode.</div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                train_config = gr.Textbox(
                    label="Train Config Path",
                    placeholder="world_model/training/configs/...yaml",
                    value=default_train_config,
                )
                checkpoint = gr.Textbox(
                    label="World Model Checkpoint",
                    placeholder=".../ckpt_*.pt",
                    value=default_checkpoint,
                )
                vae_ckpt = gr.Textbox(
                    label="VAE Checkpoint",
                    placeholder=".../vae checkpoint",
                    value=default_vae_ckpt,
                )
                data_path = gr.Textbox(
                    label="Eval Shards Manifest or Shards Dir",
                    placeholder="world_model/training/manifests/.../eval_shards.txt OR /path/to/shards",
                    value=default_data_path,
                )
            with gr.Column(scale=1):
                device = gr.Dropdown(["cuda", "cpu"], value="cuda", label="Device")
                sample_source = gr.Dropdown(["eval", "train"], value="eval", label="Prompt Source")
                seed = gr.Number(value=42, precision=0, label="Seed")
                preview_id = gr.Textbox(label="Preview ID (optional)", placeholder="auto")

        with gr.Row():
            horizon = gr.Slider(8, 128, value=32, step=1, label="Horizon")
            ddim_steps = gr.Slider(5, 50, value=20, step=1, label="DDIM Steps")
            fps = gr.Slider(6, 30, value=12, step=1, label="Video FPS")

        with gr.Row():
            use_preset = gr.Checkbox(value=False, label="Override dataset actions with preset")
            preset_name = gr.Dropdown(["single_shot", "random_shots", "bank_left", "bank_right", "chaos_burst"], value="single_shot", label="Preset")
            num_shots = gr.Slider(1, 8, value=2, step=1, label="Num Shots (random presets)")
            max_force = gr.Slider(1.0, 20.0, value=12.0, step=0.25, label="Max Force")

        with gr.Row():
            shot_frame = gr.Slider(0, 127, value=0, step=1, label="Shot Frame")
            force_x = gr.Slider(-20.0, 20.0, value=0.0, step=0.25, label="Force X")
            force_y = gr.Slider(-20.0, 20.0, value=-8.0, step=0.25, label="Force Y")

        run_btn = gr.Button("Generate Preview", variant="primary")

        with gr.Row():
            video = gr.Video(label="Rollout Preview")
            timeline = gr.Image(label="Action Timeline", type="numpy")
        with gr.Row():
            summary = gr.JSON(label="Summary")
            out_dir = gr.Textbox(label="Output Directory")

        gr.Markdown("### Live Session (Stateful, User-Driven)")
        with gr.Row():
            live_action_mode = gr.Dropdown(
                ["manual", "dataset"],
                value="manual",
                label="Live Action Mode",
            )
            live_chunk = gr.Slider(1, 24, value=4, step=1, label="Frames Per Step")
            live_trigger = gr.Checkbox(value=True, label="Trigger On Next Step")
        with gr.Row():
            live_force_x = gr.Slider(-20.0, 20.0, value=3.5, step=0.25, label="Live Force X")
            live_force_y = gr.Slider(-20.0, 20.0, value=-8.5, step=0.25, label="Live Force Y")
            live_max_force = gr.Slider(2.0, 20.0, value=12.0, step=0.25, label="Canvas Max Force")
            live_export_fps = gr.Slider(6, 30, value=12, step=1, label="Export FPS")
        with gr.Row():
            live_seconds = gr.Slider(2, 20, value=10, step=1, label="Auto-Run Seconds")
            aim_canvas = gr.Image(
                label="Aim Canvas (click to set force from center cue ball)",
                value=_make_aim_canvas(),
                type="numpy",
                interactive=True,
            )
        with gr.Row():
            live_start_btn = gr.Button("Start Live Session", variant="secondary")
            live_step_btn = gr.Button("Step Live", variant="primary")
            live_run_btn = gr.Button("Run 10s Live", variant="primary")
            live_export_btn = gr.Button("Export Live Video", variant="secondary")
        with gr.Row():
            live_frame = gr.Image(label="Latest Live Frame", type="numpy")
            live_video = gr.Video(label="Exported Live Video")
        with gr.Row():
            live_log = gr.JSON(label="Live Logs")
            live_out_dir = gr.Textbox(label="Live Session Output")

        live_state = gr.State(value=None)

        run_btn.click(
            fn=_infer,
            inputs=[
                train_config,
                checkpoint,
                vae_ckpt,
                data_path,
                device,
                sample_source,
                horizon,
                ddim_steps,
                fps,
                use_preset,
                preset_name,
                shot_frame,
                force_x,
                force_y,
                num_shots,
                max_force,
                seed,
                preview_id,
            ],
            outputs=[video, summary, timeline, out_dir],
        )

        live_start_btn.click(
            fn=_session_init,
            inputs=[
                train_config,
                checkpoint,
                vae_ckpt,
                data_path,
                device,
                sample_source,
                seed,
            ],
            outputs=[live_state, live_frame, live_log, live_out_dir],
        )

        live_step_btn.click(
            fn=_session_step,
            inputs=[
                live_state,
                train_config,
                checkpoint,
                vae_ckpt,
                device,
                ddim_steps,
                live_chunk,
                live_action_mode,
                live_trigger,
                live_force_x,
                live_force_y,
            ],
            outputs=[live_state, live_frame, live_log, live_out_dir],
        )

        live_run_btn.click(
            fn=_session_run_for_duration,
            inputs=[
                live_state,
                train_config,
                checkpoint,
                vae_ckpt,
                device,
                ddim_steps,
                live_chunk,
                live_action_mode,
                live_trigger,
                live_force_x,
                live_force_y,
                live_seconds,
                live_export_fps,
            ],
            outputs=[live_state, live_frame, live_log, live_out_dir, live_video],
        )

        live_export_btn.click(
            fn=_session_export_video,
            inputs=[live_state, live_export_fps],
            outputs=[live_video, live_log],
        )

        aim_canvas.select(
            fn=_aim_click_to_action,
            inputs=[aim_canvas, live_max_force],
            outputs=[live_force_x, live_force_y, live_trigger, live_log],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Gradio app for world-model inference previews")
    parser.add_argument("--host", default=os.environ.get("WM_INF_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WM_INF_PORT", "7860")))
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=int(args.port), show_error=True)


if __name__ == "__main__":
    main()
