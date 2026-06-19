from __future__ import annotations

"""Minimal Gradio game route for step-wise interactive world-model play.

The route keeps a lightweight state machine:
- `rolling`: auto-generate one frame per timer tick
- `capture`: paused while waiting for drag end click
- `stopped`: no further generation
"""

import argparse
import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from world_model_training.video_io import write_video
from PIL import Image, ImageDraw, ImageFont

from .config import DEFAULT_PREVIEW_CONFIG
from .pipeline import InferenceEngine, build_prompt_from_config, load_engine
from world_model_training import eval_rollout as wm_eval

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None


_ENGINE_CACHE: Dict[Tuple[str, str, str, str], InferenceEngine] = {}
HARD_MIN_ACTION_HOLD_FRAMES = 6
ACTION_VIS_RADIUS_PX = 14
DEFAULT_DISPLAY_SCALE = 10


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp scalar to [lo, hi]."""
    return max(lo, min(hi, v))


def _draw_disc(img: np.ndarray, x: float, y: float, radius: int, color: tuple[int, int, int]) -> None:
    """Rasterize a filled disc on RGB numpy image."""
    h, w = int(img.shape[0]), int(img.shape[1])
    cx = int(round(_clamp(float(x), 0.0, float(max(0, w - 1)))))
    cy = int(round(_clamp(float(y), 0.0, float(max(0, h - 1)))))
    r = int(max(1, radius))

    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * r)
    patch = img[y0:y1, x0:x1]
    patch[mask] = np.array(color, dtype=np.uint8)


def _blend_disc(
    img: np.ndarray,
    x: float,
    y: float,
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Rasterize a filled translucent disc on RGB numpy image."""
    h, w = int(img.shape[0]), int(img.shape[1])
    cx = int(round(_clamp(float(x), 0.0, float(max(0, w - 1)))))
    cy = int(round(_clamp(float(y), 0.0, float(max(0, h - 1)))))
    r = int(max(1, radius))
    a = float(_clamp(float(alpha), 0.0, 1.0))

    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0 or a <= 0.0:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * r)
    if not np.any(mask):
        return

    patch = img[y0:y1, x0:x1].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    patch[mask] = patch[mask] * (1.0 - a) + color_arr * a
    img[y0:y1, x0:x1] = np.clip(patch, 0.0, 255.0).astype(np.uint8)


def _draw_line(
    img: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw thick line by sampling points and stamping discs."""
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    steps = int(max(abs(dx), abs(dy), 1.0))
    xs = np.linspace(float(x0), float(x1), steps + 1)
    ys = np.linspace(float(y0), float(y1), steps + 1)
    rr = int(max(1, thickness))
    for xx, yy in zip(xs, ys):
        _draw_disc(img, xx, yy, rr, color)


def _upscale_nn(img: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor integer upscale for crisp low-res rendering."""
    s = int(max(1, scale))
    if s == 1:
        return img
    return np.repeat(np.repeat(img, s, axis=0), s, axis=1)


def _get_base_frame(state: Dict[str, Any] | None) -> np.ndarray:
    """Return latest generated frame or fallback canvas."""
    if state and state.get("frames"):
        return np.array(state["frames"][-1], dtype=np.uint8, copy=True)
    return _make_table_canvas()


def _render_display_frame(state: Dict[str, Any] | None, base_frame: np.ndarray | None) -> np.ndarray:
    """Overlay interaction guides on top of latest frame for UI display."""
    img = np.array(base_frame if base_frame is not None else _make_table_canvas(), dtype=np.uint8, copy=True)
    if not state:
        return img

    # Current capture point while paused for second click.
    start_xy = state.get("capture_start_xy")
    if isinstance(start_xy, list) and len(start_xy) == 2 and str(state.get("phase", "")) == "capture":
        _blend_disc(
            img,
            float(start_xy[0]),
            float(start_xy[1]),
            radius=ACTION_VIS_RADIUS_PX,
            color=(255, 40, 40),
            alpha=0.26,
        )
        _draw_disc(img, float(start_xy[0]), float(start_xy[1]), radius=4, color=(230, 165, 60))
        _draw_disc(img, float(start_xy[0]), float(start_xy[1]), radius=2, color=(255, 255, 255))

    # Last committed drag action (short-lived visual).
    now_ts = float(time.time())
    drag = state.get("overlay_drag")
    if isinstance(drag, dict):
        expire_ts = float(drag.get("expire_ts", 0.0))
        if expire_ts >= now_ts:
            sx, sy = drag.get("start_xy", [0.0, 0.0])
            ex, ey = drag.get("end_xy", [0.0, 0.0])
            _blend_disc(img, float(sx), float(sy), radius=ACTION_VIS_RADIUS_PX, color=(255, 40, 40), alpha=0.26)
            _blend_disc(img, float(ex), float(ey), radius=ACTION_VIS_RADIUS_PX, color=(255, 40, 40), alpha=0.22)
            _draw_line(img, float(sx), float(sy), float(ex), float(ey), color=(240, 140, 40), thickness=1)
            _draw_disc(img, float(sx), float(sy), radius=3, color=(250, 205, 90))
            _draw_disc(img, float(ex), float(ey), radius=3, color=(255, 255, 255))
        else:
            state["overlay_drag"] = None

    display_scale = int(max(1, int(state.get("display_scale", 1))))
    return _upscale_nn(img, display_scale)


def _action_hud_text(state: Dict[str, Any] | None) -> str:
    """Compact action/debug text shown outside gameplay area."""
    if not state:
        return "Action HUD: idle"
    phase = str(state.get("phase", "rolling"))
    frame_idx = int(state.get("steps_generated", 0))
    src = str(state.get("last_action_source", "idle"))
    rem = int(state.get("action_repeat_remaining", 0))
    used = int(state.get("interactions_used", 0))
    max_used = int(state.get("max_interactions", 0))
    queued = state.get("last_queued_action", [0.0, 0.0, 0.0])
    applied = state.get("last_applied_action", [0.0, 0.0, 0.0])
    mag = float(np.sqrt(float(applied[0]) * float(applied[0]) + float(applied[1]) * float(applied[1])))
    return (
        "Action HUD: "
        f"frame={frame_idx} phase={phase} src={src} rem={rem} interactions={used}/{max_used} | "
        f"queued=({float(queued[0]):+.1f},{float(queued[1]):+.1f}) "
        f"applied=({float(applied[0]):+.1f},{float(applied[1]):+.1f}) |a|={mag:.1f}"
    )


def _append_console_line(state: Dict[str, Any] | None, line: str, max_lines: int = 240) -> None:
    """Append one timestamped line to rolling in-memory console."""
    if not state:
        return
    lines = list(state.get("console_lines", []))
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    lines.append(f"[{stamp}] {line}")
    state["console_lines"] = lines[-int(max(20, max_lines)) :]


def _console_text(state: Dict[str, Any] | None, last_n: int = 30) -> str:
    """Return tail of console lines as plain text for UI."""
    if not state:
        return ""
    lines = list(state.get("console_lines", []))
    return "\n".join(lines[-int(max(5, last_n)) :])


def _draw_action_hud(img: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    """Draw a compact fixed overlay strip with runtime action info."""
    out = np.array(img, dtype=np.uint8, copy=True)
    h, w = out.shape[:2]
    panel_w = int(max(210, min(w - 12, 420)))
    panel_h = int(max(44, min(62, int(0.16 * h))))
    x0 = 8
    y1 = h - 8
    y0 = max(4, y1 - panel_h)
    x1 = min(w - 4, x0 + panel_w)
    if x1 <= x0 or y1 <= y0:
        return out

    # Dark translucent panel.
    panel = out[y0:y1, x0:x1].astype(np.float32)
    panel = panel * 0.35 + np.array([10.0, 10.0, 10.0], dtype=np.float32) * 0.65
    out[y0:y1, x0:x1] = np.clip(panel, 0.0, 255.0).astype(np.uint8)

    # Border (green when action is actively being injected).
    rem = int(state.get("action_repeat_remaining", 0))
    border = (40, 190, 90) if rem > 0 else (135, 135, 135)
    out[y0:y0 + 2, x0:x1] = border
    out[y1 - 2:y1, x0:x1] = border
    out[y0:y1, x0:x0 + 2] = border
    out[y0:y1, x1 - 2:x1] = border

    queued = state.get("last_queued_action", [0.0, 0.0, 0.0])
    applied = state.get("last_applied_action", [0.0, 0.0, 0.0])
    phase = str(state.get("phase", "rolling"))
    frame_idx = int(state.get("steps_generated", 0))
    src = str(state.get("last_action_source", "idle"))
    mag = float(np.sqrt(float(applied[0]) * float(applied[0]) + float(applied[1]) * float(applied[1])))
    used = int(state.get("interactions_used", 0))
    max_used = int(state.get("max_interactions", 0))

    try:
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        font = ImageFont.load_default()
        lines = [
            f"f:{frame_idx} {phase[:4]} src:{src[:8]} rem:{rem} int:{used}/{max_used}",
            f"q({float(queued[0]):+.1f},{float(queued[1]):+.1f}) a({float(applied[0]):+.1f},{float(applied[1]):+.1f}) |a|:{mag:.1f}",
        ]
        yy = y0 + 6
        for line in lines:
            draw.text((x0 + 8, yy), line, fill=(240, 240, 240), font=font)
            yy += 16
        out = np.array(pil)
    except Exception:
        # Keep overlay box even if PIL text fails.
        pass

    return out


def _resolve_data_cfg(data_manifest_or_dir: str, sample_source: str) -> Dict[str, Any]:
    """Normalize data path input into preview-compatible data config."""
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


def _get_engine(train_config_path: str, checkpoint_path: str, vae_checkpoint_path: str, device: str) -> InferenceEngine:
    """Get cached inference engine or load it once per unique key."""
    key = (
        str(Path(str(train_config_path).strip()).resolve()),
        str(Path(str(checkpoint_path).strip()).resolve()),
        str(Path(str(vae_checkpoint_path).strip()).resolve()),
        str(device).lower(),
    )
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = load_engine(
            model_name="game_route_model",
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


def _make_table_canvas(width: int = 640, height: int = 360) -> np.ndarray:
    """Create fallback pool-table canvas before first decoded frame."""
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
    rr = 10
    yy, xx = np.ogrid[:height, :width]
    ball = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
    img[ball] = np.array([245, 245, 245], dtype=np.uint8)
    return img


def _decode_one_frame(vae, latent_1x1: torch.Tensor, device: torch.device) -> np.ndarray:
    """Decode a single latent frame [1,1,C,H,W] -> uint8 RGB image."""
    frame_t = wm_eval._decode_latents(vae, latent_1x1, device=device, batch_size=1)[0, 0]
    return torch.clamp(frame_t, 0.0, 1.0).permute(1, 2, 0).mul(255.0).byte().numpy()


def _start_game(
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    data_manifest_or_dir: str,
    device: str,
    sample_source: str,
    seed: int,
    display_scale: int,
    action_hold_frames: int,
    max_interactions: int,
    action_gain: float,
):
    """Initialize session state, sample prompt, and decode first frame."""
    t0 = time.time()
    try:
        engine = _get_engine(train_config_path, checkpoint_path, vae_checkpoint_path, device)
    except Exception as e:
        return None, _make_table_canvas(), {
            "error": "Engine initialization failed",
            "detail": str(e),
            "hint": "Open Advanced and set valid file paths for Train Config, Checkpoint, and VAE Checkpoint.",
        }, "", _action_hud_text(None), ""

    cfg = dict(DEFAULT_PREVIEW_CONFIG)
    cfg["run"] = dict(DEFAULT_PREVIEW_CONFIG["run"])
    cfg["data"] = dict(DEFAULT_PREVIEW_CONFIG["data"])
    cfg["actions"] = dict(DEFAULT_PREVIEW_CONFIG["actions"])
    cfg["run"]["horizon"] = 8
    cfg["run"]["seed"] = int(seed)
    cfg["data"].update(_resolve_data_cfg(data_manifest_or_dir, sample_source))
    cfg["actions"]["source"] = "dataset"

    prompt = build_prompt_from_config(cfg, engine)
    context = np.array(
        prompt.context_max[:1, -engine.bundle.context_len :],
        dtype=np.float32,
        copy=True,
    )
    ctx_t = torch.from_numpy(context[:, -1:, ...]).float()
    first_frame = _decode_one_frame(engine.vae, ctx_t, engine.device)

    session_id = f"game_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}"
    output_dir = Path("./world_model_inference/runs/game_sessions").resolve() / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "session_id": session_id,
        "output_dir": str(output_dir),
        "context": context,
        "frames": [first_frame],
        "steps_generated": 0,
        "running": True,
        "phase": "rolling",  # rolling | capture | stopped
        "pending_action": [0.0, 0.0, 0.0],
        "capture_start_xy": None,
        "capture_started_ts": 0.0,
        "interactions_used": 0,
        "max_interactions": int(max(1, int(max_interactions))),
        "action_hold_frames": int(max(HARD_MIN_ACTION_HOLD_FRAMES, int(action_hold_frames))),
        "action_gain": float(max(0.1, float(action_gain))),
        "action_repeat_remaining": 0,
        "action_events": [],
        "frame_action_log": [],
        "console_lines": [],
        "display_scale": int(max(1, int(display_scale))),
        "overlay_drag": None,
        "last_queued_action": [0.0, 0.0, 0.0],
        "last_applied_action": [0.0, 0.0, 0.0],
        "last_action_source": "idle",
        "prompt_meta": prompt.meta[0] if prompt.meta else {},
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    display_frame = _render_display_frame(state, first_frame)
    log = {
        "event": "start_game",
        "session_id": session_id,
        "startup_sec": round(time.time() - t0, 3),
        "display_scale": int(state["display_scale"]),
        "action_hold_frames": int(state["action_hold_frames"]),
        "max_interactions": int(state["max_interactions"]),
        "action_gain": float(state["action_gain"]),
        "prompt_meta": state["prompt_meta"],
        "output_dir": str(output_dir),
    }
    _append_console_line(
        state,
        (
            "start "
            f"ddim=? hold={int(state['action_hold_frames'])} "
            f"(hard-min={HARD_MIN_ACTION_HOLD_FRAMES}) "
            f"gain={float(state['action_gain']):.2f} max_interactions={int(state['max_interactions'])}"
        ),
    )
    return state, display_frame, log, str(output_dir), _action_hud_text(state), _console_text(state)


def _queue_click_action(
    state: Dict[str, Any] | None,
    canvas_img: np.ndarray | None,
    max_force: float,
    evt: "gr.SelectData",
):
    """Handle click-drag interaction as a two-click capture workflow."""
    base_frame = _get_base_frame(state)
    if not state:
        return state, _render_display_frame(state, base_frame), {"error": "Start game first."}, _action_hud_text(state), _console_text(state)
    if canvas_img is None:
        return state, _render_display_frame(state, base_frame), {"error": "Canvas missing."}, _action_hud_text(state), _console_text(state)

    try:
        px_display = float(evt.index[0])
        py_display = float(evt.index[1])
    except Exception:
        return state, _render_display_frame(state, base_frame), {"error": "Click parse failed."}, _action_hud_text(state), _console_text(state)

    h_model, w_model = int(base_frame.shape[0]), int(base_frame.shape[1])
    # Map click from displayed canvas resolution back to model-frame resolution.
    display_h, display_w = int(canvas_img.shape[0]), int(canvas_img.shape[1])
    if display_w > 1 and display_h > 1 and w_model > 1 and h_model > 1:
        px = _clamp(px_display * float(w_model - 1) / float(display_w - 1), 0.0, float(w_model - 1))
        py = _clamp(py_display * float(h_model - 1) / float(display_h - 1), 0.0, float(h_model - 1))
    else:
        # Fallback to configured display scale if dimensions are degenerate.
        scale = int(max(1, int(state.get("display_scale", 1))))
        px = _clamp(px_display / float(scale), 0.0, float(max(0, w_model - 1)))
        py = _clamp(py_display / float(scale), 0.0, float(max(0, h_model - 1)))

    if int(state.get("interactions_used", 0)) >= int(state.get("max_interactions", 2)):
        _append_console_line(state, "interaction ignored: max interactions reached")
        return state, _render_display_frame(state, base_frame), {
            "event": "interaction_ignored",
            "reason": "max_interactions_reached",
            "max_interactions": int(state.get("max_interactions", 2)),
        }, _action_hud_text(state), _console_text(state)

    phase = str(state.get("phase", "rolling"))
    if phase == "rolling":
        state["phase"] = "capture"
        state["capture_start_xy"] = [px, py]
        state["capture_started_ts"] = float(time.time())
        _append_console_line(state, f"capture start at ({px:.1f},{py:.1f})")
        return state, _render_display_frame(state, base_frame), {
            "event": "capture_started",
            "click_start_xy": [px, py],
            "click_display_xy": [px_display, py_display],
            "display_size_hw": [display_h, display_w],
            "model_size_hw": [h_model, w_model],
            "hint": "Generation paused. Click second point to commit action and resume.",
            "interactions_used": int(state.get("interactions_used", 0)),
            "max_interactions": int(state.get("max_interactions", 2)),
        }, _action_hud_text(state), _console_text(state)

    if phase != "capture":
        return state, _render_display_frame(state, base_frame), {"event": "click_ignored", "phase": phase}, _action_hud_text(state), _console_text(state)

    start_xy = state.get("capture_start_xy")
    if not (isinstance(start_xy, list) and len(start_xy) == 2):
        state["phase"] = "rolling"
        _append_console_line(state, "capture aborted: missing start point")
        return state, _render_display_frame(state, base_frame), {"event": "capture_aborted", "reason": "missing_start_point"}, _action_hud_text(state), _console_text(state)

    sx, sy = float(start_xy[0]), float(start_xy[1])
    ex, ey = float(px), float(py)
    dx = ex - sx
    dy = ey - sy
    dist = float(np.sqrt(dx * dx + dy * dy))

    h, w = h_model, w_model
    norm = max(float(np.sqrt(float(w * w + h * h))), 1.0)
    ux = dx / norm
    uy = dy / norm
    drag_scale = min(dist / (0.35 * norm), 1.0)

    fmax = float(max(0.0, max_force))
    gain = float(max(0.1, float(state.get("action_gain", 1.0))))
    fx = float(ux * fmax * drag_scale * gain)
    fy = float(uy * fmax * drag_scale * gain)

    state["pending_action"] = [fx, fy, 1.0]
    state["last_queued_action"] = [fx, fy, 1.0]
    enforced_hold = int(
        max(
            HARD_MIN_ACTION_HOLD_FRAMES,
            int(state.get("action_hold_frames", HARD_MIN_ACTION_HOLD_FRAMES)),
        )
    )
    state["action_repeat_remaining"] = enforced_hold
    state["phase"] = "rolling"
    state["capture_start_xy"] = None
    state["capture_started_ts"] = 0.0
    state["interactions_used"] = int(state.get("interactions_used", 0)) + 1
    state["overlay_drag"] = {
        "start_xy": [sx, sy],
        "end_xy": [ex, ey],
        "expire_ts": float(time.time()) + 4.0,
    }

    action_mag = float(np.sqrt(fx * fx + fy * fy))
    state["action_events"].append(
        {
            "event": "user_drag",
            "start_xy": [sx, sy],
            "end_xy": [ex, ey],
            "drag_dist_px": float(dist),
            "queued_action": [fx, fy, 1.0],
            "queued_action_magnitude": action_mag,
            "queued_action_hold_frames": int(state.get("action_repeat_remaining", 0)),
            "queued_action_gain": gain,
            "frame_idx": int(state.get("steps_generated", 0)),
            "ts_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
    )
    _append_console_line(
        state,
        (
            f"action queued from ({sx:.1f},{sy:.1f})->({ex:.1f},{ey:.1f}) "
            f"fx={fx:+.2f} fy={fy:+.2f} |a|={action_mag:.2f} hold={enforced_hold}"
        ),
    )

    return state, _render_display_frame(state, base_frame), {
        "event": "action_queued",
        "click_start_xy": [sx, sy],
        "click_end_xy": [ex, ey],
        "click_end_display_xy": [px_display, py_display],
        "action": [round(fx, 3), round(fy, 3), 1.0],
        "action_magnitude": round(action_mag, 3),
        "action_hold_frames": int(state.get("action_hold_frames", 1)),
        "action_repeat_remaining": int(state.get("action_repeat_remaining", 0)),
        "action_gain": round(gain, 3),
        "interactions_used": int(state.get("interactions_used", 0)),
        "max_interactions": int(state.get("max_interactions", 2)),
    }, _action_hud_text(state), _console_text(state)


def _step_once(
    state: Dict[str, Any] | None,
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    device: str,
    ddim_steps: int,
):
    """Generate exactly one next frame using queued user action."""
    if not state:
        return state, None, {"error": "Start game first."}, ""
    if not bool(state.get("running", False)):
        state["phase"] = "stopped"
        return state, state.get("frames", [None])[-1], {"event": "stopped"}, str(state.get("output_dir", ""))

    # Pause generation while waiting for second click (drag end).
    if str(state.get("phase", "rolling")) == "capture":
        return (
            state,
            state.get("frames", [None])[-1],
            {
                "event": "paused_for_capture",
                "capture_start_xy": state.get("capture_start_xy"),
                "interactions_used": int(state.get("interactions_used", 0)),
                "max_interactions": int(state.get("max_interactions", 2)),
            },
            str(state.get("output_dir", "")),
        )

    t0 = time.time()
    try:
        engine = _get_engine(train_config_path, checkpoint_path, vae_checkpoint_path, device)
    except Exception as e:
        state["running"] = False
        state["phase"] = "stopped"
        return state, state.get("frames", [None])[-1], {
            "error": "Engine reload failed",
            "detail": str(e),
        }, str(state.get("output_dir", ""))

    context_np = np.array(state["context"], dtype=np.float32, copy=True)
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    action_source = "idle"
    pending_action = np.array(state.get("pending_action", [0.0, 0.0, 0.0]), dtype=np.float32)
    repeat_remaining = int(state.get("action_repeat_remaining", 0))
    if repeat_remaining > 0 and float(pending_action[2]) > 0.5:
        action = pending_action.copy()
        action_source = "user_drag"
        state["action_repeat_remaining"] = max(0, repeat_remaining - 1)
        if int(state["action_repeat_remaining"]) <= 0:
            state["pending_action"] = [0.0, 0.0, 0.0]

    # Fast path for interactive inference: keep tensors on GPU for single-step generation.
    with torch.inference_mode():
        context_t = torch.from_numpy(context_np).to(device=engine.device, dtype=torch.float32)
        context_t = context_t[:, -engine.bundle.context_len :]
        action_t = torch.from_numpy(action.reshape(1, -1)).to(device=engine.device, dtype=torch.float32)

        pred_next = wm_eval._sample_next_latent_ddim(
            bundle=engine.bundle,
            context=context_t,
            action=action_t,
            ddim_steps=int(ddim_steps),
        )  # [1, C, h, w]

        new_context_t = torch.cat([context_t, pred_next.unsqueeze(1)], dim=1)[:, -engine.bundle.context_len :]
        state["context"] = new_context_t.detach().cpu().numpy().astype(np.float32, copy=False)

    frame = _decode_one_frame(engine.vae, pred_next.unsqueeze(1), engine.device)
    state["frames"].append(frame)
    state["frames"] = state["frames"][-1200:]  # cap memory
    state["steps_generated"] = int(state.get("steps_generated", 0)) + 1
    state["frame_action_log"].append(
        {
            "frame_idx": int(state["steps_generated"]),
            "action_used": [float(action[0]), float(action[1]), float(action[2])],
            "action_source": action_source,
            "action_magnitude": float(np.sqrt(float(action[0] * action[0] + action[1] * action[1]))),
            "ts_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
    )
    state["frame_action_log"] = state["frame_action_log"][-1200:]

    if float(action[2]) > 0.5:
        state["action_events"].append(
            {
                "frame_idx": int(state["steps_generated"]),
                "action": [float(action[0]), float(action[1]), float(action[2])],
                "action_source": action_source,
                "ts_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }
        )
    state["last_applied_action"] = [float(action[0]), float(action[1]), float(action[2])]
    state["last_action_source"] = action_source
    _append_console_line(
        state,
        (
            f"frame={int(state['steps_generated'])} src={action_source} "
            f"a=({float(action[0]):+.2f},{float(action[1]):+.2f}) "
            f"rem={int(state.get('action_repeat_remaining', 0))} "
            f"lat={round(time.time()-t0,3):.3f}s"
        ),
    )

    log = {
        "event": "frame",
        "session_id": state.get("session_id", ""),
        "frame_idx": int(state["steps_generated"]),
        "action_used": [float(action[0]), float(action[1]), float(action[2])],
        "action_source": action_source,
        "action_magnitude": float(np.sqrt(float(action[0] * action[0] + action[1] * action[1]))),
        "action_repeat_remaining": int(state.get("action_repeat_remaining", 0)),
        "last_queued_action": [float(x) for x in state.get("last_queued_action", [0.0, 0.0, 0.0])],
        "interactions_used": int(state.get("interactions_used", 0)),
        "max_interactions": int(state.get("max_interactions", 0)),
        "latency_sec": round(time.time() - t0, 3),
    }
    if torch.cuda.is_available() and str(device).lower() == "cuda":
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            log["gpu_mem_used_gb"] = round((total_b - free_b) / (1024**3), 2)
            log["gpu_mem_total_gb"] = round(total_b / (1024**3), 2)
        except Exception:
            pass

    return state, frame, log, str(state.get("output_dir", ""))


def _run_game_loop(
    state: Dict[str, Any] | None,
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    device: str,
    ddim_steps: int,
    run_seconds: float,
    fps_target: int,
):
    """Generate a fixed-duration clip and auto-export it at the end."""
    if not state:
        yield state, None, {"error": "Start game first."}, "", None
        return

    total_frames = int(max(1, round(float(run_seconds) * float(max(1, fps_target)))))
    generated = 0
    last_frame = state.get("frames", [None])[-1]

    while bool(state.get("running", False)) and generated < total_frames:
        frame_start = time.time()
        state, frame, log, out_dir = _step_once(
            state=state,
            train_config_path=train_config_path,
            checkpoint_path=checkpoint_path,
            vae_checkpoint_path=vae_checkpoint_path,
            device=device,
            ddim_steps=int(ddim_steps),
        )
        generated += 1
        last_frame = frame
        if isinstance(log, dict):
            log["target_frames"] = int(total_frames)
            log["generated_frames"] = int(generated)
        yield state, frame, log, out_dir, None

        elapsed = time.time() - frame_start
        target_dt = 1.0 / float(max(1, fps_target))
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    video_path, export_log = _export_game(state, int(max(1, fps_target)))
    if isinstance(export_log, dict):
        export_log["event"] = "run_finished"
        export_log["target_frames"] = int(total_frames)
        export_log["generated_frames"] = int(generated)
    yield state, last_frame, export_log, str(state.get("output_dir", "")), video_path


def _tick_game(
    state: Dict[str, Any] | None,
    train_config_path: str,
    checkpoint_path: str,
    vae_checkpoint_path: str,
    device: str,
    ddim_steps: int,
):
    """Single timer tick wrapper used by Gradio Timer."""
    state, frame, log, out_dir = _step_once(
        state=state,
        train_config_path=train_config_path,
        checkpoint_path=checkpoint_path,
        vae_checkpoint_path=vae_checkpoint_path,
        device=device,
        ddim_steps=ddim_steps,
    )
    if state and frame is None and state.get("frames"):
        frame = state["frames"][-1]
    if state and isinstance(log, dict):
        log["phase"] = str(state.get("phase", "rolling"))
        log["pending_action"] = [float(x) for x in state.get("pending_action", [0.0, 0.0, 0.0])]
        log["last_applied_action"] = [float(x) for x in state.get("last_applied_action", [0.0, 0.0, 0.0])]
    display_frame = _render_display_frame(state, frame)
    return state, display_frame, log, out_dir, _action_hud_text(state), _console_text(state)


def _stop_game(state: Dict[str, Any] | None):
    """Mark session as stopped and return status payload."""
    if state:
        state["running"] = False
        state["phase"] = "stopped"
        _append_console_line(state, "stop requested")
        return state, {"event": "stop_requested", "session_id": state.get("session_id", "")}, _action_hud_text(state), _console_text(state)
    return state, {"event": "stop_requested", "warning": "no active state"}, _action_hud_text(state), _console_text(state)


def _export_game(state: Dict[str, Any] | None, fps: int):
    """Export current session frames and action logs to disk."""
    if not state:
        return None, {"error": "No game state."}
    frames = list(state.get("frames", []))
    if not frames:
        return None, {"error": "No frames to export."}

    out_dir = Path(str(state.get("output_dir", "./world_model_inference/runs/game_sessions")))
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "gameplay.mp4"
    actions_path = out_dir / "actions_log.json"
    frame_actions_path = out_dir / "frame_actions_log.json"

    video = torch.from_numpy(np.stack(frames, axis=0))
    write_video(str(video_path), video, fps=int(max(1, fps)))

    actions_path.write_text(json.dumps(state.get("action_events", []), indent=2), encoding="utf-8")
    frame_actions_path.write_text(json.dumps(state.get("frame_action_log", []), indent=2), encoding="utf-8")
    return str(video_path), {
        "event": "export",
        "video_path": str(video_path),
        "actions_path": str(actions_path),
        "frame_actions_path": str(frame_actions_path),
        "frames": int(video.shape[0]),
        "fps": int(max(1, fps)),
    }


def build_game_demo() -> "gr.Blocks":
    """Build the Gradio app with controls + timer-driven generation."""
    if gr is None:
        raise RuntimeError("gradio is not installed. Install with: pip install -r world_model_inference/requirements.txt")

    default_train_config = os.environ.get("WM_INF_DEFAULT_TRAIN_CONFIG", "").strip()
    default_checkpoint = os.environ.get("WM_INF_DEFAULT_CHECKPOINT", "").strip()
    default_vae_ckpt = os.environ.get("WM_INF_DEFAULT_VAE_CHECKPOINT", "").strip()
    default_data_path = os.environ.get("WM_INF_DEFAULT_DATA_PATH", "").strip()

    image_kwargs: Dict[str, Any] = {"sources": []}
    try:
        params = inspect.signature(gr.Image).parameters
        # Gradio >=6 uses unified buttons list.
        if "buttons" in params:
            image_kwargs["buttons"] = []
        # Gradio 4/5 uses dedicated toggles.
        if "show_download_button" in params:
            image_kwargs["show_download_button"] = False
        if "show_share_button" in params:
            image_kwargs["show_share_button"] = False
        if "show_fullscreen_button" in params:
            image_kwargs["show_fullscreen_button"] = False
    except Exception:
        pass

    css = """
    .game-canvas-fixed img, .game-canvas-fixed canvas {
      -webkit-user-drag: none !important;
      user-select: none !important;
      cursor: crosshair !important;
    }
    """

    with gr.Blocks(title="Pool Game Route", css=css) as demo:
        gr.Markdown("## Pool Game Route")
        gr.Markdown(
            "Start Game to begin rolling generation. First click pauses generation and sets drag start. Second click sets drag end, queues action for multiple frames, then resumes generation."
        )

        game_state = gr.State(value=None)
        tick_timer = gr.Timer(value=0.25, active=False)

        with gr.Row():
            start_btn = gr.Button("Start Game", variant="primary")
            stop_btn = gr.Button("Stop", variant="secondary")
            export_btn = gr.Button("Export", variant="secondary")
        action_hud = gr.Markdown("Action HUD: idle")

        game_canvas = gr.Image(
            value=_make_table_canvas(),
            type="numpy",
            interactive=False,
            label="Game Canvas (click start point, then end point for drag)",
            elem_classes=["game-canvas-fixed"],
            **image_kwargs,
        )
        with gr.Row():
            status = gr.JSON(label="Logs")
            out_dir = gr.Textbox(label="Session Output")
        frame_console = gr.Textbox(label="Frame Console", lines=12, max_lines=20, interactive=False)
        with gr.Row():
            exported_video = gr.Video(label="Exported Gameplay")

        with gr.Accordion("Advanced", open=False):
            train_config = gr.Textbox(label="Train Config", value=default_train_config)
            checkpoint = gr.Textbox(label="Checkpoint", value=default_checkpoint)
            vae_ckpt = gr.Textbox(label="VAE Checkpoint", value=default_vae_ckpt)
            data_path = gr.Textbox(label="Eval Manifest or Shards Dir", value=default_data_path)
            device = gr.Dropdown(["cuda", "cpu"], value="cuda", label="Device")
            sample_source = gr.Dropdown(["eval", "train"], value="eval", label="Prompt Source")
            seed = gr.Number(value=42, precision=0, label="Seed")
            ddim_steps = gr.Slider(5, 50, value=8, step=1, label="DDIM Steps")
            max_force = gr.Slider(2.0, 300.0, value=180.0, step=1.0, label="Max Click Force")
            action_gain = gr.Slider(0.5, 8.0, value=2.0, step=0.1, label="Action Gain")
            action_hold_frames = gr.Slider(
                HARD_MIN_ACTION_HOLD_FRAMES,
                20,
                value=8,
                step=1,
                label=f"Action Hold Frames (hard-min {HARD_MIN_ACTION_HOLD_FRAMES})",
            )
            max_interactions = gr.Slider(1, 30, value=12, step=1, label="Max Interactions")
            display_scale = gr.Slider(
                6,
                16,
                value=DEFAULT_DISPLAY_SCALE,
                step=1,
                label="Display Scale (pixelated; affects click mapping)",
            )
            gen_fps = gr.Slider(0.5, 8.0, value=1.5, step=0.5, label="Generation FPS (slower for debugging)")
            export_fps = gr.Slider(4, 20, value=10, step=1, label="Export FPS")

        gen_fps.change(
            fn=lambda x: gr.update(value=float(max(0.05, 1.0 / max(0.1, float(x))))),
            inputs=[gen_fps],
            outputs=[tick_timer],
        )

        start_evt = start_btn.click(
            fn=_start_game,
            inputs=[
                train_config,
                checkpoint,
                vae_ckpt,
                data_path,
                device,
                sample_source,
                seed,
                display_scale,
                action_hold_frames,
                max_interactions,
                action_gain,
            ],
            outputs=[game_state, game_canvas, status, out_dir, action_hud, frame_console],
        )
        start_evt.then(fn=lambda: gr.update(active=True), outputs=[tick_timer])

        tick_timer.tick(
            fn=_tick_game,
            inputs=[game_state, train_config, checkpoint, vae_ckpt, device, ddim_steps],
            outputs=[game_state, game_canvas, status, out_dir, action_hud, frame_console],
        )

        game_canvas.select(
            fn=_queue_click_action,
            inputs=[game_state, game_canvas, max_force],
            outputs=[game_state, game_canvas, status, action_hud, frame_console],
        )

        stop_evt = stop_btn.click(fn=_stop_game, inputs=[game_state], outputs=[game_state, status, action_hud, frame_console])
        stop_evt.then(fn=lambda: gr.update(active=False), outputs=[tick_timer])

        export_btn.click(fn=_export_game, inputs=[game_state, export_fps], outputs=[exported_video, status])

    return demo


def main() -> None:
    """Launch minimal game route web app."""
    parser = argparse.ArgumentParser(description="Minimal game-style route for world-model interaction")
    parser.add_argument("--host", default=os.environ.get("WM_GAME_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WM_GAME_PORT", "7862")))
    args = parser.parse_args()

    demo = build_game_demo()
    demo.launch(server_name=args.host, server_port=int(args.port), show_error=True)


if __name__ == "__main__":
    main()
