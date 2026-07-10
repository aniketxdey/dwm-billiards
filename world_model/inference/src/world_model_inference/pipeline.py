from __future__ import annotations

"""Reusable inference pipeline for checkpoint previews and interactive apps.

This module keeps model/VAE loading, prompt sampling, latent rollout, decode,
and preview artifact writing in one place so CLI/UI routes share one behavior.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from world_model_training import eval_rollout as wm_eval
from world_model_training.video_io import write_video
from world_model_training.data import (
    resolve_eval_shards_from_data_cfg,
    resolve_train_val_shards_from_data_cfg,
)
from world_model_training.utils import ensure_dir, now_utc_stamp, save_json, set_seed

from .action_presets import build_action_sequence
from .config import load_preview_config
from .viz import render_action_timeline


@dataclass
class InferencePrompt:
    """Prompt package consumed by latent rollout."""

    context_max: np.ndarray  # [N, Lmax, C, H, W]
    actions: np.ndarray  # [N, H, A]
    gt_future: Optional[np.ndarray]  # [N, H, C, H, W] if dataset actions
    meta: List[Dict[str, Any]]
    source: str


@dataclass
class InferenceEngine:
    """Loaded world model bundle + optional VAE on one device."""

    bundle: wm_eval.ModelBundle
    device: torch.device
    vae: Optional[Any] = None


@dataclass
class PreviewArtifacts:
    """Filesystem outputs produced by a preview run."""

    preview_id: str
    output_dir: Path
    summary_path: Path
    video_paths: List[Path]
    action_timeline_paths: List[Path]


def _resolve_device(requested: str) -> torch.device:
    """Resolve requested runtime device with CUDA fallback to CPU."""
    req = str(requested or "cuda").lower()
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _require_file_path(path_like: str | Path, label: str) -> Path:
    """Validate path exists and points to a file (not a directory)."""
    raw = str(path_like or "").strip()
    if not raw:
        raise ValueError(f"{label} is empty. Please provide a valid file path.")
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{label} must be a file, but got a directory: {p}")
    return p


def load_engine(
    *,
    model_name: str,
    checkpoint_path: str | Path,
    train_config_path: str | Path,
    device: str = "cuda",
    vae_cfg: Optional[Dict[str, Any]] = None,
) -> InferenceEngine:
    """Load world model bundle and optional VAE for inference."""
    dev = _resolve_device(device)
    checkpoint_file = _require_file_path(checkpoint_path, "World-model checkpoint")
    train_config_file = _require_file_path(train_config_path, "World-model train config")
    bundle = wm_eval._load_model_bundle(
        {
            "name": model_name,
            "checkpoint_path": str(checkpoint_file),
            "config_path": str(train_config_file),
        },
        device=dev,
    )
    vae = None
    if vae_cfg and bool(vae_cfg.get("enabled", False)):
        vae_ckpt = _require_file_path(vae_cfg.get("checkpoint_path", ""), "VAE checkpoint")
        vae_cfg_local = dict(vae_cfg)
        vae_cfg_local["checkpoint_path"] = str(vae_ckpt)
        vae = wm_eval._load_vae(vae_cfg=vae_cfg_local, device=dev)
    return InferenceEngine(bundle=bundle, device=dev, vae=vae)


def load_engine_from_preview_config(cfg_path: str | Path) -> tuple[Dict[str, Any], InferenceEngine]:
    """Load preview config then construct a matching inference engine."""
    cfg = load_preview_config(cfg_path)
    engine = load_engine(
        model_name=str(cfg["model"]["name"]),
        checkpoint_path=cfg["model"]["checkpoint_path"],
        train_config_path=cfg["model"]["train_config_path"],
        device=str(cfg["run"].get("device", "cuda")),
        vae_cfg=dict(cfg.get("vae", {})),
    )
    return cfg, engine


def _sample_dataset_prompt(
    *,
    engine: InferenceEngine,
    data_cfg: Dict[str, Any],
    horizon: int,
    num_clips: int,
    seed: int,
) -> InferencePrompt:
    """Sample context/actions from dataset shards for preview/inference."""
    sample_from = str(data_cfg.get("sample_from", "eval")).strip().lower()
    if sample_from == "train":
        train_shards, _ = resolve_train_val_shards_from_data_cfg(data_cfg)
        shards = train_shards
    else:
        shards = resolve_eval_shards_from_data_cfg(data_cfg)

    clips = wm_eval._sample_eval_clips(
        shards=shards,
        num_clips=int(num_clips),
        max_context_len=int(engine.bundle.context_len),
        max_horizon=int(horizon),
        seed=int(seed),
    )
    return InferencePrompt(
        context_max=clips.context_max,
        actions=clips.actions,
        gt_future=clips.gt_future,
        meta=clips.meta,
        source="dataset",
    )


def build_prompt_from_config(cfg: Dict[str, Any], engine: InferenceEngine) -> InferencePrompt:
    """Build prompt from dataset and optional action preset override."""
    run_cfg = cfg["run"]
    data_cfg = dict(cfg.get("data", {}))
    act_cfg = dict(cfg.get("actions", {}))
    horizon = int(run_cfg["horizon"])
    num_clips = int(run_cfg.get("num_clips", 1))
    seed = int(run_cfg.get("seed", 42))

    prompt = _sample_dataset_prompt(
        engine=engine,
        data_cfg=data_cfg,
        horizon=horizon,
        num_clips=num_clips,
        seed=seed,
    )

    if str(act_cfg.get("source", "dataset")).strip().lower() == "preset":
        preset = dict(act_cfg.get("preset", {}))
        seq = build_action_sequence(preset, fallback_horizon=horizon)
        if seq.shape[0] != horizon:
            raise ValueError(f"Preset action horizon mismatch: got {seq.shape[0]}, expected {horizon}")
        prompt.actions = np.repeat(seq[None, :, :], repeats=prompt.context_max.shape[0], axis=0)
        prompt.gt_future = None
        prompt.source = f"preset:{preset.get('name', 'unknown')}"
        for item in prompt.meta:
            item["actions_source"] = prompt.source
    return prompt


def rollout_latents(
    *,
    engine: InferenceEngine,
    prompt: InferencePrompt,
    ddim_steps: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Generate future latent rollout with DDIM from context + actions."""
    clips = wm_eval.EvalClips(
        context_max=prompt.context_max,
        actions=prompt.actions,
        gt_future=prompt.gt_future if prompt.gt_future is not None else np.zeros_like(prompt.context_max[:, : prompt.actions.shape[1]]),
        meta=prompt.meta,
    )
    return wm_eval._rollout_predictions(
        bundle=engine.bundle,
        clips=clips,
        max_context_len=int(engine.bundle.context_len),
        max_horizon=int(prompt.actions.shape[1]),
        ddim_steps=int(ddim_steps),
        device=engine.device,
        batch_size=int(batch_size),
    )


def decode_latents(engine: InferenceEngine, latents: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Decode latent clips into RGB frame tensors via VAE."""
    if engine.vae is None:
        raise RuntimeError("VAE is not loaded. Set vae.enabled=true and provide vae.checkpoint_path.")
    return wm_eval._decode_latents(engine.vae, latents, device=engine.device, batch_size=int(batch_size))


def _write_pred_only_videos(
    output_dir: Path,
    pred_frames: torch.Tensor,
    fps: int,
    prefix: str = "clip",
) -> List[Path]:
    """Write prediction-only videos (or fallback NPY if codec fails)."""
    ensure_dir(output_dir)
    out_paths: List[Path] = []
    for clip_idx in range(pred_frames.shape[0]):
        frames = []
        for t in range(pred_frames.shape[1]):
            frame = torch.clamp(pred_frames[clip_idx, t], 0.0, 1.0).permute(1, 2, 0)
            frames.append((frame * 255.0).byte())
        video = torch.stack(frames, dim=0)
        out_path = output_dir / f"{prefix}_{clip_idx:03d}.mp4"
        try:
            write_video(str(out_path), video, fps=fps)
            out_paths.append(out_path)
        except Exception:
            fallback = output_dir / f"{prefix}_{clip_idx:03d}_frames.npy"
            np.save(fallback, video.numpy())
            out_paths.append(fallback)
    return out_paths


def run_preview_from_config(cfg: Dict[str, Any], engine: InferenceEngine) -> PreviewArtifacts:
    """Run one full preview job and persist artifacts + summary JSON."""
    run_cfg = cfg["run"]
    viz_cfg = dict(cfg.get("viz", {}))
    vae_cfg = dict(cfg.get("vae", {}))
    set_seed(int(run_cfg.get("seed", 42)))

    preview_id = str(run_cfg.get("preview_id") or f"preview_{now_utc_stamp()}")
    out_root = ensure_dir(Path(str(run_cfg.get("output_root", "./world_model_inference/runs"))).resolve())
    out_dir = ensure_dir(out_root / preview_id)

    prompt = build_prompt_from_config(cfg, engine)
    start = time.time()
    preds = rollout_latents(
        engine=engine,
        prompt=prompt,
        ddim_steps=int(run_cfg.get("ddim_steps", 20)),
        batch_size=8,
    )
    rollout_seconds = time.time() - start

    summary: Dict[str, Any] = {
        "preview_id": preview_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": {
            "name": engine.bundle.name,
            "checkpoint_path": str(engine.bundle.checkpoint_path),
            "train_config_path": str(engine.bundle.config_path),
            "context_len": int(engine.bundle.context_len),
        },
        "source": prompt.source,
        "num_clips": int(prompt.context_max.shape[0]),
        "horizon": int(prompt.actions.shape[1]),
        "ddim_steps": int(run_cfg.get("ddim_steps", 20)),
        "rollout_seconds": float(rollout_seconds),
        "device": str(engine.device),
        "clip_meta": prompt.meta,
    }

    # Save raw action sequences for reproducibility.
    np.savez_compressed(
        out_dir / "prompt_actions.npz",
        actions=prompt.actions.astype(np.float32),
        source=np.array([prompt.source]),
    )

    action_timeline_paths: List[Path] = []
    if bool(viz_cfg.get("write_action_timeline", True)):
        for i in range(prompt.actions.shape[0]):
            p = out_dir / "actions" / f"clip_{i:03d}_timeline.png"
            try:
                render_action_timeline(prompt.actions[i], p, title=f"Actions | clip {i:03d} | {prompt.source}")
                action_timeline_paths.append(p)
            except Exception as exc:
                summary.setdefault("warnings", []).append(
                    f"action timeline render failed for clip {i}: {exc}"
                )

    video_paths: List[Path] = []
    if bool(vae_cfg.get("enabled", False)) and bool(viz_cfg.get("write_video", True)):
        decode_bs = int(run_cfg.get("decode_batch_size", 128))
        pred_frames = decode_latents(engine, preds, batch_size=decode_bs)

        if prompt.gt_future is not None and bool(viz_cfg.get("include_gt_if_available", True)):
            gt_future = torch.from_numpy(prompt.gt_future).float()
            gt_frames = decode_latents(engine, gt_future, batch_size=decode_bs)
            video_out = wm_eval._write_comparison_videos(
                output_dir=out_dir / "videos",
                model_names=[engine.bundle.name],
                gt_frames=gt_frames,
                pred_frames_by_model={engine.bundle.name: pred_frames},
                fps=int(run_cfg.get("video_fps", 12)),
            )
            video_paths = [Path(item["path"]) for item in video_out]

            metric_horizons = [h for h in [1, 4, 8, 16, 32] if h <= preds.shape[1]]
            latent_metrics = wm_eval._latent_metrics(preds=preds, gt=gt_future, horizons=metric_horizons)
            frame_metrics = wm_eval._frame_metrics(
                vae=engine.vae,
                preds=preds,
                gt=gt_future,
                horizons=metric_horizons,
                device=engine.device,
                decode_batch_size=decode_bs,
            )
            summary["metrics"] = {**latent_metrics, **frame_metrics}
            save_json(
                {
                    "columns": ["ground_truth", engine.bundle.name],
                    "description": "GT | model prediction",
                },
                out_dir / "video_legend.json",
            )
        else:
            video_paths = _write_pred_only_videos(out_dir / "videos", pred_frames, fps=int(run_cfg.get("video_fps", 12)))

        # Save prompt reference frame(s) for UI/debug.
        ctx = torch.from_numpy(prompt.context_max[:, -1:, ...]).float()
        ctx_frames = decode_latents(engine, ctx, batch_size=decode_bs).squeeze(1)
        _write_pred_only_videos(out_dir / "prompt_frames", ctx_frames[:, None, ...], fps=1, prefix="prompt")

    save_json(summary, out_dir / "summary.json")
    save_json(cfg, out_dir / "config_resolved.json")

    return PreviewArtifacts(
        preview_id=preview_id,
        output_dir=out_dir,
        summary_path=out_dir / "summary.json",
        video_paths=video_paths,
        action_timeline_paths=action_timeline_paths,
    )
