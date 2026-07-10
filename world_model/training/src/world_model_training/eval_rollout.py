from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from vae_training.model import ConvVAE
from world_model_training.video_io import write_video
from world_model_training.data import resolve_eval_shards_from_data_cfg
from world_model_training.diffusion import GaussianDiffusion
from world_model_training.model import ActionConditionedDiT
from world_model_training.utils import ensure_dir, now_utc_stamp, save_json, set_seed


DEFAULT_CFG: Dict[str, Any] = {
    "run": {
        "eval_id": "",
        "seed": 42,
        "output_root": "./world_model/training/evals",
        "device": "cuda",
        "num_clips": 64,
        "num_viz_clips": 6,
        "horizons": [1, 4, 8, 16, 32],
        "ddim_steps": 10,
        "decode_batch_size": 128,
        "video_fps": 12,
    },
    "data": {
        "shards_dir": "",
        "val_shards": 50,
    },
    "models": [],
    "vae": {
        "enabled": False,
        "checkpoint_path": "",
        "base_channels": 64,
        "latent_channels": 4,
    },
}


@dataclass
class ModelBundle:
    name: str
    checkpoint_path: Path
    config_path: Path
    context_len: int
    action_dim: int
    latent_shape: tuple[int, int, int]
    diffusion_steps: int
    model: ActionConditionedDiT
    diffusion: GaussianDiffusion


@dataclass
class EvalClips:
    context_max: np.ndarray
    actions: np.ndarray
    gt_future: np.ndarray
    meta: List[Dict[str, Any]]


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_structured(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_eval_config(path: str | Path) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    loaded = _load_structured(Path(path))
    _deep_update(cfg, loaded)
    return cfg


def _load_model_bundle(spec: Dict[str, Any], device: torch.device) -> ModelBundle:
    name = str(spec["name"])
    checkpoint_path = Path(spec["checkpoint_path"]).resolve()
    config_path = Path(spec["config_path"]).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for model '{name}': {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found for model '{name}': {config_path}")

    train_cfg = _load_structured(config_path)
    model_cfg = train_cfg["model"]
    data_cfg = train_cfg["data"]
    diff_cfg = train_cfg["diffusion"]

    model = ActionConditionedDiT(
        latent_channels=int(model_cfg["latent_channels"]),
        latent_h=int(model_cfg["latent_h"]),
        latent_w=int(model_cfg["latent_w"]),
        context_len=int(data_cfg["context_len"]),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        action_dim=int(model_cfg["action_dim"]),
        diffusion_steps=int(diff_cfg["timesteps"]),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
        device=device,
    )

    return ModelBundle(
        name=name,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        context_len=int(data_cfg["context_len"]),
        action_dim=int(model_cfg["action_dim"]),
        latent_shape=(
            int(model_cfg["latent_channels"]),
            int(model_cfg["latent_h"]),
            int(model_cfg["latent_w"]),
        ),
        diffusion_steps=int(diff_cfg["timesteps"]),
        model=model,
        diffusion=diffusion,
    )


def _sample_eval_clips(
    shards: List[Path],
    num_clips: int,
    max_context_len: int,
    max_horizon: int,
    seed: int,
) -> EvalClips:
    rng = np.random.default_rng(seed)
    context_max_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    gt_future_list: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    attempts = 0
    max_attempts = max(5000, num_clips * 200)

    while len(context_max_list) < num_clips and attempts < max_attempts:
        attempts += 1
        shard_path = shards[int(rng.integers(0, len(shards)))]

        with np.load(shard_path, allow_pickle=False) as d:
            latents = d["latents"].astype(np.float32, copy=False)  # [E, T, C, H, W]
            actions = d["actions"].astype(np.float32, copy=False)  # [E, T, A]
            lengths = d["lengths"].astype(np.int64, copy=False)  # [E]

            valid_eps = np.where(lengths >= (max_context_len + max_horizon + 1))[0]
            if len(valid_eps) == 0:
                continue

            ep = int(valid_eps[int(rng.integers(0, len(valid_eps)))])
            ep_len = int(lengths[ep])

            t_min = max_context_len - 1
            t_max = ep_len - 1 - max_horizon
            if t_max < t_min:
                continue

            t = int(rng.integers(t_min, t_max + 1))

            context_max = np.array(
                latents[ep, t - max_context_len + 1 : t + 1],
                dtype=np.float32,
                copy=True,
            )
            act = np.array(actions[ep, t : t + max_horizon], dtype=np.float32, copy=True)
            gt_future = np.array(
                latents[ep, t + 1 : t + max_horizon + 1],
                dtype=np.float32,
                copy=True,
            )

            context_max_list.append(context_max)
            actions_list.append(act)
            gt_future_list.append(gt_future)
            meta.append(
                {
                    "shard": str(shard_path),
                    "episode": ep,
                    "anchor_t": t,
                    "episode_length": ep_len,
                }
            )

    if len(context_max_list) < num_clips:
        raise RuntimeError(
            f"Could only sample {len(context_max_list)} clips out of requested {num_clips}. "
            "Increase val shard count or reduce num_clips/horizon/context." 
        )

    return EvalClips(
        context_max=np.stack(context_max_list, axis=0),
        actions=np.stack(actions_list, axis=0),
        gt_future=np.stack(gt_future_list, axis=0),
        meta=meta,
    )


def _ddim_schedule_indices(timesteps: int, ddim_steps: int) -> np.ndarray:
    if ddim_steps < 1:
        raise ValueError("ddim_steps must be >= 1")
    idx = np.linspace(timesteps - 1, -1, ddim_steps + 1)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, -1, timesteps - 1)
    return idx


def _sample_next_latent_ddim(
    bundle: ModelBundle,
    context: torch.Tensor,
    action: torch.Tensor,
    ddim_steps: int,
) -> torch.Tensor:
    # context: [B, L, C, H, W], action: [B, A]
    b, _, c, h, w = context.shape
    x = torch.randn((b, c, h, w), device=context.device, dtype=context.dtype)

    schedule = _ddim_schedule_indices(bundle.diffusion_steps, ddim_steps)
    alpha_bar = bundle.diffusion.alpha_bar

    for i in range(ddim_steps):
        t_val = int(schedule[i])
        t_next = int(schedule[i + 1])
        t = torch.full((b,), t_val, dtype=torch.long, device=context.device)

        pred_noise = bundle.model(context=context, action=action, noisy_target=x, t_idx=t)

        a_t = alpha_bar[t_val]
        sqrt_a_t = torch.sqrt(a_t)
        sqrt_one_minus_a_t = torch.sqrt(1.0 - a_t)

        x0 = (x - sqrt_one_minus_a_t * pred_noise) / sqrt_a_t

        if t_next >= 0:
            a_next = alpha_bar[t_next]
        else:
            a_next = torch.tensor(1.0, device=context.device, dtype=x.dtype)

        x = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * pred_noise

    return x


def _rollout_predictions(
    bundle: ModelBundle,
    clips: EvalClips,
    max_context_len: int,
    max_horizon: int,
    ddim_steps: int,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    n = clips.context_max.shape[0]
    c, h, w = clips.context_max.shape[2], clips.context_max.shape[3], clips.context_max.shape[4]

    preds = torch.empty((n, max_horizon, c, h, w), dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            context_max = torch.from_numpy(clips.context_max[start:end]).to(device=device, dtype=torch.float32)
            actions = torch.from_numpy(clips.actions[start:end]).to(device=device, dtype=torch.float32)

            context = context_max[:, -bundle.context_len :]
            out_steps: List[torch.Tensor] = []

            for step in range(max_horizon):
                act = actions[:, step]
                pred_next = _sample_next_latent_ddim(
                    bundle=bundle,
                    context=context,
                    action=act,
                    ddim_steps=ddim_steps,
                )
                out_steps.append(pred_next.detach().cpu())
                context = torch.cat([context[:, 1:], pred_next.unsqueeze(1)], dim=1)

            pred_batch = torch.stack(out_steps, dim=1)  # [B, H, C, h, w]
            preds[start:end] = pred_batch

    return preds


def _latent_metrics(
    preds: torch.Tensor,
    gt: torch.Tensor,
    horizons: List[int],
) -> Dict[str, Dict[str, float]]:
    out_mse: Dict[str, float] = {}
    out_mae: Dict[str, float] = {}

    for h in horizons:
        pred_h = preds[:, h - 1]
        gt_h = gt[:, h - 1]
        diff = pred_h - gt_h
        mse = torch.mean(diff * diff).item()
        mae = torch.mean(torch.abs(diff)).item()
        out_mse[str(h)] = float(mse)
        out_mae[str(h)] = float(mae)

    return {"latent_mse": out_mse, "latent_mae": out_mae}


def _load_vae(vae_cfg: Dict[str, Any], device: torch.device) -> ConvVAE:
    ckpt_path = Path(str(vae_cfg["checkpoint_path"])).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")

    vae = ConvVAE(
        in_channels=3,
        base_channels=int(vae_cfg.get("base_channels", 64)),
        latent_channels=int(vae_cfg.get("latent_channels", 4)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    vae.load_state_dict(state, strict=True)
    vae.eval()
    return vae


def _decode_latents(
    vae: ConvVAE,
    latents: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    # latents: [N, H, C, h, w], returns [N, H, 3, Himg, Wimg] in [0, 1]
    n, t, c, h, w = latents.shape
    flat = latents.reshape(n * t, c, h, w)
    out: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, flat.shape[0], batch_size):
            z = flat[i : i + batch_size].to(device=device, dtype=torch.float32)
            x = vae.decode(z)
            x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
            out.append(x.detach().cpu())

    all_frames = torch.cat(out, dim=0)
    return all_frames.reshape(n, t, 3, all_frames.shape[-2], all_frames.shape[-1])


def _frame_metrics(
    vae: ConvVAE,
    preds: torch.Tensor,
    gt: torch.Tensor,
    horizons: List[int],
    device: torch.device,
    decode_batch_size: int,
) -> Dict[str, Dict[str, float]]:
    psnr_map: Dict[str, float] = {}
    l1_map: Dict[str, float] = {}

    with torch.no_grad():
        for h in horizons:
            pred_h = preds[:, h - 1 : h]
            gt_h = gt[:, h - 1 : h]

            pred_img = _decode_latents(vae, pred_h, device=device, batch_size=decode_batch_size).squeeze(1)
            gt_img = _decode_latents(vae, gt_h, device=device, batch_size=decode_batch_size).squeeze(1)

            diff = pred_img - gt_img
            mse = torch.mean(diff * diff).item()
            l1 = torch.mean(torch.abs(diff)).item()
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))

            psnr_map[str(h)] = float(psnr)
            l1_map[str(h)] = float(l1)

    return {"frame_psnr": psnr_map, "frame_l1": l1_map}


def _write_comparison_videos(
    output_dir: Path,
    model_names: List[str],
    gt_frames: torch.Tensor,
    pred_frames_by_model: Dict[str, torch.Tensor],
    fps: int,
) -> List[Dict[str, Any]]:
    ensure_dir(output_dir)
    outputs: List[Dict[str, Any]] = []

    num_clips = gt_frames.shape[0]
    horizon = gt_frames.shape[1]

    for clip_idx in range(num_clips):
        frames = []
        for t in range(horizon):
            panels = [gt_frames[clip_idx, t]]
            for model_name in model_names:
                panels.append(pred_frames_by_model[model_name][clip_idx, t])
            panel = torch.cat(panels, dim=2)  # [3, H, W_total]
            frame = torch.clamp(panel, 0.0, 1.0).permute(1, 2, 0)
            frame = (frame * 255.0).byte()
            frames.append(frame)

        video = torch.stack(frames, dim=0)
        out_path = output_dir / f"clip_{clip_idx:03d}.mp4"
        try:
            write_video(str(out_path), video, fps=fps)
            outputs.append({"clip": clip_idx, "path": str(out_path), "status": "ok"})
        except Exception as exc:
            fallback = output_dir / f"clip_{clip_idx:03d}_frames.npy"
            np.save(fallback, video.numpy())
            outputs.append(
                {
                    "clip": clip_idx,
                    "path": str(fallback),
                    "status": "fallback_npy",
                    "error": str(exc),
                }
            )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout evaluation for action-conditioned world models")
    parser.add_argument("--config", required=True, help="Path to rollout eval YAML/JSON config")
    parser.add_argument("--eval-id", default="", help="Optional eval id override")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    if args.eval_id:
        cfg["run"]["eval_id"] = args.eval_id

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    vae_cfg = cfg["vae"]

    eval_id = str(run_cfg.get("eval_id") or f"rollout_eval_{now_utc_stamp()}")
    output_root = ensure_dir(Path(run_cfg["output_root"]).resolve())
    eval_dir = output_root / eval_id
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise FileExistsError(f"Eval output dir already exists and is not empty: {eval_dir}")
    ensure_dir(eval_dir)

    set_seed(int(run_cfg["seed"]))

    requested_device = str(run_cfg.get("device", "cuda")).lower()
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    horizons = sorted({int(h) for h in list(run_cfg["horizons"]) if int(h) >= 1})
    if not horizons:
        raise ValueError("run.horizons must include at least one positive horizon")
    max_horizon = max(horizons)

    model_specs = list(cfg.get("models", []))
    if not model_specs:
        raise ValueError("No models provided in config under 'models'")

    bundles = [_load_model_bundle(m, device=device) for m in model_specs]

    latent_shape_set = {b.latent_shape for b in bundles}
    action_dim_set = {b.action_dim for b in bundles}
    if len(latent_shape_set) != 1:
        raise ValueError(f"All models must share latent shape, got: {latent_shape_set}")
    if len(action_dim_set) != 1:
        raise ValueError(f"All models must share action dim, got: {action_dim_set}")

    max_context_len = max(b.context_len for b in bundles)

    eval_shards = resolve_eval_shards_from_data_cfg(data_cfg)

    clips = _sample_eval_clips(
        shards=eval_shards,
        num_clips=int(run_cfg["num_clips"]),
        max_context_len=max_context_len,
        max_horizon=max_horizon,
        seed=int(run_cfg["seed"]),
    )

    gt_future = torch.from_numpy(clips.gt_future).float()

    manifest = {
        "eval_id": eval_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "num_clips": int(run_cfg["num_clips"]),
        "max_horizon": max_horizon,
        "horizons": horizons,
        "ddim_steps": int(run_cfg["ddim_steps"]),
        "models": [
            {
                "name": b.name,
                "checkpoint_path": str(b.checkpoint_path),
                "config_path": str(b.config_path),
                "context_len": b.context_len,
            }
            for b in bundles
        ],
        "data_shards_dir": str(Path(str(data_cfg.get("shards_dir", "") or ".")).resolve())
        if str(data_cfg.get("shards_dir", "") or "").strip()
        else "",
        "data_shards_manifest": str(data_cfg.get("shards_manifest", "") or ""),
        "data_eval_shards_manifest": str(data_cfg.get("eval_shards_manifest", "") or ""),
        "data_val_shards_manifest": str(data_cfg.get("val_shards_manifest", "") or ""),
        "val_shards": int(data_cfg["val_shards"]),
        "clip_meta": clips.meta,
    }
    save_json(manifest, eval_dir / "manifest.json")

    per_model_metrics: Dict[str, Dict[str, Any]] = {}
    preds_by_model: Dict[str, torch.Tensor] = {}

    for bundle in bundles:
        print(f"[eval] rolling out model={bundle.name} context_len={bundle.context_len}")
        preds = _rollout_predictions(
            bundle=bundle,
            clips=clips,
            max_context_len=max_context_len,
            max_horizon=max_horizon,
            ddim_steps=int(run_cfg["ddim_steps"]),
            device=device,
            batch_size=16,
        )
        preds_by_model[bundle.name] = preds

        latent_metrics = _latent_metrics(preds=preds, gt=gt_future, horizons=horizons)
        per_model_metrics[bundle.name] = latent_metrics

    if bool(vae_cfg.get("enabled", False)):
        vae = _load_vae(vae_cfg=vae_cfg, device=device)
        decode_batch_size = int(run_cfg.get("decode_batch_size", 128))

        for bundle in bundles:
            name = bundle.name
            frame_metrics = _frame_metrics(
                vae=vae,
                preds=preds_by_model[name],
                gt=gt_future,
                horizons=horizons,
                device=device,
                decode_batch_size=decode_batch_size,
            )
            per_model_metrics[name].update(frame_metrics)

        num_viz = max(0, min(int(run_cfg.get("num_viz_clips", 0)), int(run_cfg["num_clips"])))
        if num_viz > 0:
            gt_viz = _decode_latents(
                vae=vae,
                latents=gt_future[:num_viz],
                device=device,
                batch_size=decode_batch_size,
            )

            pred_viz: Dict[str, torch.Tensor] = {}
            for name in preds_by_model:
                pred_viz[name] = _decode_latents(
                    vae=vae,
                    latents=preds_by_model[name][:num_viz],
                    device=device,
                    batch_size=decode_batch_size,
                )

            video_outputs = _write_comparison_videos(
                output_dir=eval_dir / "videos",
                model_names=[b.name for b in bundles],
                gt_frames=gt_viz,
                pred_frames_by_model=pred_viz,
                fps=int(run_cfg.get("video_fps", 12)),
            )
            save_json(video_outputs, eval_dir / "video_outputs.json")

            legend = {
                "columns": ["ground_truth"] + [b.name for b in bundles],
                "description": "Each video frame is horizontally concatenated as GT | model_1 | model_2 | ...",
            }
            save_json(legend, eval_dir / "video_legend.json")

    summary = {
        "eval_id": eval_id,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "horizons": horizons,
        "models": per_model_metrics,
    }

    # Simple ranking at key horizons.
    key_horizons = [h for h in [16, 32] if h in horizons]
    rankings: Dict[str, Any] = {}
    for h in key_horizons:
        key = str(h)
        by_mse = sorted(
            (
                (name, metrics["latent_mse"][key])
                for name, metrics in per_model_metrics.items()
            ),
            key=lambda x: x[1],
        )
        rankings[f"latent_mse_h{h}"] = by_mse

        if all("frame_psnr" in m for m in per_model_metrics.values()):
            by_psnr = sorted(
                (
                    (name, metrics["frame_psnr"][key])
                    for name, metrics in per_model_metrics.items()
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            rankings[f"frame_psnr_h{h}"] = by_psnr

    summary["rankings"] = rankings
    save_json(summary, eval_dir / "summary.json")

    print(f"Rollout eval complete: {eval_dir}")


if __name__ == "__main__":
    main()
