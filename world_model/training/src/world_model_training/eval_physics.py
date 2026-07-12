"""Physics-grounded rollout evaluation with per-clip statistics.

Extends the latent/frame-metric rollout eval (`eval_rollout.py`) with
object-level physics metrics computed against simulator ground truth
(`sim_state` stored in latent shards):

  - ball recall / missing-ball ("swallowing") rate
  - spurious-ball ("hallucination") rate
  - matched ball position error (Hungarian assignment, pixels)
  - color-identity persistence across the rollout

All metrics are computed per clip so that paired model comparisons can be
tested for significance (Wilcoxon signed-rank) with bootstrap confidence
intervals and effect sizes. The same detector is applied to VAE-decoded
ground-truth frames so that detector and VAE artifacts cancel in paired
comparisons.

Usage:
  PYTHONPATH=world_model/training/src:vae_training/src \
    python -m world_model_training.eval_physics --config <yaml>

Config schema is a superset of eval_rollout's; see DEFAULT_CFG below.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

from vae_training.model import ConvVAE
from world_model_training.data import resolve_eval_shards_from_data_cfg
from world_model_training.eval_rollout import (
    DEFAULT_CFG as ROLLOUT_DEFAULT_CFG,
    _decode_latents,
    _deep_update,
    _latent_metrics,
    _load_model_bundle,
    _load_structured,
    _load_vae,
    _rollout_predictions,
    _write_comparison_videos,
)
from world_model_training.utils import ensure_dir, save_json, set_seed

# Canonical ball palette from rl_data_gen/record_billiard.py (RGB).
BALL_COLORS: List[Tuple[int, int, int]] = [
    (255, 255, 255), (255, 215, 0), (0, 0, 180), (255, 50, 50),
    (128, 0, 128), (255, 120, 0), (34, 139, 34), (139, 69, 19),
    (20, 20, 20), (255, 230, 100), (100, 100, 220), (255, 130, 130),
    (180, 100, 180), (255, 180, 100), (100, 180, 100), (180, 120, 80),
]
# Legacy shards generated before the pocket recolor used black pockets identical
# to the 8-ball; set excluded_ball_ids: [8] in the eval config for those runs.
EXCLUDED_BALL_IDS: set[int] = set()

DEFAULT_CFG: Dict[str, Any] = json.loads(json.dumps(ROLLOUT_DEFAULT_CFG))
DEFAULT_CFG["physics"] = {
    "color_match_threshold": 90,   # L1 RGB distance for color-keyed masks
    "min_component_area": 5,
    "max_component_area": 80,
    "match_radius_px": 4.0,        # max GT<->detection distance to count a match
    "excluded_ball_ids": [],       # e.g. [8] for legacy black-pocket data
    "bootstrap_iters": 10000,
    "alpha": 0.05,
}


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    _deep_update(cfg, _load_structured(Path(path)))
    return cfg


# ---------------------------------------------------------------------------
# Clip sampling (mirrors eval_rollout but also returns sim_state slices)
# ---------------------------------------------------------------------------

def sample_eval_clips_with_state(
    shards: List[Path],
    num_clips: int,
    max_context_len: int,
    max_horizon: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    ctx_list, act_list, gt_list, state_list, meta = [], [], [], [], []

    attempts, max_attempts = 0, max(5000, num_clips * 200)
    while len(ctx_list) < num_clips and attempts < max_attempts:
        attempts += 1
        shard_path = shards[int(rng.integers(0, len(shards)))]
        with np.load(shard_path, allow_pickle=False) as d:
            latents = d["latents"]
            actions = d["actions"]
            sim_state = d["sim_state"]
            lengths = d["lengths"].astype(np.int64)

            valid = np.where(lengths >= (max_context_len + max_horizon + 1))[0]
            if len(valid) == 0:
                continue
            ep = int(valid[int(rng.integers(0, len(valid)))])
            t_min = max_context_len - 1
            t_max = int(lengths[ep]) - 1 - max_horizon
            if t_max < t_min:
                continue
            t = int(rng.integers(t_min, t_max + 1))

            ctx_list.append(np.array(latents[ep, t - max_context_len + 1 : t + 1], dtype=np.float32))
            act_list.append(np.array(actions[ep, t : t + max_horizon], dtype=np.float32))
            gt_list.append(np.array(latents[ep, t + 1 : t + max_horizon + 1], dtype=np.float32))
            state_list.append(np.array(sim_state[ep, t + 1 : t + max_horizon + 1], dtype=np.float32))
            meta.append({"shard": str(shard_path), "episode": ep, "anchor_t": t})

    if len(ctx_list) < num_clips:
        raise RuntimeError(f"Sampled only {len(ctx_list)}/{num_clips} clips; loosen constraints.")

    return {
        "context_max": np.stack(ctx_list),
        "actions": np.stack(act_list),
        "gt_future": np.stack(gt_list),
        "sim_state": np.stack(state_list),  # [N, H, 16, 4]
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Color-keyed ball detection
# ---------------------------------------------------------------------------

def detect_balls(
    frame_rgb_u8: np.ndarray,
    color_thresh: int,
    min_area: int,
    max_area: int,
    excluded_ball_ids: Optional[set[int]] = None,
) -> Dict[int, Tuple[float, float]]:
    """Detect balls by color in a [H, W, 3] uint8 frame.

    Returns {ball_color_id: (x, y)} for the largest valid component per color.
    """
    skip = excluded_ball_ids if excluded_ball_ids is not None else EXCLUDED_BALL_IDS
    img = frame_rgb_u8.astype(np.int16)
    out: Dict[int, Tuple[float, float]] = {}
    for ci, c in enumerate(BALL_COLORS):
        if ci in skip:
            continue
        dist = np.abs(img - np.array(c, dtype=np.int16)).sum(axis=-1)
        mask = dist < color_thresh
        if not mask.any():
            continue
        lab, n = ndimage.label(mask)
        best: Optional[Tuple[float, float, int]] = None
        for i in range(1, n + 1):
            comp = lab == i
            area = int(comp.sum())
            if min_area <= area <= max_area and (best is None or area > best[2]):
                cy, cx = ndimage.center_of_mass(comp)
                best = (float(cx), float(cy), area)
        if best is not None:
            out[ci] = (best[0], best[1])
    return out


def _frame_physics(
    pred_dets: Dict[int, Tuple[float, float]],
    ref_dets: Dict[int, Tuple[float, float]],
    match_radius: float,
) -> Dict[str, float]:
    """Compare predicted-frame detections against reference-frame detections.

    The reference is the detector output on the VAE-decoded ground-truth frame,
    so detector recall limitations cancel in paired comparisons.
    """
    ref_ids = set(ref_dets)
    pred_ids = set(pred_dets)
    n_ref = len(ref_ids)

    if n_ref == 0:
        return {
            "ball_recall": float("nan"),
            "missing_rate": float("nan"),
            "spurious_rate": float("nan"),
            "identity_recall": float("nan"),
            "position_err_px": float("nan"),
        }

    # Identity-aware position match: same color id within radius.
    matched, pos_errs = 0, []
    for ci in ref_ids & pred_ids:
        rx, ry = ref_dets[ci]
        px, py = pred_dets[ci]
        err = float(np.hypot(rx - px, ry - py))
        if err <= match_radius:
            matched += 1
            pos_errs.append(err)

    # Identity-agnostic count match (Hungarian) for swallow/hallucinate rates.
    ref_xy = np.array(list(ref_dets.values()), dtype=np.float64)
    pred_xy = (
        np.array(list(pred_dets.values()), dtype=np.float64)
        if pred_dets
        else np.zeros((0, 2))
    )
    count_matched = 0
    if len(pred_xy) > 0:
        cost = np.hypot(
            ref_xy[:, None, 0] - pred_xy[None, :, 0],
            ref_xy[:, None, 1] - pred_xy[None, :, 1],
        )
        ri, ci_ = linear_sum_assignment(cost)
        count_matched = int((cost[ri, ci_] <= match_radius).sum())

    return {
        "ball_recall": count_matched / n_ref,
        "missing_rate": (n_ref - count_matched) / n_ref,
        "spurious_rate": max(0, len(pred_ids) - count_matched) / n_ref,
        "identity_recall": matched / n_ref,
        "position_err_px": float(np.mean(pos_errs)) if pos_errs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _bootstrap_ci(x: np.ndarray, iters: int, alpha: float, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    iters: int,
    alpha: float,
) -> Dict[str, float]:
    """Paired stats for per-clip metric arrays of two models (same clips)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    out: Dict[str, float] = {
        "n": int(len(d)),
        "mean_a": float(np.mean(a)) if len(a) else float("nan"),
        "mean_b": float(np.mean(b)) if len(b) else float("nan"),
        "mean_diff": float(np.mean(d)) if len(d) else float("nan"),
    }
    lo, hi = _bootstrap_ci(d, iters, alpha)
    out["diff_ci95_lo"], out["diff_ci95_hi"] = lo, hi
    if len(d) >= 6 and np.any(d != 0):
        try:
            _, p = wilcoxon(d)
            out["wilcoxon_p"] = float(p)
            # Rank-biserial from signed ranks directly: positive means a > b.
            nz = d[d != 0]
            ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
            total = ranks.sum()
            r_pos = ranks[nz > 0].sum()
            r_neg = ranks[nz < 0].sum()
            out["rank_biserial"] = float((r_pos - r_neg) / total) if total > 0 else float("nan")
        except ValueError:
            out["wilcoxon_p"] = float("nan")
            out["rank_biserial"] = float("nan")
    else:
        out["wilcoxon_p"] = float("nan")
        out["rank_biserial"] = float("nan")
    sd = np.std(d, ddof=1) if len(d) > 1 else float("nan")
    out["cohens_d_paired"] = float(np.mean(d) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    return out


def holm_correct(pvals: Dict[str, float]) -> Dict[str, float]:
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adj[k] = running
    for k, v in pvals.items():
        if k not in adj:
            adj[k] = float("nan")
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Physics-grounded rollout eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-id", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_cfg, data_cfg, vae_cfg, phys_cfg = cfg["run"], cfg["data"], cfg["vae"], cfg["physics"]

    eval_id = args.eval_id or str(run_cfg.get("eval_id") or f"physics_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    eval_dir = Path(str(run_cfg["output_root"])).resolve() / eval_id
    ensure_dir(eval_dir)
    set_seed(int(run_cfg["seed"]))

    requested = str(run_cfg.get("device", "cuda")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested in ("cuda", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    horizons = sorted({int(h) for h in run_cfg["horizons"] if int(h) >= 1})
    max_horizon = max(horizons)

    bundles = [_load_model_bundle(m, device=device) for m in cfg["models"]]
    max_context_len = max(b.context_len for b in bundles)

    if not bool(vae_cfg.get("enabled", True)):
        raise ValueError("Physics eval requires a VAE for decoding (vae.enabled: true)")
    vae = _load_vae(vae_cfg=vae_cfg, device=device)
    decode_bs = int(run_cfg.get("decode_batch_size", 128))

    shards = resolve_eval_shards_from_data_cfg(data_cfg)
    clips = sample_eval_clips_with_state(
        shards=shards,
        num_clips=int(run_cfg["num_clips"]),
        max_context_len=max_context_len,
        max_horizon=max_horizon,
        seed=int(run_cfg["seed"]),
    )

    class _Clips:
        pass

    clip_obj = _Clips()
    clip_obj.context_max = clips["context_max"]
    clip_obj.actions = clips["actions"]
    clip_obj.gt_future = clips["gt_future"]
    clip_obj.meta = clips["meta"]

    n_clips = clips["context_max"].shape[0]
    det_kwargs = dict(
        color_thresh=int(phys_cfg["color_match_threshold"]),
        min_area=int(phys_cfg["min_component_area"]),
        max_area=int(phys_cfg["max_component_area"]),
        excluded_ball_ids=set(int(x) for x in phys_cfg.get("excluded_ball_ids", [])),
    )
    match_radius = float(phys_cfg["match_radius_px"])

    # Reference detections: detector applied to VAE-decoded GT latents.
    gt_frames = _decode_latents(
        vae, torch.from_numpy(clips["gt_future"]).float(), device=device, batch_size=decode_bs
    )  # [N, H, 3, Himg, Wimg], values in [0, 1]
    gt_u8 = (gt_frames.numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
    ref_dets = [
        [detect_balls(gt_u8[i, h], **det_kwargs) for h in range(max_horizon)]
        for i in range(n_clips)
    ]

    metric_names = ["ball_recall", "missing_rate", "spurious_rate", "identity_recall", "position_err_px"]
    per_model: Dict[str, Dict[str, Any]] = {}
    per_clip_store: Dict[str, Any] = {}
    preds_by_model: Dict[str, torch.Tensor] = {}
    gt_latent = torch.from_numpy(clips["gt_future"]).float()
    rollout_bs = int(run_cfg.get("rollout_batch_size", 32))

    for bundle in bundles:
        print(f"[physics-eval] rolling out model={bundle.name}")
        preds = _rollout_predictions(
            bundle=bundle,
            clips=clip_obj,
            max_context_len=max_context_len,
            max_horizon=max_horizon,
            ddim_steps=int(run_cfg["ddim_steps"]),
            device=device,
            batch_size=rollout_bs,
        )
        preds_by_model[bundle.name] = preds
        pred_frames = _decode_latents(vae, preds, device=device, batch_size=decode_bs)
        pred_u8 = (pred_frames.numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)

        # per_clip[metric][h] -> [n_clips]
        per_clip = {m: {h: np.full(n_clips, np.nan) for h in horizons} for m in metric_names}
        for i in range(n_clips):
            for h in horizons:
                dets = detect_balls(pred_u8[i, h - 1], **det_kwargs)
                stats = _frame_physics(dets, ref_dets[i][h - 1], match_radius)
                for m in metric_names:
                    per_clip[m][h][i] = stats[m]

        agg: Dict[str, Any] = {}
        for m in metric_names:
            agg[m] = {}
            for h in horizons:
                x = per_clip[m][h]
                lo, hi = _bootstrap_ci(x, int(phys_cfg["bootstrap_iters"]), float(phys_cfg["alpha"]))
                agg[m][str(h)] = {
                    "mean": float(np.nanmean(x)) if np.isfinite(x).any() else float("nan"),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n": int(np.isfinite(x).sum()),
                }

        # Latent + pixel metrics from the same rollout (avoids a separate eval_rollout pass).
        agg.update(_latent_metrics(preds=preds, gt=gt_latent, horizons=horizons))
        psnr_map: Dict[str, float] = {}
        l1_map: Dict[str, float] = {}
        with torch.no_grad():
            for h in horizons:
                ph = pred_frames[:, h - 1]
                gh = gt_frames[:, h - 1]
                diff = ph - gh
                mse = (diff * diff).mean(dim=(1, 2, 3)).clamp(min=1e-10)
                psnr_map[str(h)] = float((10.0 * torch.log10(1.0 / mse)).mean().item())
                l1_map[str(h)] = float(diff.abs().mean().item())
        agg["frame_psnr"] = psnr_map
        agg["frame_l1"] = l1_map

        per_model[bundle.name] = agg
        per_clip_store[bundle.name] = {
            m: {str(h): per_clip[m][h].tolist() for h in horizons} for m in metric_names
        }

    num_viz = max(0, min(int(run_cfg.get("num_viz_clips", 0)), n_clips))
    if num_viz > 0:
        pred_viz_decoded = {
            name: _decode_latents(vae, preds[:num_viz], device=device, batch_size=decode_bs)
            for name, preds in preds_by_model.items()
        }
        video_outputs = _write_comparison_videos(
            output_dir=eval_dir / "videos",
            model_names=[b.name for b in bundles],
            gt_frames=gt_frames[:num_viz],
            pred_frames_by_model=pred_viz_decoded,
            fps=int(run_cfg.get("video_fps", 12)),
        )
        save_json(video_outputs, eval_dir / "video_outputs.json")

    # Pairwise model comparisons with Holm correction across horizons per metric.
    comparisons: Dict[str, Any] = {}
    names = [b.name for b in bundles]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            key = f"{a_name}_vs_{b_name}"
            comp: Dict[str, Any] = {}
            for m in metric_names:
                pvals: Dict[str, float] = {}
                per_h: Dict[str, Any] = {}
                for h in horizons:
                    a = np.array(per_clip_store[a_name][m][str(h)])
                    b = np.array(per_clip_store[b_name][m][str(h)])
                    stats = paired_comparison(
                        a, b, int(phys_cfg["bootstrap_iters"]), float(phys_cfg["alpha"])
                    )
                    per_h[str(h)] = stats
                    pvals[str(h)] = stats["wilcoxon_p"]
                adj = holm_correct(pvals)
                for h_key, stats in per_h.items():
                    stats["wilcoxon_p_holm"] = adj[h_key]
                comp[m] = per_h
            comparisons[key] = comp

    summary = {
        "eval_id": eval_id,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "horizons": horizons,
        "num_clips": n_clips,
        "detector": {
            **det_kwargs,
            "match_radius_px": match_radius,
            "excluded_ball_ids": sorted(det_kwargs["excluded_ball_ids"]),
        },
        "models": per_model,
        "paired_comparisons": comparisons,
    }
    save_json(summary, eval_dir / "physics_summary.json")
    save_json(per_clip_store, eval_dir / "physics_per_clip.json")
    save_json({"clip_meta": clips["meta"]}, eval_dir / "manifest_physics.json")
    print(f"Physics eval complete: {eval_dir}")


if __name__ == "__main__":
    main()
