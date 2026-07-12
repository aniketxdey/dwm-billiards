from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae_training.config import load_config
from vae_training.data import (
    FrameCacheDataset,
    ShardStreamDataset,
    sample_preview_frames_from_shards,
)
from vae_training.losses import LPIPSWrapper, compute_vae_loss
from vae_training.model import ConvVAE
from vae_training.preview import save_recon_grid, save_recon_video
from vae_training.utils import append_jsonl, ensure_dir, now_utc_stamp, save_json, set_seed


RUN_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _dtype_from_name(name: str) -> torch.dtype:
    n = name.lower()
    if n == "fp16":
        return torch.float16
    if n == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_run_id(id_prefix: str, cli_run_id: str) -> str:
    run_id = cli_run_id.strip() if cli_run_id else f"{id_prefix}_{now_utc_stamp()}"
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            f"Invalid run id '{run_id}'. Allowed chars: letters, digits, dot, underscore, hyphen"
        )
    return run_id


def _make_run_dirs(root: Path, run_id: str) -> Dict[str, Path]:
    run_dir = root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Use a new run id or clean the old directory first."
        )

    ensure_dir(run_dir)
    dirs = {
        "run": run_dir,
        "checkpoints": ensure_dir(run_dir / "checkpoints"),
        "previews": ensure_dir(run_dir / "previews"),
        "metrics": ensure_dir(run_dir / "metrics"),
        "config": ensure_dir(run_dir / "config"),
    }
    return dirs


def _save_checkpoint(
    path: Path,
    model: ConvVAE,
    optimizer: torch.optim.Optimizer,
    step: int,
    processed_frames: int,
    save_optimizer_state: bool,
) -> None:
    payload: Dict[str, Any] = {
        "step": step,
        "processed_frames": processed_frames,
        "model_state": model.state_dict(),
    }
    if save_optimizer_state:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conv VAE on frame cache")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-id", default="", help="Explicit run id (recommended)")
    parser.add_argument("--resume", default="", help="Optional checkpoint path")
    parser.add_argument("--notes", default="", help="Optional run notes")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_seed = int(cfg["run"]["seed"])
    set_seed(run_seed)
    run_id = _resolve_run_id(str(cfg["run"]["id_prefix"]), args.run_id)

    output_root = Path(cfg["run"]["output_root"]).resolve()
    ensure_dir(output_root)
    dirs = _make_run_dirs(output_root, run_id)

    save_json(cfg, dirs["config"] / "resolved_config.json")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    mp_dtype = _dtype_from_name(str(cfg["training"]["mixed_precision"]))

    data_source = str(cfg["data"].get("source", "frame_cache")).lower()
    if data_source == "shards":
        dataset = ShardStreamDataset(
            shards_dir=str(cfg["data"]["shards_dir"]),
            seed=run_seed,
            shuffle_shards=bool(cfg["data"].get("shuffle_shards", True)),
            shuffle_frames_within_shard=bool(
                cfg["data"].get("shuffle_frames_within_shard", True)
            ),
            repeat=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["data"]["batch_size"]),
            drop_last=bool(cfg["data"]["drop_last"]),
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=bool(cfg["data"]["pin_memory"]),
            persistent_workers=bool(cfg["data"]["persistent_workers"]),
            prefetch_factor=int(cfg["data"]["prefetch_factor"]),
        )
    else:
        dataset = FrameCacheDataset(
            frame_cache_path=str(cfg["data"]["frame_cache_path"]),
            frame_index_path=str(cfg["data"].get("frame_index_path", "")),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["data"]["batch_size"]),
            shuffle=bool(cfg["data"]["shuffle"]),
            drop_last=bool(cfg["data"]["drop_last"]),
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=bool(cfg["data"]["pin_memory"]),
            persistent_workers=bool(cfg["data"]["persistent_workers"]),
            prefetch_factor=int(cfg["data"]["prefetch_factor"]),
        )

    model = ConvVAE(
        in_channels=3,
        base_channels=int(cfg["model"]["base_channels"]),
        latent_channels=int(cfg["model"]["latent_channels"]),
    ).to(device)

    if bool(cfg["training"].get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    start_step = 0
    processed_frames = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt.get("step", 0))
        processed_frames = int(ckpt.get("processed_frames", 0))

    lpips_weight = float(cfg["loss"].get("lpips_weight", 0.0))
    lpips_fn = LPIPSWrapper(enabled=lpips_weight > 0).to(device)

    use_autocast = device.type == "cuda" and mp_dtype in (torch.float16, torch.bfloat16)

    wandb_run = None
    if bool(cfg["wandb"].get("enabled", False)):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(cfg["wandb"]["project"]),
                entity=str(cfg["wandb"].get("entity", "")) or None,
                name=run_id,
                config=cfg,
                notes=args.notes or None,
            )
        except Exception as exc:
            print(f"W&B init skipped: {exc}")

    preview_count = int(cfg["preview"]["fixed_preview_images"])
    video_frames = int(cfg["preview"]["fixed_preview_video_frames"])

    if data_source == "shards":
        shards_dir = str(cfg["data"]["shards_dir"])
        fixed_preview = sample_preview_frames_from_shards(
            shards_dir=shards_dir, count=preview_count, seed=run_seed
        ).to(device)
        fixed_video = sample_preview_frames_from_shards(
            shards_dir=shards_dir, count=video_frames, seed=run_seed + 1
        ).to(device)
    else:
        fixed_preview = torch.stack([dataset[i] for i in range(preview_count)], dim=0).to(device)
        fixed_video = torch.stack([dataset[i] for i in range(video_frames)], dim=0).to(device)

    target_frames = int(cfg["run"]["target_frames"])
    raw_milestones = cfg["run"].get("milestone_frames", [])
    milestone_frames = sorted(
        {
            int(m)
            for m in raw_milestones
            if int(m) > 0 and int(m) <= target_frames
        }
    )
    ckpt_every = int(cfg["run"]["checkpoint_every_frames"])
    preview_every = int(cfg["run"]["preview_every_frames"])
    log_every_steps = int(cfg["run"]["log_every_steps"])

    next_ckpt = ((processed_frames // ckpt_every) + 1) * ckpt_every
    next_preview = ((processed_frames // preview_every) + 1) * preview_every
    milestone_idx = 0
    while milestone_idx < len(milestone_frames) and milestone_frames[milestone_idx] <= processed_frames:
        milestone_idx += 1
    next_milestone = (
        milestone_frames[milestone_idx] if milestone_idx < len(milestone_frames) else target_frames + 1
    )

    metrics_path = dirs["metrics"] / "train_metrics.jsonl"
    run_manifest_path = dirs["run"] / "manifest.json"
    summary_path = dirs["run"] / "summary.json"
    registry_path = output_root / "run_registry.jsonl"

    manifest = {
        "run_id": run_id,
        "config_path": str(Path(args.config).resolve()),
        "resume_from": args.resume or None,
        "notes": args.notes or None,
        "device": str(device),
        "dataset_s3_root": cfg["run"].get("dataset_s3_root", ""),
        "dataset_local_root": cfg["run"].get("dataset_local_root", ""),
        "data_source": data_source,
        "milestone_frames": milestone_frames,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    save_json(manifest, run_manifest_path)
    append_jsonl({"event": "run_started", **manifest}, registry_path)

    step = start_step
    t0 = time.time()
    model.train()

    pbar = tqdm(total=target_frames, initial=processed_frames, desc="frames", unit="frm")

    while processed_frames < target_frames:
        for batch in loader:
            if processed_frames >= target_frames:
                break

            step += 1
            batch = batch.to(device, non_blocking=True)

            # Enforce exact preview/checkpoint/milestone frame boundaries.
            next_boundary = min(target_frames, next_ckpt, next_preview, next_milestone)
            max_take = min(target_frames - processed_frames, next_boundary - processed_frames)
            if max_take <= 0:
                max_take = min(target_frames - processed_frames, batch.shape[0])
            if batch.shape[0] > max_take:
                batch = batch[:max_take]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_autocast):
                recon, mu, logvar = model(batch)
                losses = compute_vae_loss(
                    x=batch,
                    recon=recon,
                    mu=mu,
                    logvar=logvar,
                    processed_frames=processed_frames,
                    cfg_loss=cfg["loss"],
                    lpips_fn=lpips_fn,
                )

            total = losses["total"]
            total.backward()

            max_grad_norm = float(cfg["training"]["max_grad_norm"])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            processed_frames += int(batch.shape[0])
            pbar.update(int(batch.shape[0]))

            did_preview = False
            did_ckpt = False
            did_milestone = False
            milestone_frame = None

            if processed_frames >= next_milestone:
                did_milestone = True
                milestone_frame = int(next_milestone)
                append_jsonl(
                    {
                        "event": "milestone",
                        "run_id": run_id,
                        "step": step,
                        "processed_frames": processed_frames,
                        "milestone_frame": milestone_frame,
                    },
                    registry_path,
                )
                milestone_idx += 1
                next_milestone = (
                    milestone_frames[milestone_idx]
                    if milestone_idx < len(milestone_frames)
                    else target_frames + 1
                )

            if processed_frames >= next_preview:
                did_preview = True
                model.eval()
                with torch.no_grad():
                    recon_img, _, _ = model(fixed_preview)
                    recon_vid, _, _ = model(fixed_video)

                preview_tag = f"{processed_frames:07d}"
                grid_path = dirs["previews"] / f"preview_grid_{preview_tag}.png"
                video_path = dirs["previews"] / f"preview_video_{preview_tag}.mp4"

                save_recon_grid(fixed_preview, recon_img, grid_path, nrow=8)
                saved_video_path = save_recon_video(fixed_video, recon_vid, video_path, fps=8)

                if wandb_run is not None:
                    try:
                        import wandb

                        video_fmt = saved_video_path.suffix.lower().lstrip(".") or "mp4"
                        wandb_run.log(
                            {
                                "preview/grid": wandb.Image(str(grid_path)),
                                "preview/video": wandb.Video(
                                    str(saved_video_path), fps=8, format=video_fmt
                                ),
                            },
                            step=step,
                        )
                    except Exception:
                        pass

                model.train()
                next_preview += preview_every

            if processed_frames >= next_ckpt:
                did_ckpt = True
                ckpt_path = dirs["checkpoints"] / f"ckpt_{processed_frames:07d}.pt"
                _save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    processed_frames=processed_frames,
                    save_optimizer_state=bool(cfg["run"]["save_optimizer_state"]),
                )
                next_ckpt += ckpt_every

            emit_log = (
                step % log_every_steps == 0
                or did_milestone
                or did_preview
                or did_ckpt
                or processed_frames >= target_frames
            )
            if emit_log:
                elapsed = max(time.time() - t0, 1e-6)
                fps = processed_frames / elapsed
                row = {
                    "step": step,
                    "processed_frames": processed_frames,
                    "fps": float(fps),
                    "loss_total": float(total.detach().cpu()),
                    "loss_l1": float(losses["l1"].cpu()),
                    "loss_lpips": float(losses["lpips"].cpu()),
                    "loss_kl": float(losses["kl"].cpu()),
                    "beta": float(losses["beta"]),
                    "event_milestone": did_milestone,
                    "milestone_frame": milestone_frame,
                    "event_preview": did_preview,
                    "event_checkpoint": did_ckpt,
                }
                append_jsonl(row, metrics_path)

                if wandb_run is not None:
                    wandb_run.log(row, step=step)

                print(
                    f"step={step} frames={processed_frames}/{target_frames} "
                    f"fps={row['fps']:.1f} total={row['loss_total']:.4f} l1={row['loss_l1']:.4f} "
                    f"lpips={row['loss_lpips']:.4f} kl={row['loss_kl']:.6f} beta={row['beta']:.6f} "
                    f"milestone={milestone_frame} preview={did_preview} ckpt={did_ckpt}"
                )

    pbar.close()

    if processed_frames % ckpt_every == 0:
        final_ckpt = dirs["checkpoints"] / f"ckpt_{processed_frames:07d}.pt"
    else:
        final_ckpt = dirs["checkpoints"] / f"ckpt_{processed_frames:07d}_final.pt"
        _save_checkpoint(
            path=final_ckpt,
            model=model,
            optimizer=optimizer,
            step=step,
            processed_frames=processed_frames,
            save_optimizer_state=bool(cfg["run"]["save_optimizer_state"]),
        )

    duration_sec = max(time.time() - t0, 1e-6)
    end_manifest = {
        "run_id": run_id,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "processed_frames": processed_frames,
        "final_step": step,
        "duration_sec": duration_sec,
        "avg_fps": processed_frames / duration_sec,
        "final_checkpoint": str(final_ckpt),
    }
    save_json(end_manifest, summary_path)
    append_jsonl({"event": "run_finished", **end_manifest}, registry_path)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Run complete: {run_id}")
    print(f"Outputs: {dirs['run']}")


if __name__ == "__main__":
    main()
