"""World-model training entrypoint.

This module trains an action-conditioned DiT in latent space, with optional
Diffusion Forcing (multi-step rollout training).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from world_model_training.config import load_config
from world_model_training.data import (
    LatentActionRolloutIterableDataset,
    LatentActionIterableDataset,
    resolve_train_val_shards_from_data_cfg,
)
from world_model_training.diffusion import GaussianDiffusion
from world_model_training.model import ActionConditionedDiT
from world_model_training.utils import append_jsonl, ensure_dir, now_utc_stamp, save_json, set_seed


RUN_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _dtype_from_name(name: str) -> torch.dtype:
    """Map config mixed-precision names to torch dtypes."""
    n = name.lower()
    if n == "fp16":
        return torch.float16
    if n == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_run_id(id_prefix: str, cli_run_id: str) -> str:
    """Build/validate a run id used for output directories and logging."""
    run_id = cli_run_id.strip() if cli_run_id else f"{id_prefix}_{now_utc_stamp()}"
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            f"Invalid run id '{run_id}'. Allowed chars: letters, digits, dot, underscore, hyphen"
        )
    return run_id


def _make_run_dirs(root: Path, run_id: str) -> Dict[str, Path]:
    """Create a fresh run directory tree and fail if the run already exists."""
    run_dir = root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Use a new run id or clean the old directory first."
        )

    ensure_dir(run_dir)
    return {
        "run": run_dir,
        "checkpoints": ensure_dir(run_dir / "checkpoints"),
        "metrics": ensure_dir(run_dir / "metrics"),
        "config": ensure_dir(run_dir / "config"),
    }


def _run_dirs(root: Path, run_id: str) -> Dict[str, Path]:
    """Return run directory paths without creating them."""
    run_dir = root / run_id
    return {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "metrics": run_dir / "metrics",
        "config": run_dir / "config",
    }


def _loader_kwargs(cfg: Dict[str, Any], workers: int) -> Dict[str, Any]:
    """Build DataLoader kwargs from config with worker-specific options."""
    kwargs: Dict[str, Any] = {
        "batch_size": int(cfg["data"]["batch_size"]),
        "drop_last": bool(cfg["data"]["drop_last"]),
        "num_workers": int(workers),
        "pin_memory": bool(cfg["data"]["pin_memory"]),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(cfg["data"]["persistent_workers"])
        kwargs["prefetch_factor"] = int(cfg["data"]["prefetch_factor"])
    return kwargs


def _is_distributed() -> bool:
    """Detect distributed mode from environment variables."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _distributed_info() -> Tuple[int, int, int]:
    """Read rank/local-rank/world-size from torchrun-style env variables."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _maybe_init_distributed(cuda_available: bool) -> Tuple[bool, int, int, int]:
    """Initialize torch.distributed when WORLD_SIZE > 1."""
    rank, local_rank, world_size = _distributed_info()
    use_distributed = _is_distributed()
    if not use_distributed:
        return False, 0, 0, 1

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this environment.")
    if cuda_available and not torch.cuda.is_available():
        raise RuntimeError("WORLD_SIZE>1 but CUDA is not available.")
    # Pin each rank to its GPU before NCCL init.
    if cuda_available:
        torch.cuda.set_device(local_rank)
    backend = "nccl" if cuda_available else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, local_rank, world_size


def _main_process(rank: int) -> bool:
    """Return True only for rank 0 (the process that writes run artifacts)."""
    return rank == 0


def _barrier(use_distributed: bool) -> None:
    """Synchronization barrier used around shared side effects."""
    if use_distributed:
        dist.barrier()


def _reduce_mean(value: torch.Tensor, use_distributed: bool) -> torch.Tensor:
    """Average a scalar tensor across ranks when distributed training is enabled."""
    if not use_distributed:
        return value
    t = value.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def _model_for_io(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP model for state_dict save/load."""
    return model.module if isinstance(model, DDP) else model


def _df_enabled(cfg: Dict[str, Any]) -> bool:
    """Check whether diffusion-forcing mode is enabled in config."""
    return bool(cfg.get("diffusion_forcing", {}).get("enabled", False))


def _df_teacher_forcing_prob(cfg: Dict[str, Any], processed_samples: int) -> float:
    """Linearly decay teacher-forcing probability over processed samples."""
    df_cfg = cfg.get("diffusion_forcing", {})
    start = float(df_cfg.get("teacher_forcing_prob_start", 1.0))
    end = float(df_cfg.get("teacher_forcing_prob_end", start))
    decay = int(df_cfg.get("teacher_forcing_decay_samples", 0))
    if decay <= 0:
        return end
    alpha = min(max(processed_samples, 0), decay) / float(decay)
    return start + (end - start) * alpha


def _corrupt_context(
    context: torch.Tensor,
    diffusion: GaussianDiffusion,
    tau_max: int,
) -> torch.Tensor:
    """GameNGen-style context corruption.

    Independently noise each context latent to a random diffusion level in
    [0, tau_max] so the model learns to correct imperfect history.
    context: [B, L, C, H, W]
    """
    b, l = context.shape[0], context.shape[1]
    tau = torch.randint(0, tau_max + 1, (b, l), device=context.device)
    alpha_bar = diffusion.alpha_bar[tau]  # [B, L]
    alpha_bar = alpha_bar.view(b, l, 1, 1, 1).to(context.dtype)
    noise = torch.randn_like(context)
    return torch.sqrt(alpha_bar) * context + torch.sqrt(1.0 - alpha_bar) * noise


def _df_rollout_training_loss(
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    context: torch.Tensor,
    actions_seq: torch.Tensor,
    targets_seq: torch.Tensor,
    teacher_forcing_prob: float,
    detach_predicted_context: bool,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-step diffusion-forcing loss.

    For each rollout step:
    1) sample diffusion timestep/noise
    2) train model to predict noise (MSE)
    3) update context with GT/predicted next latent mixture
    """
    # context: [B, L, C, H, W], actions_seq: [B, K, A], targets_seq: [B, K, C, H, W]
    bsz = int(context.shape[0])
    rollout_steps = int(actions_seq.shape[1])
    ctx = context
    losses = []
    teacher_frames = 0
    predicted_frames = 0

    for k in range(rollout_steps):
        action_k = actions_seq[:, k]
        target_k = targets_seq[:, k]
        t_idx = diffusion.sample_timesteps(bsz, device=target_k.device)
        noise = torch.randn_like(target_k)
        noisy = diffusion.q_sample(target_k, t_idx, noise)
        pred_noise = model(context=ctx, action=action_k, noisy_target=noisy, t_idx=t_idx)
        loss_k = torch.mean((pred_noise - noise) ** 2)
        losses.append(loss_k)

        if k >= rollout_steps - 1:
            continue

        pred_x0 = diffusion.predict_x0_from_eps(noisy, t_idx, pred_noise)
        if detach_predicted_context:
            pred_x0 = pred_x0.detach()

        if teacher_forcing_prob >= 1.0:
            next_latent = target_k
            teacher_frames += bsz
        elif teacher_forcing_prob <= 0.0:
            next_latent = pred_x0
            predicted_frames += bsz
        else:
            teacher_mask = (torch.rand(bsz, 1, 1, 1, device=target_k.device) < teacher_forcing_prob)
            next_latent = torch.where(teacher_mask, target_k, pred_x0)
            teacher_frames += int(teacher_mask.sum().item())
            predicted_frames += bsz - int(teacher_mask.sum().item())

        ctx = torch.cat([ctx[:, 1:], next_latent.unsqueeze(1)], dim=1)

    total_loss = torch.stack(losses).mean()
    aux = {
        "df_rollout_steps": float(rollout_steps),
        "df_loss_step0": float(losses[0].detach().cpu()),
        "df_loss_last": float(losses[-1].detach().cpu()),
        "df_teacher_forcing_prob": float(teacher_forcing_prob),
        "df_teacher_context_frames": float(teacher_frames),
        "df_pred_context_frames": float(predicted_frames),
    }
    return total_loss, aux


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    processed_samples: int,
    save_optimizer_state: bool,
) -> None:
    """Save model (and optional optimizer) checkpoint with training progress."""
    payload: Dict[str, Any] = {
        "step": step,
        "processed_samples": processed_samples,
        "model_state": _model_for_io(model).state_dict(),
    }
    if save_optimizer_state:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def evaluate(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    mp_dtype: torch.dtype,
    use_autocast: bool,
    num_batches: int,
    cfg: Dict[str, Any],
    processed_samples: int,
) -> float:
    """Run short validation loop and return average validation loss."""
    model.eval()
    total = 0.0
    n = 0
    use_df = _df_enabled(cfg)
    df_detach = bool(cfg.get("diffusion_forcing", {}).get("detach_predicted_context", True))
    teacher_forcing_prob = _df_teacher_forcing_prob(cfg, processed_samples)
    with torch.no_grad():
        it = iter(loader)
        for _ in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            if use_df:
                context, actions_seq, targets_seq = batch
                context = context.to(device, non_blocking=True)
                actions_seq = actions_seq.to(device, non_blocking=True)
                targets_seq = targets_seq.to(device, non_blocking=True)
            else:
                context, action, target = batch
                context = context.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_autocast):
                if use_df:
                    loss, _ = _df_rollout_training_loss(
                        model=model,
                        diffusion=diffusion,
                        context=context,
                        actions_seq=actions_seq,
                        targets_seq=targets_seq,
                        teacher_forcing_prob=teacher_forcing_prob,
                        detach_predicted_context=df_detach,
                    )
                else:
                    t = diffusion.sample_timesteps(target.shape[0], device=device)
                    loss, _, _ = diffusion.training_loss(model, context, action, target, t)
            total += float(loss.detach().cpu())
            n += 1
    model.train()
    if n == 0:
        return float("nan")
    return total / n


def main() -> None:
    """Train the world model end-to-end from config."""
    # --- CLI/config/bootstrap -------------------------------------------------
    parser = argparse.ArgumentParser(description="Train action-conditioned DiT baseline on latent shards")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-id", default="", help="Explicit run id")
    parser.add_argument("--resume", default="", help="Optional checkpoint path")
    parser.add_argument("--notes", default="", help="Optional run notes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_distributed, rank, local_rank, world_size = _maybe_init_distributed(
        cuda_available=torch.cuda.is_available()
    )
    is_main = _main_process(rank)

    seed = int(cfg["run"]["seed"]) + rank
    set_seed(seed)

    # Ensure all ranks agree on one run id (broadcast from rank 0).
    if use_distributed:
        run_id_holder = [_resolve_run_id(str(cfg["run"]["id_prefix"]), args.run_id) if is_main else ""]
        dist.broadcast_object_list(run_id_holder, src=0)
        run_id = str(run_id_holder[0])
    else:
        run_id = _resolve_run_id(str(cfg["run"]["id_prefix"]), args.run_id)

    output_root = Path(cfg["run"]["output_root"]).resolve()
    if is_main:
        ensure_dir(output_root)
        dirs = _make_run_dirs(output_root, run_id)
        save_json(cfg, dirs["config"] / "resolved_config.json")
    _barrier(use_distributed)
    dirs = _run_dirs(output_root, run_id)

    if use_distributed and int(cfg["run"]["target_samples"]) % world_size != 0:
        raise ValueError(
            f"run.target_samples ({cfg['run']['target_samples']}) must be divisible by WORLD_SIZE ({world_size})"
        )

    # Device + mixed precision setup.
    if torch.cuda.is_available():
        if use_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available() and not use_distributed:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mp_dtype = _dtype_from_name(str(cfg["training"]["mixed_precision"]))
    use_autocast = device.type == "cuda" and mp_dtype in (torch.float16, torch.bfloat16)

    # --- Data pipeline --------------------------------------------------------
    train_shards, val_shards = resolve_train_val_shards_from_data_cfg(cfg["data"])

    use_df = _df_enabled(cfg)
    df_rollout_steps = int(cfg.get("diffusion_forcing", {}).get("rollout_steps", 1))

    if use_df:
        train_ds = LatentActionRolloutIterableDataset(
            shard_paths=train_shards,
            context_len=int(cfg["data"]["context_len"]),
            rollout_steps=df_rollout_steps,
            seed=int(cfg["run"]["seed"]),
            distributed_rank=rank,
            distributed_world_size=world_size,
            repeat=True,
            shuffle_shards=bool(cfg["data"]["shuffle_shards"]),
            shuffle_within_episode=bool(cfg["data"]["shuffle_within_episode"]),
        )
    else:
        train_ds = LatentActionIterableDataset(
            shard_paths=train_shards,
            context_len=int(cfg["data"]["context_len"]),
            seed=int(cfg["run"]["seed"]),
            distributed_rank=rank,
            distributed_world_size=world_size,
            repeat=True,
            shuffle_shards=bool(cfg["data"]["shuffle_shards"]),
            shuffle_within_episode=bool(cfg["data"]["shuffle_within_episode"]),
        )
    train_workers = int(cfg["data"]["num_workers"])
    train_loader = DataLoader(train_ds, **_loader_kwargs(cfg, workers=train_workers))

    # Optional validation loader (kept deterministic/no shuffle within episodes).
    val_loader = None
    if val_shards:
        if use_df:
            val_ds = LatentActionRolloutIterableDataset(
                shard_paths=val_shards,
                context_len=int(cfg["data"]["context_len"]),
                rollout_steps=df_rollout_steps,
                seed=int(cfg["run"]["seed"]) + 999,
                distributed_rank=0,
                distributed_world_size=1,
                repeat=True,
                shuffle_shards=False,
                shuffle_within_episode=False,
            )
        else:
            val_ds = LatentActionIterableDataset(
                shard_paths=val_shards,
                context_len=int(cfg["data"]["context_len"]),
                seed=int(cfg["run"]["seed"]) + 999,
                distributed_rank=0,
                distributed_world_size=1,
                repeat=True,
                shuffle_shards=False,
                shuffle_within_episode=False,
            )
        val_loader = DataLoader(
            val_ds,
            **_loader_kwargs(cfg, workers=max(1, train_workers // 2)),
        )

    # --- Model/optimizer/diffusion -------------------------------------------
    model = ActionConditionedDiT(
        latent_channels=int(cfg["model"]["latent_channels"]),
        latent_h=int(cfg["model"]["latent_h"]),
        latent_w=int(cfg["model"]["latent_w"]),
        context_len=int(cfg["data"]["context_len"]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        mlp_ratio=float(cfg["model"]["mlp_ratio"]),
        dropout=float(cfg["model"]["dropout"]),
        action_dim=int(cfg["model"]["action_dim"]),
        diffusion_steps=int(cfg["diffusion"]["timesteps"]),
    ).to(device)

    if bool(cfg["training"].get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    if use_distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    diffusion = GaussianDiffusion(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_start=float(cfg["diffusion"]["beta_start"]),
        beta_end=float(cfg["diffusion"]["beta_end"]),
        device=device,
    )

    step = 0
    processed_samples = 0

    # Resume checkpoint if requested.
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        _model_for_io(model).load_state_dict(ckpt["model_state"], strict=True)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        step = int(ckpt.get("step", 0))
        processed_samples = int(ckpt.get("processed_samples", 0))
    start_processed_samples = processed_samples

    wandb_run = None
    # Initialize W&B on main process only to avoid duplicate runs.
    if is_main and bool(cfg["wandb"].get("enabled", False)):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(cfg["wandb"]["project"]),
                entity=str(cfg["wandb"].get("entity", "")) or None,
                group=str(cfg["wandb"].get("group", "")) or None,
                tags=list(cfg["wandb"].get("tags", [])),
                name=run_id,
                config=cfg,
                notes=args.notes or None,
            )
        except Exception as exc:
            print(f"W&B init skipped: {exc}")

    # Validate scheduling knobs early.
    target_samples = int(cfg["run"]["target_samples"])
    ckpt_every = int(cfg["run"]["checkpoint_every_samples"])
    log_every = int(cfg["run"]["log_every_steps"])
    eval_every = int(cfg["run"]["eval_every_steps"])
    val_batches = int(cfg["eval"]["val_batches"])
    if target_samples <= 0:
        raise ValueError("run.target_samples must be > 0")
    if ckpt_every <= 0:
        raise ValueError("run.checkpoint_every_samples must be > 0")
    if log_every <= 0:
        raise ValueError("run.log_every_steps must be > 0")
    if eval_every <= 0:
        raise ValueError("run.eval_every_steps must be > 0")

    next_ckpt = ((processed_samples // ckpt_every) + 1) * ckpt_every

    # Paths for metrics/summary/manifest bookkeeping.
    metrics_path = dirs["metrics"] / "train_metrics.jsonl"
    summary_path = dirs["run"] / "summary.json"
    manifest_path = dirs["run"] / "manifest.json"
    registry_path = output_root / "run_registry.jsonl"

    manifest = {
        "run_id": run_id,
        "training_mode": "dit_df_v0" if use_df else "dit_baseline",
        "config_path": str(Path(args.config).resolve()),
        "resume_from": args.resume or None,
        "notes": args.notes or None,
        "device": str(device),
        "rank": rank,
        "world_size": world_size,
        "source_latent_s3_root": cfg["run"].get("source_latent_s3_root", ""),
        "source_latent_local_root": cfg["run"].get("source_latent_local_root", ""),
        "shards_dir": cfg["data"].get("shards_dir", ""),
        "shards_manifest": cfg["data"].get("shards_manifest", ""),
        "train_shards_manifest": cfg["data"].get("train_shards_manifest", ""),
        "val_shards_manifest": cfg["data"].get("val_shards_manifest", ""),
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "df_enabled": use_df,
        "df_rollout_steps": df_rollout_steps if use_df else 0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    if is_main:
        save_json(manifest, manifest_path)
        append_jsonl({"event": "run_started", **manifest}, registry_path)

    # --- Training loop --------------------------------------------------------
    t0 = time.time()
    model.train()

    loader_iter = iter(train_loader)
    while processed_samples < target_samples:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            continue

        step += 1

        if use_df:
            context, actions_seq, targets_seq = batch
        else:
            context, action, target = batch

        # Clip final batch so total processed_samples hits target exactly.
        local_batch = int(context.shape[0])
        remaining = target_samples - processed_samples
        if use_distributed:
            local_take = remaining // world_size
            if local_take <= 0:
                break
            if local_batch > local_take:
                context = context[:local_take]
                if use_df:
                    actions_seq = actions_seq[:local_take]
                    targets_seq = targets_seq[:local_take]
                else:
                    action = action[:local_take]
                    target = target[:local_take]
                local_batch = local_take
            global_batch = local_batch * world_size
        else:
            if local_batch > remaining:
                context = context[:remaining]
                if use_df:
                    actions_seq = actions_seq[:remaining]
                    targets_seq = targets_seq[:remaining]
                else:
                    action = action[:remaining]
                    target = target[:remaining]
                local_batch = remaining
            global_batch = local_batch

        # Move tensors to device.
        context = context.to(device, non_blocking=True)
        if use_df:
            actions_seq = actions_seq.to(device, non_blocking=True)
            targets_seq = targets_seq.to(device, non_blocking=True)
        else:
            action = action.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        df_aux: Dict[str, float] = {}
        # Forward loss: baseline (single-step) or DF (multi-step rollout).
        with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_autocast):
            if use_df:
                teacher_forcing_prob = _df_teacher_forcing_prob(cfg, processed_samples)
                loss, df_aux = _df_rollout_training_loss(
                    model=model,
                    diffusion=diffusion,
                    context=context,
                    actions_seq=actions_seq,
                    targets_seq=targets_seq,
                    teacher_forcing_prob=teacher_forcing_prob,
                    detach_predicted_context=bool(
                        cfg.get("diffusion_forcing", {}).get("detach_predicted_context", True)
                    ),
                )
            else:
                ctx_noise_cfg = cfg.get("context_noise", {})
                if bool(ctx_noise_cfg.get("enabled", False)):
                    context = _corrupt_context(
                        context, diffusion, int(ctx_noise_cfg.get("tau_max", 150))
                    )
                t_idx = diffusion.sample_timesteps(target.shape[0], device=device)
                loss, _, _ = diffusion.training_loss(model, context, action, target, t_idx)

        loss.backward()
        # Global grad clipping stabilizes training at scale.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["training"]["max_grad_norm"]))
        optimizer.step()

        processed_samples += global_batch
        train_loss_log = float(_reduce_mean(loss.detach(), use_distributed).cpu())

        did_ckpt = False
        # Save checkpoints by sample count, not by step count.
        if processed_samples >= next_ckpt and is_main:
            did_ckpt = True
            ckpt_path = dirs["checkpoints"] / f"ckpt_{processed_samples:09d}.pt"
            _save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                step,
                processed_samples,
                bool(cfg["run"]["save_optimizer_state"]),
            )
            next_ckpt += ckpt_every
        elif processed_samples >= next_ckpt:
            next_ckpt += ckpt_every

        do_eval = val_loader is not None and (step % eval_every == 0)
        val_loss = float("nan")
        # Evaluation runs on main process; result is broadcast to all ranks.
        if do_eval:
            if use_distributed:
                dist.barrier()
            if is_main:
                val_loss = evaluate(
                    model=model,
                    diffusion=diffusion,
                    loader=val_loader,
                    device=device,
                    mp_dtype=mp_dtype,
                    use_autocast=use_autocast,
                    num_batches=val_batches,
                    cfg=cfg,
                    processed_samples=processed_samples,
                )
            if use_distributed:
                val_tensor = torch.tensor([val_loss if is_main else 0.0], device=device, dtype=torch.float32)
                dist.broadcast(val_tensor, src=0)
                val_loss = float(val_tensor.item())
                dist.barrier()

        emit_log = (
            step % log_every == 0
            or did_ckpt
            or do_eval
            or processed_samples >= target_samples
        )

        if emit_log and is_main:
            elapsed = max(time.time() - t0, 1e-6)
            current_run_samples = max(processed_samples - start_processed_samples, 1)
            sps = current_run_samples / elapsed
            row = {
                "step": step,
                "processed_samples": processed_samples,
                "samples_per_sec": float(sps),
                "train_loss": train_loss_log,
                "val_loss": float(val_loss),
                "event_checkpoint": did_ckpt,
                "world_size": world_size,
                "global_batch_size": int(global_batch),
            }
            if use_df:
                row.update(df_aux)
            append_jsonl(row, metrics_path)
            if wandb_run is not None:
                wandb_run.log(row, step=step)

            print(
                f"step={step} samples={processed_samples}/{target_samples} "
                f"sps={row['samples_per_sec']:.1f} train_loss={row['train_loss']:.6f} "
                f"val_loss={row['val_loss']:.6f} ckpt={did_ckpt} world_size={world_size}"
            )

    # --- Finalization ---------------------------------------------------------
    _barrier(use_distributed)
    final_ckpt = dirs["checkpoints"] / f"ckpt_{processed_samples:09d}.pt"
    if is_main and not final_ckpt.exists():
        _save_checkpoint(
            final_ckpt,
            model,
            optimizer,
            step,
            processed_samples,
            bool(cfg["run"]["save_optimizer_state"]),
        )

    duration_sec = max(time.time() - t0, 1e-6)
    current_run_samples = max(processed_samples - start_processed_samples, 0)
    summary = {
        "run_id": run_id,
        "training_mode": "dit_df_v0" if use_df else "dit_baseline",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "processed_samples": processed_samples,
        "processed_samples_this_run": current_run_samples,
        "final_step": step,
        "duration_sec": duration_sec,
        "avg_samples_per_sec": (current_run_samples / duration_sec) if current_run_samples > 0 else 0.0,
        "final_checkpoint": str(final_ckpt),
        "world_size": world_size,
    }
    if use_df:
        summary["df_rollout_steps"] = df_rollout_steps
    if is_main:
        save_json(summary, summary_path)
        append_jsonl({"event": "run_finished", **summary}, registry_path)

        if wandb_run is not None:
            wandb_run.finish()

        print(f"Run complete: {run_id}")
        print(f"Outputs: {dirs['run']}")

    if use_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
