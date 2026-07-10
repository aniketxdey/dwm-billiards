"""Teacher-to-student distillation trainer for action-conditioned DiT world models.

This script trains a smaller student DiT to match a frozen teacher's noise
prediction on the same diffusion task. The optimization objective is:

  L = w_kd * MSE(eps_student, eps_teacher) + w_gt * MSE(eps_student, eps_true)

where eps_true is the sampled diffusion noise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

from world_model_training.config import load_config
from world_model_training.data import LatentActionIterableDataset, resolve_train_val_shards_from_data_cfg
from world_model_training.diffusion import GaussianDiffusion
from world_model_training.model import ActionConditionedDiT
from world_model_training.utils import append_jsonl, ensure_dir, now_utc_stamp, save_json, set_seed


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
    return {
        "run": run_dir,
        "checkpoints": ensure_dir(run_dir / "checkpoints"),
        "metrics": ensure_dir(run_dir / "metrics"),
        "config": ensure_dir(run_dir / "config"),
    }


def _run_dirs(root: Path, run_id: str) -> Dict[str, Path]:
    run_dir = root / run_id
    return {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "metrics": run_dir / "metrics",
        "config": run_dir / "config",
    }


def _loader_kwargs(cfg: Dict[str, Any], workers: int) -> Dict[str, Any]:
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
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _distributed_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _maybe_init_distributed(cuda_available: bool) -> Tuple[bool, int, int, int]:
    rank, local_rank, world_size = _distributed_info()
    use_distributed = _is_distributed()
    if not use_distributed:
        return False, 0, 0, 1

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this environment.")
    if cuda_available and not torch.cuda.is_available():
        raise RuntimeError("WORLD_SIZE>1 but CUDA is not available.")
    if cuda_available:
        torch.cuda.set_device(local_rank)
    backend = "nccl" if cuda_available else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, local_rank, world_size


def _main_process(rank: int) -> bool:
    return rank == 0


def _barrier(use_distributed: bool) -> None:
    if use_distributed:
        dist.barrier()


def _reduce_mean(value: torch.Tensor, use_distributed: bool) -> torch.Tensor:
    if not use_distributed:
        return value
    t = value.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def _model_for_io(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _load_structured(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f) or {}


def _build_model_from_cfg(cfg: Dict[str, Any], device: torch.device) -> ActionConditionedDiT:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    diff_cfg = cfg["diffusion"]
    return ActionConditionedDiT(
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


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    processed_samples: int,
    save_optimizer_state: bool,
) -> None:
    payload: Dict[str, Any] = {
        "step": step,
        "processed_samples": processed_samples,
        "model_state": _model_for_io(model).state_dict(),
    }
    if save_optimizer_state:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def evaluate(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    mp_dtype: torch.dtype,
    use_autocast: bool,
    num_batches: int,
    kd_weight: float,
    gt_weight: float,
) -> Dict[str, float]:
    student.eval()
    teacher.eval()
    total = 0.0
    total_kd = 0.0
    total_gt = 0.0
    n = 0
    with torch.no_grad():
        it = iter(loader)
        for _ in range(num_batches):
            try:
                context, action, target = next(it)
            except StopIteration:
                break
            context = context.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            bsz = int(target.shape[0])

            t_idx = diffusion.sample_timesteps(bsz, device=device)
            noise = torch.randn_like(target)
            noisy = diffusion.q_sample(target, t_idx, noise)

            with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_autocast):
                eps_teacher = teacher(context=context, action=action, noisy_target=noisy, t_idx=t_idx)
                eps_student = student(context=context, action=action, noisy_target=noisy, t_idx=t_idx)
                loss_kd = torch.mean((eps_student - eps_teacher) ** 2)
                loss_gt = torch.mean((eps_student - noise) ** 2)
                loss = kd_weight * loss_kd + gt_weight * loss_gt

            total += float(loss.detach().cpu())
            total_kd += float(loss_kd.detach().cpu())
            total_gt += float(loss_gt.detach().cpu())
            n += 1
    student.train()
    if n == 0:
        return {"loss": float("nan"), "kd_loss": float("nan"), "gt_loss": float("nan")}
    return {"loss": total / n, "kd_loss": total_kd / n, "gt_loss": total_gt / n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill teacher DiT into smaller student DiT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-id", default="", help="Explicit run id")
    parser.add_argument("--resume", default="", help="Optional student checkpoint to resume")
    parser.add_argument("--notes", default="", help="Optional run notes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    distill_cfg = dict(cfg.get("distill", {}))
    teacher_ckpt = str(distill_cfg.get("teacher_checkpoint_path", "")).strip()
    teacher_cfg_path = str(distill_cfg.get("teacher_config_path", "")).strip()
    if not teacher_ckpt or not teacher_cfg_path:
        raise ValueError(
            "distill.teacher_checkpoint_path and distill.teacher_config_path are required."
        )

    kd_weight = float(distill_cfg.get("kd_weight", 0.7))
    gt_weight = float(distill_cfg.get("gt_weight", 0.3))
    if kd_weight < 0.0 or gt_weight < 0.0 or (kd_weight + gt_weight) <= 0.0:
        raise ValueError("Invalid distill weights. Need kd_weight>=0, gt_weight>=0, and sum>0.")

    use_distributed, rank, local_rank, world_size = _maybe_init_distributed(
        cuda_available=torch.cuda.is_available()
    )
    is_main = _main_process(rank)
    set_seed(int(cfg["run"]["seed"]) + rank)

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

    if torch.cuda.is_available():
        if use_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    mp_dtype = _dtype_from_name(str(cfg["training"]["mixed_precision"]))
    use_autocast = device.type == "cuda" and mp_dtype in (torch.float16, torch.bfloat16)

    train_shards, val_shards = resolve_train_val_shards_from_data_cfg(cfg["data"])
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
    train_loader = DataLoader(train_ds, **_loader_kwargs(cfg, workers=int(cfg["data"]["num_workers"])))

    val_loader = None
    if val_shards:
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
            **_loader_kwargs(cfg, workers=max(1, int(cfg["data"]["num_workers"]) // 2)),
        )

    student = _build_model_from_cfg(cfg, device=device)
    if bool(cfg["training"].get("compile_model", False)) and hasattr(torch, "compile"):
        student = torch.compile(student)  # type: ignore[assignment]

    if use_distributed:
        student = DDP(
            student,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    teacher_cfg = _load_structured(Path(teacher_cfg_path).expanduser().resolve())
    teacher = _build_model_from_cfg(teacher_cfg, device=device)
    teacher_state = torch.load(Path(teacher_ckpt).expanduser().resolve(), map_location="cpu")
    teacher_state = teacher_state["model_state"] if isinstance(teacher_state, dict) and "model_state" in teacher_state else teacher_state
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Sanity checks for distillation compatibility.
    t_model = teacher_cfg["model"]
    s_model = cfg["model"]
    for key in ("latent_channels", "latent_h", "latent_w", "action_dim"):
        if int(t_model[key]) != int(s_model[key]):
            raise ValueError(f"Teacher/student mismatch for '{key}': {t_model[key]} vs {s_model[key]}")
    if int(teacher_cfg["data"]["context_len"]) != int(cfg["data"]["context_len"]):
        raise ValueError(
            f"Teacher/student mismatch for data.context_len: "
            f"{teacher_cfg['data']['context_len']} vs {cfg['data']['context_len']}"
        )

    optimizer = torch.optim.AdamW(
        student.parameters(),
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        _model_for_io(student).load_state_dict(ckpt["model_state"], strict=True)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        step = int(ckpt.get("step", 0))
        processed_samples = int(ckpt.get("processed_samples", 0))
    start_processed_samples = processed_samples

    wandb_run = None
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

    target_samples = int(cfg["run"]["target_samples"])
    ckpt_every = int(cfg["run"]["checkpoint_every_samples"])
    log_every = int(cfg["run"]["log_every_steps"])
    eval_every = int(cfg["run"]["eval_every_steps"])
    val_batches = int(cfg["eval"]["val_batches"])
    next_ckpt = ((processed_samples // ckpt_every) + 1) * ckpt_every

    metrics_path = dirs["metrics"] / "train_metrics.jsonl"
    summary_path = dirs["run"] / "summary.json"
    manifest_path = dirs["run"] / "manifest.json"
    registry_path = output_root / "run_registry.jsonl"

    manifest = {
        "run_id": run_id,
        "training_mode": "distill_v0",
        "teacher_checkpoint_path": str(Path(teacher_ckpt).expanduser().resolve()),
        "teacher_config_path": str(Path(teacher_cfg_path).expanduser().resolve()),
        "config_path": str(Path(args.config).resolve()),
        "resume_from": args.resume or None,
        "notes": args.notes or None,
        "device": str(device),
        "rank": rank,
        "world_size": world_size,
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "kd_weight": kd_weight,
        "gt_weight": gt_weight,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    if is_main:
        save_json(manifest, manifest_path)
        append_jsonl({"event": "run_started", **manifest}, registry_path)

    t0 = time.time()
    student.train()
    loader_iter = iter(train_loader)
    while processed_samples < target_samples:
        try:
            context, action, target = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            continue

        step += 1
        local_batch = int(context.shape[0])
        remaining = target_samples - processed_samples
        if use_distributed:
            local_take = remaining // world_size
            if local_take <= 0:
                break
            if local_batch > local_take:
                context = context[:local_take]
                action = action[:local_take]
                target = target[:local_take]
                local_batch = local_take
            global_batch = local_batch * world_size
        else:
            if local_batch > remaining:
                context = context[:remaining]
                action = action[:remaining]
                target = target[:remaining]
                local_batch = remaining
            global_batch = local_batch

        context = context.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        t_idx = diffusion.sample_timesteps(target.shape[0], device=device)
        noise = torch.randn_like(target)
        noisy = diffusion.q_sample(target, t_idx, noise)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_autocast):
            with torch.no_grad():
                eps_teacher = teacher(context=context, action=action, noisy_target=noisy, t_idx=t_idx)
            eps_student = student(context=context, action=action, noisy_target=noisy, t_idx=t_idx)
            loss_kd = torch.mean((eps_student - eps_teacher) ** 2)
            loss_gt = torch.mean((eps_student - noise) ** 2)
            loss = kd_weight * loss_kd + gt_weight * loss_gt

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(cfg["training"]["max_grad_norm"]))
        optimizer.step()

        processed_samples += global_batch
        did_ckpt = processed_samples >= next_ckpt
        if did_ckpt:
            if is_main:
                _save_checkpoint(
                    dirs["checkpoints"] / f"ckpt_{processed_samples:09d}.pt",
                    student,
                    optimizer,
                    step,
                    processed_samples,
                    bool(cfg["run"]["save_optimizer_state"]),
                )
            next_ckpt += ckpt_every

        do_eval = val_loader is not None and (step % eval_every == 0)
        eval_stats = {"loss": float("nan"), "kd_loss": float("nan"), "gt_loss": float("nan")}
        if do_eval:
            if use_distributed:
                dist.barrier()
            if is_main:
                eval_stats = evaluate(
                    student=student,
                    teacher=teacher,
                    diffusion=diffusion,
                    loader=val_loader,
                    device=device,
                    mp_dtype=mp_dtype,
                    use_autocast=use_autocast,
                    num_batches=val_batches,
                    kd_weight=kd_weight,
                    gt_weight=gt_weight,
                )
            if use_distributed:
                vals = torch.tensor(
                    [
                        float(eval_stats["loss"]) if is_main else 0.0,
                        float(eval_stats["kd_loss"]) if is_main else 0.0,
                        float(eval_stats["gt_loss"]) if is_main else 0.0,
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                dist.broadcast(vals, src=0)
                eval_stats = {"loss": float(vals[0].item()), "kd_loss": float(vals[1].item()), "gt_loss": float(vals[2].item())}
                dist.barrier()

        emit_log = step % log_every == 0 or did_ckpt or do_eval or processed_samples >= target_samples
        train_loss = float("nan")
        train_kd = float("nan")
        train_gt = float("nan")
        if emit_log:
            # Must execute on all ranks to avoid distributed collective desync.
            train_loss = float(_reduce_mean(loss.detach(), use_distributed).cpu())
            train_kd = float(_reduce_mean(loss_kd.detach(), use_distributed).cpu())
            train_gt = float(_reduce_mean(loss_gt.detach(), use_distributed).cpu())

        if emit_log and is_main:
            elapsed = max(time.time() - t0, 1e-6)
            current_run_samples = max(processed_samples - start_processed_samples, 1)
            sps = current_run_samples / elapsed
            row = {
                "step": step,
                "processed_samples": processed_samples,
                "samples_per_sec": float(sps),
                "train_loss": train_loss,
                "train_kd_loss": train_kd,
                "train_gt_loss": train_gt,
                "val_loss": float(eval_stats["loss"]),
                "val_kd_loss": float(eval_stats["kd_loss"]),
                "val_gt_loss": float(eval_stats["gt_loss"]),
                "event_checkpoint": did_ckpt,
                "world_size": world_size,
                "global_batch_size": int(global_batch),
            }
            append_jsonl(row, metrics_path)
            if wandb_run is not None:
                wandb_run.log(row, step=step)
            print(
                f"step={step} samples={processed_samples}/{target_samples} sps={sps:.1f} "
                f"train={train_loss:.6f} kd={train_kd:.6f} gt={train_gt:.6f} "
                f"val={row['val_loss']:.6f} ckpt={did_ckpt}"
            )

    _barrier(use_distributed)
    final_ckpt = dirs["checkpoints"] / f"ckpt_{processed_samples:09d}.pt"
    if is_main and not final_ckpt.exists():
        _save_checkpoint(
            final_ckpt,
            student,
            optimizer,
            step,
            processed_samples,
            bool(cfg["run"]["save_optimizer_state"]),
        )

    duration_sec = max(time.time() - t0, 1e-6)
    current_run_samples = max(processed_samples - start_processed_samples, 0)
    summary = {
        "run_id": run_id,
        "training_mode": "distill_v0",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "processed_samples": processed_samples,
        "processed_samples_this_run": current_run_samples,
        "final_step": step,
        "duration_sec": duration_sec,
        "avg_samples_per_sec": (current_run_samples / duration_sec) if current_run_samples > 0 else 0.0,
        "final_checkpoint": str(final_ckpt),
        "world_size": world_size,
        "teacher_checkpoint_path": str(Path(teacher_ckpt).expanduser().resolve()),
        "teacher_config_path": str(Path(teacher_cfg_path).expanduser().resolve()),
        "kd_weight": kd_weight,
        "gt_weight": gt_weight,
    }
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
