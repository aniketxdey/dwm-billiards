from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml

from vae_training.export_latents import (
    numpy_latent_dtype,
    parse_shard_id,
    torch_amp_dtype,
)
from vae_training.model import ConvVAE
from vae_training.utils import append_jsonl, ensure_dir, save_json, set_seed


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_raw_shards(shards_dir: Path) -> List[Path]:
    if not shards_dir.exists():
        return []
    return sorted(shards_dir.glob("shard_*.npz"))


def latent_out_path(out_shards_dir: Path, shard_id: int) -> Path:
    return out_shards_dir / f"latent_shard_{shard_id:05d}.npz"


def is_file_stable(path: Path, min_age_sec: float) -> bool:
    if not path.exists():
        return False
    if min_age_sec <= 0:
        return True
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age >= min_age_sec


def atomic_save_npz(path: Path, compress: bool, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp.npz")
    if compress:
        np.savez_compressed(tmp_path, **payload)
    else:
        np.savez(tmp_path, **payload)
    tmp_path.replace(path)


def encode_one_shard(
    *,
    shard_path: Path,
    out_path: Path,
    model: ConvVAE,
    device: torch.device,
    batch_size: int,
    deterministic_latent: bool,
    include_actions: bool,
    include_sim_state: bool,
    include_lengths: bool,
    include_episode_meta: bool,
    mp_dtype: torch.dtype,
    out_np_dtype: np.dtype,
    compress: bool,
    use_autocast: bool,
) -> Dict[str, Any]:
    t0 = time.time()
    global_sum = 0.0
    global_sumsq = 0.0
    global_count = 0

    with np.load(shard_path, allow_pickle=True) as d:
        frames = d["frames"]  # [N, T, H, W, 3] uint8
        n_eps, n_t = int(frames.shape[0]), int(frames.shape[1])
        h, w, c = int(frames.shape[2]), int(frames.shape[3]), int(frames.shape[4])
        if c != 3:
            raise ValueError(f"Expected RGB frames with C=3, got {frames.shape}")

        flat = frames.reshape(-1, h, w, c)
        n_frames = int(flat.shape[0])
        latents = None

        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch_np = np.ascontiguousarray(flat[start:end])
            x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            x = x.div(127.5).sub(1.0)

            with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_autocast):
                mu, logvar = model.encode(x)
                z = mu if deterministic_latent else model.reparameterize(mu, logvar)

            z_cpu = z.detach().cpu().float().numpy()
            if latents is None:
                lc, lh, lw = int(z_cpu.shape[1]), int(z_cpu.shape[2]), int(z_cpu.shape[3])
                latents = np.empty((n_frames, lc, lh, lw), dtype=out_np_dtype)
            latents[start:end] = z_cpu.astype(out_np_dtype, copy=False)
            global_sum += float(z_cpu.sum(dtype=np.float64))
            global_sumsq += float(np.square(z_cpu, dtype=np.float64).sum())
            global_count += int(z_cpu.size)

        if latents is None:
            raise RuntimeError(f"No latent batches produced for shard {shard_path}")

        latents = latents.reshape(n_eps, n_t, latents.shape[1], latents.shape[2], latents.shape[3])
        keys = set(d.files)
        payload: Dict[str, Any] = {"latents": latents}
        if include_actions and "actions" in keys:
            payload["actions"] = d["actions"]
        if include_sim_state and "sim_state" in keys:
            payload["sim_state"] = d["sim_state"]
        if include_lengths and "lengths" in keys:
            payload["lengths"] = d["lengths"]
        if include_episode_meta and "episode_meta" in keys:
            payload["episode_meta"] = d["episode_meta"]

    atomic_save_npz(out_path, compress=compress, payload=payload)
    sec = max(time.time() - t0, 1e-6)
    return {
        "frames": n_frames,
        "seconds": sec,
        "frames_per_sec": n_frames / sec,
        "latent_sum": global_sum,
        "latent_sumsq": global_sumsq,
        "latent_count": global_count,
    }


def count_output_shards(out_shards_dir: Path) -> int:
    return sum(1 for _ in out_shards_dir.glob("latent_shard_*.npz"))


def iter_pending_shards(
    *,
    source_shards_dir: Path,
    out_shards_dir: Path,
    skip_existing: bool,
    stable_age_sec: float,
    max_shards: int,
) -> Iterable[Path]:
    shards = discover_raw_shards(source_shards_dir)
    if max_shards > 0:
        shards = shards[:max_shards]
    for shard_path in shards:
        if not is_file_stable(shard_path, stable_age_sec):
            continue
        shard_id = parse_shard_id(shard_path)
        out_path = latent_out_path(out_shards_dir, shard_id)
        if skip_existing and out_path.exists():
            continue
        yield shard_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch raw shards and export VAE latents incrementally")
    parser.add_argument("--config", required=True, help="Path to latent export watch YAML config")
    parser.add_argument("--once", action="store_true", help="Process currently available shards then exit")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    export_cfg = cfg["export"]
    watch_cfg = cfg.get("watch", {})

    export_id = str(run_cfg["export_id"])
    seed = int(run_cfg.get("seed", 42))
    set_seed(seed)

    source_shards_dir = Path(run_cfg["source_shards_dir"]).resolve()
    output_root = Path(run_cfg["output_root"]).resolve()
    run_dir = ensure_dir(output_root / export_id)
    out_shards_dir = ensure_dir(run_dir / "shards")
    out_logs_dir = ensure_dir(run_dir / "logs")

    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    progress_path = out_logs_dir / "export_progress.jsonl"

    checkpoint_path = Path(run_cfg["checkpoint_path"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    max_shards = int(run_cfg.get("max_shards", 0))
    skip_existing = bool(run_cfg.get("skip_existing", True))
    log_every_shards = int(run_cfg.get("log_every_shards", 5))

    batch_size = int(data_cfg["batch_size"])
    include_actions = bool(data_cfg.get("include_actions", True))
    include_sim_state = bool(data_cfg.get("include_sim_state", True))
    include_lengths = bool(data_cfg.get("include_lengths", True))
    include_episode_meta = bool(data_cfg.get("include_episode_meta", True))

    deterministic_latent = bool(export_cfg.get("deterministic_latent", True))
    compress = bool(export_cfg.get("compress", False))
    mp_dtype = torch_amp_dtype(str(export_cfg.get("mixed_precision", "bf16")))
    out_np_dtype = numpy_latent_dtype(str(export_cfg.get("output_dtype", "float16")))

    poll_seconds = float(watch_cfg.get("poll_seconds", 30))
    stable_age_sec = float(watch_cfg.get("stable_age_sec", 15))
    expected_shards = int(watch_cfg.get("expected_shards", 0))
    idle_timeout_sec = float(watch_cfg.get("idle_timeout_sec", 0))
    stop_when_metadata_complete = bool(watch_cfg.get("stop_when_metadata_complete", False))
    metadata_path = Path(str(watch_cfg.get("metadata_path", source_shards_dir.parent.parent / "meta" / "metadata.json"))).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = device.type == "cuda" and mp_dtype in (torch.float16, torch.bfloat16)

    model = ConvVAE(
        in_channels=3,
        base_channels=int(model_cfg["base_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
    ).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    expected_shards_from_meta = 0
    if stop_when_metadata_complete and metadata_path.exists():
        try:
            meta = load_yaml(metadata_path) if metadata_path.suffix in {".yml", ".yaml"} else None
            if meta is None:
                import json

                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            expected_shards_from_meta = int(meta.get("total_shards", 0))
        except Exception:
            expected_shards_from_meta = 0

    target_shards = expected_shards if expected_shards > 0 else expected_shards_from_meta

    manifest = {
        "export_id": export_id,
        "mode": "watch",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "source_shards_dir": str(source_shards_dir),
        "source_dataset_local_root": run_cfg.get("source_dataset_local_root", ""),
        "source_dataset_s3_root": run_cfg.get("source_dataset_s3_root", ""),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "batch_size": batch_size,
        "deterministic_latent": deterministic_latent,
        "output_dtype": str(out_np_dtype),
        "compress": compress,
        "include_actions": include_actions,
        "include_sim_state": include_sim_state,
        "include_lengths": include_lengths,
        "include_episode_meta": include_episode_meta,
        "max_shards": max_shards,
        "watch": {
            "poll_seconds": poll_seconds,
            "stable_age_sec": stable_age_sec,
            "expected_shards": expected_shards,
            "idle_timeout_sec": idle_timeout_sec,
            "stop_when_metadata_complete": stop_when_metadata_complete,
            "metadata_path": str(metadata_path),
            "expected_shards_from_meta": expected_shards_from_meta,
            "target_shards": target_shards,
        },
    }
    save_json(manifest, manifest_path)

    total_frames = 0
    total_shards_written = 0
    global_sum = 0.0
    global_sumsq = 0.0
    global_count = 0
    t_start = time.time()
    last_progress_at = t_start
    encoded_this_session: set[int] = set()
    stop_reason = ""

    with torch.no_grad():
        while True:
            made_progress = False
            seen_raw = discover_raw_shards(source_shards_dir)
            if max_shards > 0:
                seen_raw = seen_raw[:max_shards]

            for shard_path in iter_pending_shards(
                source_shards_dir=source_shards_dir,
                out_shards_dir=out_shards_dir,
                skip_existing=skip_existing,
                stable_age_sec=stable_age_sec,
                max_shards=max_shards,
            ):
                shard_id = parse_shard_id(shard_path)
                out_path = latent_out_path(out_shards_dir, shard_id)
                shard_stats = encode_one_shard(
                    shard_path=shard_path,
                    out_path=out_path,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    deterministic_latent=deterministic_latent,
                    include_actions=include_actions,
                    include_sim_state=include_sim_state,
                    include_lengths=include_lengths,
                    include_episode_meta=include_episode_meta,
                    mp_dtype=mp_dtype,
                    out_np_dtype=out_np_dtype,
                    compress=compress,
                    use_autocast=use_autocast,
                )
                total_frames += int(shard_stats["frames"])
                total_shards_written += 1
                global_sum += float(shard_stats["latent_sum"])
                global_sumsq += float(shard_stats["latent_sumsq"])
                global_count += int(shard_stats["latent_count"])
                encoded_this_session.add(shard_id)
                made_progress = True
                last_progress_at = time.time()

                append_jsonl(
                    {
                        "event": "shard_done",
                        "shard_id": shard_id,
                        "source_shard": str(shard_path),
                        "output_shard": str(out_path),
                        "frames": int(shard_stats["frames"]),
                        "seconds": float(shard_stats["seconds"]),
                        "frames_per_sec": float(shard_stats["frames_per_sec"]),
                    },
                    progress_path,
                )

                if total_shards_written % log_every_shards == 0:
                    elapsed = max(time.time() - t_start, 1e-6)
                    print(
                        f"[watch] shards_written={total_shards_written} "
                        f"frames={total_frames} avg_fps={total_frames/elapsed:.1f}"
                    )

            total_output_shards = count_output_shards(out_shards_dir)
            total_raw_shards_seen = len(seen_raw)

            if target_shards > 0 and total_output_shards >= target_shards:
                stop_reason = "target_shards_reached"
                break

            if args.once:
                stop_reason = "once_complete"
                break

            if made_progress:
                continue

            if idle_timeout_sec > 0 and (time.time() - last_progress_at) >= idle_timeout_sec:
                stop_reason = "idle_timeout"
                break

            append_jsonl(
                {
                    "event": "wait",
                    "raw_shards_seen": total_raw_shards_seen,
                    "output_shards_seen": total_output_shards,
                    "sleep_sec": poll_seconds,
                },
                progress_path,
            )
            time.sleep(max(poll_seconds, 1.0))

    elapsed = max(time.time() - t_start, 1e-6)
    mean = global_sum / max(global_count, 1)
    var = (global_sumsq / max(global_count, 1)) - (mean * mean)
    std = float(np.sqrt(max(var, 0.0)))

    summary = {
        "export_id": export_id,
        "mode": "watch",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "stop_reason": stop_reason,
        "session_shards_written": total_shards_written,
        "session_frames_encoded": total_frames,
        "session_duration_sec": elapsed,
        "session_avg_frames_per_sec": total_frames / elapsed if total_frames > 0 else 0.0,
        "total_output_shards_present": count_output_shards(out_shards_dir),
        "total_raw_shards_seen": len(discover_raw_shards(source_shards_dir)),
        "latent_mean_session": float(mean),
        "latent_std_session": std,
        "encoded_shard_ids_sample": sorted(list(encoded_this_session))[:20],
    }
    save_json(summary, summary_path)
    append_jsonl({"event": "watch_finished", **summary}, progress_path)
    print(f"Latent watch export complete: {run_dir} ({stop_reason})")


if __name__ == "__main__":
    main()
