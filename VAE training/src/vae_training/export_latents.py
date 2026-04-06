from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

from vae_training.model import ConvVAE
from vae_training.utils import append_jsonl, ensure_dir, save_json, set_seed


SHARD_RE = re.compile(r"shard_(\d+)\.npz$")


def parse_shard_id(path: Path) -> int:
    m = SHARD_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected shard filename: {path.name}")
    return int(m.group(1))


def discover_shards(shards_dir: Path) -> List[Path]:
    shards = sorted(shards_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz files found in {shards_dir}")
    return shards


def torch_amp_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n == "fp16":
        return torch.float16
    if n == "bf16":
        return torch.bfloat16
    return torch.float32


def numpy_latent_dtype(name: str) -> np.dtype:
    n = name.lower()
    if n == "float16":
        return np.float16
    if n == "float32":
        return np.float32
    raise ValueError(f"Unsupported latent output dtype: {name}")


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VAE latents from raw frame shards")
    parser.add_argument("--config", required=True, help="Path to latent export YAML config")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    export_cfg = cfg["export"]

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

    shards = discover_shards(source_shards_dir)
    if max_shards > 0:
        shards = shards[:max_shards]

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

    manifest = {
        "export_id": export_id,
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
        "total_shards": len(shards),
    }
    save_json(manifest, manifest_path)

    global_sum = 0.0
    global_sumsq = 0.0
    global_count = 0
    total_frames = 0
    total_shards_written = 0
    t0 = time.time()

    with torch.no_grad():
        for idx, shard_path in enumerate(tqdm(shards, desc="shards", unit="shard"), start=1):
            shard_id = parse_shard_id(shard_path)
            out_path = out_shards_dir / f"latent_shard_{shard_id:05d}.npz"

            if skip_existing and out_path.exists():
                append_jsonl(
                    {
                        "event": "skip_existing",
                        "shard_id": shard_id,
                        "source_shard": str(shard_path),
                        "output_shard": str(out_path),
                    },
                    progress_path,
                )
                continue

            shard_start = time.time()
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
                    try:
                        payload["episode_meta"] = d["episode_meta"]
                    except Exception as exc:
                        append_jsonl(
                            {
                                "event": "episode_meta_skip",
                                "shard_id": shard_id,
                                "reason": str(exc),
                            },
                            progress_path,
                        )

            if compress:
                np.savez_compressed(out_path, **payload)
            else:
                np.savez(out_path, **payload)

            total_frames += n_frames
            total_shards_written += 1
            shard_sec = max(time.time() - shard_start, 1e-6)
            append_jsonl(
                {
                    "event": "shard_done",
                    "shard_id": shard_id,
                    "source_shard": str(shard_path),
                    "output_shard": str(out_path),
                    "frames": n_frames,
                    "seconds": shard_sec,
                    "frames_per_sec": n_frames / shard_sec,
                },
                progress_path,
            )

            if idx % log_every_shards == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"[{idx}/{len(shards)}] frames={total_frames} "
                    f"avg_fps={total_frames/elapsed:.1f} shards_written={total_shards_written}"
                )

    elapsed = max(time.time() - t0, 1e-6)
    mean = global_sum / max(global_count, 1)
    var = (global_sumsq / max(global_count, 1)) - (mean * mean)
    std = float(np.sqrt(max(var, 0.0)))

    summary = {
        "export_id": export_id,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_shards_seen": len(shards),
        "total_shards_written": total_shards_written,
        "total_frames_encoded": total_frames,
        "duration_sec": elapsed,
        "avg_frames_per_sec": total_frames / elapsed,
        "latent_mean": float(mean),
        "latent_std": std,
    }
    save_json(summary, summary_path)

    append_jsonl({"event": "export_finished", **summary}, progress_path)
    print(f"Latent export complete: {run_dir}")


if __name__ == "__main__":
    main()
