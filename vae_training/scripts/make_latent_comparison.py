#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from vae_training.model import ConvVAE


def to_u8_from_minus1_1(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(-1, 1)
    x = (x + 1.0) * 127.5
    return x.round().to(torch.uint8).cpu().numpy()


def latent_to_rgb(latents: np.ndarray) -> np.ndarray:
    # latents: [T, C, H, W], expected C>=3
    t, c, h, w = latents.shape
    if c < 3:
        pad = np.zeros((t, 3 - c, h, w), dtype=latents.dtype)
        latents = np.concatenate([latents, pad], axis=1)

    rgb = latents[:, :3].astype(np.float32)
    lo = np.percentile(rgb, 1.0, axis=(0, 2, 3), keepdims=True)
    hi = np.percentile(rgb, 99.0, axis=(0, 2, 3), keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (rgb * 255.0).round().astype(np.uint8)
    # [T, 3, H, W] -> [T, H, W, 3]
    rgb = np.transpose(rgb, (0, 2, 3, 1))
    # 9x16 -> 72x128 by nearest-neighbor repeat
    rgb = np.repeat(np.repeat(rgb, 8, axis=1), 8, axis=2)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real/recon/latent comparison video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-shard", required=True)
    parser.add_argument("--latent-shard", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out-video", required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    raw_shard = Path(args.raw_shard)
    latent_shard = Path(args.latent_shard)
    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(in_channels=3, base_channels=64, latent_channels=4).to(device)
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()

    with np.load(raw_shard, allow_pickle=False) as d:
        raw_frames = d["frames"][args.episode_index, : args.num_frames]  # [T, 72, 128, 3] uint8

    with np.load(latent_shard, allow_pickle=False) as d:
        z = d["latents"][args.episode_index, : args.num_frames]  # [T, 4, 9, 16]

    z_t = torch.from_numpy(z).to(device=device, dtype=torch.float32)
    recon_batches: list[np.ndarray] = []
    bs = 64
    with torch.no_grad():
        for s in range(0, z_t.shape[0], bs):
            e = min(s + bs, z_t.shape[0])
            recon = model.decode(z_t[s:e])
            recon_u8 = to_u8_from_minus1_1(recon)  # [B, 3, H, W]
            recon_u8 = np.transpose(recon_u8, (0, 2, 3, 1))  # [B, H, W, 3]
            recon_batches.append(recon_u8)
    recon_frames = np.concatenate(recon_batches, axis=0)

    latent_rgb = latent_to_rgb(z)

    panels = []
    for i in range(raw_frames.shape[0]):
        panel = np.concatenate([raw_frames[i], recon_frames[i], latent_rgb[i]], axis=1)
        panels.append(panel)

    imageio.mimsave(str(out_video), panels, fps=args.fps)
    print(f"Wrote: {out_video}")


if __name__ == "__main__":
    main()

