from __future__ import annotations

from pathlib import Path
import warnings

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clamp(-1, 1)
    x = (x + 1.0) * 127.5
    return x.round().to(torch.uint8)


def save_recon_grid(
    x: torch.Tensor,
    recon: torch.Tensor,
    out_path: str | Path,
    nrow: int = 8,
) -> None:
    # [B, C, H, W]
    inp = to_uint8(x).float() / 255.0
    rec = to_uint8(recon).float() / 255.0
    stacked = torch.cat([inp, rec], dim=0)
    grid = torchvision.utils.make_grid(stacked, nrow=nrow)
    torchvision.utils.save_image(grid, str(out_path))


def save_recon_video(
    x: torch.Tensor,
    recon: torch.Tensor,
    out_path: str | Path,
    fps: int = 8,
) -> Path:
    # x/recon are [T, C, H, W]
    x_u8 = to_uint8(x).permute(0, 2, 3, 1).cpu().numpy()
    r_u8 = to_uint8(recon).permute(0, 2, 3, 1).cpu().numpy()

    frames = []
    for i in range(x_u8.shape[0]):
        side = np.concatenate([x_u8[i], r_u8[i]], axis=1)
        frames.append(side)

    out = Path(out_path)
    try:
        imageio.mimsave(str(out), frames, fps=fps)
        return out
    except ValueError as exc:
        # Some environments miss an MP4-capable backend; fallback keeps training alive.
        if out.suffix.lower() == ".mp4":
            fallback = out.with_suffix(".gif")
            warnings.warn(
                f"MP4 backend unavailable ({exc}); falling back to GIF preview at {fallback}",
                RuntimeWarning,
            )
            imageio.mimsave(str(fallback), frames, fps=fps)
            return fallback
        raise
