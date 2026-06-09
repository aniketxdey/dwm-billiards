from __future__ import annotations

"""Video writing compatibility shim.

`torchvision.io.write_video` was removed in newer torchvision releases
(the video IO API was deprecated). This module provides a drop-in
replacement with the same call signature, backed by imageio + ffmpeg.
"""

from pathlib import Path
from typing import Any

import numpy as np

try:  # Prefer torchvision when it still ships write_video.
    from torchvision.io import write_video as _tv_write_video  # type: ignore
except Exception:  # pragma: no cover - exercised on torchvision>=0.22
    _tv_write_video = None


def write_video(filename: str, video_array: Any, fps: float, **kwargs: Any) -> None:
    """Write a [T, H, W, C] uint8 tensor/array to a video file.

    Mirrors the torchvision.io.write_video signature closely enough for the
    callers in this repo. Falls back to imageio when torchvision lacks it.
    """
    if _tv_write_video is not None:
        _tv_write_video(filename, video_array, fps, **kwargs)
        return

    import imageio.v2 as imageio

    if hasattr(video_array, "detach"):
        frames = video_array.detach().cpu().numpy()
    else:
        frames = np.asarray(video_array)

    frames = np.ascontiguousarray(frames).astype(np.uint8, copy=False)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 preserves exact frame dimensions (panels here are not
    # multiples of 16); fps must be an int for the ffmpeg writer.
    imageio.mimsave(filename, list(frames), fps=int(round(float(fps))), macro_block_size=1)
