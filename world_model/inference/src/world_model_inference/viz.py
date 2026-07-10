from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


BG = (15, 18, 24)
PANEL = (23, 29, 38)
GRID = (58, 70, 88)
TXT = (225, 229, 235)
FX = (60, 190, 255)
FY = (255, 156, 86)
TR = (141, 244, 132)


def _safe_font(size: int):
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def render_action_timeline(actions: np.ndarray, out_path: str | Path, title: str = "Action Timeline") -> str:
    if Image is None:
        raise RuntimeError("Pillow is required for action timeline rendering. Install pillow.")

    acts = np.asarray(actions, dtype=np.float32)
    if acts.ndim != 2 or acts.shape[1] < 3:
        raise ValueError(f"Expected actions shape [T,3], got {acts.shape}")

    t = acts.shape[0]
    w, h = 1400, 360
    pad = 24
    chart_x0, chart_y0 = 90, 64
    chart_x1, chart_y1 = w - 24, h - 80
    chart_w = chart_x1 - chart_x0
    chart_h = chart_y1 - chart_y0

    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    f_title = _safe_font(24)
    f_body = _safe_font(15)

    d.rounded_rectangle((8, 8, w - 8, h - 8), radius=16, fill=PANEL, outline=(35, 43, 56), width=2)
    d.text((24, 20), title, fill=TXT, font=f_title)

    # grid
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = chart_y0 + int((1.0 - frac) * chart_h)
        d.line((chart_x0, y, chart_x1, y), fill=GRID, width=1)
    for i in range(0, max(t, 2), max(1, t // 8)):
        x = chart_x0 + int((i / max(t - 1, 1)) * chart_w)
        d.line((x, chart_y0, x, chart_y1), fill=(42, 51, 65), width=1)
        d.text((x - 8, chart_y1 + 8), str(i), fill=(170, 180, 195), font=f_body)

    max_abs = float(np.max(np.abs(acts[:, :2]))) if t else 1.0
    max_abs = max(max_abs, 1e-6)

    def to_xy(i: int, val: float):
        x = chart_x0 + int((i / max(t - 1, 1)) * chart_w)
        y = chart_y0 + int((0.5 - 0.5 * (val / max_abs)) * chart_h)
        return x, y

    # zero line
    zx0, zy = to_xy(0, 0.0)
    zx1, _ = to_xy(max(t - 1, 1), 0.0)
    d.line((zx0, zy, zx1, zy), fill=(120, 128, 140), width=1)

    for col, color in ((0, FX), (1, FY)):
        pts = [to_xy(i, float(acts[i, col])) for i in range(t)]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=3, joint="curve")

    # trigger bars
    trig_top = chart_y1 + 32
    trig_h = 14
    d.text((24, trig_top - 2), "trigger", fill=TR, font=f_body)
    for i in range(t):
        if acts[i, 2] > 0.5:
            x = chart_x0 + int((i / max(t - 1, 1)) * chart_w)
            d.rectangle((x - 2, trig_top, x + 2, trig_top + trig_h), fill=TR)

    legend_x = w - 280
    d.rounded_rectangle((legend_x, 20, w - 20, 92), radius=12, fill=(28, 35, 46), outline=(45, 56, 72))
    d.line((legend_x + 14, 40, legend_x + 52, 40), fill=FX, width=3)
    d.text((legend_x + 60, 31), "force_x", fill=TXT, font=f_body)
    d.line((legend_x + 14, 61, legend_x + 52, 61), fill=FY, width=3)
    d.text((legend_x + 60, 52), "force_y", fill=TXT, font=f_body)
    d.line((legend_x + 14, 82, legend_x + 52, 82), fill=TR, width=4)
    d.text((legend_x + 60, 73), "trigger", fill=TXT, font=f_body)

    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def render_action_timeline_array(actions: np.ndarray, title: str = "Action Timeline") -> Optional[np.ndarray]:
    if Image is None:
        return None
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        p = render_action_timeline(actions, tmp.name, title=title)
    arr = np.asarray(Image.open(p).convert("RGB"))
    try:
        Path(p).unlink(missing_ok=True)
    except Exception:
        pass
    return arr
