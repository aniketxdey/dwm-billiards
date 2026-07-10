from __future__ import annotations

from typing import Any, Dict

import numpy as np


ACTION_DIM = 3


def zeros(horizon: int) -> np.ndarray:
    return np.zeros((int(horizon), ACTION_DIM), dtype=np.float32)


def single_shot(horizon: int, shot_frame: int, force_x: float, force_y: float) -> np.ndarray:
    arr = zeros(horizon)
    idx = int(np.clip(shot_frame, 0, horizon - 1))
    arr[idx, 0] = np.float32(force_x)
    arr[idx, 1] = np.float32(force_y)
    arr[idx, 2] = np.float32(1.0)
    return arr


def random_shots(
    horizon: int,
    seed: int = 42,
    num_shots: int = 2,
    max_force: float = 12.0,
    min_gap: int = 6,
) -> np.ndarray:
    horizon = int(horizon)
    arr = zeros(horizon)
    rng = np.random.default_rng(int(seed))
    n = max(1, min(int(num_shots), horizon))

    candidates = list(range(horizon))
    chosen: list[int] = []
    while candidates and len(chosen) < n:
        idx = int(rng.choice(candidates))
        chosen.append(idx)
        candidates = [c for c in candidates if abs(c - idx) >= int(min_gap)]
    chosen.sort()

    for idx in chosen:
        angle = float(rng.uniform(-np.pi, np.pi))
        mag = float(rng.uniform(max_force * 0.35, max_force))
        arr[idx, 0] = np.float32(np.cos(angle) * mag)
        arr[idx, 1] = np.float32(np.sin(angle) * mag)
        arr[idx, 2] = np.float32(1.0)
    return arr


def bank_left(horizon: int, shot_frame: int = 0, force: float = 10.0) -> np.ndarray:
    return single_shot(horizon=horizon, shot_frame=shot_frame, force_x=-abs(force), force_y=-abs(force) * 0.35)


def bank_right(horizon: int, shot_frame: int = 0, force: float = 10.0) -> np.ndarray:
    return single_shot(horizon=horizon, shot_frame=shot_frame, force_x=abs(force), force_y=-abs(force) * 0.35)


def chaos_burst(horizon: int, seed: int = 42, max_force: float = 12.0) -> np.ndarray:
    return random_shots(horizon=horizon, seed=seed, num_shots=max(2, horizon // 10), max_force=max_force, min_gap=2)


def build_action_sequence(spec: Dict[str, Any], fallback_horizon: int) -> np.ndarray:
    preset = dict(spec or {})
    name = str(preset.get("name", "single_shot")).strip().lower()
    horizon = int(preset.get("horizon", fallback_horizon))

    if name == "single_shot":
        return single_shot(
            horizon=horizon,
            shot_frame=int(preset.get("shot_frame", 0)),
            force_x=float(preset.get("force_x", 0.0)),
            force_y=float(preset.get("force_y", -8.0)),
        )
    if name == "random_shots":
        return random_shots(
            horizon=horizon,
            seed=int(preset.get("seed", 42)),
            num_shots=int(preset.get("num_shots", 2)),
            max_force=float(preset.get("max_force", 12.0)),
            min_gap=int(preset.get("min_gap", 6)),
        )
    if name == "bank_left":
        return bank_left(horizon=horizon, shot_frame=int(preset.get("shot_frame", 0)), force=float(preset.get("force", 10.0)))
    if name == "bank_right":
        return bank_right(horizon=horizon, shot_frame=int(preset.get("shot_frame", 0)), force=float(preset.get("force", 10.0)))
    if name == "chaos_burst":
        return chaos_burst(horizon=horizon, seed=int(preset.get("seed", 42)), max_force=float(preset.get("max_force", 12.0)))
    raise ValueError(f"Unknown action preset: {name}")
