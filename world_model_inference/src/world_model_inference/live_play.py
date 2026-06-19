from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image

from .config import DEFAULT_PREVIEW_CONFIG
from .pipeline import InferenceEngine, build_prompt_from_config, load_engine
from world_model_training import eval_rollout as wm_eval


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Pool Live Play</title>
  <style>
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: #0a0c10; color: #e8ecf1; }
    .top { display: flex; gap: 8px; padding: 12px; align-items: center; border-bottom: 1px solid #202634; background: #0f1420; }
    .btn { background: #ff6a00; color: white; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 600; }
    .btn.blue { background: #1e65ff; }
    .btn.gray { background: #3b4253; }
    .inp { width: 76px; border-radius: 6px; border: 1px solid #344055; background: #111722; color: #e8ecf1; padding: 6px 8px; }
    .row { display: flex; gap: 6px; align-items: center; }
    .wrap { padding: 10px; }
    .stage { display: grid; place-items: center; background: #06080d; border: 1px solid #202634; border-radius: 10px; min-height: calc(100vh - 130px); }
    canvas { image-rendering: pixelated; max-width: 98vw; max-height: calc(100vh - 170px); border: 1px solid #2a3448; border-radius: 8px; background: #0c111a; }
    .stats { margin-left: auto; opacity: .9; font-size: 13px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  </style>
</head>
<body>
  <div class=\"top\">
    <button id=\"start\" class=\"btn\">Start</button>
    <button id=\"restart\" class=\"btn blue\">Start Over</button>
    <button id=\"stop\" class=\"btn gray\">Stop</button>
    <div class=\"row\"><span>Mode</span>
      <select id=\"mode\" class=\"inp\" style=\"width: 110px;\">
        <option value=\"fast\">fast</option>
        <option value=\"balanced\" selected>balanced</option>
        <option value=\"quality\">quality</option>
      </select>
    </div>
    <div class=\"row\"><span>DDIM</span><input id=\"ddim\" class=\"inp\" type=\"number\" min=\"5\" max=\"50\" step=\"1\" value=\"8\" /></div>
    <div class=\"row\"><span>FPS</span><input id=\"fps\" class=\"inp\" type=\"number\" min=\"4\" max=\"30\" step=\"1\" value=\"15\" /></div>
    <div class=\"row\"><span>MaxForce</span><input id=\"force\" class=\"inp\" type=\"number\" min=\"2\" max=\"300\" step=\"1\" value=\"180\" /></div>
    <div class=\"row\"><span>JPEG</span><input id=\"jpeg\" class=\"inp\" type=\"number\" min=\"40\" max=\"95\" step=\"1\" value=\"75\" /></div>
    <div class=\"row\"><span>Scale</span><input id=\"scale\" class=\"inp\" type=\"number\" min=\"1\" max=\"12\" step=\"1\" value=\"6\" /></div>
    <div class=\"row\"><label><input id=\"compile\" type=\"checkbox\" />compile</label></div>
    <div class=\"row\"><label><input id=\"bench\" type=\"checkbox\" checked />bench</label></div>
    <div class=\"stats mono\" id=\"stats\">disconnected</div>
  </div>
  <div class=\"wrap\">
    <div class=\"stage\">
      <canvas id=\"canvas\" width=\"640\" height=\"360\"></canvas>
    </div>
  </div>

<script>
(() => {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const stats = document.getElementById('stats');

  const startBtn = document.getElementById('start');
  const restartBtn = document.getElementById('restart');
  const stopBtn = document.getElementById('stop');
  const modeEl = document.getElementById('mode');
  const ddimEl = document.getElementById('ddim');
  const fpsEl = document.getElementById('fps');
  const forceEl = document.getElementById('force');
  const jpegEl = document.getElementById('jpeg');
  const scaleEl = document.getElementById('scale');
  const compileEl = document.getElementById('compile');
  const benchEl = document.getElementById('bench');

  let ws = null;
  let lastFrame = null;
  let drag = null;
  let lastShot = null;
  let lastActionInfo = 'none';
  let actionSeq = 0;
  let isRunning = false;
  let displayScale = Math.max(1, parseInt(scaleEl.value || '6', 10));

  ctx.imageSmoothingEnabled = false;

  function applyCanvasDisplaySize() {
    canvas.style.width = `${Math.max(1, Math.round(canvas.width * displayScale))}px`;
    canvas.style.height = `${Math.max(1, Math.round(canvas.height * displayScale))}px`;
  }

  function draw() {
    ctx.imageSmoothingEnabled = false;
    if (lastFrame) {
      ctx.drawImage(lastFrame, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#101722';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    if (drag) {
      ctx.strokeStyle = '#ff5a00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(drag.sx, drag.sy);
      ctx.lineTo(drag.ex, drag.ey);
      ctx.stroke();

      ctx.fillStyle = '#ffd34d';
      ctx.beginPath();
      ctx.arc(drag.sx, drag.sy, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(drag.ex, drag.ey, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    if (lastShot && Date.now() < lastShot.untilTs) {
      ctx.strokeStyle = '#3ddc97';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(lastShot.sx, lastShot.sy);
      ctx.lineTo(lastShot.ex, lastShot.ey);
      ctx.stroke();

      ctx.fillStyle = '#8ef0c6';
      ctx.beginPath();
      ctx.arc(lastShot.sx, lastShot.sy, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(lastShot.ex, lastShot.ey, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function normPos(ev) {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / Math.max(rect.width, 1);
    const y = (ev.clientY - rect.top) / Math.max(rect.height, 1);
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
  }

  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => { stats.textContent = 'connected'; };
    ws.onclose = () => { stats.textContent = 'disconnected'; setTimeout(connectWS, 1000); };
    ws.onerror = () => { stats.textContent = 'socket error'; };

    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.type === 'frame') {
        const img = new Image();
        img.onload = () => {
          if (canvas.width !== msg.width || canvas.height !== msg.height) {
            canvas.width = msg.width;
            canvas.height = msg.height;
            applyCanvasDisplaySize();
          }
          lastFrame = img;
          draw();
        };
        img.src = `data:image/jpeg;base64,${msg.jpeg_b64}`;
        const actionUsed = Array.isArray(msg.action_used) ? ` action=[${msg.action_used.map(v => Number(v).toFixed(2)).join(',')}]` : '';
        const actionMag = Number(msg.action_mag ?? 0).toFixed(1);
        const actionSummary = msg.action_summary || {};
        const actionMagP95 = Number(actionSummary.action_mag_p95 ?? 0).toFixed(1);
        const shotRatePct = (Number(actionSummary.shot_frame_rate ?? 0) * 100.0).toFixed(1);
        const perf = msg.perf || {};
        const p50 = (perf.end_to_end_p50_ms ?? 0).toFixed ? Number(perf.end_to_end_p50_ms).toFixed(0) : '0';
        const p95 = (perf.end_to_end_p95_ms ?? 0).toFixed ? Number(perf.end_to_end_p95_ms).toFixed(0) : '0';
        stats.textContent = `running=${msg.running} frame=${msg.frame_idx} mode=${msg.runtime_mode || 'n/a'} ddim=${msg.ddim_steps_used} mfwd=${msg.model_forward_ms}ms ddim=${msg.ddim_total_ms}ms dec=${msg.vae_decode_ms}ms jpg=${msg.jpeg_encode_ms}ms loop=${msg.loop_latency_ms}ms send~${msg.send_latency_ms}ms p50/p95=${p50}/${p95} target_fps=${msg.target_fps}${actionUsed} |mag|=${actionMag} p95|a|=${actionMagP95} shot%=${shotRatePct} last_action=${lastActionInfo}`;
      } else if (msg.type === 'action_ack') {
        if (msg.ok) {
          lastActionInfo = `ack#${msg.client_seq ?? '?'} fx=${Number(msg.fx || 0).toFixed(2)} fy=${Number(msg.fy || 0).toFixed(2)} |mag|=${Number(msg.mag || 0).toFixed(2)}`;
        } else {
          lastActionInfo = `ack_failed#${msg.client_seq ?? '?'} ${msg.message || ''}`;
        }
        if (msg.start && msg.end) {
          lastShot = {
            sx: Number(msg.start.x) * canvas.width,
            sy: Number(msg.start.y) * canvas.height,
            ex: Number(msg.end.x) * canvas.width,
            ey: Number(msg.end.y) * canvas.height,
            untilTs: Date.now() + 1500,
          };
          draw();
        }
      } else if (msg.type === 'info') {
        stats.textContent = msg.message || 'info';
      } else if (msg.type === 'error') {
        stats.textContent = 'error: ' + (msg.message || 'unknown');
      }
    };
  }

  const startPayload = () => ({
    runtime_mode: String(modeEl.value || 'balanced'),
    ddim_steps: parseInt(ddimEl.value || '8', 10),
    target_fps: parseInt(fpsEl.value || '15', 10),
    max_force: parseFloat(forceEl.value || '12'),
    jpeg_quality: parseInt(jpegEl.value || '75', 10),
    compile_model: Boolean(compileEl.checked),
    enable_benchmark: Boolean(benchEl.checked),
    device: 'cuda',
    sample_source: 'eval',
    seed: 42,
  });

  startBtn.onclick = async () => {
    const body = {
      ...startPayload(),
    };
    const r = await fetch('/api/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) {
      stats.textContent = 'start failed: ' + (j.detail || 'unknown');
      return;
    }
    isRunning = true;
    stats.textContent = 'started';
  };

  restartBtn.onclick = async () => {
    const body = {
      ...startPayload(),
    };
    const r = await fetch('/api/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) {
      stats.textContent = 'restart failed: ' + (j.detail || 'unknown');
      return;
    }
    isRunning = true;
    lastActionInfo = 'none';
    lastShot = null;
    stats.textContent = 'restarted';
  };

  stopBtn.onclick = async () => {
    const r = await fetch('/api/stop', { method: 'POST' });
    const j = await r.json();
    isRunning = false;
    stats.textContent = j.message || 'stopped';
  };

  const onScaleChange = () => {
    displayScale = Math.max(1, parseInt(scaleEl.value || '6', 10));
    applyCanvasDisplaySize();
    draw();
  };
  scaleEl.addEventListener('change', onScaleChange);
  scaleEl.addEventListener('input', onScaleChange);

  canvas.addEventListener('pointerdown', (ev) => {
    if (!isRunning) return;
    const p = normPos(ev);
    drag = { sx: p.x * canvas.width, sy: p.y * canvas.height, ex: p.x * canvas.width, ey: p.y * canvas.height, nsx: p.x, nsy: p.y, nex: p.x, ney: p.y };
    draw();
  });

  canvas.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const p = normPos(ev);
    drag.ex = p.x * canvas.width;
    drag.ey = p.y * canvas.height;
    drag.nex = p.x;
    drag.ney = p.y;
    draw();
  });

  canvas.addEventListener('pointerup', () => {
    if (!drag || !ws || ws.readyState !== WebSocket.OPEN) {
      drag = null;
      draw();
      return;
    }
    actionSeq += 1;
    ws.send(JSON.stringify({
      type: 'action_drag',
      start: { x: drag.nsx, y: drag.nsy },
      end: { x: drag.nex, y: drag.ney },
      client_seq: actionSeq,
    }));
    lastActionInfo = `sent#${actionSeq} waiting_ack`;
    lastShot = {
      sx: drag.sx,
      sy: drag.sy,
      ex: drag.ex,
      ey: drag.ey,
      untilTs: Date.now() + 1200,
    };
    drag = null;
    draw();
  });

  connectWS();
  applyCanvasDisplaySize();
  draw();
})();
</script>
</body>
</html>
"""


@dataclass
class LiveState:
    session_id: str
    context_t: torch.Tensor  # [1, L, C, h, w], GPU tensor
    frame_idx: int
    pending_action: np.ndarray  # [3]
    max_force: float
    target_fps: int
    ddim_steps: int
    runtime_mode: str
    jpeg_quality: int
    running: bool


class StartRequest(BaseModel):
    train_config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    vae_checkpoint_path: Optional[str] = None
    data_manifest_or_dir: Optional[str] = None
    device: str = Field(default="cuda")
    sample_source: str = Field(default="eval")
    seed: int = Field(default=42)
    runtime_mode: str = Field(default="balanced")
    ddim_steps: int = Field(default=8, ge=5, le=50)
    target_fps: int = Field(default=15, ge=4, le=30)
    max_force: float = Field(default=180.0, ge=2.0, le=300.0)
    jpeg_quality: int = Field(default=75, ge=40, le=95)
    compile_model: bool = Field(default=False)
    enable_benchmark: bool = Field(default=True)


@dataclass
class BenchSession:
    session_id: str
    out_dir: Path
    metrics_path: Path
    actions_path: Path
    summary_path: Path
    rows: int
    started_wall_ts: float
    started_mono_ts: float
    series: Dict[str, list[float]]


class LiveGameServer:
    def __init__(self) -> None:
        self.engine: Optional[InferenceEngine] = None
        self._engine_sig: Optional[Tuple[str, str, str, str, bool]] = None
        self.state: Optional[LiveState] = None
        self._lock = threading.Lock()
        self._loop_task: Optional[asyncio.Task[Any]] = None
        self._clients: Set[WebSocket] = set()
        self._last_frame: Optional[np.ndarray] = None
        self._prev_send_latency_ms: int = 0
        self._bench_session: Optional[BenchSession] = None
        self._last_perf_snapshot: Dict[str, float] = {
            "end_to_end_p50_ms": 0.0,
            "end_to_end_p95_ms": 0.0,
            "actual_fps_p50": 0.0,
        }
        self._last_action_snapshot: Dict[str, float] = {
            "action_mag_p50": 0.0,
            "action_mag_p95": 0.0,
            "action_mag_max": 0.0,
            "shot_frame_rate": 0.0,
        }
        self._last_action_event: Dict[str, Any] = {"event": "none"}

    @staticmethod
    def _defaults() -> Dict[str, str]:
        return {
            "train_config_path": os.environ.get("WM_INF_DEFAULT_TRAIN_CONFIG", "").strip(),
            "checkpoint_path": os.environ.get("WM_INF_DEFAULT_CHECKPOINT", "").strip(),
            "vae_checkpoint_path": os.environ.get("WM_INF_DEFAULT_VAE_CHECKPOINT", "").strip(),
            "data_manifest_or_dir": os.environ.get("WM_INF_DEFAULT_DATA_PATH", "").strip(),
        }

    @staticmethod
    def _normalize_runtime_mode(mode: str) -> str:
        m = str(mode or "balanced").strip().lower()
        if m not in {"fast", "balanced", "quality"}:
            return "balanced"
        return m

    def _resolve_start(self, req: StartRequest) -> Dict[str, Any]:
        d = self._defaults()
        out = {
            "train_config_path": (req.train_config_path or d["train_config_path"]),
            "checkpoint_path": (req.checkpoint_path or d["checkpoint_path"]),
            "vae_checkpoint_path": (req.vae_checkpoint_path or d["vae_checkpoint_path"]),
            "data_manifest_or_dir": (req.data_manifest_or_dir or d["data_manifest_or_dir"]),
            "device": req.device,
            "sample_source": req.sample_source,
            "seed": int(req.seed),
            "runtime_mode": self._normalize_runtime_mode(req.runtime_mode),
            "ddim_steps": int(req.ddim_steps),
            "target_fps": int(req.target_fps),
            "max_force": float(req.max_force),
            "jpeg_quality": int(req.jpeg_quality),
            "compile_model": bool(req.compile_model),
            "enable_benchmark": bool(req.enable_benchmark),
        }
        # Runtime mode presets prioritize responsive defaults while still letting
        # callers override with explicit values.
        if out["runtime_mode"] == "fast":
            out["ddim_steps"] = min(out["ddim_steps"], 6)
            out["target_fps"] = max(out["target_fps"], 18)
            out["jpeg_quality"] = min(out["jpeg_quality"], 70)
        elif out["runtime_mode"] == "quality":
            out["ddim_steps"] = max(out["ddim_steps"], 12)
            out["target_fps"] = min(out["target_fps"], 12)
            out["jpeg_quality"] = max(out["jpeg_quality"], 85)
        return out

    def _load_or_get_engine(self, cfg: Dict[str, Any]) -> InferenceEngine:
        sig = (
            str(cfg["checkpoint_path"]),
            str(cfg["train_config_path"]),
            str(cfg["vae_checkpoint_path"]),
            str(cfg["device"]),
            bool(cfg.get("compile_model", False)),
        )
        with self._lock:
            if self.engine is not None and self._engine_sig == sig:
                return self.engine

        engine = load_engine(
            model_name="live_play_model",
            checkpoint_path=cfg["checkpoint_path"],
            train_config_path=cfg["train_config_path"],
            device=cfg["device"],
            vae_cfg={
                "enabled": True,
                "checkpoint_path": cfg["vae_checkpoint_path"],
                "base_channels": int(os.environ.get("WM_INF_VAE_BASE_CHANNELS", "64")),
                "latent_channels": 4,
            },
        )
        if bool(cfg.get("compile_model", False)) and hasattr(torch, "compile"):
            try:
                engine.bundle.model = torch.compile(engine.bundle.model)  # type: ignore[assignment]
                engine.bundle.model.eval()
            except Exception:
                # Keep serving even if compile is unavailable/unstable.
                pass
        with self._lock:
            self.engine = engine
            self._engine_sig = sig
        return engine

    def _bench_root(self) -> Path:
        root = os.environ.get("WM_LIVE_BENCH_ROOT", "./world_model_inference/runs/live_play_bench").strip()
        return Path(root).resolve()

    @staticmethod
    def _pctl(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        return float(np.percentile(np.asarray(vals, dtype=np.float64), q))

    def _start_bench(self, session_id: str, cfg: Dict[str, Any], engine: InferenceEngine) -> None:
        if not bool(cfg.get("enable_benchmark", True)):
            self._bench_session = None
            self._last_perf_snapshot = {
                "end_to_end_p50_ms": 0.0,
                "end_to_end_p95_ms": 0.0,
                "actual_fps_p50": 0.0,
            }
            self._last_action_snapshot = {
                "action_mag_p50": 0.0,
                "action_mag_p95": 0.0,
                "action_mag_max": 0.0,
                "shot_frame_rate": 0.0,
            }
            return
        out_dir = self._bench_root() / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "timings.jsonl"
        actions_path = out_dir / "actions.jsonl"
        summary_path = out_dir / "summary.json"
        manifest = {
            "session_id": session_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "runtime_mode": str(cfg["runtime_mode"]),
            "target_fps": int(cfg["target_fps"]),
            "ddim_steps": int(cfg["ddim_steps"]),
            "jpeg_quality": int(cfg["jpeg_quality"]),
            "compile_model": bool(cfg.get("compile_model", False)),
            "model_checkpoint": str(engine.bundle.checkpoint_path),
            "train_config": str(engine.bundle.config_path),
            "vae_enabled": bool(engine.vae is not None),
            "device": str(engine.device),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._bench_session = BenchSession(
            session_id=session_id,
            out_dir=out_dir,
            metrics_path=metrics_path,
            actions_path=actions_path,
            summary_path=summary_path,
            rows=0,
            started_wall_ts=time.time(),
            started_mono_ts=time.perf_counter(),
            series={
                "end_to_end_ms": [],
                "ddim_total_ms": [],
                "model_forward_ms": [],
                "vae_decode_ms": [],
                "jpeg_encode_ms": [],
                "send_ms": [],
                "actual_fps": [],
                "action_mag": [],
                "action_shot": [],
            },
        )
        self._last_perf_snapshot = {
            "end_to_end_p50_ms": 0.0,
            "end_to_end_p95_ms": 0.0,
            "actual_fps_p50": 0.0,
        }
        self._last_action_snapshot = {
            "action_mag_p50": 0.0,
            "action_mag_p95": 0.0,
            "action_mag_max": 0.0,
            "shot_frame_rate": 0.0,
        }

    def _append_bench(self, row: Dict[str, Any]) -> None:
        s = self._bench_session
        if s is None:
            return
        with s.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        s.rows += 1
        for k in list(s.series.keys()):
            v = float(row.get(k, 0.0))
            s.series[k].append(v)
        self._last_perf_snapshot = {
            "end_to_end_p50_ms": self._pctl(s.series["end_to_end_ms"], 50),
            "end_to_end_p95_ms": self._pctl(s.series["end_to_end_ms"], 95),
            "actual_fps_p50": self._pctl(s.series["actual_fps"], 50),
        }
        mags = s.series.get("action_mag", [])
        shots = s.series.get("action_shot", [])
        self._last_action_snapshot = {
            "action_mag_p50": self._pctl(mags, 50),
            "action_mag_p95": self._pctl(mags, 95),
            "action_mag_max": float(max(mags)) if mags else 0.0,
            "shot_frame_rate": float(np.mean(shots)) if shots else 0.0,
        }

    def _append_action_event(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row["ts_unix"] = float(time.time())
        self._last_action_event = row
        s = self._bench_session
        if s is None:
            return
        with s.actions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _finalize_bench(self, reason: str) -> None:
        s = self._bench_session
        if s is None:
            return
        duration_sec = max(1e-9, time.perf_counter() - s.started_mono_ts)
        summary = {
            "session_id": s.session_id,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "reason": reason,
            "rows": int(s.rows),
            "duration_sec": float(duration_sec),
            "metrics": {
                k: {
                    "mean": float(np.mean(v)) if v else 0.0,
                    "p50": self._pctl(v, 50),
                    "p95": self._pctl(v, 95),
                }
                for k, v in s.series.items()
            },
        }
        s.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._bench_session = None

    def _build_start_context(self, engine: InferenceEngine, cfg: Dict[str, Any]) -> torch.Tensor:
        prompt_cfg = dict(DEFAULT_PREVIEW_CONFIG)
        prompt_cfg["run"] = dict(DEFAULT_PREVIEW_CONFIG["run"])
        prompt_cfg["data"] = dict(DEFAULT_PREVIEW_CONFIG["data"])
        prompt_cfg["actions"] = dict(DEFAULT_PREVIEW_CONFIG["actions"])

        prompt_cfg["run"]["horizon"] = 8
        prompt_cfg["run"]["seed"] = int(cfg["seed"])
        prompt_cfg["actions"]["source"] = "dataset"

        p = str(cfg["data_manifest_or_dir"]).strip()
        if p:
            if p.endswith(".txt") or p.endswith(".json") or p.endswith(".yaml") or p.endswith(".yml"):
                prompt_cfg["data"]["eval_shards_manifest"] = p
            else:
                prompt_cfg["data"]["shards_dir"] = p
        prompt_cfg["data"]["sample_from"] = str(cfg["sample_source"]).strip().lower() or "eval"

        prompt = build_prompt_from_config(prompt_cfg, engine)
        context_np = np.array(prompt.context_max[:1, -engine.bundle.context_len :], dtype=np.float32, copy=True)
        return torch.from_numpy(context_np).to(device=engine.device, dtype=torch.float32)

    def _sample_next_latent_ddim_profiled(
        self,
        bundle: wm_eval.ModelBundle,
        context: torch.Tensor,
        action: torch.Tensor,
        ddim_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """DDIM sampling with timing breakdown for model-forward and full loop."""
        b, _, c, h, w = context.shape
        x = torch.randn((b, c, h, w), device=context.device, dtype=context.dtype)
        schedule = wm_eval._ddim_schedule_indices(bundle.diffusion_steps, ddim_steps)
        alpha_bar = bundle.diffusion.alpha_bar

        t_ddim0 = time.perf_counter()
        model_forward_sec = 0.0
        for i in range(ddim_steps):
            t_val = int(schedule[i])
            t_next = int(schedule[i + 1])
            t = torch.full((b,), t_val, dtype=torch.long, device=context.device)

            t_fw0 = time.perf_counter()
            pred_noise = bundle.model(context=context, action=action, noisy_target=x, t_idx=t)
            model_forward_sec += time.perf_counter() - t_fw0

            a_t = alpha_bar[t_val]
            sqrt_a_t = torch.sqrt(a_t)
            sqrt_one_minus_a_t = torch.sqrt(1.0 - a_t)
            x0 = (x - sqrt_one_minus_a_t * pred_noise) / sqrt_a_t

            if t_next >= 0:
                a_next = alpha_bar[t_next]
            else:
                a_next = torch.tensor(1.0, device=context.device, dtype=x.dtype)
            x = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * pred_noise

        ddim_total_ms = (time.perf_counter() - t_ddim0) * 1000.0
        return x, {
            "model_forward_ms": model_forward_sec * 1000.0,
            "ddim_total_ms": ddim_total_ms,
        }

    def _decode_frame_rgb(self, pred_next: torch.Tensor) -> Tuple[np.ndarray, float]:
        assert self.engine is not None and self.engine.vae is not None
        t0 = time.perf_counter()
        frame_t = wm_eval._decode_latents(
            self.engine.vae,
            pred_next.unsqueeze(1),
            device=self.engine.device,
            batch_size=1,
        )[0, 0]
        frame = torch.clamp(frame_t, 0.0, 1.0).permute(1, 2, 0).mul(255.0).byte().numpy()
        return frame, (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _encode_jpeg_b64(frame: np.ndarray, jpeg_quality: int) -> Tuple[str, int, int, float]:
        t0 = time.perf_counter()
        with io.BytesIO() as buf:
            Image.fromarray(frame).save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, int(frame.shape[1]), int(frame.shape[0]), (time.perf_counter() - t0) * 1000.0

    def _step_once_sync(self) -> Dict[str, Any]:
        with self._lock:
            if self.engine is None or self.state is None or not self.state.running:
                return {"type": "info", "message": "not running"}
            state = self.state
            engine = self.engine
            action = np.array(state.pending_action, dtype=np.float32, copy=True)
            state.pending_action[:] = 0.0
            ddim_steps = int(state.ddim_steps)
            target_fps = int(state.target_fps)
            runtime_mode = str(state.runtime_mode)
            jpeg_quality = int(state.jpeg_quality)
            context_t = state.context_t
            frame_idx = int(state.frame_idx)
            perf_snapshot = dict(self._last_perf_snapshot)
            action_snapshot = dict(self._last_action_snapshot)
            prev_send_ms = int(self._prev_send_latency_ms)

        action_fx = float(action[0])
        action_fy = float(action[1])
        action_trigger = float(action[2])
        action_mag = float(np.hypot(action_fx, action_fy))
        action_is_shot = 1.0 if (action_trigger > 0.5 and action_mag > 1e-6) else 0.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            action_t = torch.from_numpy(action.reshape(1, -1)).to(device=engine.device, dtype=torch.float32)
            pred_next, ddim_stats = self._sample_next_latent_ddim_profiled(
                bundle=engine.bundle,
                context=context_t,
                action=action_t,
                ddim_steps=ddim_steps,
            )
            new_context_t = torch.cat([context_t, pred_next.unsqueeze(1)], dim=1)[:, -engine.bundle.context_len :]

        frame, decode_ms = self._decode_frame_rgb(pred_next)
        self._last_frame = frame
        b64, width, height, jpeg_ms = self._encode_jpeg_b64(frame, jpeg_quality=jpeg_quality)
        step_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            if self.state is not None:
                self.state.context_t = new_context_t
                self.state.frame_idx = frame_idx + 1

        self._append_action_event(
            {
                "event": "action_consumed",
                "frame_idx": int(frame_idx + 1),
                "fx": action_fx,
                "fy": action_fy,
                "trigger": action_trigger,
                "mag": action_mag,
                "is_shot": action_is_shot,
            }
        )

        return {
            "type": "frame",
            "jpeg_b64": b64,
            "width": width,
            "height": height,
            "frame_idx": frame_idx + 1,
            "running": True,
            "runtime_mode": runtime_mode,
            "ddim_steps_used": int(ddim_steps),
            "model_latency_ms": int(ddim_stats["ddim_total_ms"]),
            "model_forward_ms": int(ddim_stats["model_forward_ms"]),
            "ddim_total_ms": int(ddim_stats["ddim_total_ms"]),
            "vae_decode_ms": int(decode_ms),
            "jpeg_encode_ms": int(jpeg_ms),
            "loop_latency_ms": int(step_ms),
            "send_latency_ms": int(prev_send_ms),
            "target_fps": target_fps,
            "action_used": [action_fx, action_fy, action_trigger],
            "action_mag": action_mag,
            "action_is_shot": action_is_shot,
            "perf": perf_snapshot,
            "action_summary": action_snapshot,
            "_bench": {
                "model_forward_ms": float(ddim_stats["model_forward_ms"]),
                "ddim_total_ms": float(ddim_stats["ddim_total_ms"]),
                "vae_decode_ms": float(decode_ms),
                "jpeg_encode_ms": float(jpeg_ms),
                "step_latency_ms": float(step_ms),
                "runtime_mode": runtime_mode,
                "ddim_steps": int(ddim_steps),
                "target_fps": int(target_fps),
                "frame_idx": int(frame_idx + 1),
                "action_mag": float(action_mag),
                "action_shot": float(action_is_shot),
            },
        }

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        msg = json.dumps(payload)
        for ws in list(self._clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _run_loop(self) -> None:
        while True:
            with self._lock:
                running = bool(self.state and self.state.running)
                target_fps = int(self.state.target_fps) if self.state else 15
            if not running:
                break

            loop_t0 = time.time()
            try:
                payload = await asyncio.to_thread(self._step_once_sync)
            except Exception as e:
                await self._broadcast({"type": "error", "message": str(e)})
                with self._lock:
                    if self.state is not None:
                        self.state.running = False
                break

            payload["loop_latency_ms"] = int((time.time() - loop_t0) * 1000)
            bench_row = dict(payload.get("_bench", {}))
            payload.pop("_bench", None)

            send_t0 = time.perf_counter()
            await self._broadcast(payload)
            send_ms = (time.perf_counter() - send_t0) * 1000.0
            self._prev_send_latency_ms = int(send_ms)

            if bench_row:
                loop_ms = float(payload.get("loop_latency_ms", 0.0))
                actual_fps = 1000.0 / max(1e-6, loop_ms + send_ms)
                bench_row.update(
                    {
                        "send_ms": float(send_ms),
                        "end_to_end_ms": float(loop_ms + send_ms),
                        "loop_ms": float(loop_ms),
                        "actual_fps": float(actual_fps),
                        "ts_unix": float(time.time()),
                    }
                )
                self._append_bench(bench_row)

            dt = time.time() - loop_t0
            sleep_for = max(0.0, (1.0 / float(max(1, target_fps))) - dt)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    async def start(self, req: StartRequest) -> Dict[str, Any]:
        cfg = self._resolve_start(req)

        # Stop previous loop cleanly.
        await self.stop(message="restarting")

        try:
            engine = await asyncio.to_thread(self._load_or_get_engine, cfg)
            context_t = await asyncio.to_thread(self._build_start_context, engine, cfg)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        session_id = f"live_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}"
        with self._lock:
            self.state = LiveState(
                session_id=session_id,
                context_t=context_t,
                frame_idx=0,
                pending_action=np.zeros((3,), dtype=np.float32),
                max_force=float(cfg["max_force"]),
                target_fps=int(cfg["target_fps"]),
                ddim_steps=int(cfg["ddim_steps"]),
                runtime_mode=str(cfg["runtime_mode"]),
                jpeg_quality=int(cfg["jpeg_quality"]),
                running=True,
            )
        self._prev_send_latency_ms = 0
        self._start_bench(session_id=session_id, cfg=cfg, engine=engine)

        # Emit first frame from prompt context tail for instant visual.
        try:
            with torch.inference_mode():
                first_latent = context_t[:, -1]
            frame, decode_ms = await asyncio.to_thread(self._decode_frame_rgb, first_latent)
            self._last_frame = frame
            b64, width, height, jpeg_ms = await asyncio.to_thread(
                self._encode_jpeg_b64, frame, int(cfg["jpeg_quality"])
            )
            await self._broadcast(
                {
                    "type": "frame",
                    "jpeg_b64": b64,
                    "width": width,
                    "height": height,
                    "frame_idx": 0,
                    "running": True,
                    "runtime_mode": str(cfg["runtime_mode"]),
                    "ddim_steps_used": int(cfg["ddim_steps"]),
                    "model_latency_ms": 0,
                    "model_forward_ms": 0,
                    "ddim_total_ms": 0,
                    "vae_decode_ms": int(decode_ms),
                    "jpeg_encode_ms": int(jpeg_ms),
                    "loop_latency_ms": 0,
                    "send_latency_ms": 0,
                    "target_fps": int(cfg["target_fps"]),
                    "action_used": [0.0, 0.0, 0.0],
                    "action_mag": 0.0,
                    "action_is_shot": 0.0,
                    "perf": dict(self._last_perf_snapshot),
                    "action_summary": dict(self._last_action_snapshot),
                }
            )
        except Exception:
            pass

        self._loop_task = asyncio.create_task(self._run_loop())
        return {
            "ok": True,
            "session_id": session_id,
            "runtime_mode": str(cfg["runtime_mode"]),
            "ddim_steps": int(cfg["ddim_steps"]),
            "target_fps": int(cfg["target_fps"]),
            "max_force": float(cfg["max_force"]),
            "jpeg_quality": int(cfg["jpeg_quality"]),
            "compile_model": bool(cfg.get("compile_model", False)),
            "bench_enabled": bool(cfg.get("enable_benchmark", True)),
            "bench_dir": str(self._bench_session.out_dir) if self._bench_session is not None else "",
            "actions_log": str(self._bench_session.actions_path) if self._bench_session is not None else "",
            "action_summary": dict(self._last_action_snapshot),
        }

    async def stop(self, message: str = "stopped") -> Dict[str, Any]:
        with self._lock:
            if self.state is not None:
                self.state.running = False

        if self._loop_task is not None:
            try:
                await asyncio.wait_for(self._loop_task, timeout=2.0)
            except Exception:
                self._loop_task.cancel()
            self._loop_task = None

        self._finalize_bench(reason=message)
        await self._broadcast({"type": "info", "message": message})
        return {"ok": True, "message": message}

    async def queue_drag_action(self, start: Dict[str, float], end: Dict[str, float]) -> Dict[str, Any]:
        with self._lock:
            if self.state is None or not self.state.running:
                return {"ok": False, "message": "not running"}
            state = self.state
            context_t = state.context_t
            w = int(context_t.shape[-1])
            h = int(context_t.shape[-2])

            sx = float(max(0.0, min(1.0, float(start.get("x", 0.0))))) * float(w - 1)
            sy = float(max(0.0, min(1.0, float(start.get("y", 0.0))))) * float(h - 1)
            ex = float(max(0.0, min(1.0, float(end.get("x", 0.0))))) * float(w - 1)
            ey = float(max(0.0, min(1.0, float(end.get("y", 0.0))))) * float(h - 1)

            dx = ex - sx
            dy = ey - sy
            dist = float(np.sqrt(dx * dx + dy * dy))
            norm = max(float(np.sqrt(float(w * w + h * h))), 1.0)
            ux = dx / norm
            uy = dy / norm
            drag_scale = min(dist / (0.35 * norm), 1.0)

            fmax = float(max(0.0, state.max_force))
            fx = float(ux * fmax * drag_scale)
            fy = float(uy * fmax * drag_scale)
            mag = float(np.hypot(fx, fy))
            state.pending_action[:] = np.array([fx, fy, 1.0], dtype=np.float32)

        self._append_action_event(
            {
                "event": "action_queued",
                "fx": fx,
                "fy": fy,
                "mag": mag,
                "trigger": 1.0,
                "drag_scale": float(drag_scale),
                "max_force": float(fmax),
                "start": {"x": float(start.get("x", 0.0)), "y": float(start.get("y", 0.0))},
                "end": {"x": float(end.get("x", 0.0)), "y": float(end.get("y", 0.0))},
            }
        )

        return {
            "ok": True,
            "fx": fx,
            "fy": fy,
            "mag": mag,
            "message": f"action queued fx={fx:.2f} fy={fy:.2f} |mag|={mag:.2f}",
            "start": {"x": float(start.get("x", 0.0)), "y": float(start.get("y", 0.0))},
            "end": {"x": float(end.get("x", 0.0)), "y": float(end.get("y", 0.0))},
        }

    async def set_params(self, ddim_steps: Optional[int], target_fps: Optional[int], max_force: Optional[float]) -> Dict[str, Any]:
        with self._lock:
            if self.state is None:
                return {"ok": False, "message": "not running"}
            if ddim_steps is not None:
                self.state.ddim_steps = int(max(5, min(50, ddim_steps)))
            if target_fps is not None:
                self.state.target_fps = int(max(4, min(30, target_fps)))
            if max_force is not None:
                self.state.max_force = float(max(2.0, min(300.0, max_force)))
            return {
                "ok": True,
                "ddim_steps": int(self.state.ddim_steps),
                "target_fps": int(self.state.target_fps),
                "max_force": float(self.state.max_force),
            }

    async def attach(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        with self._lock:
            state = self.state
        await ws.send_text(
            json.dumps(
                {
                    "type": "info",
                    "message": "connected",
                    "running": bool(state and state.running),
                }
            )
        )

        if self._last_frame is not None:
            jpeg_quality = int(state.jpeg_quality) if state is not None else 75
            b64 = base64.b64encode(self._jpeg_bytes(self._last_frame, quality=jpeg_quality)).decode("ascii")
            await ws.send_text(
                json.dumps(
                    {
                        "type": "frame",
                        "jpeg_b64": b64,
                        "width": int(self._last_frame.shape[1]),
                        "height": int(self._last_frame.shape[0]),
                        "frame_idx": int(state.frame_idx) if state else 0,
                        "running": bool(state and state.running),
                        "runtime_mode": str(state.runtime_mode) if state else "n/a",
                        "ddim_steps_used": int(state.ddim_steps) if state else 0,
                        "model_latency_ms": 0,
                        "model_forward_ms": 0,
                        "ddim_total_ms": 0,
                        "vae_decode_ms": 0,
                        "jpeg_encode_ms": 0,
                        "loop_latency_ms": 0,
                        "send_latency_ms": int(self._prev_send_latency_ms),
                        "target_fps": int(state.target_fps) if state else 0,
                        "action_used": [0.0, 0.0, 0.0],
                        "action_mag": 0.0,
                        "action_is_shot": 0.0,
                        "perf": dict(self._last_perf_snapshot),
                        "action_summary": dict(self._last_action_snapshot),
                    }
                )
            )

    @staticmethod
    def _jpeg_bytes(frame: np.ndarray, quality: int = 75) -> bytes:
        with io.BytesIO() as buf:
            Image.fromarray(frame).save(
                buf,
                format="JPEG",
                quality=int(max(40, min(95, quality))),
                optimize=True,
            )
            return buf.getvalue()

    async def detach(self, ws: WebSocket) -> None:
        self._clients.discard(ws)


server = LiveGameServer()
app = FastAPI(title="Pool Live Play", version="1.0")


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/status")
async def status() -> Dict[str, Any]:
    with server._lock:
        st = server.state
        pending_mag = 0.0
        if st is not None:
            pending_mag = float(np.hypot(float(st.pending_action[0]), float(st.pending_action[1])))
        return {
            "running": bool(st and st.running),
            "frame_idx": int(st.frame_idx) if st else 0,
            "ddim_steps": int(st.ddim_steps) if st else None,
            "target_fps": int(st.target_fps) if st else None,
            "max_force": float(st.max_force) if st else None,
            "runtime_mode": str(st.runtime_mode) if st else None,
            "jpeg_quality": int(st.jpeg_quality) if st else None,
            "perf": dict(server._last_perf_snapshot),
            "action_summary": dict(server._last_action_snapshot),
            "last_action_event": dict(server._last_action_event),
            "pending_action_mag": pending_mag,
            "bench_dir": str(server._bench_session.out_dir) if server._bench_session is not None else "",
        }


@app.post("/api/start")
async def start(req: StartRequest) -> Dict[str, Any]:
    return await server.start(req)


@app.post("/api/stop")
async def stop() -> Dict[str, Any]:
    return await server.stop()


@app.post("/api/params")
async def params(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await server.set_params(
        ddim_steps=payload.get("ddim_steps"),
        target_fps=payload.get("target_fps"),
        max_force=payload.get("max_force"),
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await server.attach(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                payload = json.loads(msg)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid json"}))
                continue

            t = str(payload.get("type", "")).strip().lower()
            if t == "action_drag":
                start = payload.get("start") or {}
                end = payload.get("end") or {}
                client_seq = payload.get("client_seq")
                out = await server.queue_drag_action(start=start, end=end)
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "action_ack",
                            "ok": bool(out.get("ok", False)),
                            "message": out.get("message", ""),
                            "fx": out.get("fx"),
                            "fy": out.get("fy"),
                            "mag": out.get("mag"),
                            "start": out.get("start"),
                            "end": out.get("end"),
                            "client_seq": client_seq,
                        }
                    )
                )
            elif t == "set_params":
                out = await server.set_params(
                    ddim_steps=payload.get("ddim_steps"),
                    target_fps=payload.get("target_fps"),
                    max_force=payload.get("max_force"),
                )
                await ws.send_text(json.dumps({"type": "info", "message": f"params updated: {out}"}))
            elif t == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
            else:
                await ws.send_text(json.dumps({"type": "error", "message": f"unknown message type: {t}"}))
    except WebSocketDisconnect:
        pass
    finally:
        await server.detach(ws)


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-latency live play route (websocket canvas)")
    parser.add_argument("--host", default=os.environ.get("WM_LIVE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WM_LIVE_PORT", "7863")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "world_model_inference.live_play:app",
        host=str(args.host),
        port=int(args.port),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
