#!/usr/bin/env python3
"""
energy_eval_pipelines.py
------------------------
Measures energy consumption of six smoke-detection pipelines on a
Jetson AGX Xavier (or any Jetson with tegrastats / jtop).

Pipelines:
  1. mobilenet        -- LBP + optical-flow + MobileNet on CPU
  2. resnet34         -- ResNet34 on GPU
  3. yolov8           -- YOLOv8 on GPU
  4. ensemble_OR      -- ResNet34 + YOLOv8 OR logic on GPU
  5. gate_from_window -- MobileNet (CPU) gates the ensemble (GPU);
                         once MobileNet fires, ensemble runs from that
                         frame onwards
  6. gate_from_start  -- MobileNet (CPU) gates the ensemble (GPU);
                         once MobileNet fires, ensemble retroactively
                         re-processes ALL frames from index 0

Two 3-hour sequences are evaluated (180 frames at 1 frame/minute):
  A. No-smoke : all frames are no-smoke
  B. Smoke    : frames before --smoke_start_idx are no-smoke, rest smoke

MobileNet always runs on CPU (assumed more energy-efficient for this
lightweight model). ResNet34 and YOLOv8 always run on GPU.

Power is read from VDD_IN via tegrastats (or jtop if installed).
An idle baseline is subtracted so reported energy reflects only the
incremental cost of inference.

Usage
-----
    python energy_eval_pipelines.py \\
        --no_smoke_dir /path/to/no_smoke_sequence \\
        --smoke_dir    /path/to/smoke_sequence \\
        --smoke_start_idx 60 \\
        --mobilenet_ckpt  sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \\
        --resnet_ckpt     ../smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt \\
        --yolo_ckpt       ../smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt \\
        --n_frames 2 --frame_gap 1 \\
        --threshold 0.5 \\
        --out_dir energy_results/pipelines
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Allow imports from this directory
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))


# ---------------------------------------------------------------------------
# Power monitoring  (shared with energy_eval.py)
# ---------------------------------------------------------------------------

class _BasePowerMonitor:
    def start(self): ...
    def stop(self): ...
    def mean_power_mw(self, t_start: float, t_end: float) -> float | None:
        raise NotImplementedError


class TegrastatsMonitor(_BasePowerMonitor):
    _VDD_IN_RE   = re.compile(r'VDD_IN\s+(\d+)mW')
    _FIRST_MW_RE = re.compile(r'(\d+)mW')

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._readings: list[tuple[float, float]] = []
        self._lock   = threading.Lock()
        self._proc   = None
        self._thread = None
        self._stop   = threading.Event()

    def _parse_power(self, line: str) -> float | None:
        m = self._VDD_IN_RE.search(line)
        if m:
            return float(m.group(1))
        m = self._FIRST_MW_RE.search(line)
        return float(m.group(1)) if m else None

    def _reader(self):
        for raw in iter(self._proc.stdout.readline, b''):
            if self._stop.is_set():
                break
            t = time.perf_counter()
            p = self._parse_power(raw.decode('utf-8', errors='ignore'))
            if p is not None:
                with self._lock:
                    self._readings.append((t, p))

    def start(self):
        self._proc = subprocess.Popen(
            ['tegrastats', '--interval', str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(0.5)
        print(f"  [TegrastatsMonitor] started  (interval={self.interval_ms}ms)")

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2)
        print(f"  [TegrastatsMonitor] stopped  ({len(self._readings)} readings)")

    def mean_power_mw(self, t_start: float, t_end: float) -> float | None:
        with self._lock:
            window = [p for t, p in self._readings if t_start <= t <= t_end]
            if window:
                return float(np.mean(window))
            mid = (t_start + t_end) / 2.0
            if self._readings:
                _, nearest = min(self._readings, key=lambda x: abs(x[0] - mid))
                return float(nearest)
        return None


class JtopMonitor(_BasePowerMonitor):
    def __init__(self, interval_ms: int = 100):
        self.interval_s = interval_ms / 1000.0
        self._readings: list[tuple[float, float]] = []
        self._lock   = threading.Lock()
        self._thread = None
        self._stop   = threading.Event()

    def _reader(self):
        from jtop import jtop
        with jtop() as jetson:
            while not self._stop.is_set():
                t = time.perf_counter()
                try:
                    p = jetson.power['tot']['power']    # jtop 7.x
                except (KeyError, TypeError, AttributeError):
                    try:
                        p = jetson.power[1]['tot']['cur']  # jtop 4.x
                    except (KeyError, TypeError, AttributeError):
                        try:
                            p = jetson.stats['Power TOT']  # jtop 3.x
                        except (KeyError, TypeError):
                            p = None
                if p is not None:
                    with self._lock:
                        self._readings.append((t, float(p)))
                time.sleep(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(0.5)
        print(f"  [JtopMonitor] started  (interval={self.interval_s*1000:.0f}ms)")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        print(f"  [JtopMonitor] stopped  ({len(self._readings)} readings)")

    def mean_power_mw(self, t_start: float, t_end: float) -> float | None:
        with self._lock:
            window = [p for t, p in self._readings if t_start <= t <= t_end]
            if window:
                return float(np.mean(window))
            mid = (t_start + t_end) / 2.0
            if self._readings:
                _, nearest = min(self._readings, key=lambda x: abs(x[0] - mid))
                return float(nearest)
        return None


def make_monitor(interval_ms: int) -> _BasePowerMonitor:
    try:
        import jtop  # noqa: F401
        print("Power monitor: jtop (jetson-stats)")
        return JtopMonitor(interval_ms=interval_ms)
    except ImportError:
        pass
    print("Power monitor: tegrastats")
    return TegrastatsMonitor(interval_ms=interval_ms)


def measure_idle_power(monitor: _BasePowerMonitor, duration_s: float = 5.0) -> float:
    print(f"\nMeasuring idle power  ({duration_s:.0f}s) ...")
    t0 = time.perf_counter()
    time.sleep(duration_s)
    t1 = time.perf_counter()
    idle_mw = monitor.mean_power_mw(t0, t1) or 0.0
    print(f"  Idle power: {idle_mw:.1f} mW")
    return idle_mw


def energy_mj(monitor, t_start, t_end, idle_mw) -> float:
    """Net energy (mJ) = (raw_power - idle) * duration, clamped to 0."""
    raw = monitor.mean_power_mw(t_start, t_end) or 0.0
    net_mw = max(0.0, raw - idle_mw)
    return net_mw * (t_end - t_start) * 1e3   # mW * s → mJ


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_frames(directory: Path, limit: int = 180) -> list[Path]:
    """Return up to `limit` image paths from directory, sorted by filename."""
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        [f for f in directory.iterdir() if f.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if len(frames) > limit:
        frames = frames[:limit]
    return frames


def is_valid(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LBP feature computation  (always on CPU)
# ---------------------------------------------------------------------------

def compute_lbp_pair(f1: Path, f2: Path, target_size=(240, 180)) -> np.ndarray | None:
    """Compute LBP-motion image for a frame pair on CPU."""
    try:
        import cv2
        from feature_extraction import make_lbp_motion_image
        bgr1 = cv2.imread(str(f1))
        bgr2 = cv2.imread(str(f2))
        if bgr1 is None or bgr2 is None:
            return None
        return make_lbp_motion_image(bgr1, bgr2, target_size)
    except Exception:
        return None


def load_lbp_pair_cached(
    f1: Path, f2: Path, pair_idx: int,
    cache_seq_dir: Path | None,
    target_size=(240, 180),
) -> np.ndarray | None:
    if cache_seq_dir is not None:
        cached = cache_seq_dir / f"pair_{pair_idx:04d}.png"
        if cached.exists():
            return np.array(Image.open(cached).convert("RGB"))
    return compute_lbp_pair(f1, f2, target_size)


def get_lbp_image(
    frames: list[Path], idx: int,
    n_frames: int, frame_gap: int,
    cache_seq_dir: Path | None,
) -> np.ndarray | None:
    """
    Build the LBP-motion image for the window ending at frames[idx].
    Window: [frames[idx - (n_frames-1)*frame_gap], ..., frames[idx]]
    Returns None if any required frame is missing.
    """
    window_size = (n_frames - 1) * frame_gap
    start = idx - window_size
    if start < 0:
        return None
    window = [frames[start + i * frame_gap] for i in range(n_frames)]

    if n_frames == 2:
        return load_lbp_pair_cached(window[0], window[1], start, cache_seq_dir)
    else:
        pairs = []
        for i in range(n_frames - 1):
            p = load_lbp_pair_cached(
                window[i], window[i + 1],
                start + i * frame_gap,
                cache_seq_dir,
            )
            if p is not None:
                pairs.append(p)
        return np.mean(pairs, axis=0).astype(np.uint8) if pairs else None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ort_session(ckpt_path: str, use_gpu: bool):
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if use_gpu else ["CPUExecutionProvider"])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(ckpt_path, sess_options=opts, providers=providers)


def load_mobilenet(ckpt_path: str, cpu_device: torch.device):
    from model import get_transforms
    transform = get_transforms(train=False)
    ext = Path(ckpt_path).suffix.lower()

    if ext == ".onnx":
        session = _ort_session(ckpt_path, use_gpu=False)
        print(f"  MobileNet loaded : {ckpt_path}  (ONNX Runtime, device=cpu)")
        return ("onnx", session), transform

    # .pt / .pth — PyTorch checkpoint
    from model import build_model
    ckpt    = torch.load(ckpt_path, map_location=cpu_device)
    variant = ckpt.get("variant", "v3_small")
    model   = build_model(variant=variant)
    model.load_state_dict(ckpt["state_dict"])
    model.to(cpu_device).eval()
    print(f"  MobileNet loaded : {ckpt_path}  (PyTorch, variant={variant}, device=cpu)")
    return ("torch", model), transform


def load_resnet(ckpt_path: str, gpu_device: torch.device):
    ext = Path(ckpt_path).suffix.lower()

    if ext == ".onnx":
        session = _ort_session(ckpt_path, use_gpu=(str(gpu_device) != "cpu"))
        print(f"  ResNet34 loaded  : {ckpt_path}  (ONNX Runtime, device={gpu_device})")
        return ("onnx", session)

    # .pt / .pth — PyTorch checkpoint
    else:
        sd = ckpt
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(sd)
    model.to(gpu_device).eval()
    print(f"  ResNet34 loaded  : {ckpt_path}  (PyTorch, device={gpu_device})")
    return ("torch", model)


def load_yolo(ckpt_path: str):
    ext = Path(ckpt_path).suffix.lower()

    if ext == ".onnx":
        # Use ONNX Runtime directly — no ultralytics needed
        session = _ort_session(ckpt_path, use_gpu=False)
        print(f"  YOLOv8 loaded    : {ckpt_path}  (ONNX Runtime)")
        return ("onnx", session)

    # .pt / .trt / .engine — use ultralytics
    import functools
    from ultralytics import YOLO

    # Ultralytics requires .engine extension for TensorRT; rename .trt → .engine
    load_path = ckpt_path
    if ext == ".trt":
        engine_path = Path(ckpt_path).with_suffix(".engine")
        if not engine_path.exists():
            engine_path.symlink_to(Path(ckpt_path).resolve())
        load_path = str(engine_path)

    orig = torch.load
    torch.load = functools.partial(orig, weights_only=False)
    try:
        model = YOLO(load_path, task="classify")
    finally:
        torch.load = orig
    print(f"  YOLOv8 loaded    : {ckpt_path}  (Ultralytics)")
    return ("ultralytics", model)


# ---------------------------------------------------------------------------
# Per-frame inference helpers
# ---------------------------------------------------------------------------

_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def infer_mobilenet(lbp_img: np.ndarray, model_tuple, transform, device) -> float:
    fmt, model = model_tuple
    img_tensor = transform(Image.fromarray(lbp_img)).unsqueeze(0)
    if fmt == "onnx":
        arr = img_tensor.numpy()
        logit = model.run(["logit"], {"input": arr})[0].squeeze()
        return float(1 / (1 + np.exp(-logit)))  # sigmoid
    # torch
    with torch.no_grad():
        logit = model(img_tensor.to(device)).squeeze()
        return torch.sigmoid(logit).item()


def infer_resnet(frame_path: Path, model_tuple, device) -> float:
    fmt, model = model_tuple
    img = Image.open(frame_path).convert("RGB")
    tensor = _resnet_transform(img).unsqueeze(0)
    if fmt == "onnx":
        arr = tensor.numpy()
        logits = model.run(None, {"input": arr})[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(probs[0, 1])
    # torch
    with torch.no_grad():
        logits = model(tensor.to(device))
        return torch.softmax(logits, dim=1)[0, 1].item()


def infer_yolo(frame_path: Path, yolo_model, imgsz: int = 224) -> float:
    fmt, model = yolo_model
    if fmt == "onnx":
        from PIL import Image as PILImage
        # Detect the input size the model was exported with
        expected_h = model.get_inputs()[0].shape[2]
        expected_w = model.get_inputs()[0].shape[3]
        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = PILImage.open(frame_path).convert("RGB").resize((expected_w, expected_h))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _mean) / _std
        arr = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        input_name = model.get_inputs()[0].name
        logits = model.run(None, {input_name: arr})[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(probs[0, 1])
    # ultralytics
    results = model.predict(source=str(frame_path), imgsz=imgsz, verbose=False)
    return float(results[0].probs.data[1].item())


# ---------------------------------------------------------------------------
# Pipeline evaluation: process one sequence, one pipeline
# ---------------------------------------------------------------------------

def _record(
    frame_idx: int, frame_path: Path, true_label: int,
    mob_energy_mj: float, ensemble_energy_mj: float,
    mob_prob: float | None, resnet_prob: float | None, yolo_prob: float | None,
    prediction: int,
) -> dict:
    return {
        "frame_idx":         frame_idx,
        "frame":             frame_path.name,
        "true_label":        true_label,
        "prediction":        prediction,
        "correct":           int(prediction == true_label),
        "mob_prob":          round(mob_prob,    4) if mob_prob    is not None else None,
        "resnet_prob":       round(resnet_prob, 4) if resnet_prob is not None else None,
        "yolo_prob":         round(yolo_prob,   4) if yolo_prob   is not None else None,
        "mob_energy_mj":     round(mob_energy_mj,      4),
        "ensemble_energy_mj":round(ensemble_energy_mj, 4),
        "total_energy_mj":   round(mob_energy_mj + ensemble_energy_mj, 4),
    }


def run_mobilenet_only(
    frames, labels, mob_model, mob_transform, cpu_device,
    n_frames, frame_gap, threshold, monitor, idle_mw,
    cache_seq_dir, warmup,
) -> list[dict]:
    records = []
    for i, (fp, lbl) in enumerate(zip(frames, labels)):
        lbp = get_lbp_image(frames, i, n_frames, frame_gap, cache_seq_dir)
        if lbp is None:
            continue

        t0 = time.perf_counter()
        prob = infer_mobilenet(lbp, mob_model, mob_transform, cpu_device)
        t1 = time.perf_counter()

        e_mj = 0.0 if i < warmup else energy_mj(monitor, t0, t1, idle_mw)
        pred = int(prob >= threshold)
        if i >= warmup:
            records.append(_record(i, fp, lbl, e_mj, 0.0, prob, None, None, pred))
    return records


def run_resnet_only(
    frames, labels, resnet_model, gpu_device,
    threshold, monitor, idle_mw, warmup,
) -> list[dict]:
    records = []
    for i, (fp, lbl) in enumerate(zip(frames, labels)):
        if not is_valid(fp):
            continue
        t0 = time.perf_counter()
        prob = infer_resnet(fp, resnet_model, gpu_device)
        t1 = time.perf_counter()

        e_mj = 0.0 if i < warmup else energy_mj(monitor, t0, t1, idle_mw)
        pred = int(prob >= threshold)
        if i >= warmup:
            records.append(_record(i, fp, lbl, 0.0, e_mj, None, prob, None, pred))
    return records


def run_yolo_only(
    frames, labels, yolo_model,
    threshold, monitor, idle_mw, warmup, imgsz=224,
) -> list[dict]:
    records = []
    for i, (fp, lbl) in enumerate(zip(frames, labels)):
        if not is_valid(fp):
            continue
        t0 = time.perf_counter()
        prob = infer_yolo(fp, yolo_model, imgsz)
        t1 = time.perf_counter()

        e_mj = 0.0 if i < warmup else energy_mj(monitor, t0, t1, idle_mw)
        pred = int(prob >= threshold)
        if i >= warmup:
            records.append(_record(i, fp, lbl, 0.0, e_mj, None, None, prob, pred))
    return records


def run_ensemble_or(
    frames, labels, resnet_model, yolo_model, gpu_device,
    threshold, monitor, idle_mw, warmup, imgsz=224,
) -> list[dict]:
    records = []
    for i, (fp, lbl) in enumerate(zip(frames, labels)):
        if not is_valid(fp):
            continue
        t0 = time.perf_counter()
        r_prob = infer_resnet(fp, resnet_model, gpu_device)
        y_prob = infer_yolo(fp, yolo_model, imgsz)
        t1 = time.perf_counter()

        e_mj = 0.0 if i < warmup else energy_mj(monitor, t0, t1, idle_mw)
        pred = int(max(r_prob, y_prob) >= threshold)
        if i >= warmup:
            records.append(_record(i, fp, lbl, 0.0, e_mj, None, r_prob, y_prob, pred))
    return records


def run_gate(
    frames, labels,
    mob_model, mob_transform, cpu_device,
    resnet_model, yolo_model, gpu_device,
    n_frames, frame_gap, threshold, monitor, idle_mw,
    cache_seq_dir, warmup, from_start: bool, imgsz=224,
) -> list[dict]:
    """
    Gate pipeline (both modes).

    from_start=False (gate_from_window):
      Once MobileNet fires at frame G, ensemble runs on frames G, G+1, G+2, ...
      Frames before G: MobileNet energy only.

    from_start=True (gate_from_start):
      Once MobileNet fires at frame G, ensemble retroactively runs on ALL
      frames 0..G (including already-seen frames), then continues on G+1, ...
      This is more expensive but gives the ensemble its best chance.
    """
    records_by_idx: dict[int, dict] = {}

    # --- Phase 1: scan all frames with MobileNet ----------------------------
    gate_fired_at: int | None = None   # frame index where MobileNet first fires
    mob_probs: dict[int, float] = {}

    for i, (fp, lbl) in enumerate(zip(frames, labels)):
        lbp = get_lbp_image(frames, i, n_frames, frame_gap, cache_seq_dir)
        if lbp is None:
            continue

        t0 = time.perf_counter()
        prob = infer_mobilenet(lbp, mob_model, mob_transform, cpu_device)
        t1 = time.perf_counter()

        mob_probs[i] = prob
        e_mj = 0.0 if i < warmup else energy_mj(monitor, t0, t1, idle_mw)

        # Store MobileNet result; ensemble fields start at 0
        records_by_idx[i] = {
            "frame_idx":          i,
            "frame":              fp.name,
            "true_label":         lbl,
            "mob_prob":           round(prob, 4),
            "resnet_prob":        None,
            "yolo_prob":          None,
            "mob_energy_mj":      e_mj,
            "ensemble_energy_mj": 0.0,
        }

        if gate_fired_at is None and prob >= threshold:
            gate_fired_at = i

    # --- Phase 2: run ensemble on relevant frames ----------------------------
    if gate_fired_at is not None:
        ensemble_start = 0 if from_start else gate_fired_at
        ensemble_frames = [
            (i, frames[i], labels[i])
            for i in range(ensemble_start, len(frames))
            if i in records_by_idx and is_valid(frames[i])
        ]

        for i, fp, lbl in ensemble_frames:
            t0 = time.perf_counter()
            r_prob = infer_resnet(fp, resnet_model, gpu_device)
            y_prob = infer_yolo(fp, yolo_model, imgsz)
            t1 = time.perf_counter()

            e_mj = 0.0 if i < warmup else energy_mj(monitor, t0, t1, idle_mw)
            records_by_idx[i]["resnet_prob"]        = round(r_prob, 4)
            records_by_idx[i]["yolo_prob"]          = round(y_prob, 4)
            records_by_idx[i]["ensemble_energy_mj"] += e_mj

    # --- Assemble final records ----------------------------------------------
    out = []
    first_detection = None
    for i in sorted(records_by_idx):
        r   = records_by_idx[i]
        mob = r["mob_prob"]
        rp  = r["resnet_prob"]
        yp  = r["yolo_prob"]

        if rp is not None and yp is not None:
            # Ensemble ran: gate + ensemble both must agree
            pred = int(mob >= threshold and max(rp, yp) >= threshold)
        else:
            pred = 0   # ensemble didn't run → no detection this frame

        if i >= warmup:
            if pred == 1 and first_detection is None:
                first_detection = i
            out.append({
                "frame_idx":          r["frame_idx"],
                "frame":              r["frame"],
                "true_label":         r["true_label"],
                "prediction":         pred,
                "correct":            int(pred == r["true_label"]),
                "mob_prob":           mob,
                "resnet_prob":        rp,
                "yolo_prob":          yp,
                "mob_energy_mj":      round(r["mob_energy_mj"],      4),
                "ensemble_energy_mj": round(r["ensemble_energy_mj"], 4),
                "total_energy_mj":    round(r["mob_energy_mj"] + r["ensemble_energy_mj"], 4),
                "gate_fired":         int(gate_fired_at is not None),
                "ensemble_ran":       int(r["resnet_prob"] is not None),
            })
    return out


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def summarise_records(pipeline: str, sequence: str, records: list[dict],
                       gate_fired_at: int | None = None) -> dict:
    if not records:
        return {"pipeline": pipeline, "sequence": sequence, "n_frames": 0}

    total_mob_mj      = sum(r["mob_energy_mj"]      for r in records)
    total_ensemble_mj = sum(r["ensemble_energy_mj"]  for r in records)
    total_mj          = sum(r["total_energy_mj"]     for r in records)
    n                 = len(records)

    frames_mob_ran      = sum(1 for r in records if r.get("mob_prob")    is not None)
    frames_ensemble_ran = sum(1 for r in records if r.get("resnet_prob") is not None)

    # Detection
    detected      = any(r["prediction"] == 1 for r in records)
    first_det_idx = next((r["frame_idx"] for r in records if r["prediction"] == 1), None)

    tp = sum(1 for r in records if r["prediction"] == 1 and r["true_label"] == 1)
    tn = sum(1 for r in records if r["prediction"] == 0 and r["true_label"] == 0)
    fp = sum(1 for r in records if r["prediction"] == 1 and r["true_label"] == 0)
    fn = sum(1 for r in records if r["prediction"] == 0 and r["true_label"] == 1)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    return {
        "pipeline":                pipeline,
        "sequence":                sequence,
        "n_frames":                n,
        "frames_mobilenet_ran":    frames_mob_ran,
        "frames_ensemble_ran":     frames_ensemble_ran,
        "total_mob_energy_mj":     round(total_mob_mj,      2),
        "total_ensemble_energy_mj":round(total_ensemble_mj, 2),
        "total_energy_mj":         round(total_mj,          2),
        "total_energy_j":          round(total_mj / 1000,   4),
        "mean_energy_per_frame_mj":round(total_mj / n,      4),
        "detected":                detected,
        "first_detection_frame_idx": first_det_idx,
        "gate_fired_at_frame_idx": gate_fired_at,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr": round(tpr, 4) if tpr is not None else None,
        "fpr": round(fpr, 4) if fpr is not None else None,
    }


def print_summary_row(s: dict) -> None:
    det_str = (f"frame {s['first_detection_frame_idx']}"
               if s["detected"] else "not detected")
    gate_str = ""
    if s["gate_fired_at_frame_idx"] is not None:
        gate_str = f"  gate@{s['gate_fired_at_frame_idx']}"
    elif "frames_mobilenet_ran" in s and s["frames_mobilenet_ran"] > 0:
        gate_str = "  gate never fired"

    print(f"  {s['pipeline']:<22} | {s['sequence']:<10} | "
          f"E_total={s['total_energy_j']:6.3f}J  "
          f"mob={s['total_mob_energy_mj']:6.1f}mJ  "
          f"ens={s['total_ensemble_energy_mj']:6.1f}mJ  "
          f"mob_frames={s['frames_mobilenet_ran']:3d}  "
          f"ens_frames={s['frames_ensemble_ran']:3d}  "
          f"{det_str}{gate_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Sequences
    parser.add_argument("--no_smoke_dir",  required=True,
                        help="Directory of images for the no-smoke 3-hour sequence")
    parser.add_argument("--smoke_dir",     required=True,
                        help="Directory of images for the smoke 3-hour sequence")
    parser.add_argument("--smoke_start_idx", type=int, default=60,
                        help="0-based index of first smoke frame in smoke_dir "
                             "(frames before this index are treated as no-smoke; "
                             "default: 60 = 1 hour in)")
    parser.add_argument("--n_images", type=int, default=180,
                        help="Max frames to use from each sequence (default: 180)")

    # Checkpoints
    parser.add_argument("--mobilenet_ckpt", default=None)
    parser.add_argument("--resnet_ckpt",    default=None)
    parser.add_argument("--yolo_ckpt",      default=None)

    # MobileNet params
    parser.add_argument("--n_frames",   type=int, default=2)
    parser.add_argument("--frame_gap",  type=int, default=1)
    parser.add_argument("--cache_root", default=None,
                        help="Gap-specific LBP cache root (optional, speeds up MobileNet)")

    # Inference params
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--imgsz",    type=int,   default=224)
    parser.add_argument("--warmup",   type=int,   default=3,
                        help="Frames used for GPU/model warmup (excluded from energy stats)")

    # Power monitoring
    parser.add_argument("--tegrastats_interval_ms", type=int, default=100)
    parser.add_argument("--idle_duration_s",        type=float, default=5.0)

    # Output
    parser.add_argument("--out_dir", default="energy_results/pipelines")

    args = parser.parse_args()

    if not any([args.mobilenet_ckpt, args.resnet_ckpt, args.yolo_ckpt]):
        parser.error("Provide at least one of --mobilenet_ckpt, --resnet_ckpt, --yolo_ckpt")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_device.type == "cpu":
        print("WARNING: CUDA not available — GPU models will run on CPU.")
    print(f"CPU device : {cpu_device}  (MobileNet)")
    print(f"GPU device : {gpu_device}  (ResNet34, YOLOv8)")

    # --- Load sequences -------------------------------------------------------
    no_smoke_frames = load_frames(Path(args.no_smoke_dir), args.n_images)
    smoke_frames    = load_frames(Path(args.smoke_dir),    args.n_images)
    no_smoke_labels = [0] * len(no_smoke_frames)
    smoke_labels    = [
        1 if i >= args.smoke_start_idx else 0
        for i in range(len(smoke_frames))
    ]
    print(f"\nNo-smoke sequence : {len(no_smoke_frames)} frames  (all no-smoke)")
    print(f"Smoke sequence    : {len(smoke_frames)} frames  "
          f"({smoke_labels.count(0)} no-smoke, {smoke_labels.count(1)} smoke, "
          f"smoke starts at idx {args.smoke_start_idx})")

    # --- Load models ----------------------------------------------------------
    print("\nLoading models ...")
    mob_model, mob_transform = None, None
    resnet_model             = None
    yolo_model               = None

    if args.mobilenet_ckpt:
        mob_model, mob_transform = load_mobilenet(args.mobilenet_ckpt, cpu_device)
    if args.resnet_ckpt:
        resnet_model = load_resnet(args.resnet_ckpt, gpu_device)
    if args.yolo_ckpt:
        yolo_model = load_yolo(args.yolo_ckpt)

    # --- Cache dirs -----------------------------------------------------------
    cache_no_smoke = None
    cache_smoke    = None
    if args.cache_root and mob_model:
        cr = Path(args.cache_root)
        # Try to find matching subdirectory; fall back to root
        ns_name = Path(args.no_smoke_dir).name
        sm_name = Path(args.smoke_dir).name
        cache_no_smoke = cr / ns_name if (cr / ns_name).is_dir() else cr
        cache_smoke    = cr / sm_name if (cr / sm_name).is_dir() else cr

    # --- Decide which pipelines to run ----------------------------------------
    pipelines = []
    if mob_model:
        pipelines.append("mobilenet")
    if resnet_model:
        pipelines.append("resnet34")
    if yolo_model:
        pipelines.append("yolov8")
    if resnet_model and yolo_model:
        pipelines.append("ensemble_OR")
    if mob_model and resnet_model and yolo_model:
        pipelines += ["gate_from_window", "gate_from_start"]

    print(f"\nPipelines to evaluate: {pipelines}")

    # --- Power monitor --------------------------------------------------------
    monitor = make_monitor(args.tegrastats_interval_ms)
    monitor.start()
    idle_mw = measure_idle_power(monitor, duration_s=args.idle_duration_s)

    # GPU warmup (run a few dummy frames through GPU models before recording)
    if (resnet_model or yolo_model) and gpu_device.type == "cuda":
        print(f"\nWarming up GPU ({args.warmup} frames) ...")
        warmup_frames_list = smoke_frames[:args.warmup]
        for fp in warmup_frames_list:
            if not is_valid(fp):
                continue
            if resnet_model:
                infer_resnet(fp, resnet_model, gpu_device)
            if yolo_model:
                infer_yolo(fp, yolo_model, args.imgsz)
        print("  GPU warmup done.")

    # --- Run each pipeline on both sequences ----------------------------------
    all_summaries = []
    all_records   = {}   # pipeline -> {sequence -> list[dict]}

    sequences = [
        ("no_smoke", no_smoke_frames, no_smoke_labels, cache_no_smoke),
        ("smoke",    smoke_frames,    smoke_labels,    cache_smoke),
    ]

    for pipeline in pipelines:
        print(f"\n{'='*70}")
        print(f"  Pipeline: {pipeline.upper()}")
        print(f"{'='*70}")
        all_records[pipeline] = {}

        for seq_name, frames, labels, cache_dir in sequences:
            print(f"\n  -- Sequence: {seq_name}  ({len(frames)} frames) --")

            gate_fired_at = None

            if pipeline == "mobilenet":
                recs = run_mobilenet_only(
                    frames, labels, mob_model, mob_transform, cpu_device,
                    args.n_frames, args.frame_gap, args.threshold,
                    monitor, idle_mw, cache_dir, args.warmup,
                )

            elif pipeline == "resnet34":
                recs = run_resnet_only(
                    frames, labels, resnet_model, gpu_device,
                    args.threshold, monitor, idle_mw, args.warmup,
                )

            elif pipeline == "yolov8":
                recs = run_yolo_only(
                    frames, labels, yolo_model,
                    args.threshold, monitor, idle_mw, args.warmup, args.imgsz,
                )

            elif pipeline == "ensemble_OR":
                recs = run_ensemble_or(
                    frames, labels, resnet_model, yolo_model, gpu_device,
                    args.threshold, monitor, idle_mw, args.warmup, args.imgsz,
                )

            elif pipeline in ("gate_from_window", "gate_from_start"):
                from_start = (pipeline == "gate_from_start")
                recs = run_gate(
                    frames, labels,
                    mob_model, mob_transform, cpu_device,
                    resnet_model, yolo_model, gpu_device,
                    args.n_frames, args.frame_gap, args.threshold,
                    monitor, idle_mw, cache_dir, args.warmup, from_start, args.imgsz,
                )
                fired_recs = [r for r in recs if r.get("gate_fired")]
                if fired_recs:
                    gate_fired_at = fired_recs[0]["frame_idx"]
            else:
                recs = []

            all_records[pipeline][seq_name] = recs
            s = summarise_records(pipeline, seq_name, recs, gate_fired_at)
            all_summaries.append(s)
            print_summary_row(s)

    monitor.stop()

    # --- Print final table ----------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  ENERGY SUMMARY  (idle baseline = {idle_mw:.1f} mW | threshold = {args.threshold})")
    print(f"{'='*100}")
    hdr = (f"{'Pipeline':<22}  {'Sequence':<10}  "
           f"{'E_total(J)':>10}  {'E_mob(mJ)':>9}  {'E_ens(mJ)':>9}  "
           f"{'#mob':>5}  {'#ens':>5}  {'Detected':>8}  {'TPR':>6}  {'FPR':>6}")
    print(hdr)
    print("-" * 100)
    for s in all_summaries:
        det_str = f"@f{s['first_detection_frame_idx']}" if s["detected"] else "   NO"
        tpr_str = f"{s['tpr']:.3f}" if s.get("tpr") is not None else "  N/A"
        fpr_str = f"{s['fpr']:.3f}" if s.get("fpr") is not None else "  N/A"
        print(
            f"{s['pipeline']:<22}  {s['sequence']:<10}  "
            f"{s['total_energy_j']:>10.4f}  "
            f"{s['total_mob_energy_mj']:>9.1f}  "
            f"{s['total_ensemble_energy_mj']:>9.1f}  "
            f"{s['frames_mobilenet_ran']:>5}  "
            f"{s['frames_ensemble_ran']:>5}  "
            f"{det_str:>8}  {tpr_str:>6}  {fpr_str:>6}"
        )
    print(f"{'='*100}")

    # --- Save outputs ---------------------------------------------------------
    # Summary JSON
    summary_path = out_dir / "energy_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"idle_mw": idle_mw, "threshold": args.threshold,
                   "results": all_summaries}, f, indent=2)
    print(f"\nSummary saved  → {summary_path}")

    # Per-frame CSVs
    import csv
    for pipeline, seq_dict in all_records.items():
        for seq_name, recs in seq_dict.items():
            if not recs:
                continue
            csv_path = out_dir / f"{pipeline}_{seq_name}_per_frame.csv"
            fieldnames = list(recs[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(recs)
    print(f"Per-frame CSVs → {out_dir}/")


if __name__ == "__main__":
    main()
