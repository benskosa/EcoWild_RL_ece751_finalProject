#!/usr/bin/env python3
"""
energy_eval.py

Measures inference latency and energy consumption of EcoWild baseline
smoke-detection models on a Jetson Orin Nano.

Three configurations are evaluated back-to-back:
  1. ResNet34 only
  2. YOLOv8 only
  3. Ensemble  (ResNet34 OR YOLOv8, "whole" mode — EcoWild baseline)

Power is sampled via tegrastats at --tegrastats_interval_ms (default 100 ms).
jtop (jetson-stats package) is used instead if it is installed, since its
Python API avoids subprocess parsing overhead.

Before inference begins, the script measures idle board power (models loaded,
GPU warmed up, no inference running) for --idle_duration_s seconds.  This
baseline is subtracted from each inference power reading so that reported
energy reflects only the incremental cost of running the model.

For each inference the script records:
  - wall-clock latency (ms)
  - mean VDD_IN power during the inference window (mW)          [raw]
  - net power = raw power − idle power (mW)                     [net]
  - energy estimate = net_power × latency (mJ)                  [net]

Accuracy (TPR / FPR) is also reported against the true labels derived from
--smoke_start.

Dataset layout (under --dataset_dir, sorted lexicographically):
  img_00001.jpg   ← no-smoke  (1 … smoke_start-1)
  ...
  img_00041.jpg   ← smoke     (smoke_start … end)
  ...

Usage:
    cd smokeDetection_baseline_ecoWild/
    python energy_eval.py \\
        --dataset_dir  Dataset/ \\
        --resnet_path  Model/Pytorch/best_resnet34_model_epoch_3.pth \\
        --yolo_path    Model/Pytorch/yolov8l_cls_whole_golden_best.pt \\
        --smoke_start  41 \\
        --warmup       3  \\
        --out_dir      energy_results/
"""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Decision thresholds (match yolo_resnet_assemble_4_edge.py "whole" mode)
# ---------------------------------------------------------------------------
RESNET_SMOKE_THRESH = 0.2    # sigmoid(logit[:,1]) >= threshold → smoke
YOLO_SMOKE_THRESH   = 0.25   # probs[1] >= threshold → smoke


# ---------------------------------------------------------------------------
# Image preprocessing  (same as original "whole" mode)
#
# Original pipeline:
#   1. cv2.resize(img, (2016, 1536))          – width=2016, height=1536
#   2. img[1536 - 1120:]                       – crop bottom 1120 rows
#   3. ResNet: Resize(224,224) + ImageNet norm
#   4. YOLO:   Resize(640,640), passed as PIL to ultralytics
# ---------------------------------------------------------------------------
_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(img_path: Path) -> tuple[torch.Tensor, Image.Image]:
    """
    Load and preprocess one image for both models.

    Returns
    -------
    resnet_tensor : (1, 3, 224, 224) float32 tensor, ImageNet-normalised
    yolo_pil      : 640×640 PIL Image ready for ultralytics YOLO
    """
    img_np = cv2.imread(str(img_path))               # BGR
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (2016, 1536))        # W×H for cv2
    img_np = img_np[1536 - 1120:, :]                 # crop → (1120, 2016, 3)
    pil_full = Image.fromarray(img_np)

    resnet_tensor = _resnet_transform(pil_full).unsqueeze(0)   # (1,3,224,224)
    yolo_pil      = pil_full.resize((640, 640), Image.BILINEAR)

    return resnet_tensor, yolo_pil


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_resnet(weights_path: str, device: torch.device) -> nn.Module:
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    print(f"  ResNet34 loaded from {weights_path}  (device={device})")
    return model


def load_yolo(weights_path: str) -> YOLO:
    model = YOLO(weights_path)
    print(f"  YOLOv8  loaded from {weights_path}")
    return model


# ---------------------------------------------------------------------------
# Power monitoring
# ---------------------------------------------------------------------------

class _BasePowerMonitor:
    """Common interface for power monitors."""

    def start(self): ...
    def stop(self): ...

    def mean_power_mw(self, t_start: float, t_end: float) -> float | None:
        raise NotImplementedError


class TegrastatsMonitor(_BasePowerMonitor):
    """
    Spawns `tegrastats` as a subprocess and accumulates
    (perf_counter_timestamp, VDD_IN_mW) readings in a background thread.
    Falls back to the first mW value in the line if VDD_IN is not present
    (e.g., different JetPack versions that use 'TOT' or 'SYS 5V').
    """

    # Primary: VDD_IN rail (total input power, Orin family JetPack 5/6)
    _VDD_IN_RE  = re.compile(r'VDD_IN\s+(\d+)mW')
    # Fallback: first mW value anywhere in the line
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
        if m:
            return float(m.group(1))
        return None

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
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(0.5)   # let tegrastats emit a few baseline readings
        print(f"  [TegrastatsMonitor] started  (interval={self.interval_ms}ms)")

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2)
        print(f"  [TegrastatsMonitor] stopped  ({len(self._readings)} readings)")

    def mean_power_mw(self, t_start: float, t_end: float) -> float | None:
        """
        Mean VDD_IN power (mW) across readings in [t_start, t_end].
        If no reading falls in the window (inference faster than the polling
        interval), returns the reading closest to the midpoint instead.
        """
        with self._lock:
            window = [p for t, p in self._readings if t_start <= t <= t_end]
            if window:
                return float(np.mean(window))
            # fallback: nearest reading to the inference midpoint
            mid = (t_start + t_end) / 2.0
            if self._readings:
                _, nearest = min(self._readings, key=lambda x: abs(x[0] - mid))
                return float(nearest)
        return None


class JtopMonitor(_BasePowerMonitor):
    """
    Uses the jtop Python library (jetson-stats) for power readings.
    Polled at --tegrastats_interval_ms; requires `pip install jetson-stats`.
    Supports both the jtop 3.x API (stats['Power TOT']) and
    the jtop 4.x API (power[1]['tot']['cur']).
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_s  = interval_ms / 1000.0
        self._readings: list[tuple[float, float]] = []
        self._lock   = threading.Lock()
        self._thread = None
        self._stop   = threading.Event()
        self._jetson = None

    def _reader(self):
        from jtop import jtop  # imported here so the class is loadable without jtop
        with jtop() as jetson:
            self._jetson = jetson
            while not self._stop.is_set():
                t = time.perf_counter()
                try:
                    # jtop 4.x
                    p = jetson.power[1]['tot']['cur']
                except (KeyError, TypeError, AttributeError):
                    try:
                        # jtop 3.x
                        p = jetson.stats['Power TOT']
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
    """Try jtop first; fall back to tegrastats."""
    try:
        import jtop  # noqa: F401
        print("Power monitor: jtop (jetson-stats)")
        return JtopMonitor(interval_ms=interval_ms)
    except ImportError:
        pass
    print("Power monitor: tegrastats")
    return TegrastatsMonitor(interval_ms=interval_ms)


# ---------------------------------------------------------------------------
# Idle power measurement
# ---------------------------------------------------------------------------

def measure_idle_power(
    monitor: _BasePowerMonitor,
    duration_s: float = 5.0,
) -> float:
    """
    Measure baseline (idle) board power with models loaded but no inference
    running.  Called once after model loading and GPU warmup, before the
    evaluation loops.

    The monitor must already be started.  We simply sleep for `duration_s`
    and average all power readings collected during that window.

    Returns
    -------
    idle_power_mW : float
        Mean VDD_IN power (mW) during the idle window.
    """
    print(f"\nMeasuring idle power  ({duration_s:.0f}s, no inference running) ...")
    t_start = time.perf_counter()
    time.sleep(duration_s)
    t_end = time.perf_counter()

    idle_mW = monitor.mean_power_mw(t_start, t_end)
    if idle_mW is None:
        print("  WARNING: no power readings received during idle window; "
              "idle_power_mW will be set to 0.")
        idle_mW = 0.0

    print(f"  Idle power: {idle_mW:.1f} mW  "
          f"(averaged over {t_end - t_start:.1f}s)")
    return idle_mW


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

def infer_resnet(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[int, float]:
    """Return (smoke_label, smoke_probability)."""
    with torch.no_grad():
        logits = model(tensor.to(device))       # (1, 2)
        prob   = torch.sigmoid(logits[:, 1]).item()
    return int(prob >= RESNET_SMOKE_THRESH), prob


def infer_yolo(model: YOLO, pil_img: Image.Image) -> tuple[int, float]:
    """Return (smoke_label, smoke_probability)."""
    results = model(pil_img, verbose=False)
    prob    = results[0].probs.data.tolist()[1]
    return int(prob >= YOLO_SMOKE_THRESH), prob


# ---------------------------------------------------------------------------
# Per-configuration evaluation loop
# ---------------------------------------------------------------------------

def evaluate_config(
    config_name: str,
    images: list[Path],
    true_labels: list[int],
    resnet_model: nn.Module | None,
    yolo_model: YOLO | None,
    device: torch.device,
    monitor: _BasePowerMonitor,
    idle_power_mw: float = 0.0,
    warmup: int = 3,
) -> pd.DataFrame:
    """
    Run config_name ('resnet' | 'yolo' | 'ensemble') on all images.

    The first `warmup` images are processed to warm up the GPU/model cache
    but are excluded from the reported statistics.

    idle_power_mw is subtracted from each raw power reading before computing
    net_energy_mJ.  Raw power is still stored in the CSV for reference.

    Returns a DataFrame with one row per image.
    """
    print(f"\n{'='*64}")
    print(f"  Config : {config_name.upper()}  |  warmup={warmup}  |  N={len(images)}")
    print(f"{'='*64}")

    records = []

    for i, (img_path, true_label) in enumerate(zip(images, true_labels)):
        is_warmup = i < warmup

        # --- Preprocessing --------------------------------------------------
        t0 = time.perf_counter()
        resnet_tensor, yolo_pil = preprocess_image(img_path)
        t_preproc_end = time.perf_counter()
        preproc_ms = (t_preproc_end - t0) * 1e3

        # --- Inference  (power window starts here) --------------------------
        t_inf_start = time.perf_counter()

        if config_name == "resnet":
            label, prob_r = infer_resnet(resnet_model, resnet_tensor, device)
            prob_y = None

        elif config_name == "yolo":
            label, prob_y = infer_yolo(yolo_model, yolo_pil)
            prob_r = None

        else:  # ensemble: OR rule
            pred_r, prob_r = infer_resnet(resnet_model, resnet_tensor, device)
            pred_y, prob_y = infer_yolo(yolo_model, yolo_pil)
            label = int(pred_r == 1 or pred_y == 1)

        t_inf_end = time.perf_counter()

        # --- Energy ----------------------------------------------------------
        inf_s     = t_inf_end - t_inf_start
        inf_ms    = inf_s * 1e3
        total_ms  = (t_inf_end - t0) * 1e3
        power_mW  = monitor.mean_power_mw(t_inf_start, t_inf_end)

        # Net power: clamp to 0 so noise can't produce negative energy
        net_power_mW  = max(0.0, power_mW - idle_power_mw) if power_mW is not None else None
        net_energy_mJ = (net_power_mW * inf_s) if net_power_mW is not None else None

        # --- Logging ---------------------------------------------------------
        if is_warmup:
            print(f"  [warmup {i+1}/{warmup}] {img_path.name:<30} "
                  f"inf={inf_ms:6.0f}ms  "
                  f"raw={power_mW or 0:5.0f}mW  "
                  f"net={net_power_mW or 0:5.0f}mW  (excluded)")
        else:
            n_measured = i - warmup + 1
            n_total    = len(images) - warmup
            print(f"  [{n_measured:3d}/{n_total}] {img_path.name:<30} "
                  f"true={true_label}  pred={label}  "
                  f"inf={inf_ms:6.0f}ms  "
                  f"raw={power_mW or 0:5.0f}mW  "
                  f"net={net_power_mW or 0:5.0f}mW  "
                  f"net_energy={net_energy_mJ or 0:6.2f}mJ")

        records.append({
            "config":         config_name,
            "image":          img_path.name,
            "true_label":     true_label,
            "prediction":     label,
            "correct":        int(label == true_label),
            "prob_resnet":    round(prob_r, 4) if prob_r is not None else None,
            "prob_yolo":      round(prob_y, 4) if prob_y is not None else None,
            "preproc_ms":     round(preproc_ms, 2),
            "inf_ms":         round(inf_ms, 2),
            "total_ms":       round(total_ms, 2),
            "power_mW":       round(power_mW, 1)     if power_mW     is not None else None,
            "net_power_mW":   round(net_power_mW, 1) if net_power_mW is not None else None,
            "net_energy_mJ":  round(net_energy_mJ, 3) if net_energy_mJ is not None else None,
            "warmup":         is_warmup,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, config_name: str) -> dict:
    measured = df[~df["warmup"]].copy()

    def _stat(col: str) -> dict:
        vals = measured[col].dropna()
        return {
            "mean":  round(float(vals.mean()),  3),
            "std":   round(float(vals.std()),   3),
            "min":   round(float(vals.min()),   3),
            "max":   round(float(vals.max()),   3),
            "total": round(float(vals.sum()),   3),
        }

    tp = int(((measured["prediction"] == 1) & (measured["true_label"] == 1)).sum())
    tn = int(((measured["prediction"] == 0) & (measured["true_label"] == 0)).sum())
    fp = int(((measured["prediction"] == 1) & (measured["true_label"] == 0)).sum())
    fn = int(((measured["prediction"] == 0) & (measured["true_label"] == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    return {
        "config":         config_name,
        "n_measured":     len(measured),
        "n_smoke":        int((measured["true_label"] == 1).sum()),
        "n_no_smoke":     int((measured["true_label"] == 0).sum()),
        "accuracy":       round(float(measured["correct"].mean()), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr":            round(tpr, 4) if tpr is not None else None,
        "fpr":            round(fpr, 4) if fpr is not None else None,
        "latency_ms":     _stat("inf_ms"),
        "total_ms":       _stat("total_ms"),
        "preproc_ms":     _stat("preproc_ms"),
        "raw_power_mW":   _stat("power_mW"),
        "net_power_mW":   _stat("net_power_mW"),
        "net_energy_mJ":  _stat("net_energy_mJ"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Energy profiling for EcoWild baseline models on Jetson Orin Nano",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Directory containing the 80 test images (sorted by filename)")
    parser.add_argument("--resnet_path",
                        default="Model/Pytorch/best_resnet34_model_epoch_3.pth",
                        help="Path to ResNet34 .pth weights")
    parser.add_argument("--yolo_path",
                        default="Model/Pytorch/yolov8l_cls_whole_golden_best.pt",
                        help="Path to YOLOv8 .pt weights")
    parser.add_argument("--smoke_start", type=int, default=41,
                        help="1-indexed position where smoke begins (images before this are no-smoke)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Images to use for GPU warmup per config (not included in stats)")
    parser.add_argument("--out_dir", default="energy_results",
                        help="Directory for output CSV and JSON")
    parser.add_argument("--tegrastats_interval_ms", type=int, default=100,
                        help="Power sampling interval in ms")
    parser.add_argument("--idle_duration_s", type=float, default=5.0,
                        help="Seconds to sample idle power after model load, before inference")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. 'cuda', 'cpu'. Auto-detected if omitted.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Torch device  : {device}")

    # --- Dataset ------------------------------------------------------------
    img_dir  = Path(args.dataset_dir)
    all_imgs = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg")) +
        list(img_dir.glob("*.png"))
    )
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # Build ground-truth labels (0 = no-smoke, 1 = smoke)
    true_labels = [
        1 if (i + 1) >= args.smoke_start else 0
        for i in range(len(all_imgs))
    ]
    n_no_smoke = true_labels.count(0)
    n_smoke    = true_labels.count(1)
    print(f"Dataset       : {len(all_imgs)} images  "
          f"({n_no_smoke} no-smoke, {n_smoke} smoke, "
          f"smoke starts at position {args.smoke_start})")

    # --- Models -------------------------------------------------------------
    print("\nLoading models ...")
    resnet_model = load_resnet(args.resnet_path, device)
    yolo_model   = load_yolo(args.yolo_path)

    # --- Power monitor ------------------------------------------------------
    monitor = make_monitor(args.tegrastats_interval_ms)
    monitor.start()

    # --- Idle power baseline ------------------------------------------------
    # Measured with models resident in GPU memory but no inference running.
    # Each config's per-image net_power_mW = raw_power_mW - idle_power_mW.
    idle_power_mw = measure_idle_power(monitor, duration_s=args.idle_duration_s)

    # --- Run all three configurations ---------------------------------------
    all_dfs       = {}
    all_summaries = []

    for config in ["resnet", "yolo", "ensemble"]:
        df = evaluate_config(
            config_name   = config,
            images        = all_imgs,
            true_labels   = true_labels,
            resnet_model  = resnet_model if config in ("resnet", "ensemble") else None,
            yolo_model    = yolo_model   if config in ("yolo",   "ensemble") else None,
            device        = device,
            monitor       = monitor,
            idle_power_mw = idle_power_mw,
            warmup        = args.warmup,
        )
        all_dfs[config] = df
        all_summaries.append(summarise(df, config))

    monitor.stop()

    # --- Save outputs -------------------------------------------------------
    combined_df = pd.concat(all_dfs.values(), ignore_index=True)
    csv_path    = out_dir / "per_image_results.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\nPer-image CSV saved → {csv_path}")

    summary_path = out_dir / "energy_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Summary JSON  saved → {summary_path}")

    # --- Print summary table ------------------------------------------------
    w = 78
    print("\n" + "=" * w)
    print(f"  ENERGY SUMMARY  (warmup images excluded | "
          f"idle baseline = {idle_power_mw:.1f} mW)")
    print("=" * w)
    hdr = (f"{'Config':<12}{'Acc':>7}{'TPR':>7}{'FPR':>7}  "
           f"{'Lat(ms)':>13}  {'RawPwr(mW)':>13}  {'NetPwr(mW)':>13}  {'NetEnergy(mJ)':>14}")
    print(hdr)
    print("-" * w)

    for s in all_summaries:
        lat  = s["latency_ms"]
        rpwr = s["raw_power_mW"]
        npwr = s["net_power_mW"]
        neng = s["net_energy_mJ"]
        tpr_str = f"{s['tpr']:.4f}" if s["tpr"] is not None else "  N/A "
        fpr_str = f"{s['fpr']:.4f}" if s["fpr"] is not None else "  N/A "
        print(
            f"{s['config']:<12}"
            f"{s['accuracy']:>7.4f}"
            f"{tpr_str:>7}"
            f"{fpr_str:>7}  "
            f"{lat['mean']:>5.0f}±{lat['std']:<6.0f}  "
            f"{rpwr['mean']:>6.0f}±{rpwr['std']:<5.0f}  "
            f"{npwr['mean']:>6.0f}±{npwr['std']:<5.0f}  "
            f"{neng['mean']:>7.2f}±{neng['std']:<5.2f}"
        )

    print("=" * w)
    print(f"\nNotes:")
    print(f"  - Idle baseline ({idle_power_mw:.1f} mW) measured for "
          f"{args.idle_duration_s:.0f}s after model load, before inference")
    print(f"  - net_power_mW  = raw_power_mW − idle_power_mW  (clamped to 0)")
    print(f"  - net_energy_mJ = net_power_mW × inference_latency_s  (per image)")
    print(f"  - Power rail: VDD_IN (total board power), "
          f"sampled every {args.tegrastats_interval_ms}ms")
    print(f"  - Latency = model inference only; "
          f"preprocessing time is in the CSV as preproc_ms")
    print(f"  - Warmup: {args.warmup} image(s) per config (GPU/model cache warm-up)")


if __name__ == "__main__":
    main()
