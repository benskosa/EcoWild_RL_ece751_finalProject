"""
pipeline_classifier_eval.py
---------------------------
Sequence-level classification metrics (Accuracy, TPR, FPR) for each
detection pipeline on the test dataset.

Each sequence is treated as a single binary classification instance:
  smoke sequence   -- positive class
    detected (any frame/window >= threshold) -> TP
    not detected                             -> FN
  no_smoke sequence -- negative class
    any frame/window fires                  -> FP
    no frame fires                          -> TN

Pipelines evaluated (depending on which checkpoints are supplied):
  - LBP + MobileNet standalone       (--mobilenet_ckpt)
  - ResNet34 standalone              (--resnet_ckpt)
  - YOLOv8 standalone                (--yolo_ckpt)
  - ResNet34 + YOLOv8 ensemble OR    (--resnet_ckpt + --yolo_ckpt)
  - LBP + MobileNet gate -> ensemble (all three checkpoints)

Usage
-----
    # All three final-comparison pipelines:
    python pipeline_classifier_eval.py \\
        --data_root   smokeDetection_baseline_ecoWild/Dataset/test \\
        --mobilenet_ckpt smokeDetection_ourExperiments/sweep_results/checkpoints/nf2_gap16/nf2_gap16_best_acc.pt \\
        --resnet_ckpt smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt \\
        --yolo_ckpt   smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt \\
        --n_frames 2 --frame_gap 16 \\
        --cache_root smokeDetection_baseline_ecoWild/lbp_cache/gap_16 \\
        --threshold 0.5 \\
        --out_dir pipeline_classifier_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Reuse scoring helpers from sequence_eval.py
import sequence_eval as _se

# Allow imports from smokeDetection_ourExperiments (model, feature_extraction)
_exp_dir = Path(__file__).parent / "smokeDetection_ourExperiments"
if str(_exp_dir) not in sys.path:
    sys.path.insert(0, str(_exp_dir))


# ---------------------------------------------------------------------------
# Load frames for a no_smoke sequence (no ignition offset needed)
# ---------------------------------------------------------------------------

def load_no_smoke_frames(seq_dir: Path) -> list[Path]:
    """Return all valid jpg frames in a no_smoke sequence directory."""
    frames = []
    for f in sorted(seq_dir.glob("*.jpg")):
        if _se.is_valid_image(f):
            frames.append(f)
    return frames


# ---------------------------------------------------------------------------
# Check if any score in a list crosses the threshold
# ---------------------------------------------------------------------------

def any_detection(scores: list, threshold: float) -> bool:
    return _se.first_detection(scores, threshold) is not None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "tpr": round(tpr, 4), "fpr": round(fpr, 4), "accuracy": round(acc, 4)}


def print_metrics(name: str, m: dict) -> None:
    print(f"\n  [{name}]")
    print(f"    TP={m['tp']}  FN={m['fn']}  FP={m['fp']}  TN={m['tn']}")
    print(f"    TPR      : {m['tpr']:.3f}  ({m['tp']}/{m['tp']+m['fn']} smoke sequences detected)")
    print(f"    FPR      : {m['fpr']:.3f}  ({m['fp']}/{m['fp']+m['tn']} no_smoke sequences triggered)")
    print(f"    Accuracy : {m['accuracy']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",       required=True,
                        help="Dataset split root (contains smoke/ and no_smoke/)")
    parser.add_argument("--mobilenet_ckpt",  default=None)
    parser.add_argument("--resnet_ckpt",     default=None)
    parser.add_argument("--yolo_ckpt",       default=None)
    parser.add_argument("--n_frames",        type=int, default=2)
    parser.add_argument("--frame_gap",       type=int, default=1)
    parser.add_argument("--cache_root",      default=None,
                        help="Gap-specific LBP cache root (e.g. lbp_cache/gap_1)")
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--imgsz",           type=int, default=224)
    parser.add_argument("--out_dir",         default="pipeline_classifier_results")
    parser.add_argument("--device",          default=None)
    parser.add_argument("--gate_from_start", action="store_true")
    args = parser.parse_args()

    if not any([args.mobilenet_ckpt, args.resnet_ckpt, args.yolo_ckpt]):
        parser.error("Provide at least one checkpoint.")

    device     = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root  = Path(args.data_root).resolve()
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).resolve() if args.cache_root else None

    smoke_dirs    = sorted(d for d in (data_root / "smoke").iterdir()    if d.is_dir())
    no_smoke_dirs = sorted(d for d in (data_root / "no_smoke").iterdir() if d.is_dir())

    print(f"\nData root          : {data_root}")
    print(f"Smoke sequences    : {len(smoke_dirs)}")
    print(f"No-smoke sequences : {len(no_smoke_dirs)}")
    print(f"Threshold          : {args.threshold}")
    print(f"Device             : {device}")

    # --- Load models ----------------------------------------------------------
    mob_model, resnet_model, yolo_model = None, None, None

    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.mobilenet_ckpt:
        from model import build_model, get_transforms
        ckpt      = torch.load(args.mobilenet_ckpt, map_location=device)
        variant   = ckpt.get("variant", "v3_small")
        mob_model = build_model(variant=variant, pretrained=False)
        mob_model.load_state_dict(ckpt["state_dict"])
        mob_model.to(device).eval()
        mob_transform = get_transforms(train=False)
        print(f"MobileNet loaded   : {args.mobilenet_ckpt}  (variant={variant})")

    if args.resnet_ckpt:
        ckpt = torch.load(args.resnet_ckpt, map_location=device)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
        else:
            state_dict = ckpt
        resnet_model = models.resnet34()
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
        resnet_model.load_state_dict(state_dict)
        resnet_model.to(device).eval()
        print(f"ResNet34 loaded    : {args.resnet_ckpt}")

    if args.yolo_ckpt:
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_ckpt)
        print(f"YOLOv8 loaded      : {args.yolo_ckpt}")

    gate_label = "gate_from_start" if args.gate_from_start else "gate_from_window"
    pipeline_names = (
        (["mobilenet"]   if mob_model                          else []) +
        (["resnet34"]    if resnet_model                       else []) +
        (["yolov8"]      if yolo_model                         else []) +
        (["ensemble_OR"] if resnet_model and yolo_model        else []) +
        ([gate_label]    if mob_model and resnet_model and yolo_model else [])
    )

    # counters: {pipeline: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}}
    counts = {p: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for p in pipeline_names}

    # --- Score a sequence and return per-pipeline detection booleans ----------
    def score_seq(frames: list[Path], post_start: int,
                  seq_dir: Path) -> dict[str, bool]:
        cache_seq_dir = None
        if cache_root is not None:
            rel = seq_dir.relative_to(data_root.parent)
            cache_seq_dir = cache_root / rel

        mob_scores, resnet_scores, yolo_scores = [], [], []

        if mob_model:
            mob_scores = _se.get_mobilenet_scores(
                frames, post_start, mob_model, cache_seq_dir,
                args.n_frames, args.frame_gap, mob_transform, device,
            )
        if resnet_model:
            resnet_scores = _se.get_resnet_scores(
                frames, post_start, resnet_model, resnet_transform, device,
            )
        if yolo_model:
            yolo_scores = _se.get_yolo_scores(frames, post_start, yolo_model, args.imgsz)

        resnet_by_offset = {o: p for o, p in resnet_scores}
        yolo_by_offset   = {o: p for o, p in yolo_scores}

        detected: dict[str, bool] = {}

        if mob_model:
            detected["mobilenet"] = any_detection(mob_scores, args.threshold)
        if resnet_model:
            detected["resnet34"] = any_detection(resnet_scores, args.threshold)
        if yolo_model:
            detected["yolov8"] = any_detection(yolo_scores, args.threshold)
        if resnet_model and yolo_model:
            ensemble_scores = [
                (offset, max(resnet_by_offset.get(offset, 0.0),
                             yolo_by_offset.get(offset, 0.0)))
                for offset in sorted(resnet_by_offset)
            ]
            detected["ensemble_OR"] = any_detection(ensemble_scores, args.threshold)
        if mob_model and resnet_model and yolo_model:
            _, ens_det = _se.gate_first_detection(
                mob_scores, resnet_by_offset, yolo_by_offset,
                args.threshold, args.threshold,
                from_start=args.gate_from_start,
            )
            detected[gate_label] = ens_det is not None

        return detected

    # --- Smoke sequences (positive class) ------------------------------------
    for seq_dir in tqdm(smoke_dirs, desc="Smoke seqs", unit="seq"):
        frames, post_start = _se.load_sequence(seq_dir)
        if post_start >= len(frames):
            continue  # no post-ignition frames — can't evaluate
        detected = score_seq(frames, post_start, seq_dir)
        for p, det in detected.items():
            if det:
                counts[p]["tp"] += 1
            else:
                counts[p]["fn"] += 1

    # --- No-smoke sequences (negative class) ---------------------------------
    for seq_dir in tqdm(no_smoke_dirs, desc="No-smoke seqs", unit="seq"):
        frames = load_no_smoke_frames(seq_dir)
        if not frames:
            continue
        # Evaluate all frames (post_start=0 means no offset filtering)
        detected = score_seq(frames, post_start=0, seq_dir=seq_dir)
        for p, det in detected.items():
            if det:
                counts[p]["fp"] += 1
            else:
                counts[p]["tn"] += 1

    # --- Print and save -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Sequence-level classifier metrics  (threshold={args.threshold})")
    print(f"{'='*60}")

    results = {}
    for p in pipeline_names:
        c = counts[p]
        m = compute_metrics(c["tp"], c["tn"], c["fp"], c["fn"])
        results[p] = m
        print_metrics(p, m)

    out_file = out_dir / "classifier_metrics.json"
    with open(out_file, "w") as f:
        json.dump({"threshold": args.threshold, "pipelines": results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
