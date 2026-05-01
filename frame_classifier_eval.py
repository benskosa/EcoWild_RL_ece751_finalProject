"""
frame_classifier_eval.py
------------------------
Frame-level classification metrics (Accuracy, TPR, FPR) for each pipeline
on the test dataset, treating every image (or MobileNet window) as an
independent sample — agnostic of which sequence it belongs to.

Labels are assigned purely by folder:
  smoke/    frames -> positive class (label = 1)
  no_smoke/ frames -> negative class (label = 0)

All frames are included regardless of ignition offset.  The gate pipeline
is omitted because it requires sequence-level context to make sense.

For MobileNet, each sliding window across a sequence produces one sample.
For ResNet34 and YOLOv8, each individual frame produces one sample.
For the OR ensemble, each frame produces one sample (max of ResNet/YOLO scores).

Usage
-----
    # All pipelines:
    python frame_classifier_eval.py \\
        --data_root   smokeDetection_baseline_ecoWild/Dataset/test \\
        --mobilenet_ckpt smokeDetection_ourExperiments/sweep_results/checkpoints/nf2_gap16/nf2_gap16_best_acc.pt \\
        --resnet_ckpt smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt \\
        --yolo_ckpt   smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt \\
        --n_frames 2 --frame_gap 16 \\
        --cache_root smokeDetection_baseline_ecoWild/lbp_cache/gap_16 \\
        --threshold 0.5 \\
        --out_dir frame_classifier_results

    # Baselines only (no MobileNet):
    python frame_classifier_eval.py \\
        --data_root   smokeDetection_baseline_ecoWild/Dataset/test \\
        --resnet_ckpt smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt \\
        --yolo_ckpt   smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt \\
        --threshold 0.5
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
from tqdm import tqdm

import sequence_eval as _se

_exp_dir = Path(__file__).parent / "smokeDetection_ourExperiments"
if str(_exp_dir) not in sys.path:
    sys.path.insert(0, str(_exp_dir))


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
    total_pos = m["tp"] + m["fn"]
    total_neg = m["fp"] + m["tn"]
    print(f"\n  [{name}]")
    print(f"    TP={m['tp']}  FN={m['fn']}  FP={m['fp']}  TN={m['tn']}")
    print(f"    TPR      : {m['tpr']:.3f}  ({m['tp']}/{total_pos} smoke frames/windows correct)")
    print(f"    FPR      : {m['fpr']:.3f}  ({m['fp']}/{total_neg} no_smoke frames triggered)")
    print(f"    Accuracy : {m['accuracy']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",       required=True)
    parser.add_argument("--mobilenet_ckpt",  default=None)
    parser.add_argument("--resnet_ckpt",     default=None)
    parser.add_argument("--yolo_ckpt",       default=None)
    parser.add_argument("--n_frames",        type=int, default=2)
    parser.add_argument("--frame_gap",       type=int, default=1)
    parser.add_argument("--cache_root",      default=None)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--imgsz",           type=int, default=224)
    parser.add_argument("--out_dir",         default="frame_classifier_results")
    parser.add_argument("--device",          default=None)
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
    print(f"Note: gate pipeline omitted (not meaningful at frame level)")

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

    pipeline_names = (
        (["mobilenet"]   if mob_model                   else []) +
        (["resnet34"]    if resnet_model                 else []) +
        (["yolov8"]      if yolo_model                   else []) +
        (["ensemble_OR"] if resnet_model and yolo_model  else [])
    )
    counts = {p: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for p in pipeline_names}

    def accumulate(pipeline: str, scores: list, label: int) -> None:
        """Tally each score in the list as a TP/FP/TN/FN."""
        for entry in scores:
            prob = entry[-1]  # works for both (offset, prob) and (o1, o2, prob)
            predicted_positive = prob >= args.threshold
            if label == 1:
                counts[pipeline]["tp" if predicted_positive else "fn"] += 1
            else:
                counts[pipeline]["fp" if predicted_positive else "tn"] += 1

    def process_sequences(seq_dirs: list[Path], label: int) -> None:
        cls_name = "smoke" if label == 1 else "no_smoke"
        for seq_dir in tqdm(seq_dirs, desc=cls_name, unit="seq"):
            # Load all frames; post_start=0 so no ignition filtering
            frames = sorted(
                f for f in seq_dir.glob("*.jpg") if _se.is_valid_image(f)
            )
            if not frames:
                continue

            cache_seq_dir = None
            if cache_root is not None:
                rel = seq_dir.relative_to(data_root.parent)
                cache_seq_dir = cache_root / rel

            if mob_model:
                mob_scores = _se.get_mobilenet_scores(
                    frames, post_start=0, model=mob_model,
                    cache_seq_dir=cache_seq_dir,
                    n_frames=args.n_frames, frame_gap=args.frame_gap,
                    transform=mob_transform, device=device,
                )
                accumulate("mobilenet", mob_scores, label)

            if resnet_model:
                resnet_scores = _se.get_resnet_scores(
                    frames, post_start=0, model=resnet_model,
                    transform=resnet_transform, device=device,
                )
                accumulate("resnet34", resnet_scores, label)

            if yolo_model:
                yolo_scores = _se.get_yolo_scores(frames, post_start=0,
                                                   yolo_model=yolo_model,
                                                   imgsz=args.imgsz)
                accumulate("yolov8", yolo_scores, label)

            if resnet_model and yolo_model:
                resnet_by_offset = {o: p for o, p in resnet_scores}
                yolo_by_offset   = {o: p for o, p in yolo_scores}
                ensemble_scores  = [
                    (offset, max(resnet_by_offset.get(offset, 0.0),
                                 yolo_by_offset.get(offset, 0.0)))
                    for offset in sorted(resnet_by_offset)
                ]
                accumulate("ensemble_OR", ensemble_scores, label)

    process_sequences(smoke_dirs,    label=1)
    process_sequences(no_smoke_dirs, label=0)

    # --- Print and save -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Frame-level classifier metrics  (threshold={args.threshold})")
    print(f"{'='*60}")

    results = {}
    for p in pipeline_names:
        c = counts[p]
        m = compute_metrics(c["tp"], c["tn"], c["fp"], c["fn"])
        results[p] = m
        print_metrics(p, m)

    out_file = out_dir / "frame_classifier_metrics.json"
    with open(out_file, "w") as f:
        json.dump({"threshold": args.threshold, "pipelines": results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
