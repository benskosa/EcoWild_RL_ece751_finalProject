"""
eval_baselines.py
-----------------
Evaluate ResNet34 and/or YOLOv8 baseline models on a dataset split,
reporting TPR, FPR, accuracy, PPV, and F1 at a configurable confidence
threshold.  Default threshold is 0.75, matching the paper:

  "Energy-Constrained Optimization for Wildfire Detection Using RGB Images"
  (target: TPR=0.50, FPR=0.19 at threshold=0.75, imgsz=224)

Also reports AUC and, when both models are supplied, an ensemble score
(average of ResNet34 + YOLOv8 smoke probabilities).

ImageFolder alphabetical class order: no_smoke=0, smoke=1

Usage
-----
    # Both models on the val set:
    python eval_baselines.py \\
        --data_root   Dataset/val \\
        --resnet_ckpt Train/checkpoints/resnet34_baseline_best_acc.pt \\
        --yolo_ckpt   Train/runs/yolov8n_baseline/weights/best.pt \\
        --threshold   0.75

    # ResNet34 only:
    python eval_baselines.py \\
        --data_root   Dataset/val \\
        --resnet_ckpt Train/checkpoints/resnet34_baseline_best_acc.pt

    # YOLOv8 only, on test set, sweep thresholds:
    python eval_baselines.py \\
        --data_root Dataset/test \\
        --yolo_ckpt Train/runs/yolov8n_baseline/weights/best.pt \\
        --threshold 0.5 0.6 0.75 0.9

    # Evaluate original EcoWild YOLOv8 model:
    python eval_baselines.py \\
        --data_root Dataset/val \\
        --yolo_ckpt Model/Pytorch/yolov8l-cls_whole_224_best_new.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total       if total       > 0 else 0.0
    tpr      = tp / (tp + fn)          if (tp + fn)   > 0 else 0.0
    fpr      = fp / (fp + tn)          if (fp + tn)   > 0 else 0.0
    ppv      = tp / (tp + fp)          if (tp + fp)   > 0 else 0.0
    f1       = 2*ppv*tpr / (ppv + tpr) if (ppv + tpr) > 0 else 0.0

    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")

    return {
        "threshold": threshold,
        "accuracy":  round(accuracy, 4),
        "tpr":       round(tpr,      4),
        "fpr":       round(fpr,      4),
        "ppv":       round(ppv,      4),
        "f1":        round(f1,       4),
        "auc":       round(auc,      4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_total": total,
    }


def print_metrics(name: str, m: dict) -> None:
    print(f"\n  [{name}]  threshold={m['threshold']}")
    print(f"    Accuracy : {m['accuracy']:.4f}")
    print(f"    TPR      : {m['tpr']:.4f}   (paper target: 0.50)")
    print(f"    FPR      : {m['fpr']:.4f}   (paper target: 0.19)")
    print(f"    PPV      : {m['ppv']:.4f}")
    print(f"    F1       : {m['f1']:.4f}")
    print(f"    AUC      : {m['auc']:.4f}")
    print(f"    TP/TN/FP/FN: {m['tp']} / {m['tn']} / {m['fp']} / {m['fn']}")


# ---------------------------------------------------------------------------
# ResNet34 inference
# ---------------------------------------------------------------------------

def run_resnet(ckpt_path: Path, data_root: Path, batch_size: int,
               num_workers: int, device: torch.device) -> np.ndarray:
    """Returns array of smoke probabilities (class-1 softmax), one per image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(root=str(data_root), transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    all_probs = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="ResNet34 inference", unit="batch"):
            imgs  = imgs.to(device)
            logits = model(imgs)                        # (B, 2)
            probs  = torch.softmax(logits, dim=1)[:, 1] # smoke class prob
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# YOLOv8 inference
# ---------------------------------------------------------------------------

def run_yolo(ckpt_path: Path, data_root: Path, batch_size: int,
             num_workers: int, imgsz: int) -> np.ndarray:
    """Returns array of smoke probabilities (class-1 confidence), one per image."""
    from ultralytics import YOLO

    # Collect image paths in the same order ImageFolder would
    dataset = ImageFolder(root=str(data_root))
    image_paths = [s[0] for s in dataset.samples]

    model  = YOLO(str(ckpt_path))
    # Predict in one call; YOLO handles batching internally
    results = model.predict(
        source  = image_paths,
        imgsz   = imgsz,
        batch   = batch_size,
        workers = num_workers,
        verbose = False,
        stream  = True,    # memory-efficient generator
    )

    all_probs = []
    for r in tqdm(results, total=len(image_paths),
                  desc="YOLOv8 inference", unit="img"):
        # Class order matches training: no_smoke=0, smoke=1 (alphabetical)
        smoke_prob = float(r.probs.data[1].item())
        all_probs.append(smoke_prob)

    return np.array(all_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root",   required=True,
                        help="Dataset split root (contains smoke/ and no_smoke/)")
    parser.add_argument("--resnet_ckpt", default=None,
                        help="Path to ResNet34 checkpoint (.pt)")
    parser.add_argument("--yolo_ckpt",   default=None,
                        help="Path to YOLOv8 checkpoint (best.pt)")
    parser.add_argument("--threshold",   type=float, nargs="+", default=[0.75],
                        help="Confidence threshold(s) to evaluate at (default: 0.75). "
                             "Pass multiple values to sweep, e.g. --threshold 0.5 0.75 0.9")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--imgsz",       type=int, default=224,
                        help="Image size for YOLOv8 inference (default: 224)")
    parser.add_argument("--out_dir",     default="eval_results",
                        help="Directory to save results JSON (default: eval_results)")
    parser.add_argument("--device",      default=None,
                        help="e.g. 'cuda', 'cpu'. Auto-detected if omitted.")
    args = parser.parse_args()

    if args.resnet_ckpt is None and args.yolo_ckpt is None:
        parser.error("Provide at least one of --resnet_ckpt or --yolo_ckpt")

    device    = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root).resolve()
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root  : {data_root}")
    print(f"Device     : {device}")
    print(f"Thresholds : {args.threshold}")

    # --- Get ground-truth labels from ImageFolder (alphabetical: no_smoke=0, smoke=1)
    dataset = ImageFolder(root=str(data_root))
    labels  = np.array([s[1] for s in dataset.samples])
    n_smoke    = int(labels.sum())
    n_no_smoke = int((labels == 0).sum())
    print(f"Images     : {len(labels)}  (smoke={n_smoke}, no_smoke={n_no_smoke})")

    results_out = {}
    probs_dict  = {}

    # --- ResNet34 ---
    if args.resnet_ckpt:
        print(f"\nRunning ResNet34 from {args.resnet_ckpt} ...")
        probs_dict["resnet34"] = run_resnet(
            Path(args.resnet_ckpt), data_root,
            args.batch_size, args.num_workers, device,
        )

    # --- YOLOv8 ---
    if args.yolo_ckpt:
        print(f"\nRunning YOLOv8 from {args.yolo_ckpt} ...")
        probs_dict["yolov8"] = run_yolo(
            Path(args.yolo_ckpt), data_root,
            args.batch_size, args.num_workers, args.imgsz,
        )

    # --- Ensemble (if both available) ---
    if "resnet34" in probs_dict and "yolov8" in probs_dict:
        probs_dict["ensemble"] = (probs_dict["resnet34"] + probs_dict["yolov8"]) / 2.0

    # --- Evaluate at each threshold ---
    print(f"\n{'='*60}")
    print(f"  Results  (paper target: TPR=0.50, FPR=0.19 @ threshold=0.75)")
    print(f"{'='*60}")

    for model_name, probs in probs_dict.items():
        results_out[model_name] = []
        for thresh in args.threshold:
            m = compute_metrics(labels, probs, thresh)
            print_metrics(model_name, m)
            results_out[model_name].append(m)

    # --- Save ---
    out_path = out_dir / "eval_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
