#!/usr/bin/env python3
"""
accuracy_eval.py

Evaluates classification accuracy of the three EcoWild baseline smoke-detection
models (ResNet34, YOLOv8, and the YOLOv8 OR ResNet34 ensemble) across multiple
FIgLib wildfire ignition sequences and across multiple detection-time intervals.

For each model x detection-interval combination the script reports:
  - Overall accuracy   (correct / total)
  - Confusion matrix   (TP, TN, FP, FN)
  - TPR, FPR, TNR, FNR

Outputs saved to --out_dir:
  - per_image_results.csv           one row per (model, interval, sequence, image)
  - summary_metrics.csv             one row per (model, interval) with aggregated metrics
  - confusion_matrix_<model>_<N>min.png  one confusion matrix plot per model x interval
  - accuracy_vs_interval.png        accuracy curves for all three models on one chart

Dataset layout expected under --dataset_dir:
    <dataset_dir>/
        <sequence_1>/          e.g. 20160604_FIRE_rm-n-mobo-c
            img_00001.jpg
            img_00002.jpg
            ...
        <sequence_2>/
            ...

If no subdirectories are found, the script treats --dataset_dir itself as a single
flat sequence.

Ground-truth labels are derived from --smoke_start (default 41, 1-indexed):
    images 1 ... smoke_start-1  ->  no-smoke (0)
    images smoke_start ... end  ->  smoke    (1)

For per-sequence overrides supply --metadata <path.csv> with columns:
    sequence,smoke_start
where "sequence" matches the subdirectory name.

At detection interval K (minutes), only images at 0-indexed positions
0, K, 2K, 3K, ... are evaluated (images are assumed to be 1 minute apart).

Usage:
    cd smokeDetection_baseline_ecoWild/
    python accuracy_eval.py \\
        --dataset_dir  /path/to/figlib_sequences/ \\
        --resnet_path  Model/Pytorch/best_resnet34_model_epoch_3.pth \\
        --yolo_path    Model/Pytorch/yolov8l_cls_whole_golden_best.pt \\
        --smoke_start  41 \\
        --out_dir      accuracy_results/
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")   # headless-safe backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Decision thresholds (match energy_eval.py / yolo_resnet_assemble_4_edge.py)
# ---------------------------------------------------------------------------
RESNET_SMOKE_THRESH = 0.2    # sigmoid(logit[:,1]) >= threshold -> smoke
YOLO_SMOKE_THRESH   = 0.25   # probs[1] >= threshold -> smoke

DEFAULT_INTERVALS   = [1, 2, 5, 10, 15]   # minutes

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Image preprocessing  (identical to energy_eval.py)
#
# 1. cv2.resize -> (2016, 1536)
# 2. crop bottom 1120 rows
# 3. ResNet: Resize(224,224) + ImageNet norm
# 4. YOLO:   Resize(640,640)  as PIL
# ---------------------------------------------------------------------------
_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(img_path: Path) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load and preprocess one image for both models.

    Returns
    -------
    resnet_tensor : (1, 3, 224, 224) float32 tensor, ImageNet-normalised
    yolo_pil      : 640x640 PIL Image ready for ultralytics YOLO
    """
    img_np = cv2.imread(str(img_path))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (2016, 1536))           # W x H for cv2
    img_np = img_np[1536 - 1120:, :]                    # crop -> (1120, 2016, 3)
    pil_full = Image.fromarray(img_np)

    resnet_tensor = _resnet_transform(pil_full).unsqueeze(0)    # (1,3,224,224)
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
    # PyTorch 2.6+ changed the default of weights_only from False to True,
    # which breaks ultralytics' internal torch.load calls.  Temporarily patch
    # torch.load to restore the old default for the duration of YOLO.__init__.
    import functools
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        model = YOLO(weights_path)
    finally:
        torch.load = original_load
    print(f"  YOLOv8  loaded from {weights_path}")
    return model


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------
def infer_resnet(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> int:
    with torch.no_grad():
        logits = model(tensor.to(device))
        prob   = torch.sigmoid(logits[:, 1]).item()
    return int(prob >= RESNET_SMOKE_THRESH)


def infer_yolo(model: YOLO, pil_img: Image.Image) -> int:
    results = model(pil_img, verbose=False)
    prob    = results[0].probs.data.tolist()[1]
    return int(prob >= YOLO_SMOKE_THRESH)


def infer_ensemble(
    resnet_model: nn.Module,
    yolo_model: YOLO,
    tensor: torch.Tensor,
    pil_img: Image.Image,
    device: torch.device,
) -> int:
    """OR rule: smoke if either model predicts smoke."""
    pred_r = infer_resnet(resnet_model, tensor, device)
    pred_y = infer_yolo(yolo_model, pil_img)
    return int(pred_r == 1 or pred_y == 1)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def discover_sequences(dataset_dir: Path) -> list[Path]:
    """
    Return sorted list of subdirectories under dataset_dir that contain images.
    Falls back to [dataset_dir] itself if no image-containing subdirectories exist.
    """
    sub_dirs  = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    sequences = [d for d in sub_dirs if any(d.glob("*.[jp][pn]g"))]
    if not sequences:
        sequences = [dataset_dir]   # flat layout
    return sequences


def get_images(seq_dir: Path) -> list[Path]:
    return sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)


def build_smoke_start_map(
    sequences: list[Path],
    default_smoke_start: int,
    metadata_csv: Optional[str],
) -> dict[str, int]:
    """
    Build {sequence_name: smoke_start (1-indexed)} for every sequence.
    Per-sequence values from the metadata CSV override the global default.
    """
    mapping = {seq.name: default_smoke_start for seq in sequences}
    if metadata_csv:
        meta = pd.read_csv(metadata_csv)
        required = {"sequence", "smoke_start"}
        if not required.issubset(meta.columns):
            raise ValueError(
                f"metadata CSV must have columns {required}, got {list(meta.columns)}"
            )
        for _, row in meta.iterrows():
            mapping[str(row["sequence"])] = int(row["smoke_start"])
    return mapping


# ---------------------------------------------------------------------------
# Confusion-matrix plot
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    cm: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """
    cm is a 2x2 array arranged as:
        [[TN, FP],
         [FN, TP]]
    Rows = true label (0=No Smoke, 1=Smoke)
    Cols = predicted label
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes    = ["No Smoke", "Smoke"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )

    ax.set_ylabel("True label",      fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(title,              fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Confusion matrix saved -> {out_path}")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def evaluate_all(
    sequences: list[Path],
    smoke_start_map: dict[str, int],
    resnet_model: nn.Module,
    yolo_model: YOLO,
    device: torch.device,
    intervals: list[int],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all three models across all sequences and all detection intervals.

    Returns
    -------
    per_image_df  : one row per (model, interval, sequence, image)
    summary_df    : one row per (model, interval) with aggregated metrics
    """
    configs = ["resnet", "yolo", "ensemble"]
    per_image_records = []
    summary_records   = []

    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        print(f"\n{'='*64}")
        print(f"  Model: {config.upper()}")
        print(f"{'='*64}")

        for interval in intervals:
            print(f"\n  -- detection interval = {interval} min --")
            tp = tn = fp = fn = 0

            for seq_dir in sequences:
                imgs = get_images(seq_dir)
                if not imgs:
                    print(f"    [SKIP] {seq_dir.name}: no images found")
                    continue

                smoke_start = smoke_start_map.get(seq_dir.name, 41)
                n = len(imgs)
                true_labels = [1 if (i + 1) >= smoke_start else 0 for i in range(n)]

                # Subsample: take every K-th image (0-indexed), since images are
                # 1 minute apart, this simulates a K-minute detection interval.
                indices        = list(range(0, n, interval))
                sampled_imgs   = [imgs[i]        for i in indices]
                sampled_labels = [true_labels[i] for i in indices]

                seq_correct = 0
                for img_path, true_label in zip(sampled_imgs, sampled_labels):
                    resnet_tensor, yolo_pil = preprocess_image(img_path)

                    if config == "resnet":
                        pred = infer_resnet(resnet_model, resnet_tensor, device)
                    elif config == "yolo":
                        pred = infer_yolo(yolo_model, yolo_pil)
                    else:
                        pred = infer_ensemble(
                            resnet_model, yolo_model,
                            resnet_tensor, yolo_pil, device,
                        )

                    correct = int(pred == true_label)
                    seq_correct += correct

                    if pred == 1 and true_label == 1:
                        tp += 1
                    elif pred == 0 and true_label == 0:
                        tn += 1
                    elif pred == 1 and true_label == 0:
                        fp += 1
                    else:
                        fn += 1

                    per_image_records.append({
                        "model":      config,
                        "interval":   interval,
                        "sequence":   seq_dir.name,
                        "image":      img_path.name,
                        "true_label": true_label,
                        "prediction": pred,
                        "correct":    correct,
                    })

                seq_acc = seq_correct / len(sampled_imgs) if sampled_imgs else float("nan")
                print(f"    {seq_dir.name:<45}  "
                      f"n_sampled={len(sampled_imgs):3d}  acc={seq_acc:.4f}")

            # Aggregate metrics across all sequences for this model x interval
            total    = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else None
            tpr      = tp / (tp + fn) if (tp + fn) > 0 else None   # sensitivity
            fpr      = fp / (fp + tn) if (fp + tn) > 0 else None
            tnr      = tn / (tn + fp) if (tn + fp) > 0 else None   # specificity
            fnr      = fn / (fn + tp) if (fn + tp) > 0 else None

            def _fmt(v):
                return f"{v:.4f}" if v is not None else "N/A"

            print(f"\n  [Summary] {config.upper()} @ {interval}min  "
                  f"Acc={_fmt(accuracy)}  TPR={_fmt(tpr)}  FPR={_fmt(fpr)}  "
                  f"TNR={_fmt(tnr)}  FNR={_fmt(fnr)}  "
                  f"TP={tp} TN={tn} FP={fp} FN={fn}")

            summary_records.append({
                "model":    config,
                "interval": interval,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "total":    total,
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "tpr":      round(tpr,      4) if tpr      is not None else None,
                "fpr":      round(fpr,      4) if fpr      is not None else None,
                "tnr":      round(tnr,      4) if tnr      is not None else None,
                "fnr":      round(fnr,      4) if fnr      is not None else None,
            })

            # Confusion matrix plot for this model x interval
            # Rows = true label, Cols = predicted label
            # [[TN, FP], [FN, TP]]
            cm       = np.array([[tn, fp], [fn, tp]])
            cm_title = f"{config.upper()}  |  {interval}-min interval"
            cm_path  = cm_dir / f"{config}_{interval}min.png"
            save_confusion_matrix(cm, cm_title, cm_path)

    per_image_df = pd.DataFrame(per_image_records)
    summary_df   = pd.DataFrame(summary_records)
    return per_image_df, summary_df


# ---------------------------------------------------------------------------
# Accuracy vs interval plot
# ---------------------------------------------------------------------------
def plot_accuracy_vs_interval(summary_df: pd.DataFrame, out_path: Path) -> None:
    """
    Line plot: accuracy (y-axis) vs detection interval in minutes (x-axis).
    One line per model.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    style = {
        "resnet":   {"color": "tab:blue",   "marker": "o", "label": "ResNet34"},
        "yolo":     {"color": "tab:orange",  "marker": "s", "label": "YOLOv8"},
        "ensemble": {"color": "tab:green",   "marker": "^", "label": "Ensemble (OR)"},
    }

    for config, s in style.items():
        sub = summary_df[summary_df["model"] == config].sort_values("interval")
        ax.plot(
            sub["interval"],
            sub["accuracy"],
            color=s["color"],
            marker=s["marker"],
            label=s["label"],
            linewidth=2,
            markersize=7,
        )

    all_intervals = sorted(summary_df["interval"].unique())
    ax.set_xticks(all_intervals)
    ax.set_xticklabels([f"{v}" for v in all_intervals], fontsize=11)
    ax.set_xlabel("Detection Interval (minutes)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Baseline Model Accuracy vs Detection Interval\n(FIgLib sequences)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nAccuracy vs interval plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Terminal summary table
# ---------------------------------------------------------------------------
def print_summary_table(summary_df: pd.DataFrame) -> None:
    w = 88
    print("\n" + "=" * w)
    print("  ACCURACY SUMMARY  (aggregated across all sequences)")
    print("=" * w)
    header = (
        f"{'Model':<12} {'Interval(min)':>13}  "
        f"{'Acc':>7} {'TPR':>7} {'FPR':>7} {'TNR':>7} {'FNR':>7}  "
        f"{'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'Total':>7}"
    )
    print(header)
    print("-" * w)

    def _f(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<12} {row['interval']:>13}  "
            f"{_f(row['accuracy']):>7} "
            f"{_f(row['tpr']):>7} "
            f"{_f(row['fpr']):>7} "
            f"{_f(row['tnr']):>7} "
            f"{_f(row['fnr']):>7}  "
            f"{row['tp']:>5} {row['tn']:>5} {row['fp']:>5} {row['fn']:>5} "
            f"{row['total']:>7}"
        )
    print("=" * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Accuracy evaluation for EcoWild baseline models "
            "(ResNet34, YOLOv8, ensemble) on FIgLib wildfire ignition sequences."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help=(
            "Root directory containing sequence subdirectories. Each subdirectory "
            "holds ~80 images (sorted lexicographically, 1 minute apart). "
            "If no subdirectories contain images, the directory itself is treated "
            "as a single flat sequence."
        ),
    )
    parser.add_argument(
        "--resnet_path",
        default="Model/Pytorch/best_resnet34_model_epoch_3.pth",
        help="Path to ResNet34 .pth weights file.",
    )
    parser.add_argument(
        "--yolo_path",
        default="Model/Pytorch/yolov8l_cls_whole_golden_best.pt",
        help="Path to YOLOv8 .pt weights file.",
    )
    parser.add_argument(
        "--smoke_start", type=int, default=41,
        help=(
            "Default 1-indexed image position where smoke begins "
            "(images before this are no-smoke). Applied to all sequences "
            "unless overridden by --metadata."
        ),
    )
    parser.add_argument(
        "--metadata", default=None,
        help=(
            "Optional CSV file with columns [sequence, smoke_start] for "
            "per-sequence smoke_start overrides. 'sequence' must match the "
            "subdirectory name exactly."
        ),
    )
    parser.add_argument(
        "--intervals", type=int, nargs="+", default=DEFAULT_INTERVALS,
        help="Detection interval(s) in minutes to evaluate.",
    )
    parser.add_argument(
        "--out_dir", default="accuracy_results",
        help="Directory where output CSVs and plots are saved.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device string (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Torch device   : {device}")

    # --- Discover sequences -------------------------------------------------
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    sequences = discover_sequences(dataset_dir)
    print(f"Dataset root   : {dataset_dir}")
    print(f"Sequences found: {len(sequences)}")
    for s in sequences[:10]:
        n_imgs = len(get_images(s))
        print(f"  {s.name}  ({n_imgs} images)")
    if len(sequences) > 10:
        print(f"  ... and {len(sequences) - 10} more")

    smoke_start_map = build_smoke_start_map(sequences, args.smoke_start, args.metadata)
    print(f"\nGlobal smoke_start : {args.smoke_start}")
    if args.metadata:
        print(f"Per-sequence overrides loaded from: {args.metadata}")

    # --- Load models --------------------------------------------------------
    print("\nLoading models ...")
    resnet_model = load_resnet(args.resnet_path, device)
    yolo_model   = load_yolo(args.yolo_path)

    # --- Run evaluation -----------------------------------------------------
    intervals = sorted(set(args.intervals))
    print(f"\nDetection intervals to evaluate: {intervals} minutes")

    per_image_df, summary_df = evaluate_all(
        sequences       = sequences,
        smoke_start_map = smoke_start_map,
        resnet_model    = resnet_model,
        yolo_model      = yolo_model,
        device          = device,
        intervals       = intervals,
        out_dir         = out_dir,
    )

    # --- Save outputs -------------------------------------------------------
    per_image_csv = out_dir / "per_image_results.csv"
    per_image_df.to_csv(per_image_csv, index=False)
    print(f"\nPer-image CSV saved  -> {per_image_csv}")

    summary_csv = out_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved    -> {summary_csv}")

    # --- Summary table ------------------------------------------------------
    print_summary_table(summary_df)

    # --- Accuracy vs interval plot ------------------------------------------
    plot_path = out_dir / "accuracy_vs_interval.png"
    plot_accuracy_vs_interval(summary_df, plot_path)

    print(f"\nDone. All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
