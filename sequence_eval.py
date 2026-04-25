"""
sequence_eval.py
----------------
Sequence-level smoke detection evaluation.

For each smoke sequence in the dataset, measures:
  1. Detection rate  -- did the model detect smoke at least once?
  2. Time to first detection -- seconds after ignition of first positive prediction

Frame offset is parsed from the EcoWild filename convention:
  <fire_id>_<unix_timestamp>_<+/-offset_seconds>.jpg
  Positive offset = after ignition, negative = before ignition.
  Only post-ignition frames (offset >= 0) are evaluated.

Pipelines evaluated (depending on which checkpoints are supplied):
  - LBP + MobileNet standalone       (--mobilenet_ckpt)
  - ResNet34 standalone              (--resnet_ckpt)
  - YOLOv8 standalone                (--yolo_ckpt)
  - ResNet34 + YOLOv8 ensemble OR    (--resnet_ckpt + --yolo_ckpt)
  - LBP + MobileNet gate -> ensemble (all three checkpoints)

Usage
-----
    # MobileNet standalone:
    python sequence_eval.py \\
        --data_root      smokeDetection_baseline_ecoWild/Dataset/test \\
        --mobilenet_ckpt smokeDetection_ourExperiments/sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \\
        --n_frames 2 --frame_gap 1 \\
        --cache_root smokeDetection_baseline_ecoWild/lbp_cache/gap_1 \\
        --out_dir    seq_eval_results/nf2_gap1

    # Baselines only (ResNet + YOLOv8 + ensemble):
    python sequence_eval.py \\
        --data_root   smokeDetection_baseline_ecoWild/Dataset/test \\
        --resnet_ckpt smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt \\
        --yolo_ckpt   smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt \\
        --out_dir     seq_eval_results/baselines

    # Full gate pipeline (all 5 pipelines at once):
    python sequence_eval.py \\
        --data_root      smokeDetection_baseline_ecoWild/Dataset/test \\
        --mobilenet_ckpt smokeDetection_ourExperiments/sweep_results/checkpoints/nf2_gap1/nf2_gap1_best_acc.pt \\
        --resnet_ckpt    smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt \\
        --yolo_ckpt      smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt \\
        --n_frames 2 --frame_gap 1 \\
        --cache_root smokeDetection_baseline_ecoWild/lbp_cache/gap_1 \\
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Allow imports from smokeDetection_ourExperiments
_exp_dir = Path(__file__).parent / "smokeDetection_ourExperiments"
if str(_exp_dir) not in sys.path:
    sys.path.insert(0, str(_exp_dir))


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_offset(path: Path) -> int:
    """Parse seconds-from-ignition from EcoWild filename. '+02226' -> 2226."""
    return int(path.stem.rsplit("_", 1)[-1])


def is_valid_image(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_sequence(seq_dir: Path) -> tuple[list[Path], int]:
    """
    Load and sort frames by ignition offset.
    Returns (sorted_frames, first_post_ignition_idx).
    Skips corrupted files.
    """
    frames = [f for f in sorted(seq_dir.glob("*.jpg"), key=parse_offset)
              if is_valid_image(f)]
    post_idx = next(
        (i for i, f in enumerate(frames) if parse_offset(f) >= 0),
        len(frames),
    )
    return frames, post_idx


# ---------------------------------------------------------------------------
# LBP + MobileNet scoring
# ---------------------------------------------------------------------------

def _load_lbp_pair(
    f1: Path, f2: Path,
    pair_idx: int,
    cache_seq_dir: Path | None,
    target_size: tuple[int, int],
) -> np.ndarray | None:
    """Load a pairwise LBP-motion image from cache or compute on-the-fly."""
    if cache_seq_dir is not None:
        cached = cache_seq_dir / f"pair_{pair_idx:04d}.png"
        if cached.exists():
            return np.array(Image.open(cached).convert("RGB"))

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


def get_mobilenet_scores(
    frames: list[Path],
    post_start: int,
    model: torch.nn.Module,
    cache_seq_dir: Path | None,
    n_frames: int,
    frame_gap: int,
    transform,
    device: torch.device,
    target_size: tuple[int, int] = (240, 180),
) -> list[tuple[int, float]]:
    """
    Returns list of (offset_seconds, smoke_probability) for each valid window
    that starts at or after post_start.
    The reported offset is that of the LAST frame in the window.
    """
    from feature_extraction import make_lbp_motion_image_nframes

    window_span = (n_frames - 1) * frame_gap  # frames needed beyond start
    scores = []

    for start in range(post_start, len(frames) - window_span):
        window = [frames[start + i * frame_gap] for i in range(n_frames)]
        last_offset = parse_offset(window[-1])

        if n_frames == 2:
            img = _load_lbp_pair(window[0], window[1], start, cache_seq_dir, target_size)
        else:
            # Average N-1 pairwise LBP images
            pairs = []
            for i in range(n_frames - 1):
                p = _load_lbp_pair(
                    window[i], window[i + 1],
                    start + i * frame_gap,
                    cache_seq_dir,
                    target_size,
                )
                if p is not None:
                    pairs.append(p)
            img = np.mean(pairs, axis=0).astype(np.uint8) if pairs else None

        if img is None:
            continue

        tensor = transform(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            prob  = torch.sigmoid(logit).item()
        scores.append((last_offset, prob))

    return scores


# ---------------------------------------------------------------------------
# ResNet34 scoring
# ---------------------------------------------------------------------------

def get_resnet_scores(
    frames: list[Path],
    post_start: int,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> list[tuple[int, float]]:
    """Returns list of (offset_seconds, smoke_probability) per post-ignition frame."""
    scores = []
    model.eval()
    with torch.no_grad():
        for f in frames[post_start:]:
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue
            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor)
            prob   = torch.softmax(logits, dim=1)[0, 1].item()
            scores.append((parse_offset(f), prob))
    return scores


# ---------------------------------------------------------------------------
# YOLOv8 scoring
# ---------------------------------------------------------------------------

def get_yolo_scores(
    frames: list[Path],
    post_start: int,
    yolo_model,
    imgsz: int = 224,
) -> list[tuple[int, float]]:
    """Returns list of (offset_seconds, smoke_probability) per post-ignition frame."""
    post_frames = [f for f in frames[post_start:] if is_valid_image(f)]
    if not post_frames:
        return []

    results = yolo_model.predict(
        source  = [str(f) for f in post_frames],
        imgsz   = imgsz,
        verbose = False,
    )
    return [
        (parse_offset(post_frames[i]), float(r.probs.data[1].item()))
        for i, r in enumerate(results)
    ]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def first_detection(scores: list[tuple[int, float]], threshold: float) -> int | None:
    """Return the offset (seconds) of first score >= threshold, or None."""
    for offset, prob in scores:
        if prob >= threshold:
            return offset
    return None


def gate_first_detection(
    mob_scores: list[tuple[int, float]],
    resnet_by_offset: dict[int, float],
    yolo_by_offset: dict[int, float],
    gate_threshold: float,
    ensemble_threshold: float,
) -> int | None:
    """
    Gate pipeline: once LBP+MobileNet fires at offset G, run the OR ensemble on
    all frames at offset >= G and return the first ensemble detection.
    This avoids exact-frame alignment issues where MobileNet and the ensemble
    fire at different offsets.
    """
    gate_offset = None
    for offset, mob_prob in mob_scores:
        if mob_prob >= gate_threshold:
            gate_offset = offset
            break

    if gate_offset is None:
        return None

    # Run ensemble on all frames from gate_offset onwards
    all_offsets = sorted(o for o in resnet_by_offset if o >= gate_offset)
    for offset in all_offsets:
        r = resnet_by_offset.get(offset, 0.0)
        y = yolo_by_offset.get(offset, 0.0)
        if max(r, y) >= ensemble_threshold:
            return offset
    return None


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(seq_results: list[dict]) -> dict:
    detected     = [r for r in seq_results if r["detected"]]
    times        = [r["first_detection_s"] for r in detected]

    return {
        "n_sequences":              len(seq_results),
        "n_detected":               len(detected),
        "detection_rate":           round(len(detected) / len(seq_results), 4) if seq_results else 0,
        "mean_time_to_detection_s": round(float(np.mean(times)),   1) if times else None,
        "median_time_to_detection_s": round(float(np.median(times)), 1) if times else None,
        "min_time_to_detection_s":  int(min(times))  if times else None,
        "max_time_to_detection_s":  int(max(times))  if times else None,
        "detection_times_s":        sorted(times),
    }


def print_summary(name: str, agg: dict) -> None:
    print(f"\n  [{name}]")
    print(f"    Sequences evaluated : {agg['n_sequences']}")
    print(f"    Detected            : {agg['n_detected']}  "
          f"(rate = {agg['detection_rate']:.1%})")
    if agg["mean_time_to_detection_s"] is not None:
        print(f"    Mean time to detect : {agg['mean_time_to_detection_s']} s  "
              f"({agg['mean_time_to_detection_s']/60:.1f} min)")
        print(f"    Median              : {agg['median_time_to_detection_s']} s  "
              f"({agg['median_time_to_detection_s']/60:.1f} min)")
        print(f"    Range               : {agg['min_time_to_detection_s']} s – "
              f"{agg['max_time_to_detection_s']} s")
    else:
        print(f"    No sequences detected — time to detection N/A")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root", required=True,
                        help="Dataset split root (contains smoke/ and no_smoke/)")
    parser.add_argument("--mobilenet_ckpt", default=None,
                        help="LBP+MobileNet checkpoint (.pt)")
    parser.add_argument("--resnet_ckpt",    default=None,
                        help="ResNet34 checkpoint (.pt)")
    parser.add_argument("--yolo_ckpt",      default=None,
                        help="YOLOv8 checkpoint (best.pt)")
    parser.add_argument("--n_frames",    type=int, default=2)
    parser.add_argument("--frame_gap",   type=int, default=1)
    parser.add_argument("--cache_root",  default=None,
                        help="Gap-specific LBP cache root (e.g. lbp_cache/gap_1)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Decision threshold for all models (default: 0.5)")
    parser.add_argument("--imgsz",       type=int, default=224)
    parser.add_argument("--out_dir",     default="seq_eval_results",
                        help="Output directory for JSON, CSV, and summary")
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    if not any([args.mobilenet_ckpt, args.resnet_ckpt, args.yolo_ckpt]):
        parser.error("Provide at least one checkpoint (--mobilenet_ckpt, --resnet_ckpt, --yolo_ckpt)")

    device    = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root).resolve()
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).resolve() if args.cache_root else None

    smoke_root = data_root / "smoke"
    seq_dirs   = sorted([d for d in smoke_root.iterdir() if d.is_dir()])
    print(f"\nData root       : {data_root}")
    print(f"Smoke sequences : {len(seq_dirs)}")
    print(f"Device          : {device}")
    print(f"Threshold       : {args.threshold}")

    # --- Load models --------------------------------------------------------
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
        print(f"MobileNet loaded: {args.mobilenet_ckpt}  (variant={variant})")

    if args.resnet_ckpt:
        ckpt = torch.load(args.resnet_ckpt, map_location=device)
        # Support: our format {"state_dict": ...}, common {"model": ...}, or raw state dict
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
        else:
            state_dict = ckpt
        resnet_model = models.resnet34()
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
        resnet_model.load_state_dict(state_dict)
        resnet_model.to(device).eval()
        print(f"ResNet34 loaded : {args.resnet_ckpt}")

    if args.yolo_ckpt:
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_ckpt)
        print(f"YOLOv8 loaded   : {args.yolo_ckpt}")

    # --- Per-sequence evaluation --------------------------------------------
    pipeline_results: dict[str, list[dict]] = {
        name: [] for name in
        (["mobilenet"]                                    if mob_model    else []) +
        (["resnet34"]                                     if resnet_model else []) +
        (["yolov8"]                                       if yolo_model   else []) +
        (["ensemble_OR"]                                  if resnet_model and yolo_model else []) +
        (["gate_mobilenet_ensemble"]                      if mob_model and resnet_model and yolo_model else [])
    }

    for seq_dir in tqdm(seq_dirs, desc="Sequences", unit="seq"):
        frames, post_start = load_sequence(seq_dir)

        if post_start >= len(frames):
            # No post-ignition frames — skip
            continue

        # Derive cache dir for this sequence
        cache_seq_dir = None
        if cache_root is not None:
            rel = seq_dir.relative_to(data_root.parent)  # e.g. test/smoke/<fire_id>
            cache_seq_dir = cache_root / rel

        # Score each model
        mob_scores     = []
        resnet_scores  = []
        yolo_scores    = []

        if mob_model:
            mob_scores = get_mobilenet_scores(
                frames, post_start, mob_model, cache_seq_dir,
                args.n_frames, args.frame_gap, mob_transform, device,
            )

        if resnet_model:
            resnet_scores = get_resnet_scores(
                frames, post_start, resnet_model, resnet_transform, device,
            )

        if yolo_model:
            yolo_scores = get_yolo_scores(frames, post_start, yolo_model, args.imgsz)

        # Build offset-indexed lookups for ensemble / gate
        resnet_by_offset = {offset: prob for offset, prob in resnet_scores}
        yolo_by_offset   = {offset: prob for offset, prob in yolo_scores}

        seq_id = seq_dir.name

        def record(pipeline: str, det_offset: int | None) -> None:
            pipeline_results[pipeline].append({
                "sequence":        seq_id,
                "detected":        det_offset is not None,
                "first_detection_s": det_offset,
            })

        if mob_model:
            record("mobilenet", first_detection(mob_scores, args.threshold))

        if resnet_model:
            record("resnet34", first_detection(resnet_scores, args.threshold))

        if yolo_model:
            record("yolov8", first_detection(yolo_scores, args.threshold))

        if resnet_model and yolo_model:
            ensemble_scores = [
                (offset, max(resnet_by_offset.get(offset, 0.0),
                             yolo_by_offset.get(offset, 0.0)))
                for offset in sorted(resnet_by_offset)
            ]
            record("ensemble_OR", first_detection(ensemble_scores, args.threshold))

        if mob_model and resnet_model and yolo_model:
            record("gate_mobilenet_ensemble",
                   gate_first_detection(mob_scores, resnet_by_offset, yolo_by_offset,
                                        args.threshold, args.threshold))

    # --- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Sequence-level results  (threshold={args.threshold})")
    print(f"{'='*60}")

    all_agg = {}
    for pipeline, results in pipeline_results.items():
        agg = aggregate(results)
        all_agg[pipeline] = agg
        print_summary(pipeline, agg)

    # --- Save ---------------------------------------------------------------
    # JSON summary
    with open(out_dir / "sequence_summary.json", "w") as f:
        json.dump({"threshold": args.threshold, "pipelines": all_agg}, f, indent=2)

    # Per-sequence CSV for each pipeline
    for pipeline, results in pipeline_results.items():
        csv_path = out_dir / f"{pipeline}_per_sequence.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sequence", "detected", "first_detection_s"])
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
