"""
run_pipeline.py
---------------
Runs the full smoke-detection training pipeline in sequence:

  1. YOLOv8  classification training
  2. LBP cache precomputation  (pairwise LBP + Farneback optical-flow images)
  3. LBP + MobileNet training
  4. ResNet34 fine-tuning

Each step streams its output live so you can watch progress.  The script
stops immediately if any step fails (non-zero exit code) so you never waste
time on a downstream step whose inputs are broken.

Usage
-----
    # Quick sanity-check run (all steps, short epochs):
    python run_pipeline.py

    # Full production run with custom epoch counts:
    python run_pipeline.py \\
        --yolo_epochs    200 \\
        --mobilenet_epochs 500 \\
        --resnet_epochs  1000

    # Skip steps you've already completed:
    python run_pipeline.py --skip_yolo --skip_cache

    # Custom dataset / cache location:
    python run_pipeline.py \\
        --dataset_root smokeDetection_baseline_ecoWild/Dataset \\
        --cache_root   smokeDetection_baseline_ecoWild/lbp_cache

    # Experiment with a different frame_gap / n_frames:
    python run_pipeline.py --frame_gap 2 --n_frames 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def run_step(label: str, cmd: list[str]) -> None:
    """
    Run a subprocess, streaming output live.
    Raises SystemExit if the step fails.
    """
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  STEP: {label}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{sep}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = fmt_time(time.time() - t0)

    if result.returncode != 0:
        print(f"\n[ERROR] '{label}' failed with exit code {result.returncode}.")
        print("Pipeline halted — fix the error above and re-run with the appropriate --skip_* flags.")
        sys.exit(result.returncode)

    print(f"\n[OK] '{label}' finished in {elapsed}.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).parent.resolve()

    # Script locations (relative to project root)
    yolo_script      = root / "smokeDetection_baseline_ecoWild" / "Train" / "yolov8_training.py"
    cache_script     = root / "smokeDetection_ourExperiments"   / "precompute_lbp_cache.py"
    mobilenet_script = root / "smokeDetection_ourExperiments"   / "train.py"
    resnet_script    = root / "smokeDetection_baseline_ecoWild" / "Train" / "simple_resnet.py"

    # Default paths
    default_dataset = root / "smokeDetection_baseline_ecoWild" / "Dataset"
    default_cache   = root / "smokeDetection_baseline_ecoWild" / "lbp_cache"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Paths ---------------------------------------------------------------
    parser.add_argument("--dataset_root", type=Path, default=default_dataset,
                        help=f"Dataset root with train/ val/ test/ (default: {default_dataset})")
    parser.add_argument("--cache_root", type=Path, default=default_cache,
                        help=f"Root directory for LBP caches (default: {default_cache})")
    parser.add_argument("--yolo_project", default="smokeDetection_baseline_ecoWild/Train/runs",
                        help="YOLOv8 project output directory")
    parser.add_argument("--yolo_name", default="yolov8n_baseline",
                        help="YOLOv8 run name")
    parser.add_argument("--resnet_save_dir", default="smokeDetection_baseline_ecoWild/Train/checkpoints",
                        help="ResNet34 checkpoint output directory")
    parser.add_argument("--resnet_run_name", default="resnet34_baseline",
                        help="ResNet34 checkpoint filename stem")
    parser.add_argument("--mobilenet_save_dir", default="smokeDetection_ourExperiments/checkpoints",
                        help="MobileNet checkpoint output directory")
    parser.add_argument("--mobilenet_run_name", default="mobilenet_baseline",
                        help="MobileNet checkpoint filename stem")

    # --- Epoch counts --------------------------------------------------------
    parser.add_argument("--yolo_epochs",      type=int, default=50)
    parser.add_argument("--mobilenet_epochs", type=int, default=30)
    parser.add_argument("--resnet_epochs",    type=int, default=50)

    # --- LBP / temporal window -----------------------------------------------
    parser.add_argument("--frame_gap", type=int, default=1,
                        help="Frame stride for LBP cache and MobileNet training (default: 1)")
    parser.add_argument("--n_frames",  type=int, default=2,
                        help="Frames per sample window for MobileNet (default: 2)")

    # --- Batch / hardware ----------------------------------------------------
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      default=None,
                        help="e.g. 'cuda', 'cpu'. Auto-detected if omitted.")

    # --- Skip flags ----------------------------------------------------------
    parser.add_argument("--skip_yolo",      action="store_true", help="Skip YOLOv8 training")
    parser.add_argument("--skip_cache",     action="store_true", help="Skip LBP cache precompute")
    parser.add_argument("--skip_mobilenet", action="store_true", help="Skip MobileNet training")
    parser.add_argument("--skip_resnet",    action="store_true", help="Skip ResNet34 training")

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    cache_dir    = (args.cache_root / f"gap_{args.frame_gap}").resolve()
    train_root   = dataset_root / "train"
    val_root     = dataset_root / "val"
    py           = sys.executable   # same Python that launched this script

    pipeline_start = time.time()
    print(f"\nPipeline configuration")
    print(f"  Dataset root  : {dataset_root}")
    print(f"  LBP cache     : {cache_dir}")
    print(f"  frame_gap     : {args.frame_gap}   n_frames : {args.n_frames}")
    print(f"  YOLOv8 epochs : {args.yolo_epochs}")
    print(f"  MobileNet ep. : {args.mobilenet_epochs}")
    print(f"  ResNet epochs : {args.resnet_epochs}")

    # -------------------------------------------------------------------------
    # Step 1 — YOLOv8
    # -------------------------------------------------------------------------
    if args.skip_yolo:
        print("\n[SKIP] YOLOv8 training")
    else:
        cmd = [
            py, str(yolo_script),
            "--data_root", str(dataset_root),
            "--project",   str(args.yolo_project),
            "--name",      args.yolo_name,
            "--epochs",    str(args.yolo_epochs),
            "--batch",     str(args.batch_size),
            "--workers",   str(args.num_workers),
        ]
        if args.device:
            cmd += ["--device", args.device]
        run_step(f"YOLOv8 training ({args.yolo_epochs} epochs)", cmd)

    # -------------------------------------------------------------------------
    # Step 2 — LBP cache precomputation
    # -------------------------------------------------------------------------
    if args.skip_cache:
        print("\n[SKIP] LBP cache precomputation")
    else:
        cmd = [
            py, str(cache_script),
            "--dataset_root", str(dataset_root),
            "--cache_root",   str(cache_dir),
            "--frame_gap",    str(args.frame_gap),
            "--splits",       "train", "val",
        ]
        run_step(f"LBP cache precompute (frame_gap={args.frame_gap})", cmd)

    # -------------------------------------------------------------------------
    # Step 3 — LBP + MobileNet
    # -------------------------------------------------------------------------
    if args.skip_mobilenet:
        print("\n[SKIP] LBP + MobileNet training")
    else:
        cmd = [
            py, str(mobilenet_script),
            "--train_root",  str(train_root),
            "--val_root",    str(val_root),
            "--cache_root",  str(cache_dir),
            "--n_frames",    str(args.n_frames),
            "--frame_gap",   str(args.frame_gap),
            "--epochs",      str(args.mobilenet_epochs),
            "--batch_size",  str(args.batch_size),
            "--save_dir",    str(args.mobilenet_save_dir),
            "--run_name",    args.mobilenet_run_name,
            "--num_workers", str(args.num_workers),
            "--pretrained",
        ]
        if args.device:
            cmd += ["--device", args.device]
        run_step(f"LBP + MobileNet training ({args.mobilenet_epochs} epochs)", cmd)

    # -------------------------------------------------------------------------
    # Step 4 — ResNet34
    # -------------------------------------------------------------------------
    if args.skip_resnet:
        print("\n[SKIP] ResNet34 training")
    else:
        cmd = [
            py, str(resnet_script),
            "--train_root",  str(train_root),
            "--val_root",    str(val_root),
            "--save_dir",    str(args.resnet_save_dir),
            "--run_name",    args.resnet_run_name,
            "--epochs",      str(args.resnet_epochs),
            "--batch_size",  str(args.batch_size),
            "--num_workers", str(args.num_workers),
        ]
        if args.device:
            cmd += ["--device", args.device]
        run_step(f"ResNet34 training ({args.resnet_epochs} epochs)", cmd)

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    total = fmt_time(time.time() - pipeline_start)
    print(f"\n{'=' * 64}")
    print(f"  Pipeline complete!  Total time: {total}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
