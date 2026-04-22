"""
run_pipeline.py
---------------
Runs the full smoke-detection training pipeline in sequence:

  1. YOLOv8  classification training
  2. Grid sweep  (LBP cache precomputation + LBP + MobileNet training for
                  every n_frames x frame_gap combination, then eval)
  3. ResNet34 fine-tuning

Each step streams its output live so you can watch progress.  The script
stops immediately if any step fails (non-zero exit code) so you never waste
time on a downstream step whose inputs are broken.

Usage
-----
    # Default run (50 yolo epochs, 30 sweep epochs, 50 resnet epochs):
    python run_pipeline.py

    # Full production run with custom epoch counts:
    python run_pipeline.py \\
        --yolo_epochs   200 \\
        --sweep_epochs  100 \\
        --resnet_epochs 1000

    # Custom sweep grid:
    python run_pipeline.py \\
        --n_frames_list  2 3 4 5 \\
        --frame_gap_list 1 2 6 16

    # Skip steps you've already completed:
    python run_pipeline.py --skip_yolo
    python run_pipeline.py --skip_yolo --skip_sweep

    # Custom dataset / cache / output locations:
    python run_pipeline.py \\
        --dataset_root smokeDetection_baseline_ecoWild/Dataset \\
        --cache_root   smokeDetection_baseline_ecoWild/lbp_cache \\
        --sweep_out    smokeDetection_ourExperiments/sweep_results
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
    """Run a subprocess, streaming output live. Exits the process on failure."""
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  STEP: {label}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{sep}\n")

    t0     = time.time()
    result = subprocess.run(cmd)
    elapsed = fmt_time(time.time() - t0)

    if result.returncode != 0:
        print(f"\n[ERROR] '{label}' failed with exit code {result.returncode}.")
        print("Pipeline halted — fix the error above and re-run with --skip_* flags.")
        sys.exit(result.returncode)

    print(f"\n[OK] '{label}' finished in {elapsed}.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).parent.resolve()

    # Script locations
    yolo_script    = root / "smokeDetection_baseline_ecoWild" / "Train" / "yolov8_training.py"
    sweep_script   = root / "smokeDetection_ourExperiments"   / "grid_sweep.py"
    resnet_script  = root / "smokeDetection_baseline_ecoWild" / "Train" / "simple_resnet.py"

    # Default paths
    default_dataset    = root / "smokeDetection_baseline_ecoWild" / "Dataset"
    default_cache      = root / "smokeDetection_baseline_ecoWild" / "lbp_cache"
    default_sweep_out  = root / "smokeDetection_ourExperiments"   / "sweep_results"
    default_resnet_dir = root / "smokeDetection_baseline_ecoWild" / "Train" / "checkpoints"
    default_yolo_proj  = root / "smokeDetection_baseline_ecoWild" / "Train" / "runs"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Paths ---------------------------------------------------------------
    parser.add_argument("--dataset_root", type=Path, default=default_dataset,
                        help="Dataset root containing train/ val/ test/")
    parser.add_argument("--cache_root", type=Path, default=default_cache,
                        help="Root for LBP caches (gap subdirs created here)")
    parser.add_argument("--sweep_out", type=Path, default=default_sweep_out,
                        help="Output directory for grid sweep results")
    parser.add_argument("--yolo_project", type=Path, default=default_yolo_proj,
                        help="YOLOv8 project output directory")
    parser.add_argument("--yolo_name", default="yolov8n_baseline",
                        help="YOLOv8 run name subfolder")
    parser.add_argument("--resnet_save_dir", type=Path, default=default_resnet_dir,
                        help="ResNet34 checkpoint output directory")
    parser.add_argument("--resnet_run_name", default="resnet34_baseline",
                        help="ResNet34 checkpoint filename stem")

    # --- Epoch counts --------------------------------------------------------
    parser.add_argument("--yolo_epochs",   type=int, default=50,
                        help="YOLOv8 training epochs (default: 50)")
    parser.add_argument("--sweep_epochs",  type=int, default=30,
                        help="Epochs per grid-sweep run (default: 30)")
    parser.add_argument("--resnet_epochs", type=int, default=50,
                        help="ResNet34 training epochs (default: 50)")

    # --- Grid sweep config ---------------------------------------------------
    parser.add_argument("--n_frames_list",  type=int, nargs="+", default=[2, 3, 4, 5],
                        help="n_frames values to sweep (default: 2 3 4 5)")
    parser.add_argument("--frame_gap_list", type=int, nargs="+", default=[1, 2, 6, 16],
                        help="frame_gap values to sweep (default: 1 2 6 16)")
    parser.add_argument("--skip_existing_cache", action="store_true",
                        help="Pass --skip_existing_cache to grid_sweep (skip cache rebuild if dir exists)")

    # --- Batch / hardware ----------------------------------------------------
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      default=None,
                        help="e.g. 'cuda', 'cpu'. Auto-detected if omitted.")

    # --- Skip flags ----------------------------------------------------------
    parser.add_argument("--skip_yolo",   action="store_true", help="Skip YOLOv8 training")
    parser.add_argument("--skip_sweep",  action="store_true", help="Skip grid sweep")
    parser.add_argument("--skip_resnet", action="store_true", help="Skip ResNet34 training")

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    train_root   = dataset_root / "train"
    val_root     = dataset_root / "val"
    py           = sys.executable

    pipeline_start = time.time()
    print(f"\nPipeline configuration")
    print(f"  Dataset root    : {dataset_root}")
    print(f"  LBP cache root  : {args.cache_root.resolve()}")
    print(f"  Sweep output    : {args.sweep_out.resolve()}")
    print(f"  n_frames list   : {args.n_frames_list}")
    print(f"  frame_gap list  : {args.frame_gap_list}")
    print(f"  YOLOv8 epochs   : {args.yolo_epochs}")
    print(f"  Sweep epochs    : {args.sweep_epochs}  (per n_frames x frame_gap run)")
    print(f"  ResNet epochs   : {args.resnet_epochs}")
    total_sweep_runs = len(args.n_frames_list) * len(args.frame_gap_list)
    print(f"  Sweep runs      : {total_sweep_runs}  "
          f"({len(args.n_frames_list)} n_frames x {len(args.frame_gap_list)} frame_gaps)")

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
    # Step 2 — Grid sweep (cache precompute + MobileNet train + eval)
    # -------------------------------------------------------------------------
    if args.skip_sweep:
        print("\n[SKIP] Grid sweep")
    else:
        cmd = [
            py, str(sweep_script),
            "--dataset_root",   str(dataset_root),
            "--cache_root",     str(args.cache_root.resolve()),
            "--out_dir",        str(args.sweep_out.resolve()),
            "--n_frames_list",  *[str(n) for n in args.n_frames_list],
            "--frame_gap_list", *[str(g) for g in args.frame_gap_list],
            "--epochs",         str(args.sweep_epochs),
            "--batch_size",     str(args.batch_size),
            "--num_workers",    str(args.num_workers),
        ]
        if args.skip_existing_cache:
            cmd.append("--skip_existing_cache")
        run_step(
            f"Grid sweep  {args.n_frames_list} n_frames x {args.frame_gap_list} gaps  "
            f"({args.sweep_epochs} epochs each, {total_sweep_runs} runs total)",
            cmd,
        )

    # -------------------------------------------------------------------------
    # Step 3 — ResNet34
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
