"""
grid_sweep.py
-------------
Automated grid sweep over n_frames x frame_gap combinations.

For each (n_frames, frame_gap) pair:
  1. Builds the LBP cache for that frame_gap (if not already cached).
  2. Trains a MobileNet smoke detector for --epochs epochs.
  3. Evaluates on the val set and records accuracy, TPR, FPR, PPV, AUC.
  4. Saves the best-accuracy checkpoint and eval results for each run.

Results are written to a summary CSV so you can easily compare all runs.

Usage
-----
    # Default grid: n_frames in {2,3,4,5}, frame_gap in {1,2,4,6}
    python grid_sweep.py \\
        --dataset_root ../smokeDetection_baseline_ecoWild/Dataset \\
        --cache_root   ../smokeDetection_baseline_ecoWild/lbp_cache \\
        --out_dir      sweep_results

    # Custom grid:
    python grid_sweep.py \\
        --dataset_root ../smokeDetection_baseline_ecoWild/Dataset \\
        --cache_root   ../smokeDetection_baseline_ecoWild/lbp_cache \\
        --n_frames_list 2 3 \\
        --frame_gap_list 1 4 8 \\
        --epochs 30

    # Skip cache rebuilding if caches already exist:
    python grid_sweep.py ... --skip_existing_cache
"""

from __future__ import annotations

import argparse
import csv
import json
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


def run(cmd: list[str], label: str) -> int:
    """Run a subprocess, streaming output live. Returns exit code."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, executable=sys.executable)
    return result.returncode


def cache_dir(cache_root: Path, gap: int) -> Path:
    return cache_root / f"gap_{gap}"


def run_name(n_frames: int, gap: int) -> str:
    return f"nf{n_frames}_gap{gap}"


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(args) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    cache_root   = Path(args.cache_root).resolve()
    out_dir      = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root = dataset_root / "train"
    val_root   = dataset_root / "val"

    script_dir = Path(__file__).parent

    grid = [
        (nf, gap)
        for gap in args.frame_gap_list
        for nf  in args.n_frames_list
    ]

    print(f"\nGrid sweep: {len(grid)} runs")
    print(f"  n_frames   : {args.n_frames_list}")
    print(f"  frame_gap  : {args.frame_gap_list}")
    print(f"  epochs     : {args.epochs}")
    print(f"  dataset    : {dataset_root}")
    print(f"  cache root : {cache_root}")
    print(f"  output     : {out_dir}\n")

    summary_rows = []
    sweep_start  = time.time()

    for run_idx, (n_frames, gap) in enumerate(grid, 1):
        rname    = run_name(n_frames, gap)
        cdir     = cache_dir(cache_root, gap)
        ckpt_dir = out_dir / "checkpoints" / rname
        eval_dir = out_dir / "eval" / rname
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{run_idx}/{len(grid)}] n_frames={n_frames}, frame_gap={gap}  ({rname})")
        t0 = time.time()

        # ---- Step 1: build LBP cache for this gap (if needed) --------------
        if cdir.is_dir() and args.skip_existing_cache:
            print(f"  Cache exists, skipping precompute: {cdir}")
        else:
            rc = run(
                [
                    sys.executable,
                    str(script_dir / "precompute_lbp_cache.py"),
                    "--dataset_root", str(dataset_root),
                    "--cache_root",   str(cdir),
                    "--frame_gap",    str(gap),
                    "--splits",       "train", "val",
                ],
                f"Precompute LBP cache  gap={gap}",
            )
            if rc != 0:
                print(f"  ERROR: precompute failed (exit {rc}), skipping run.")
                continue

        # ---- Step 2: train --------------------------------------------------
        rc = run(
            [
                sys.executable,
                str(script_dir / "train.py"),
                "--train_root",  str(train_root),
                "--val_root",    str(val_root),
                "--cache_root",  str(cdir),
                "--n_frames",    str(n_frames),
                "--epochs",      str(args.epochs),
                "--batch_size",  str(args.batch_size),
                "--pretrained",
                "--save_dir",    str(ckpt_dir),
                "--run_name",    rname,
                "--num_workers", str(args.num_workers),
            ] + (["--preload_cache"] if args.preload_cache else []),
            f"Train  n_frames={n_frames}, frame_gap={gap}",
        )
        if rc != 0:
            print(f"  ERROR: training failed (exit {rc}), skipping eval.")
            continue

        # ---- Step 3: evaluate best-accuracy checkpoint ----------------------
        ckpt_path = ckpt_dir / f"{rname}_best_acc.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found at {ckpt_path}, skipping eval.")
            continue

        rc = run(
            [
                sys.executable,
                str(script_dir / "eval.py"),
                "--checkpoint",  str(ckpt_path),
                "--data_root",   str(val_root),
                "--frame_gap",   str(gap),
                "--out_dir",     str(eval_dir),
                "--batch_size",  str(args.batch_size),
                "--num_workers", str(args.num_workers),
            ],
            f"Eval  n_frames={n_frames}, frame_gap={gap}",
        )
        if rc != 0:
            print(f"  WARNING: eval failed (exit {rc}).")

        # ---- Step 4: collect results from eval JSON -------------------------
        results_path = eval_dir / "eval_results.json"
        row = {
            "n_frames":  n_frames,
            "frame_gap": gap,
            "run_name":  rname,
            "elapsed":   fmt_time(time.time() - t0),
        }
        if results_path.exists():
            with open(results_path) as f:
                res = json.load(f)
            m = res.get("gate_metrics", {})
            row.update({
                "threshold": m.get("threshold", ""),
                "accuracy":  round(m.get("accuracy", 0), 4),
                "tpr":       round(m.get("tpr", 0), 4),
                "fpr":       round(m.get("fpr", 0), 4),
                "ppv":       round(m.get("ppv", 0), 4),
                "f1":        round(m.get("f1",  0), 4),
                "auc":       round(res.get("auc", 0), 4),
                "checkpoint": str(ckpt_path),
            })
        else:
            row.update({k: "" for k in
                        ["threshold","accuracy","tpr","fpr","ppv","f1","auc","checkpoint"]})

        summary_rows.append(row)

        # Write summary CSV after every run so progress is never lost
        _write_csv(out_dir / "sweep_summary.csv", summary_rows)
        print(f"\n  Run complete in {row['elapsed']}")
        print(f"  acc={row.get('accuracy','')}  tpr={row.get('tpr','')}  "
              f"fpr={row.get('fpr','')}  auc={row.get('auc','')}")

    # ---- Final summary ------------------------------------------------------
    total_elapsed = fmt_time(time.time() - sweep_start)
    print(f"\n{'='*60}")
    print(f"  Sweep complete in {total_elapsed}")
    print(f"  Summary CSV: {out_dir / 'sweep_summary.csv'}")
    print(f"{'='*60}\n")

    if summary_rows:
        best_acc = max(summary_rows, key=lambda r: r.get("accuracy", 0))
        best_tpr = max(summary_rows, key=lambda r: r.get("tpr", 0))
        best_fpr = min((r for r in summary_rows if r.get("fpr", "") != ""),
                       key=lambda r: r.get("fpr", 1))
        print(f"  Best accuracy : {best_acc['accuracy']}  ({best_acc['run_name']})")
        print(f"  Best TPR      : {best_tpr['tpr']}  ({best_tpr['run_name']})")
        print(f"  Best FPR      : {best_fpr['fpr']}  ({best_fpr['run_name']})")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument("--dataset_root", required=True,
                        help="Root of the dataset (must contain train/ and val/)")
    parser.add_argument("--cache_root", required=True,
                        help="Root directory for LBP caches (gap subdirs created here)")
    parser.add_argument("--out_dir", default="sweep_results",
                        help="Output directory for checkpoints, eval, and summary CSV")

    # Grid definition
    parser.add_argument("--n_frames_list", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="List of n_frames values to sweep (default: 2 3 4 5)")
    parser.add_argument("--frame_gap_list", type=int, nargs="+", default=[1, 2, 6, 16],
                        help="List of frame_gap values to sweep (default: 1 2 6 16)")

    # Training options
    parser.add_argument("--epochs",      type=int, default=30,
                        help="Training epochs per run (default: 30)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--preload_cache", action="store_true",
                        help="Preload LBP cache into RAM (faster if enough RAM)")

    # Cache options
    parser.add_argument("--skip_existing_cache", action="store_true",
                        help="Skip precompute if cache dir already exists")

    args = parser.parse_args()
    sweep(args)


if __name__ == "__main__":
    main()
