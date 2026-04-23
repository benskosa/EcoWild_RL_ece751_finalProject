"""
summarize_sweep_eval.py
-----------------------
Reads all eval_results.json files from a sweep eval directory, prints a
sorted summary table, and saves heatmaps for TPR, FPR, F1, and AUC across
the n_frames x frame_gap grid.

Usage
-----
    # Val results (from grid_sweep.py):
    python summarize_sweep_eval.py --eval_dir sweep_results/eval

    # Test results (from eval_mobilenet_test.sh):
    python summarize_sweep_eval.py --eval_dir sweep_results/eval_test

    # Sort table by a different column:
    python summarize_sweep_eval.py --eval_dir sweep_results/eval_test --sort f1

    # Show TP/TN/FP/FN counts in table:
    python summarize_sweep_eval.py --eval_dir sweep_results/eval_test --verbose

    # Save heatmaps to a custom directory:
    python summarize_sweep_eval.py --eval_dir sweep_results/eval_test --out_dir my_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(eval_dir: Path) -> list[dict]:
    rows = []
    for result_file in sorted(eval_dir.glob("*/eval_results.json")):
        run_name = result_file.parent.name
        try:
            parts = run_name.split("_")
            nf  = int(parts[0].replace("nf", ""))
            gap = int(parts[1].replace("gap", ""))
        except (IndexError, ValueError):
            nf, gap = -1, -1

        with open(result_file) as f:
            data = json.load(f)

        m = data.get("gate_metrics", {})
        rows.append({
            "run":      run_name,
            "n_frames": nf,
            "gap":      gap,
            "thresh":   m.get("threshold", "?"),
            "acc":      m.get("accuracy",  float("nan")),
            "tpr":      m.get("tpr",       float("nan")),
            "fpr":      m.get("fpr",       float("nan")),
            "ppv":      m.get("ppv",       float("nan")),
            "f1":       m.get("f1",        float("nan")),
            "auc":      data.get("auc",    float("nan")),
            "tp":       m.get("tp", "?"),
            "tn":       m.get("tn", "?"),
            "fp":       m.get("fp", "?"),
            "fn":       m.get("fn", "?"),
        })
    return rows


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], sort_by: str, verbose: bool) -> None:
    if not rows:
        print("No eval_results.json files found.")
        return

    rows = sorted(rows, key=lambda r: r.get(sort_by, 0), reverse=(sort_by != "fpr"))

    if verbose:
        print(f"{'Run':<20} {'nf':>3} {'gap':>4} {'thr':>5} "
              f"{'Acc':>6} {'TPR':>6} {'FPR':>6} {'PPV':>6} {'F1':>6} {'AUC':>6} "
              f"{'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
        print("-" * 110)
    else:
        print(f"{'Run':<20} {'nf':>3} {'gap':>4} {'thr':>5} "
              f"{'Acc':>6} {'TPR':>6} {'FPR':>6} {'PPV':>6} {'F1':>6} {'AUC':>6}")
        print("-" * 72)

    for r in rows:
        base = (f"{r['run']:<20} {r['n_frames']:>3} {r['gap']:>4} {r['thresh']:>5.2f} "
                f"{r['acc']:>6.3f} {r['tpr']:>6.3f} {r['fpr']:>6.3f} "
                f"{r['ppv']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f}")
        if verbose:
            print(f"{base} {r['tp']:>6} {r['tn']:>6} {r['fp']:>6} {r['fn']:>6}")
        else:
            print(base)

    print()
    best_acc = max(rows, key=lambda r: r["acc"])
    best_tpr = max(rows, key=lambda r: r["tpr"])
    best_fpr = min(rows, key=lambda r: r["fpr"])
    best_f1  = max(rows, key=lambda r: r["f1"])
    best_auc = max(rows, key=lambda r: r["auc"])

    print("Best runs:")
    print(f"  Accuracy : {best_acc['acc']:.3f}  ({best_acc['run']})")
    print(f"  TPR      : {best_tpr['tpr']:.3f}  ({best_tpr['run']})")
    print(f"  FPR      : {best_fpr['fpr']:.3f}  ({best_fpr['run']})  <- lower is better")
    print(f"  F1       : {best_f1['f1']:.3f}  ({best_f1['run']})")
    print(f"  AUC      : {best_auc['auc']:.3f}  ({best_auc['run']})")
    print()
    print("Paper target (threshold=0.75): TPR=0.50, FPR=0.19")


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(
    rows: list[dict],
    metric: str,
    out_path: Path,
    title: str,
    higher_is_better: bool = True,
) -> None:
    all_nf   = sorted({r["n_frames"] for r in rows})
    all_gaps = sorted({r["gap"]      for r in rows})

    # Build grid: rows=n_frames, cols=frame_gap
    grid = np.full((len(all_nf), len(all_gaps)), np.nan)
    for r in rows:
        if r["n_frames"] in all_nf and r["gap"] in all_gaps:
            i = all_nf.index(r["n_frames"])
            j = all_gaps.index(r["gap"])
            grid[i, j] = r[metric]

    cmap = "YlGn" if higher_is_better else "YlOrRd_r"
    fig, ax = plt.subplots(figsize=(max(6, len(all_gaps) * 1.8),
                                    max(4, len(all_nf) * 1.3)))

    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    ax.set_xticks(range(len(all_gaps)))
    ax.set_xticklabels([f"gap={g}" for g in all_gaps], fontsize=10)
    ax.set_yticks(range(len(all_nf)))
    ax.set_yticklabels([f"nf={n}" for n in all_nf], fontsize=10)
    ax.set_xlabel("frame_gap", fontsize=11)
    ax.set_ylabel("n_frames",  fontsize=11)

    # Annotate each cell
    for i in range(len(all_nf)):
        for j in range(len(all_gaps)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="black" if 0.2 < val < 0.85 else "white")

    # Mark the best cell with a blue border
    if not np.all(np.isnan(grid)):
        best_idx = np.unravel_index(
            np.nanargmax(grid) if higher_is_better else np.nanargmin(grid),
            grid.shape,
        )
        ax.add_patch(plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=2.5, label="Best",
        ))

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_heatmaps(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(rows, "tpr", out_dir / "heatmap_tpr.png",
                 "Test TPR across n_frames x frame_gap",
                 higher_is_better=True)
    plot_heatmap(rows, "fpr", out_dir / "heatmap_fpr.png",
                 "Test FPR across n_frames x frame_gap  (lower is better)",
                 higher_is_better=False)
    plot_heatmap(rows, "f1",  out_dir / "heatmap_f1.png",
                 "Test F1 across n_frames x frame_gap",
                 higher_is_better=True)
    plot_heatmap(rows, "auc", out_dir / "heatmap_auc.png",
                 "Test AUC across n_frames x frame_gap",
                 higher_is_better=True)
    plot_heatmap(rows, "acc", out_dir / "heatmap_acc.png",
                 "Test Accuracy across n_frames x frame_gap",
                 higher_is_better=True)

    print(f"\nAll heatmaps saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval_dir", default="sweep_results/eval",
                        help="Directory containing run subdirs with eval_results.json")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save heatmaps (default: <eval_dir>/plots)")
    parser.add_argument("--sort", default="tpr",
                        choices=["tpr", "fpr", "acc", "f1", "auc"],
                        help="Column to sort table by (default: tpr)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also show TP/TN/FP/FN counts in table")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip heatmap generation, print table only")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"Directory not found: {eval_dir}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "plots"

    rows = load_results(eval_dir)
    print(f"\nResults from: {eval_dir.resolve()}  ({len(rows)} runs)\n")
    print_table(rows, sort_by=args.sort, verbose=args.verbose)

    if not args.no_plots:
        save_heatmaps(rows, out_dir)


if __name__ == "__main__":
    main()
