"""
summarize_sequence_eval.py
--------------------------
Reads all sequence_summary.json files from a sequence eval directory,
prints a sorted summary table, and saves heatmaps for detection_rate
and mean_time_to_detection across the n_frames x frame_gap grid.

Usage
-----
    python summarize_sequence_eval.py --eval_dir seq_eval_results

    # Sort by detection rate (default):
    python summarize_sequence_eval.py --eval_dir seq_eval_results --sort det_rate

    # Sort by mean time to detect (lower is better):
    python summarize_sequence_eval.py --eval_dir seq_eval_results --sort mean_time

    # Skip heatmaps:
    python summarize_sequence_eval.py --eval_dir seq_eval_results --no_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(eval_dir: Path) -> list[dict]:
    rows = []
    for result_file in sorted(eval_dir.glob("mobilenet_*/sequence_summary.json")):
        run_name = result_file.parent.name  # e.g. mobilenet_nf2_gap1
        try:
            parts = run_name.split("_")    # ['mobilenet', 'nf2', 'gap1']
            nf  = int(parts[1].replace("nf",  ""))
            gap = int(parts[2].replace("gap", ""))
        except (IndexError, ValueError):
            nf, gap = -1, -1

        with open(result_file) as f:
            data = json.load(f)

        mob = data.get("pipelines", {}).get("mobilenet", {})
        rows.append({
            "run":       run_name,
            "n_frames":  nf,
            "gap":       gap,
            "thresh":    data.get("threshold", "?"),
            "n_seq":     mob.get("n_sequences",               0),
            "n_det":     mob.get("n_detected",                0),
            "det_rate":  mob.get("detection_rate",            float("nan")),
            "mean_time": mob.get("mean_time_to_detection_s",  float("nan")),
            "med_time":  mob.get("median_time_to_detection_s",float("nan")),
            "min_time":  mob.get("min_time_to_detection_s",   float("nan")),
            "max_time":  mob.get("max_time_to_detection_s",   float("nan")),
        })
    return rows


def print_table(rows: list[dict], sort_by: str) -> None:
    if not rows:
        print("No sequence_summary.json files found.")
        return

    reverse = sort_by != "mean_time"
    rows = sorted(rows, key=lambda r: (r.get(sort_by) or 0), reverse=reverse)

    print(f"{'Run':<26} {'nf':>3} {'gap':>4} {'thr':>5} "
          f"{'N_seq':>6} {'N_det':>6} {'DetRate':>8} "
          f"{'MeanT(s)':>10} {'MedT(s)':>9} {'MinT(s)':>9} {'MaxT(s)':>9}")
    print("-" * 105)

    for r in rows:
        mean_t = f"{r['mean_time']:>10.0f}" if r['mean_time'] == r['mean_time'] else f"{'N/A':>10}"
        med_t  = f"{r['med_time']:>9.0f}"  if r['med_time']  == r['med_time']  else f"{'N/A':>9}"
        min_t  = f"{r['min_time']:>9.0f}"  if r['min_time']  == r['min_time']  else f"{'N/A':>9}"
        max_t  = f"{r['max_time']:>9.0f}"  if r['max_time']  == r['max_time']  else f"{'N/A':>9}"

        print(f"{r['run']:<26} {r['n_frames']:>3} {r['gap']:>4} {r['thresh']:>5.2f} "
              f"{r['n_seq']:>6} {r['n_det']:>6} {r['det_rate']:>8.3f} "
              f"{mean_t}{med_t}{min_t}{max_t}")

    print()
    best_dr = max(rows, key=lambda r: r["det_rate"])
    valid   = [r for r in rows if r["mean_time"] == r["mean_time"]]
    best_mt = min(valid, key=lambda r: r["mean_time"]) if valid else None

    print("Best runs:")
    print(f"  Detection rate   : {best_dr['det_rate']:.3f}  ({best_dr['run']})")
    if best_mt:
        print(f"  Mean time (lower): {best_mt['mean_time']:.0f} s  ({best_mt['run']})")
    print()


def plot_heatmap(
    rows: list[dict],
    metric: str,
    out_path: Path,
    title: str,
    higher_is_better: bool = True,
    fmt: str = ".3f",
) -> None:
    all_nf   = sorted({r["n_frames"] for r in rows})
    all_gaps = sorted({r["gap"]      for r in rows})

    grid = np.full((len(all_nf), len(all_gaps)), np.nan)
    for r in rows:
        if r["n_frames"] in all_nf and r["gap"] in all_gaps:
            i = all_nf.index(r["n_frames"])
            j = all_gaps.index(r["gap"])
            val = r[metric]
            if val is not None and val == val:   # not None and not NaN
                grid[i, j] = float(val)

    cmap = "YlGn" if higher_is_better else "YlOrRd_r"
    fig, ax = plt.subplots(figsize=(max(6, len(all_gaps) * 1.8),
                                    max(4, len(all_nf) * 1.3)))

    vmin = np.nanmin(grid) if not np.all(np.isnan(grid)) else 0
    vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    ax.set_xticks(range(len(all_gaps)))
    ax.set_xticklabels([f"gap={g}" for g in all_gaps], fontsize=10)
    ax.set_yticks(range(len(all_nf)))
    ax.set_yticklabels([f"nf={n}" for n in all_nf], fontsize=10)
    ax.set_xlabel("frame_gap", fontsize=11)
    ax.set_ylabel("n_frames",  fontsize=11)

    for i in range(len(all_nf)):
        for j in range(len(all_gaps)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="black" if 0.2 < (val - vmin) / max(vmax - vmin, 1e-9) < 0.85 else "white")

    if not np.all(np.isnan(grid)):
        best_idx = np.unravel_index(
            np.nanargmax(grid) if higher_is_better else np.nanargmin(grid),
            grid.shape,
        )
        ax.add_patch(plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=2.5,
        ))

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_heatmaps(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(rows, "det_rate",  out_dir / "heatmap_det_rate.png",
                 "Detection Rate across n_frames x frame_gap",
                 higher_is_better=True, fmt=".3f")
    plot_heatmap(rows, "mean_time", out_dir / "heatmap_mean_time.png",
                 "Mean Time to Detection (s) across n_frames x frame_gap  (lower is better)",
                 higher_is_better=False, fmt=".0f")
    plot_heatmap(rows, "med_time",  out_dir / "heatmap_median_time.png",
                 "Median Time to Detection (s) across n_frames x frame_gap  (lower is better)",
                 higher_is_better=False, fmt=".0f")

    print(f"\nAll heatmaps saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval_dir", default="seq_eval_results",
                        help="Directory containing mobilenet_nfN_gapM/ subdirs")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save heatmaps (default: <eval_dir>/plots)")
    parser.add_argument("--sort", default="det_rate",
                        choices=["det_rate", "mean_time"],
                        help="Column to sort table by (default: det_rate)")
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
    print_table(rows, sort_by=args.sort)

    if not args.no_plots:
        save_heatmaps(rows, out_dir)


if __name__ == "__main__":
    main()
