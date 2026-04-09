"""
plot_sweep.py
-------------
Visualize training curves and final metrics from a grid_sweep.py run.

Produces four output files in --out_dir:
  1. curves_by_gap.png    -- training curves grouped by frame_gap
                             (one subplot per gap, lines = different n_frames)
  2. curves_by_nframes.png -- training curves grouped by n_frames
                             (one subplot per n_frames, lines = different gaps)
  3. heatmap_acc.png      -- grid heatmap of final val accuracy
  4. heatmap_tpr.png      -- grid heatmap of final val TPR

Usage
-----
    python plot_sweep.py --sweep_dir sweep_results
    python plot_sweep.py --sweep_dir sweep_results --metric val_tpr --smooth 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_histories(sweep_dir: Path) -> dict[tuple[int,int], dict]:
    """
    Scan sweep_dir/checkpoints/ for history.json files.
    Returns dict keyed by (n_frames, frame_gap) → {epochs, series}.
    """
    ckpt_root = sweep_dir / "checkpoints"
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"No checkpoints/ folder found under {sweep_dir}")

    runs = {}
    for run_dir in sorted(ckpt_root.iterdir()):
        if not run_dir.is_dir():
            continue
        hist_path = run_dir / "history.json"
        if not hist_path.exists():
            continue
        # Parse run name: nf{n}_gap{g}
        name = run_dir.name
        try:
            parts  = name.split("_")
            nf     = int(parts[0].replace("nf", ""))
            gap    = int(parts[1].replace("gap", ""))
        except (IndexError, ValueError):
            print(f"  [skip] could not parse run name: {name}")
            continue

        with open(hist_path) as f:
            rows = json.load(f)

        epochs = [r["epoch"] for r in rows]
        series = {k: [r[k] for r in rows] for k in rows[0] if k != "epoch"}
        runs[(nf, gap)] = {"epochs": epochs, "series": series, "name": name}

    if not runs:
        raise RuntimeError(f"No valid history.json files found under {ckpt_root}")

    print(f"Loaded {len(runs)} runs: {sorted(runs.keys())}")
    return runs


def _moving_avg(vals: list[float], w: int) -> list[float]:
    if w <= 1:
        return vals
    half = w // 2
    out  = []
    for i in range(len(vals)):
        lo = max(0, i - half)
        hi = min(len(vals), i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def final_val(series: dict, metric: str) -> float:
    """Return the best (max) value achieved during training for a metric."""
    vals = series.get(metric, [])
    return max(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Plot 1 & 2: Training curves grouped by gap / n_frames
# ---------------------------------------------------------------------------

def plot_grouped_curves(
    runs: dict,
    group_key: str,           # "gap" or "nframes"
    metric: str,
    smooth: int,
    out_path: Path,
    ecowild_baseline: float | None,
) -> None:
    """
    group_key="gap"     → one subplot per gap,    lines = n_frames values
    group_key="nframes" → one subplot per nframes, lines = gap values
    """
    assert group_key in ("gap", "nframes")

    all_nf   = sorted({k[0] for k in runs})
    all_gaps = sorted({k[1] for k in runs})

    if group_key == "gap":
        groups      = all_gaps
        group_label = "frame_gap"
        line_vals   = all_nf
        line_label  = "n_frames"
        get_key     = lambda grp, lv: (lv, grp)
    else:
        groups      = all_nf
        group_label = "n_frames"
        line_vals   = all_gaps
        line_label  = "frame_gap"
        get_key     = lambda grp, lv: (grp, lv)

    n_cols  = 2
    n_rows  = (len(groups) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4 * n_rows),
                             squeeze=False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    metric_label = metric.replace("val_", "val ").replace("_", " ").upper()

    for idx, grp in enumerate(groups):
        ax  = axes[idx // n_cols][idx % n_cols]
        for li, lv in enumerate(line_vals):
            key  = get_key(grp, lv)
            if key not in runs:
                continue
            data   = runs[key]
            epochs = data["epochs"]
            vals   = _moving_avg(data["series"].get(metric, []), smooth)
            ax.plot(epochs, vals, color=colors[li % len(colors)],
                    linewidth=1.8, label=f"{line_label}={lv}")

        if ecowild_baseline is not None:
            ax.axhline(ecowild_baseline, color="red", linestyle="--",
                       linewidth=1.0, alpha=0.7, label="EcoWild baseline")

        ax.set_title(f"{group_label}={grp}", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=1)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8, loc="lower right")

    # Hide unused subplots
    for idx in range(len(groups), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    title = (f"{metric_label} — grouped by {group_label}"
             + (f"  (smoothed w={smooth})" if smooth > 1 else ""))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 & 4: Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(
    runs: dict,
    metric: str,
    out_path: Path,
    title: str,
    higher_is_better: bool = True,
) -> None:
    all_nf   = sorted({k[0] for k in runs})
    all_gaps = sorted({k[1] for k in runs})

    grid = np.full((len(all_nf), len(all_gaps)), np.nan)
    for i, nf in enumerate(all_nf):
        for j, gap in enumerate(all_gaps):
            key = (nf, gap)
            if key in runs:
                grid[i, j] = final_val(runs[key]["series"], metric)

    cmap = "YlGn" if higher_is_better else "YlOrRd_r"
    fig, ax = plt.subplots(figsize=(max(6, len(all_gaps) * 1.6),
                                    max(4, len(all_nf) * 1.2)))
    im = ax.imshow(grid, cmap=cmap, vmin=np.nanmin(grid), vmax=np.nanmax(grid),
                   aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    ax.set_xticks(range(len(all_gaps)))
    ax.set_xticklabels([f"gap={g}" for g in all_gaps], fontsize=10)
    ax.set_yticks(range(len(all_nf)))
    ax.set_yticklabels([f"nf={n}" for n in all_nf], fontsize=10)
    ax.set_xlabel("frame_gap", fontsize=11)
    ax.set_ylabel("n_frames",  fontsize=11)

    # Annotate each cell with its value
    for i in range(len(all_nf)):
        for j in range(len(all_gaps)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if 0.2 < val < 0.85 else "white")

    # Mark the best cell
    best_idx = np.unravel_index(
        np.nanargmax(grid) if higher_is_better else np.nanargmin(grid), grid.shape
    )
    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        fill=False, edgecolor="blue", linewidth=2.5, label="Best"
    ))

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep_dir", default="sweep_results",
                        help="Output directory from grid_sweep.py (default: sweep_results)")
    parser.add_argument("--metric", default="val_acc",
                        choices=["val_acc", "val_tpr", "val_fpr", "val_ppv"],
                        help="Primary metric to plot in curve figures (default: val_acc)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Moving-average smoothing window (default: 1 = none)")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save plots (default: inside --sweep_dir)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    out_dir   = Path(args.out_dir).resolve() if args.out_dir else sweep_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_histories(sweep_dir)

    # EcoWild baselines for reference lines
    baselines = {"val_tpr": 0.90, "val_fpr": 0.58, "val_acc": None, "val_ppv": None}
    baseline  = baselines.get(args.metric)

    # 1. Curves grouped by frame_gap
    plot_grouped_curves(
        runs, group_key="gap", metric=args.metric, smooth=args.smooth,
        out_path=out_dir / "curves_by_gap.png", ecowild_baseline=baseline,
    )

    # 2. Curves grouped by n_frames
    plot_grouped_curves(
        runs, group_key="nframes", metric=args.metric, smooth=args.smooth,
        out_path=out_dir / "curves_by_nframes.png", ecowild_baseline=baseline,
    )

    # 3. Heatmap — val accuracy
    plot_heatmap(
        runs, metric="val_acc",
        out_path=out_dir / "heatmap_acc.png",
        title="Best Val Accuracy across n_frames x frame_gap grid",
        higher_is_better=True,
    )

    # 4. Heatmap — val TPR
    plot_heatmap(
        runs, metric="val_tpr",
        out_path=out_dir / "heatmap_tpr.png",
        title="Best Val TPR across n_frames x frame_gap grid",
        higher_is_better=True,
    )

    # 5. Heatmap — val FPR (lower is better)
    plot_heatmap(
        runs, metric="val_fpr",
        out_path=out_dir / "heatmap_fpr.png",
        title="Best Val FPR across n_frames x frame_gap grid (lower is better)",
        higher_is_better=False,
    )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
