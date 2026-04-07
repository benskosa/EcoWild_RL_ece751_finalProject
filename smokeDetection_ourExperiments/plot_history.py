"""
plot_history.py
---------------
Plot training curves from the history.json saved by train.py.

Produces a single figure with four subplots:
  1. Accuracy      — train_acc vs val_acc
  2. TPR           — val_tpr (sensitivity / recall)
  3. FPR           — val_fpr (false alarm rate)
  4. PPV           — val_ppv (precision)

Vertical dashed lines mark the epoch of the best val_acc and best val_tpr
checkpoints so you can see exactly when each was saved.

Usage
-----
    # Plot from default checkpoint directory:
    python plot_history.py

    # Specify a run explicitly:
    python plot_history.py --history checkpoints/history.json

    # Compare multiple runs on the same axes:
    python plot_history.py \\
        --history checkpoints_gap1/history.json \\
                  checkpoints_gap2/history.json \\
                  checkpoints_gap4/history.json \\
        --labels  gap=1 gap=2 gap=4

    # Save to a custom path:
    python plot_history.py --out training_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


METRICS = [
    ("val_acc",  "train_acc", "Accuracy",  "val_acc"),
    ("val_tpr",  None,        "TPR (Sensitivity)", "val_tpr"),
    ("val_fpr",  None,        "FPR (False Alarm Rate)", "val_fpr"),
    ("val_ppv",  None,        "PPV (Precision)", "val_ppv"),
]


def load_history(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    with open(path) as f:
        rows = json.load(f)
    epochs = [r["epoch"] for r in rows]
    series = {key: [r[key] for r in rows] for key in rows[0] if key != "epoch"}
    return epochs, series


def best_epoch(series: dict, key: str) -> int:
    vals = series[key]
    return vals.index(max(vals)) + 1   # epochs are 1-indexed


def plot_histories(
    history_paths: list[Path],
    labels: list[str],
    out_path: Path,
    smooth: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run_idx, (hpath, label) in enumerate(zip(history_paths, labels)):
        epochs, series = load_history(hpath)
        color = colors[run_idx % len(colors)]

        ep_best_acc = best_epoch(series, "val_acc")
        ep_best_tpr = best_epoch(series, "val_tpr")

        for ax_idx, (val_key, train_key, title, _) in enumerate(METRICS):
            ax = axes[ax_idx]
            val_vals = series.get(val_key, [])

            if smooth > 1:
                val_smooth = _moving_avg(val_vals, smooth)
            else:
                val_smooth = val_vals

            # Val metric (solid)
            ax.plot(epochs, val_smooth, color=color, linewidth=1.8,
                    label=label if ax_idx == 0 else "_nolegend_")

            # Train metric (dashed, only for accuracy subplot)
            if train_key and train_key in series:
                train_vals = series[train_key]
                if smooth > 1:
                    train_vals = _moving_avg(train_vals, smooth)
                ax.plot(epochs, train_vals, color=color, linewidth=1.2,
                        linestyle="--", alpha=0.6,
                        label=f"{label} (train)" if ax_idx == 0 else "_nolegend_")

            # Mark best-acc and best-tpr epochs (only on first run for clarity
            # if comparing multiple runs, skip to avoid clutter)
            if len(history_paths) == 1:
                y_acc = series[val_key][ep_best_acc - 1]
                y_tpr = series[val_key][ep_best_tpr - 1]
                ax.axvline(ep_best_acc, color="steelblue", linestyle=":",
                           linewidth=1.2, alpha=0.8,
                           label=f"best acc (ep {ep_best_acc})" if ax_idx == 0 else "_nolegend_")
                ax.axvline(ep_best_tpr, color="tomato", linestyle=":",
                           linewidth=1.2, alpha=0.8,
                           label=f"best TPR (ep {ep_best_tpr})" if ax_idx == 0 else "_nolegend_")
                ax.scatter([ep_best_acc], [y_acc], color="steelblue", zorder=5, s=40)
                ax.scatter([ep_best_tpr], [y_tpr], color="tomato",    zorder=5, s=40)

            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=1)
            ax.set_ylim(0, 1.05)

    # Single legend on the accuracy subplot
    axes[0].legend(fontsize=9, loc="lower right")

    # Mark EcoWild baselines on TPR and FPR subplots for reference
    axes[1].axhline(0.90, color="red", linestyle="--", linewidth=1,
                    alpha=0.6, label="EcoWild baseline (0.90)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[2].axhline(0.58, color="red", linestyle="--", linewidth=1,
                    alpha=0.6, label="EcoWild baseline (0.58)")
    axes[2].legend(fontsize=8, loc="upper right")

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


def _moving_avg(vals: list[float], window: int) -> list[float]:
    """Simple centred moving average, preserving list length."""
    half = window // 2
    smoothed = []
    for i in range(len(vals)):
        lo = max(0, i - half)
        hi = min(len(vals), i + half + 1)
        smoothed.append(sum(vals[lo:hi]) / (hi - lo))
    return smoothed


def main() -> None:
    default_history = Path("checkpoints") / "history.json"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--history", nargs="+", type=Path, default=[default_history],
        help="Path(s) to history.json file(s) (default: checkpoints/history.json)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Legend labels for each history file (default: filename stems)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("training_curves.png"),
        help="Output PNG path (default: training_curves.png)",
    )
    parser.add_argument(
        "--smooth", type=int, default=1,
        help="Moving-average window for smoothing noisy curves (default: 1 = no smoothing)",
    )
    args = parser.parse_args()

    # Validate inputs
    for p in args.history:
        if not p.exists():
            raise FileNotFoundError(f"history.json not found: {p}")

    labels = args.labels or [p.parent.name or p.stem for p in args.history]
    if len(labels) != len(args.history):
        raise ValueError("--labels count must match --history count")

    plot_histories(args.history, labels, args.out, args.smooth)


if __name__ == "__main__":
    main()
