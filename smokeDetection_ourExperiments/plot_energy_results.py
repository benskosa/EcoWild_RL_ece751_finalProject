"""
plot_energy_results.py
----------------------
Generates two visualisations from energy_eval_pipelines.py output:

  1. Cumulative energy over time (one line per pipeline, smoke + no-smoke)
  2. Bar chart of total energy in Wh/day per pipeline

Usage
-----
    python smokeDetection_ourExperiments/plot_energy_results.py \
        --results_dir energy_results/pipelines \
        --out_dir     energy_results/plots

    # Show only smoke sequence plots:
    python smokeDetection_ourExperiments/plot_energy_results.py \
        --results_dir energy_results/pipelines \
        --sequences   smoke
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
import pandas as pd


# ---------------------------------------------------------------------------
# Colour / style config
# ---------------------------------------------------------------------------

PIPELINE_STYLES = {
    "mobilenet":        {"color": "#2196F3", "linestyle": "-",  "label": "MobileNet (CPU)"},
    "resnet34":         {"color": "#4CAF50", "linestyle": "-",  "label": "ResNet34"},
    "yolov8":           {"color": "#FF9800", "linestyle": "-",  "label": "YOLOv8"},
    "ensemble_OR":      {"color": "#F44336", "linestyle": "-",  "label": "Ensemble OR"},
    "gate_from_window": {"color": "#9C27B0", "linestyle": "--", "label": "Gate (from window)"},
    "gate_from_start":  {"color": "#795548", "linestyle": ":",  "label": "Gate (from start)"},
}

FRAMES_PER_DAY = 60 * 24   # 1 frame/min × 1440 min/day


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_frame_csvs(results_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Returns {pipeline: {sequence: DataFrame}} from all *_per_frame.csv files.
    """
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for csv_path in sorted(results_dir.glob("*_per_frame.csv")):
        # Filename: {pipeline}_{sequence}_per_frame.csv
        stem = csv_path.stem.replace("_per_frame", "")
        # Split on known sequence names to find pipeline
        for seq in ("no_smoke", "smoke"):
            if stem.endswith(f"_{seq}"):
                pipeline = stem[: -(len(seq) + 1)]
                df = pd.read_csv(csv_path)
                data.setdefault(pipeline, {})[seq] = df
                break
    return data


def load_summary(results_dir: Path) -> dict:
    summary_path = results_dir / "energy_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Cumulative energy over time
# ---------------------------------------------------------------------------

def plot_cumulative_energy(
    data: dict[str, dict[str, pd.DataFrame]],
    sequence: str,
    out_path: Path,
    frames_per_minute: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    any_plotted = False
    for pipeline, seq_dict in data.items():
        if sequence not in seq_dict:
            continue
        df = seq_dict[sequence].copy()
        if "total_energy_mj" not in df.columns:
            continue

        style = PIPELINE_STYLES.get(pipeline, {"color": "gray", "linestyle": "-",
                                                "label": pipeline})

        # Time axis in minutes
        time_min = df["frame_idx"] / frames_per_minute

        # Cumulative energy in mJ → J
        cumulative_j = df["total_energy_mj"].cumsum() / 1000.0

        ax.plot(time_min, cumulative_j,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=style["label"])
        any_plotted = True

    if not any_plotted:
        print(f"  No data found for sequence '{sequence}' — skipping plot.")
        plt.close(fig)
        return

    seq_title = "Smoke sequence" if sequence == "smoke" else "No-smoke sequence"
    ax.set_title(f"Cumulative Energy over Time — {seq_title}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("Cumulative Energy (J)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Total Wh/day bar chart
# ---------------------------------------------------------------------------

def plot_wh_per_day(
    data: dict[str, dict[str, pd.DataFrame]],
    out_path: Path,
) -> None:
    """
    Extrapolates total energy per 180-frame sequence to a full day (1440 frames)
    and converts to Wh.
    """
    sequences = ["no_smoke", "smoke"]
    seq_labels = {"no_smoke": "No-smoke sequence", "smoke": "Smoke sequence"}

    pipelines = [p for p in PIPELINE_STYLES if p in data]
    if not pipelines:
        print("  No pipeline data found — skipping Wh/day plot.")
        return

    x = np.arange(len(pipelines))
    width = 0.35
    n_seq = sum(1 for s in sequences if any(s in data[p] for p in pipelines))

    fig, ax = plt.subplots(figsize=(max(8, len(pipelines) * 1.6), 5))

    offsets = np.linspace(-(n_seq - 1) * width / 2,
                          (n_seq - 1) * width / 2, n_seq)

    for idx, seq in enumerate(sequences):
        wh_per_day = []
        for pipeline in pipelines:
            df = data[pipeline].get(seq)
            if df is None or "total_energy_mj" not in df.columns:
                wh_per_day.append(0.0)
                continue
            n_frames = len(df)
            if n_frames == 0:
                wh_per_day.append(0.0)
                continue
            total_j = df["total_energy_mj"].sum() / 1000.0
            # Scale to full day
            j_per_day = total_j * (FRAMES_PER_DAY / n_frames)
            wh = j_per_day / 3600.0
            wh_per_day.append(wh)

        bars = ax.bar(x + offsets[idx], wh_per_day, width,
                      label=seq_labels[seq],
                      alpha=0.85,
                      color=["#2196F3" if seq == "no_smoke" else "#F44336"] * len(pipelines))

        # Annotate bar values
        for bar, val in zip(bars, wh_per_day):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Estimated Energy Consumption per Day\n(extrapolated from 3-hour sequence)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Energy (Wh/day)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_STYLES.get(p, {}).get("label", p) for p in pipelines],
                       rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

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
    parser.add_argument("--results_dir", default="energy_results/pipelines",
                        help="Directory containing *_per_frame.csv and energy_summary.json")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save plots (default: <results_dir>/plots)")
    parser.add_argument("--sequences", nargs="+", default=["smoke", "no_smoke"],
                        choices=["smoke", "no_smoke"],
                        help="Which sequences to plot cumulative energy for")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    data = load_per_frame_csvs(results_dir)

    if not data:
        print("No per-frame CSV files found. Run energy_eval_pipelines.py first.")
        return

    print(f"Found pipelines : {list(data.keys())}")
    print(f"Output dir      : {out_dir}\n")

    # Plot 1: Cumulative energy over time (one plot per sequence)
    for seq in args.sequences:
        plot_cumulative_energy(
            data, seq,
            out_dir / f"cumulative_energy_{seq}.png",
        )

    # Plot 2: Wh/day bar chart
    plot_wh_per_day(data, out_dir / "wh_per_day.png")

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
