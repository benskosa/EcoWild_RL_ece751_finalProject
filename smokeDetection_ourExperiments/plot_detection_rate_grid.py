"""
plot_detection_rate_grid.py
---------------------------
Plots a colour-coded grid of smoke-sequence detection rates from the
seq_eval_results sweep, with each cell annotated as both a percentage
and a fraction (detected / total sequences).

Usage
-----
    python plot_detection_rate_grid.py

    # Point at a different results directory:
    python plot_detection_rate_grid.py --eval_dir seq_eval_results

    # Save to a custom output directory:
    python plot_detection_rate_grid.py --out_dir my_plots
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
            continue

        with open(result_file) as f:
            data = json.load(f)

        mob = data.get("pipelines", {}).get("mobilenet", {})
        n_seq = mob.get("n_sequences", 0)
        n_det = mob.get("n_detected",  0)
        rate  = mob.get("detection_rate", float("nan"))
        rows.append({"n_frames": nf, "gap": gap,
                     "n_seq": n_seq, "n_det": n_det, "rate": rate})
    return rows


def plot_grid(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("No data found.")
        return

    all_nf   = sorted({r["n_frames"] for r in rows})
    all_gaps = sorted({r["gap"]      for r in rows})
    lookup   = {(r["n_frames"], r["gap"]): r for r in rows}

    n_rows, n_cols = len(all_nf), len(all_gaps)
    grid = np.full((n_rows, n_cols), np.nan)
    for r in rows:
        i = all_nf.index(r["n_frames"])
        j = all_gaps.index(r["gap"])
        grid[i, j] = r["rate"]

    fig, ax = plt.subplots(figsize=(max(5, n_cols * 2.2), max(3.5, n_rows * 1.8)))

    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    im = ax.imshow(grid, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Detection Rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    # Annotate each cell with percentage + fraction
    for i, nf in enumerate(all_nf):
        for j, gap in enumerate(all_gaps):
            r = lookup.get((nf, gap))
            if r is None:
                continue
            rate = r["rate"]
            pct_str  = f"{rate:.1%}"
            frac_str = f"{r['n_det']}/{r['n_seq']}"

            # Choose white or black text based on background brightness
            norm_val = (rate - vmin) / max(vmax - vmin, 1e-9)
            txt_color = "white" if norm_val > 0.75 else "black"

            ax.text(j, i - 0.1, pct_str,
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color=txt_color)
            ax.text(j, i + 0.22, frac_str,
                    ha="center", va="center", fontsize=9,
                    color=txt_color)

    # Highlight best cell with a blue border
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle(
        (best[1] - 0.5, best[0] - 0.5), 1, 1,
        fill=False, edgecolor="royalblue", linewidth=2.5, label="Best",
    ))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"gap={g}" for g in all_gaps], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"nf={n}" for n in all_nf], fontsize=11)
    ax.set_xlabel("frame_gap", fontsize=12)
    ax.set_ylabel("n_frames",  fontsize=12)
    ax.set_title("Smoke-sequence Detection Rate\n(% detected  |  detected / total sequences)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval_dir", default="seq_eval_results",
                        help="Directory with mobilenet_nfN_gapM/ subdirs (default: seq_eval_results)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: <eval_dir>/plots)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"Directory not found: {eval_dir}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "plots"
    rows = load_results(eval_dir)
    print(f"Loaded {len(rows)} runs from {eval_dir.resolve()}")
    plot_grid(rows, out_dir / "detection_rate_grid.png")


if __name__ == "__main__":
    main()
