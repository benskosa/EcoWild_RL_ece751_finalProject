"""
visualize_lbp.py
----------------
Visualize the LBP-motion features extracted from consecutive frames in
the EcoWild smoke-detection dataset.

For each selected fire sequence the script renders a row:
    [frame_1 | frame_2 | ... | frame_N | LBP-motion image]

where the LBP-motion image is the (averaged) output fed to MobileNet.

The H/S/V channels of the LBP-motion image are also shown separately so
you can see what each channel encodes:
    H  = optical-flow angle   (motion direction)
    S  = optical-flow magnitude (motion speed)
    V  = LBP of frame_1       (texture / edges)

Usage examples
--------------
# Quick check — 2-frame (paper default), 3 examples per class:
    python visualize_lbp.py --train_root ../smokeDetection_baseline_ecoWild/Dataset/train

# 4-frame window, show 5 examples per class, save to my_vis/:
    python visualize_lbp.py --train_root ../smokeDetection_baseline_ecoWild/Dataset/train \\
                            --n_frames 4 --n_examples 5 --out_dir my_vis

# Inspect a specific fire sequence by name:
    python visualize_lbp.py --train_root ../smokeDetection_baseline_ecoWild/Dataset/train \\
                            --fire_id 20160604_FIRE_rm-n-mobo-c --n_frames 3
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # no display required — saves to PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from feature_extraction import make_lbp_motion_image_nframes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames_bgr(paths: list[Path]) -> list[np.ndarray]:
    """Load a list of image paths as BGR uint8 arrays (for OpenCV)."""
    frames = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return frames


def hsv_channels(lbp_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose an LBP-motion RGB image back into its H, S, V channels
    for interpretability visualisation.

    Returns three (H, W) uint8 arrays, each converted to a grey-mapped image.
    """
    bgr  = cv2.cvtColor(lbp_rgb, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return h, s, v


def collect_sequences(
    class_dir: Path,
    n_frames: int,
    frame_gap: int,
    fire_id: str | None,
    n_examples: int,
    seed: int,
) -> list[tuple[str, list[Path]]]:
    """
    Return a list of (fire_id, frame_paths) tuples.

    If fire_id is specified, only sequences from that video are returned.
    Otherwise, n_examples sequences are chosen at random from all videos.

    Each returned sequence contains exactly n_frames paths separated by
    frame_gap (the first valid window from each video is used unless the
    video was specifically requested).
    """
    rng = random.Random(seed)
    results: list[tuple[str, list[Path]]] = []
    window_span = frame_gap * (n_frames - 1)

    video_dirs = sorted(
        d for d in class_dir.iterdir()
        if d.is_dir()
    )

    if fire_id is not None:
        # Find the specific video
        matched = [d for d in video_dirs if d.name == fire_id]
        if not matched:
            raise ValueError(
                f"Fire ID '{fire_id}' not found under {class_dir}.\n"
                f"Available: {[d.name for d in video_dirs[:10]]} ..."
            )
        video_dirs = matched

    # Collect candidate (video_dir, first_window) pairs
    candidates = []
    for vd in video_dirs:
        frames = sorted(vd.glob("*.[jp][pn]g"))
        if len(frames) < n_frames:
            continue
        # All valid starting indices
        starts = list(range(len(frames) - window_span))
        if not starts:
            continue
        candidates.append((vd.name, frames, starts))

    if not candidates:
        raise RuntimeError(
            f"No sequences with n_frames={n_frames}, frame_gap={frame_gap} "
            f"found under {class_dir}"
        )

    # Sample n_examples (or all if fewer exist)
    chosen_candidates = rng.sample(candidates, min(n_examples, len(candidates)))

    for fire_name, frames, starts in chosen_candidates:
        # Use a random starting position within the video for variety
        start = rng.choice(starts)
        seq = [frames[start + j * frame_gap] for j in range(n_frames)]
        results.append((fire_name, seq))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sequence(
    fire_name: str,
    label: str,              # "smoke" or "no_smoke"
    frame_paths: list[Path],
    lbp_rgb: np.ndarray,
    out_path: Path,
    show_channels: bool = True,
) -> None:
    """
    Save a figure showing the raw frames, the LBP-motion image, and (if
    show_channels=True) the three individual H/S/V channel images.

    Layout (n_frames=3, show_channels=True):
        Row 0: frame1 | frame2 | frame3 | LBP-motion (RGB)
        Row 1: H channel | S channel | V channel | (empty)
    """
    n = len(frame_paths)
    n_cols = n + 1          # frames + LBP-motion composite
    n_rows = 2 if show_channels else 1

    fig = plt.figure(figsize=(3.2 * n_cols, 3.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            wspace=0.05, hspace=0.35)

    # ---- Row 0: raw frames + composite LBP-motion image -------------------
    for j, fp in enumerate(frame_paths):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.array(Image.open(fp).convert("RGB")))
        ax.set_title(f"frame {j + 1}", fontsize=9)
        ax.axis("off")

    ax_lbp = fig.add_subplot(gs[0, n])
    ax_lbp.imshow(lbp_rgb)
    ax_lbp.set_title("LBP-motion\n(RGB composite)", fontsize=9)
    ax_lbp.axis("off")

    # ---- Row 1: individual channels ----------------------------------------
    if show_channels:
        h, s, v = hsv_channels(lbp_rgb)
        channel_data = [
            (h, "H: flow angle\n(motion direction)",   "hsv"),
            (s, "S: flow magnitude\n(motion speed)",   "hot"),
            (v, "V: LBP of frame 1\n(texture/edges)", "gray"),
        ]
        for j, (ch, title, cmap) in enumerate(channel_data):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(ch, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        # Leave the 4th cell in row 1 blank (or label it)
        ax_blank = fig.add_subplot(gs[1, n])
        ax_blank.axis("off")

    suptitle = (
        f"[{label.upper()}]  {fire_name}\n"
        f"n_frames={n}, frame_gap=1"
    )
    fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=1.01)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize(args) -> None:
    train_root = Path(args.train_root)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_size = (args.width, args.height)

    for label in ["smoke", "no_smoke"]:
        class_dir = train_root / label
        if not class_dir.is_dir():
            print(f"[skip] {class_dir} not found")
            continue

        print(f"\n--- {label} ---")
        try:
            sequences = collect_sequences(
                class_dir=class_dir,
                n_frames=args.n_frames,
                frame_gap=args.frame_gap,
                fire_id=args.fire_id,
                n_examples=args.n_examples,
                seed=args.seed,
            )
        except (ValueError, RuntimeError) as e:
            print(f"  ERROR: {e}")
            continue

        for i, (fire_name, frame_paths) in enumerate(sequences):
            frames_bgr = load_frames_bgr(frame_paths)
            lbp_rgb    = make_lbp_motion_image_nframes(frames_bgr, target_size)

            fname = f"{label}_{i:02d}_{fire_name[:50]}.png"
            out_path = out_dir / fname
            plot_sequence(
                fire_name=fire_name,
                label=label,
                frame_paths=frame_paths,
                lbp_rgb=lbp_rgb,
                out_path=out_path,
                show_channels=not args.no_channels,
            )

    print(f"\nAll figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train_root", required=True,
        help="Path to the train (or val) split root containing smoke/ and no_smoke/",
    )
    parser.add_argument(
        "--n_frames", type=int, default=2,
        help="Number of consecutive frames per LBP-motion sample (default: 2)",
    )
    parser.add_argument(
        "--frame_gap", type=int, default=1,
        help="Stride between consecutive frames in the window (default: 1)",
    )
    parser.add_argument(
        "--n_examples", type=int, default=3,
        help="Number of random example sequences to visualize per class (default: 3)",
    )
    parser.add_argument(
        "--fire_id", default=None,
        help=(
            "Optional: show only this specific fire sequence by name "
            "(e.g. 20160604_FIRE_rm-n-mobo-c). "
            "Overrides --n_examples."
        ),
    )
    parser.add_argument(
        "--width",  type=int, default=240,
        help="LBP-motion image width  (default: 240, matching the paper)",
    )
    parser.add_argument(
        "--height", type=int, default=180,
        help="LBP-motion image height (default: 180, matching the paper)",
    )
    parser.add_argument(
        "--no_channels", action="store_true",
        help="Skip the per-channel H/S/V breakdown row",
    )
    parser.add_argument(
        "--out_dir", default="lbp_visualizations",
        help="Output directory for saved figures (default: lbp_visualizations/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for selecting example sequences (default: 42)",
    )

    args = parser.parse_args()
    visualize(args)
