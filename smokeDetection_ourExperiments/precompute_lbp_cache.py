"""
precompute_lbp_cache.py
-----------------------
Pre-compute pairwise LBP-motion images for the entire dataset and save them
to disk.  This is a one-time cost that makes subsequent training runs much
faster by eliminating on-the-fly optical flow and LBP computation.

Cache structure
---------------
    <cache_root>/
      train/
        smoke/<fire_id>/pair_0000.png   <- LBP(frame[0], frame[frame_gap])
        smoke/<fire_id>/pair_0001.png   <- LBP(frame[1], frame[1+frame_gap])
        ...
        no_smoke/<fire_id>/pair_0000.png
        ...
      val/
        ...

The frame_gap is baked into the cache.  To experiment with multiple gaps,
run this script once per gap and store results in separate directories:

    python precompute_lbp_cache.py --frame_gap 1 --cache_root lbp_cache/gap_1
    python precompute_lbp_cache.py --frame_gap 2 --cache_root lbp_cache/gap_2

Then train with e.g.:
    python train.py --cache_root lbp_cache/gap_2 --n_frames 3 ...

Varying --n_frames within a fixed cache is free (no recomputation needed).

Usage
-----
    # Default: gap=1, both train+val, default dataset path
    python precompute_lbp_cache.py

    # Custom gap and paths
    python precompute_lbp_cache.py \\
        --dataset_root ../smokeDetection_baseline_ecoWild/Dataset \\
        --cache_root   ../smokeDetection_baseline_ecoWild/lbp_cache/gap_2 \\
        --frame_gap    2

    # Only recompute train split
    python precompute_lbp_cache.py --splits train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from feature_extraction import make_lbp_motion_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def precompute(
    dataset_root: Path,
    cache_root: Path,
    splits: list[str],
    classes: list[str],
    frame_gap: int,
    target_size: tuple[int, int],
    overwrite: bool,
) -> None:
    total_pairs = 0
    skipped     = 0
    written     = 0

    for split in splits:
        for cls in classes:
            class_dir = dataset_root / split / cls
            if not class_dir.is_dir():
                print(f"  [skip] {class_dir} not found")
                continue

            video_dirs = sorted(d for d in class_dir.iterdir() if d.is_dir())
            print(f"\n{split}/{cls}: {len(video_dirs)} video sequences")

            for video_dir in tqdm(video_dirs, desc=f"  {split}/{cls}", unit="seq"):
                frame_paths = sorted(
                    p for p in video_dir.glob("*.[jp][pn]g")
                    if p.stat().st_size > 0
                )
                if len(frame_paths) < frame_gap + 1:
                    continue   # not enough frames for this gap

                cache_video_dir = cache_root / split / cls / video_dir.name
                cache_video_dir.mkdir(parents=True, exist_ok=True)

                # Iterate over all valid (frame[i], frame[i+frame_gap]) pairs
                for i in range(len(frame_paths) - frame_gap):
                    pair_idx   = i
                    cache_path = cache_video_dir / f"pair_{pair_idx:04d}.png"
                    total_pairs += 1

                    if cache_path.exists() and not overwrite:
                        skipped += 1
                        continue

                    p1 = frame_paths[i]
                    p2 = frame_paths[i + frame_gap]

                    f1_rgb = np.array(Image.open(p1).convert("RGB"))
                    f2_rgb = np.array(Image.open(p2).convert("RGB"))
                    f1_bgr = cv2.cvtColor(f1_rgb, cv2.COLOR_RGB2BGR)
                    f2_bgr = cv2.cvtColor(f2_rgb, cv2.COLOR_RGB2BGR)

                    lbp_rgb = make_lbp_motion_image(f1_bgr, f2_bgr, target_size)
                    Image.fromarray(lbp_rgb).save(cache_path, optimize=False)
                    written += 1

    print(f"\nDone.")
    print(f"  Total pairs : {total_pairs}")
    print(f"  Written     : {written}")
    print(f"  Skipped     : {skipped}  (already cached; use --overwrite to redo)")
    print(f"  Cache root  : {cache_root.resolve()}")


def main() -> None:
    default_dataset = Path(__file__).parent.parent / "smokeDetection_baseline_ecoWild" / "Dataset"
    default_cache   = Path(__file__).parent.parent / "smokeDetection_baseline_ecoWild" / "lbp_cache" / "gap_1"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_root", type=Path, default=default_dataset,
        help=f"Dataset root containing train/ and val/ splits (default: {default_dataset})",
    )
    parser.add_argument(
        "--cache_root", type=Path, default=default_cache,
        help=f"Where to write the cache (default: {default_cache})",
    )
    parser.add_argument(
        "--frame_gap", type=int, default=1,
        help="Gap between the two frames in each pair (default: 1 = adjacent frames)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which splits to process (default: train val)",
    )
    parser.add_argument(
        "--classes", nargs="+", default=["smoke", "no_smoke"],
        help="Which class folders to process (default: smoke no_smoke)",
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
        "--overwrite", action="store_true",
        help="Recompute and overwrite already-cached files (default: skip existing)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    cache_root   = args.cache_root.expanduser().resolve()

    if not dataset_root.is_dir():
        parser.error(f"Dataset root not found: {dataset_root}")

    print(f"Dataset root : {dataset_root}")
    print(f"Cache root   : {cache_root}")
    print(f"Frame gap    : {args.frame_gap}")
    print(f"Target size  : {args.width}x{args.height}")
    print(f"Splits       : {args.splits}")
    print(f"Overwrite    : {args.overwrite}")

    precompute(
        dataset_root=dataset_root,
        cache_root=cache_root,
        splits=args.splits,
        classes=args.classes,
        frame_gap=args.frame_gap,
        target_size=(args.width, args.height),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
