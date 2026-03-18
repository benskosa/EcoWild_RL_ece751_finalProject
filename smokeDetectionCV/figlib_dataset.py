"""
figlib_dataset.py

FIgLib-aware PyTorch Dataset for the LBP-motion smoke detector.

FIgLib (Fire Ignition Library) differences from a generic video dataset:
  1. Images are still frames captured at ~1-minute intervals, NOT a continuous
     video — consecutive images may have gaps larger than 1 minute.
  2. File names encode timestamps, e.g.:
         20190601_000104_Rm_HPWREN.mp4.00001.jpg
         (YYYYMMDD_HHMMSS_<tag>_<station>.mp4.<frame>.jpg)
  3. Each sequence (fire or no-fire clip) lives in its own sub-folder under
     smoke/ or no_smoke/.
  4. Because optical flow is meaningless across sequence boundaries or across
     long time gaps, this dataset:
       a. Only pairs frames from the SAME sub-folder (same event / camera).
       b. Skips pairs whose timestamp difference exceeds max_gap_minutes.

Supported layout variants
--------------------------
Variant A — sub-folder per sequence (preferred):
    root/
        smoke/
            20190601_Rm_HPWREN/
                20190601_000104_Rm_HPWREN.mp4.00001.jpg
                20190601_000204_Rm_HPWREN.mp4.00002.jpg
                ...
        no_smoke/
            20190701_Bj_HPWREN/
                ...

Variant B — flat folders with no sub-folders (all images in smoke/ directly):
    root/
        smoke/   img_001.jpg  img_002.jpg  ...
        no_smoke/  ...
    → All images in a flat folder are treated as a SINGLE sequence and paired
      in sorted order.  Pass flat_layout=True to enable this variant.

Timestamp parsing
-----------------
The dataset tries to extract a datetime from each filename using two strategies:
  1. strptime pattern "%Y%m%d_%H%M%S" (FIgLib native).
  2. Fall back to lexicographic sort (no filtering by time gap).

If timestamps cannot be parsed, frame pairs are formed in sorted filename order
and max_gap_minutes filtering is disabled.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from feature_extraction import make_lbp_motion_image


# ---------------------------------------------------------------------------
# Timestamp extraction
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"(\d{8})_(\d{6})")   # matches YYYYMMDD_HHMMSS anywhere in filename


def _parse_timestamp(path: Path) -> Optional[datetime]:
    """
    Try to extract a datetime from a FIgLib-style filename.
    Returns None if no timestamp pattern is found.
    """
    m = _TS_RE.search(path.name)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1) + "_" + m.group(2), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FIgLibDataset(Dataset):
    """
    PyTorch Dataset that builds LBP-motion images from consecutive FIgLib
    still-frame pairs.

    Parameters
    ----------
    root : str
        Dataset root containing smoke/ and no_smoke/ sub-directories.
    transform : transforms.Compose, optional
        Applied to each LBP-motion PIL Image before returning.
    max_gap_minutes : float
        If both frames have parseable timestamps and their gap exceeds this
        value, the pair is skipped.  Use float('inf') to disable filtering.
        Default: 5 minutes  (skips across clip boundaries or missing frames).
    flat_layout : bool
        If True, treat all images in smoke/ and no_smoke/ as single flat
        sequences (no sub-folder structure).
    precomputed : bool
        If True, images under smoke/ and no_smoke/ are already LBP-motion
        images — skip feature extraction and load them directly.  In this
        mode, max_gap_minutes and flat_layout are irrelevant.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        max_gap_minutes: float = 5.0,
        flat_layout: bool = False,
        precomputed: bool = False,
    ):
        self.root            = Path(root)
        self.transform       = transform
        self.max_gap_seconds = max_gap_minutes * 60.0
        self.flat_layout     = flat_layout
        self.precomputed     = precomputed

        # Each element: (path_or_pair, label)
        # path_or_pair is either a single Path (precomputed) or (Path, Path) pair
        self.samples: list[tuple] = []

        for label, class_dir in enumerate(["no_smoke", "smoke"]):
            class_path = self.root / class_dir
            if not class_path.exists():
                raise FileNotFoundError(f"Expected directory: {class_path}")

            if precomputed:
                for img_path in sorted(class_path.rglob("*.[jp][pn]g")):
                    self.samples.append((img_path, label))
            elif flat_layout:
                # All images in the class dir are one flat sequence
                seqs = [sorted(class_path.glob("*.[jp][pn]g"))]
                self._add_pairs_from_sequences(seqs, label)
            else:
                # Sub-folders = individual sequences
                seq_dirs = sorted(d for d in class_path.iterdir() if d.is_dir())
                if seq_dirs:
                    seqs = [sorted(d.glob("*.[jp][pn]g")) for d in seq_dirs]
                else:
                    # Flat fallback: no sub-dirs found
                    seqs = [sorted(class_path.glob("*.[jp][pn]g"))]
                self._add_pairs_from_sequences(seqs, label)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found under {root}. "
                "Check the directory layout and max_gap_minutes setting."
            )

    def _add_pairs_from_sequences(
        self,
        sequences: list[list[Path]],
        label: int,
    ) -> None:
        """
        For each sequence (ordered list of frame paths), form consecutive pairs
        and add valid ones to self.samples.
        """
        for frame_paths in sequences:
            frame_paths = list(frame_paths)
            if len(frame_paths) < 2:
                continue

            # Try to extract timestamps for gap filtering
            timestamps = [_parse_timestamp(p) for p in frame_paths]
            has_timestamps = all(t is not None for t in timestamps)

            # Sort by timestamp if available, otherwise keep sorted filename order
            if has_timestamps:
                paired = sorted(zip(timestamps, frame_paths), key=lambda x: x[0])
                timestamps = [p[0] for p in paired]
                frame_paths = [p[1] for p in paired]

            for i in range(len(frame_paths) - 1):
                p1, p2 = frame_paths[i], frame_paths[i + 1]

                # Skip if time gap is too large
                if has_timestamps:
                    gap = (timestamps[i + 1] - timestamps[i]).total_seconds()
                    if gap > self.max_gap_seconds or gap < 0:
                        continue  # Cross-boundary or out-of-order

                self.samples.append(((p1, p2), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample, label = self.samples[idx]

        if self.precomputed:
            img = Image.open(sample).convert("RGB")
        else:
            path1, path2 = sample
            # Load as BGR (cv2-compatible) via PIL→numpy→bgr
            frame1 = np.array(Image.open(path1).convert("RGB"))
            frame2 = np.array(Image.open(path2).convert("RGB"))
            f1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            f2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
            lbp_motion = make_lbp_motion_image(f1_bgr, f2_bgr)
            img = Image.fromarray(lbp_motion)   # already RGB

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

    # --- Introspection helpers ----------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the dataset split."""
        n_smoke    = sum(1 for _, lbl in self.samples if lbl == 1)
        n_no_smoke = sum(1 for _, lbl in self.samples if lbl == 0)
        return (
            f"FIgLibDataset: {len(self.samples)} frame pairs  "
            f"(smoke={n_smoke}, no_smoke={n_no_smoke})"
        )

    @staticmethod
    def describe_layout(root: str) -> None:
        """
        Print the discovered structure under root to help diagnose layout issues.
        """
        root_path = Path(root)
        for class_dir in ["smoke", "no_smoke"]:
            class_path = root_path / class_dir
            if not class_path.exists():
                print(f"  MISSING: {class_path}")
                continue
            sub_dirs = [d for d in class_path.iterdir() if d.is_dir()]
            flat_imgs = list(class_path.glob("*.[jp][pn]g"))
            print(
                f"  {class_dir}/  →  {len(sub_dirs)} sub-dirs, "
                f"{len(flat_imgs)} flat images"
            )
            for d in sub_dirs[:5]:
                imgs = list(d.glob("*.[jp][pn]g"))
                ts_ok = sum(1 for p in imgs if _parse_timestamp(p) is not None)
                print(f"    {d.name}/  {len(imgs)} images  ({ts_ok} with parseable timestamps)")
            if len(sub_dirs) > 5:
                print(f"    ... and {len(sub_dirs) - 5} more sub-dirs")


# ---------------------------------------------------------------------------
# Quick smoke test / CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a FIgLib dataset layout")
    parser.add_argument("root", help="Dataset root (contains smoke/ and no_smoke/)")
    parser.add_argument("--max_gap", type=float, default=5.0,
                        help="Max gap in minutes between paired frames (default: 5)")
    parser.add_argument("--flat", action="store_true",
                        help="Use flat layout (no sub-folder sequences)")
    args = parser.parse_args()

    print("Layout inspection:")
    FIgLibDataset.describe_layout(args.root)

    print("\nBuilding dataset...")
    ds = FIgLibDataset(root=args.root, max_gap_minutes=args.max_gap, flat_layout=args.flat)
    print(ds.summary())

    # Show a few sample paths
    for i in range(min(3, len(ds.samples))):
        (p1, p2), lbl = ds.samples[i]
        t1 = _parse_timestamp(p1)
        t2 = _parse_timestamp(p2)
        gap = (t2 - t1).total_seconds() / 60 if (t1 and t2) else "N/A"
        print(f"  [{i}] label={lbl}  gap={gap}min  {p1.name} → {p2.name}")
