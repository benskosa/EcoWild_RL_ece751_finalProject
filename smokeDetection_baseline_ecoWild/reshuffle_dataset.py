"""
reshuffle_dataset.py
--------------------
Reshuffles the entire dataset into a 70% / 15% / 15% train/val/test split,
splitting by fire sequence (not by individual frame) so that no single fire
event appears in more than one split.

The script collects every sequence directory from ALL existing splits (train,
val, test), shuffles them with a fixed random seed, then moves them into
their new locations.  Any existing split structure is fully replaced.

Usage
-----
    # Preview what would happen without moving anything:
    python reshuffle_dataset.py --dataset_root Dataset --dry-run

    # Apply the reshuffle:
    python reshuffle_dataset.py --dataset_root Dataset

    # Custom split ratios:
    python reshuffle_dataset.py --dataset_root Dataset --train 0.70 --val 0.15
    # (test gets the remainder automatically)

    # Different random seed:
    python reshuffle_dataset.py --dataset_root Dataset --seed 123
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path


CLASSES  = ["smoke", "no_smoke"]
SPLITS   = ["train", "val", "test"]


def collect_sequences(dataset_root: Path, cls: str) -> list[Path]:
    """
    Gather every sequence directory for a class across all existing splits.
    Returns a sorted list of Path objects (the sequence dirs themselves).
    """
    sequences = []
    for split in SPLITS:
        class_dir = dataset_root / split / cls
        if not class_dir.is_dir():
            continue
        for seq_dir in class_dir.iterdir():
            if seq_dir.is_dir():
                sequences.append(seq_dir)
    return sorted(sequences)   # sort for determinism before shuffle


def compute_split_sizes(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    n_train = math.floor(n * train_ratio)
    n_val   = math.floor(n * val_ratio)
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test


def reshuffle(
    dataset_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    dry_run: bool,
) -> None:
    rng = random.Random(seed)

    print(f"Dataset root : {dataset_root.resolve()}")
    print(f"Split ratios : train={train_ratio:.0%}  val={val_ratio:.0%}  "
          f"test={1-train_ratio-val_ratio:.0%}")
    print(f"Random seed  : {seed}")
    print(f"Dry run      : {dry_run}\n")

    for cls in CLASSES:
        sequences = collect_sequences(dataset_root, cls)
        if not sequences:
            print(f"[{cls}] No sequences found — skipping.")
            continue

        rng.shuffle(sequences)
        n_train, n_val, n_test = compute_split_sizes(
            len(sequences), train_ratio, val_ratio
        )

        assignment: dict[str, list[Path]] = {
            "train": sequences[:n_train],
            "val":   sequences[n_train : n_train + n_val],
            "test":  sequences[n_train + n_val :],
        }

        print(f"[{cls}]  total={len(sequences)}  "
              f"train={n_train}  val={n_val}  test={n_test}")

        for split, seqs in assignment.items():
            dest_class_dir = dataset_root / split / cls
            for seq_dir in seqs:
                dest = dest_class_dir / seq_dir.name
                if seq_dir == dest:
                    # Already in the right place
                    if dry_run:
                        print(f"  [keep]  {split}/{cls}/{seq_dir.name}")
                    continue

                if dry_run:
                    print(f"  [move]  {seq_dir.parent.parent.name}/{seq_dir.parent.name}/"
                          f"{seq_dir.name}  ->  {split}/{cls}/{seq_dir.name}")
                else:
                    dest_class_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(seq_dir), str(dest))

        print()

    if not dry_run:
        # Clean up any empty split/class directories left behind
        for split in SPLITS:
            for cls in CLASSES:
                d = dataset_root / split / cls
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            split_dir = dataset_root / split
            if split_dir.is_dir() and not any(split_dir.iterdir()):
                split_dir.rmdir()

        print("Reshuffle complete. Final counts:")
        for split in SPLITS:
            for cls in CLASSES:
                d = dataset_root / split / cls
                if d.is_dir():
                    n_seqs  = sum(1 for p in d.iterdir() if p.is_dir())
                    n_frames = sum(
                        len(list(seq.glob("*.jpg")))
                        for seq in d.iterdir() if seq.is_dir()
                    )
                    print(f"  {split}/{cls}: {n_seqs} sequences, {n_frames} frames")
    else:
        print("Dry run complete — no files were moved.")
        print("Run without --dry-run to apply.")


def main() -> None:
    default_root = Path(__file__).parent / "Dataset"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_root", type=Path, default=default_root,
        help=f"Dataset root containing train/, val/, test/ (default: {default_root})",
    )
    parser.add_argument("--train", type=float, default=0.70, dest="train_ratio",
                        help="Fraction for training split (default: 0.70)")
    parser.add_argument("--val",   type=float, default=0.15, dest="val_ratio",
                        help="Fraction for validation split (default: 0.15)")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview moves without touching any files")
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train + --val must be less than 1.0 (test gets the remainder)")

    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    reshuffle(root, args.train_ratio, args.val_ratio, args.seed, args.dry_run)


if __name__ == "__main__":
    main()
