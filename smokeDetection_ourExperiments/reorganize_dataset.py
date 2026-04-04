"""
reorganize_dataset.py
---------------------
Reorganizes a flat EcoWild smoke-detection dataset into per-fire subdirectories
so it is compatible with SmokeDataset() in model.py.

BEFORE:
    <root>/{split}/{class}/20160604_FIRE_rm-n-mobo-c_1465065600_+00000.jpg
    <root>/{split}/{class}/20160604_FIRE_rm-n-mobo-c_1465065660_+00060.jpg
    ...

AFTER:
    <root>/{split}/{class}/20160604_FIRE_rm-n-mobo-c/20160604_FIRE_rm-n-mobo-c_1465065600_+00000.jpg
    <root>/{split}/{class}/20160604_FIRE_rm-n-mobo-c/20160604_FIRE_rm-n-mobo-c_1465065660_+00060.jpg
    ...

Fire ID is derived from the filename by dropping the last two '_'-separated
fields (unix timestamp and signed offset), e.g.:
    "20160604_FIRE_rm-n-mobo-c_1465065600_+00000.jpg"
     └─── fire_id ───────────┘ └─unix ts──┘ └─off─┘
    → fire_id = "20160604_FIRE_rm-n-mobo-c"

Usage:
    python reorganize_dataset.py                          # default dataset path
    python reorganize_dataset.py --root path/to/Dataset   # custom path
    python reorganize_dataset.py --dry-run                # preview only
    python reorganize_dataset.py --copy                   # copy instead of move
    python reorganize_dataset.py --splits train val       # only certain splits
    python reorganize_dataset.py --classes smoke          # only certain classes
"""

import argparse
import os
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Matches: <fire_id>_<10-digit unix ts>_<offset>.<ext>
# Captures fire_id as group 1.
FILENAME_RE = re.compile(r"^(.+)_\d{10}_[+\-]\d+(\.\w+)$")


def parse_fire_id(filename: str):
    """Return the fire ID portion of an EcoWild image filename, or None if unrecognized."""
    m = FILENAME_RE.match(filename)
    if m:
        return m.group(1)
    # Fallback: drop last two underscore-delimited fields
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return None


def reorganize(root, splits, classes, dry_run, copy):
    op = shutil.copy2 if copy else shutil.move

    total_files = 0
    total_fires = 0
    skipped = 0

    for split in splits:
        for cls in classes:
            class_dir = root / split / cls
            if not class_dir.is_dir():
                print(f"  [skip] {class_dir} - not found")
                continue

            flat_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if not flat_files:
                print(f"  [info] {class_dir} - no flat images found (already organized?)")
                continue

            # Group by fire ID
            groups = {}
            for fpath in flat_files:
                fire_id = parse_fire_id(fpath.name)
                if fire_id is None:
                    print(f"  [warn] cannot parse fire ID from: {fpath.name}")
                    skipped += 1
                    continue
                groups.setdefault(fire_id, []).append(fpath)

            print(f"\n{split}/{cls}: {len(flat_files)} images -> {len(groups)} fire sequence(s)")
            total_fires += len(groups)

            for fire_id, frames in groups.items():
                dest_dir = class_dir / fire_id
                frames_sorted = sorted(frames, key=lambda p: p.name)

                if dry_run:
                    print(f"  [dry-run] would create {dest_dir.name}/ "
                          f"with {len(frames_sorted)} frames")
                else:
                    dest_dir.mkdir(exist_ok=True)
                    for fpath in frames_sorted:
                        op(str(fpath), str(dest_dir / fpath.name))

                total_files += len(frames_sorted)

    print(f"\n{'[DRY RUN] Would process' if dry_run else 'Processed'} "
          f"{total_files} images across {total_fires} fire sequence(s) "
          f"({skipped} skipped).")
    if dry_run:
        print("Run without --dry-run to apply changes.")


def main():
    default_root = Path(__file__).parent.parent / "smokeDetection_baseline_ecoWild" / "Dataset"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=default_root,
                        help=f"Dataset root directory (default: {default_root})")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to process (default: train val test)")
    parser.add_argument("--classes", nargs="+", default=["smoke", "no_smoke"],
                        help="Class folders to process (default: smoke no_smoke)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would happen without moving any files")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them (preserves originals)")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset root not found: {root}")

    print(f"Dataset root : {root}")
    print(f"Splits       : {args.splits}")
    print(f"Classes      : {args.classes}")
    print(f"Operation    : {'copy' if args.copy else 'move'}"
          f"{' (dry run)' if args.dry_run else ''}")

    reorganize(root, args.splits, args.classes, args.dry_run, args.copy)


if __name__ == "__main__":
    main()
