"""
yolov8_training.py
------------------
Fine-tunes YOLOv8-nano (or any YOLO variant) in classification mode for
binary smoke detection on the EcoWild dataset.

Our dataset has an extra fire_id subdirectory level that confuses YOLOv8:

    <data_root>/
      train/
        smoke/     <fire_id>/  *.jpg  ...
        no_smoke/  <fire_id>/  *.jpg  ...

This script automatically builds a temporary flat symlink tree that
YOLOv8 can read correctly (no files are copied or moved):

    <tmp>/
      train/
        smoke/     *.jpg  (symlinks to originals)
        no_smoke/  *.jpg

The temp directory is deleted when training finishes.

Usage
-----
    python yolov8_training.py \\
        --data_root ../Dataset \\
        --project   runs \\
        --name      yolov8n_baseline

    # Larger model, more epochs, custom batch:
    python yolov8_training.py \\
        --data_root ../Dataset \\
        --model     yolov8s-cls.pt \\
        --epochs    300 \\
        --batch     64 \\
        --patience  50
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path


SPLITS  = ["train", "val", "test"]
CLASSES = ["smoke", "no_smoke"]


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def build_flat_symlink_tree(data_root: Path) -> Path:
    """
    Create a temporary directory with a flat class/image structure for YOLOv8.

    Our layout:   split/class/<fire_id>/*.jpg
    YOLOv8 wants: split/class/*.jpg

    Symlinks are created so no data is duplicated.
    Returns the path to the temp root (caller must delete it when done).
    """
    tmp = Path(tempfile.mkdtemp(prefix="yolo_flat_"))
    n_links = 0
    for split in SPLITS:
        for cls in CLASSES:
            src_class = data_root / split / cls
            if not src_class.is_dir():
                continue
            dst_class = tmp / split / cls
            dst_class.mkdir(parents=True, exist_ok=True)
            for img in sorted(src_class.rglob("*.jpg")):
                # Use the full relative path as the filename to guarantee uniqueness
                unique_name = img.relative_to(src_class).as_posix().replace("/", "__")
                dst = dst_class / unique_name
                os.symlink(img, dst)
                n_links += 1
    print(f"Flat symlink tree built: {tmp}  ({n_links} symlinks)")
    return tmp


def train(args) -> None:
    # Import here so argparse --help works even without ultralytics installed
    from ultralytics import YOLO

    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"Model         : {args.model}")
    print(f"Data root     : {data_root}")
    print(f"Project dir   : {args.project}")
    print(f"Run name      : {args.name}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch}")
    print(f"Image size    : {args.imgsz}")
    print(f"Device        : {args.device}")
    print(f"Patience      : {args.patience}")
    print(f"Workers       : {args.workers}")

    # Build a temporary flat view of the dataset that YOLOv8 can parse correctly
    print("\nBuilding flat symlink tree for YOLOv8...")
    flat_root = build_flat_symlink_tree(data_root)

    model   = YOLO(args.model)
    t_start = time.time()
    print("\nSTART TRAINING")

    try:
        model.train(
            data     = str(flat_root),
            epochs   = args.epochs,
            batch    = args.batch,
            imgsz    = args.imgsz,
            device   = args.device,
            resume   = args.resume,
            save     = True,
            patience = args.patience,
            workers  = args.workers,
            project  = args.project,
            name     = args.name,
            optimizer= args.optimizer,
            plots    = True,
            exist_ok = True,
        )
    finally:
        # Always clean up the temp tree, even if training crashes
        shutil.rmtree(flat_root, ignore_errors=True)
        print(f"Temp symlink tree removed: {flat_root}")

    elapsed = fmt_time(time.time() - t_start)
    print(f"\nTraining complete.")
    print(f"  Results saved to : {args.project}/{args.name}")
    print(f"  Total time       : {elapsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Dataset root containing train/ and val/ (with smoke/ and no_smoke/ inside)",
    )
    parser.add_argument(
        "--model", default="yolov8n-cls.pt",
        help="YOLO model weights to start from (default: yolov8n-cls.pt). "
             "Other options: yolov8s-cls.pt, yolov8m-cls.pt, etc.",
    )
    parser.add_argument(
        "--project", default="runs",
        help="Parent directory for saving results (default: runs)",
    )
    parser.add_argument(
        "--name", default="yolov8_baseline",
        help="Sub-directory name inside --project for this run (default: yolov8_baseline)",
    )
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--batch",     type=int,   default=64,
                        help="Batch size (default: 64). Use -1 for auto.")
    parser.add_argument("--imgsz",     type=int,   default=224,
                        help="Input image size in pixels (default: 224)")
    parser.add_argument("--device",    default=None,
                        help="Device to use, e.g. '0', '0,1', 'cpu'. Auto-detected if omitted.")
    parser.add_argument("--patience",  type=int,   default=50,
                        help="Early stopping patience in epochs (default: 50)")
    parser.add_argument("--workers",   type=int,   default=4,
                        help="DataLoader worker processes (default: 4; use 0 on Windows)")
    parser.add_argument("--optimizer", default="auto",
                        help="Optimizer name (default: auto)")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume training from the last checkpoint")
    args = parser.parse_args()
    train(args)
