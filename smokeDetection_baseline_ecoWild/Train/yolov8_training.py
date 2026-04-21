"""
yolov8_training.py
------------------
Fine-tunes YOLOv8-nano (or any YOLO variant) in classification mode for
binary smoke detection on the EcoWild dataset.

YOLOv8 classification expects the dataset root to contain class-named
subdirectories under each split:

    <data_root>/
      train/
        smoke/     <fire_id>/  *.jpg  ...
        no_smoke/  <fire_id>/  *.jpg  ...
      val/
        smoke/     <fire_id>/  *.jpg  ...
        no_smoke/  <fire_id>/  *.jpg  ...

YOLOv8 finds images recursively, so the extra fire_id subdirectory level
is handled automatically.

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
import time
from pathlib import Path


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


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

    model = YOLO(args.model)

    t_start = time.time()
    print("\nSTART TRAINING")

    model.train(
        data     = str(data_root),
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
