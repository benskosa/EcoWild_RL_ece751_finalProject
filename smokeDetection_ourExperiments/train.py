"""
train.py

Training and evaluation loop for the LBP-motion + MobileNet smoke detector.

Reproduces the experimental setup from Shi et al. (2020):
  - 500 epochs
  - Adam optimizer, lr=0.001
  - BCE loss (BCEWithLogitsLoss for numerical stability)
  - Batch size 32
  - Reports accuracy, TPR, PPV, FPR at the end of each epoch

Extended features vs. paper:
  - Accepts separate --train_root and --val_root (mirrors the pre-split
    Dataset/train/ and Dataset/val/ directory structure from EcoWild).
  - --n_frames N  uses N consecutive frames per LBP-motion image (N=2
    reproduces the paper; N>2 averages N-1 pairwise LBP images).
  - Auto-detects safe num_workers (0 on Windows to avoid multiprocessing
    deadlocks with the default 'spawn' start method).

Usage:
    # Recommended — use pre-split train/val directories:
    python train.py --train_root ../smokeDetection_baseline_ecoWild/Dataset/train \\
                    --val_root   ../smokeDetection_baseline_ecoWild/Dataset/val

    # Paper-exact settings (MobileNetV2, no pretrained weights):
    python train.py --train_root ... --val_root ... --variant v2 --epochs 500

    # Experiment with 4-frame temporal window:
    python train.py --train_root ... --val_root ... --n_frames 4

    # Quick sanity check (few epochs, pretrained weights):
    python train.py --train_root ... --val_root ... --epochs 5 --pretrained

Directory structure expected under --train_root / --val_root:
    <root>/
      smoke/
          <fire_id>/   frame_0001.jpg  frame_0002.jpg  ...
          ...
      no_smoke/
          <fire_id>/   ...
          ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import SmokeDataset, build_model, get_transforms


# ---------------------------------------------------------------------------
# Safe num_workers: Windows 'spawn' start method deadlocks with workers > 0
# inside a script that is not protected by  if __name__ == "__main__"
# ---------------------------------------------------------------------------
def safe_num_workers(requested: int) -> int:
    if platform.system() == "Windows":
        return 0
    return requested


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: torch.Tensor,
    y_pred_logits: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute accuracy, TPR (sensitivity), PPV (precision), and FPR.

    Parameters
    ----------
    y_true         : 1-D tensor of ground-truth labels {0, 1}
    y_pred_logits  : 1-D tensor of raw model outputs (before sigmoid)
    threshold      : decision threshold applied after sigmoid
    """
    probs = torch.sigmoid(y_pred_logits)
    preds = (probs >= threshold).float()

    tp = ((preds == 1) & (y_true == 1)).sum().item()
    tn = ((preds == 0) & (y_true == 0)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total       if total       > 0 else 0.0
    tpr      = tp / (tp + fn)          if (tp + fn)   > 0 else 0.0   # recall
    ppv      = tp / (tp + fp)          if (tp + fp)   > 0 else 0.0   # precision
    fpr      = fp / (fp + tn)          if (fp + tn)   > 0 else 0.0   # false alarm

    return {"accuracy": accuracy, "tpr": tpr, "ppv": ppv, "fpr": fpr}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    train_root: str,
    val_root: str | None = None,
    variant: str = "v3_small",
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.2,
    n_frames: int = 2,
    frame_gap: int = 1,
    pretrained: bool = False,
    save_dir: str = "checkpoints",
    num_workers: int = 4,
    device: str | None = None,
    cache_root: str | None = None,
    preload_cache: bool = False,
    run_name: str = "best_model",
):
    """
    Parameters
    ----------
    train_root  : path to training split root (smoke/ and no_smoke/ inside)
    val_root    : path to validation split root.  If None, val_split fraction
                  of the training set is held out as validation.
    variant     : "v3_small" or "v2" (paper uses v2)
    epochs      : training epochs
    batch_size  : mini-batch size
    lr          : Adam learning rate
    val_split   : fraction of train set used for val when val_root is None
    n_frames    : consecutive frames per LBP-motion sample (2 = paper default)
    frame_gap   : stride between frames in each window (1 = adjacent)
    pretrained  : use ImageNet pretrained weights (speeds up convergence)
    save_dir    : directory for checkpoints and history JSON
    num_workers : DataLoader workers (auto-capped to 0 on Windows)
    device      : torch device string; auto-detected if None
    cache_root    : path to pre-computed pairwise LBP cache (from
                    precompute_lbp_cache.py).  Speeds up training dramatically
                    by skipping on-the-fly optical flow + LBP computation.
                    Changing n_frames when using the cache is nearly free.
    preload_cache : load all cached pair PNGs into RAM at startup.  Eliminates
                    disk I/O for every subsequent epoch.  Recommended on servers
                    with >=8 GB free RAM.  Ignored if cache_root is not set.
    """
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device  : {device}")
    print(f"n_frames      : {n_frames}  (frame_gap={frame_gap})")
    print(f"Variant       : {variant}   pretrained={pretrained}")
    print(f"Cache root    : {cache_root if cache_root else 'None (on-the-fly)'}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    workers = safe_num_workers(num_workers)

    # --- Datasets -----------------------------------------------------------
    train_transform = get_transforms(train=True)
    val_transform   = get_transforms(train=False)

    train_ds = SmokeDataset(
        root=train_root,
        n_frames=n_frames,
        frame_gap=frame_gap,
        transform=train_transform,
        cache_root=cache_root,
        preload_cache=preload_cache,
    )

    if val_root is not None:
        # Use the pre-split validation directory (recommended)
        val_ds = SmokeDataset(
            root=val_root,
            n_frames=n_frames,
            frame_gap=frame_gap,
            transform=val_transform,
            cache_root=cache_root,
            preload_cache=preload_cache,
        )
        print(f"Train root    : {train_root}")
        print(f"Val root      : {val_root}")
    else:
        # Fall back to random split from the training set
        n_val   = int(len(train_ds) * val_split)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        # Rebuild a clean val dataset with val transforms (no augmentation)
        val_ds_clean = SmokeDataset(
            root=train_root,
            n_frames=n_frames,
            frame_gap=frame_gap,
            transform=val_transform,
            cache_root=cache_root,
            preload_cache=preload_cache,
        )
        val_ds.dataset = val_ds_clean   # redirect val subset to no-aug dataset
        print(f"Train root    : {train_root}  (internal {val_split:.0%} val split)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")

    # --- Model, loss, optimiser ---------------------------------------------
    model = build_model(variant=variant, pretrained=pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_val_tpr = 0.0
    history: list[dict] = []
    t_start = time.time()

    # --- Epoch loop ---------------------------------------------------------
    for epoch in range(1, epochs + 1):

        # ---- Train ---------------------------------------------------------
        model.train()
        train_logits, train_labels = [], []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{epochs} [train]",
                         leave=False, unit="batch")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)       # (B,)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_logits.append(logits.detach().cpu())
            train_labels.append(labels.cpu())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_metrics = compute_metrics(
            torch.cat(train_labels), torch.cat(train_logits)
        )

        # ---- Validate ------------------------------------------------------
        model.eval()
        val_logits, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:4d}/{epochs} [val]  ",
                                     leave=False, unit="batch"):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs).squeeze(1)
                val_logits.append(logits.cpu())
                val_labels.append(labels.cpu())

        val_metrics = compute_metrics(
            torch.cat(val_labels), torch.cat(val_logits)
        )

        # ---- Logging -------------------------------------------------------
        row = {
            "epoch":      epoch,
            "train_acc":  train_metrics["accuracy"],
            "val_acc":    val_metrics["accuracy"],
            "val_tpr":    val_metrics["tpr"],
            "val_ppv":    val_metrics["ppv"],
            "val_fpr":    val_metrics["fpr"],
        }
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"TPR={val_metrics['tpr']:.4f} | "
                f"PPV={val_metrics['ppv']:.4f} | "
                f"FPR={val_metrics['fpr']:.4f}"
            )

        # ---- Save best checkpoints -----------------------------------------
        ckpt = {
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "val_acc":    val_metrics["accuracy"],
            "val_tpr":    val_metrics["tpr"],
            "variant":    variant,
            "n_frames":   n_frames,
            "frame_gap":  frame_gap,
        }

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(ckpt, Path(save_dir) / f"{run_name}_best_acc.pt")

        if val_metrics["tpr"] > best_val_tpr:
            best_val_tpr = val_metrics["tpr"]
            torch.save(ckpt, Path(save_dir) / f"{run_name}_best_tpr.pt")

    # ---- Save training history as JSON -------------------------------------
    with open(Path(save_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)

    print(f"\nTraining complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}  -> {save_dir}/{run_name}_best_acc.pt")
    print(f"  Best val TPR      : {best_val_tpr:.4f}  -> {save_dir}/{run_name}_best_tpr.pt")
    print(f"  Total training time: {h:02d}h {m:02d}m {s:02d}s")
    return history


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def predict(
    model: nn.Module,
    lbp_motion_image,            # PIL Image or np.ndarray (H, W, 3) RGB
    device: str = "cpu",
    threshold: float = 0.5,
) -> tuple[int, float]:
    """
    Run inference on a single LBP-motion image.

    Returns
    -------
    label       : int  (1 = smoke detected, 0 = no smoke)
    probability : float in [0, 1]
    """
    from PIL import Image as PILImage
    import numpy as np

    if isinstance(lbp_motion_image, np.ndarray):
        img = PILImage.fromarray(lbp_motion_image)
    else:
        img = lbp_motion_image

    transform = get_transforms(train=False)
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob  = torch.sigmoid(logit).item()

    label = int(prob >= threshold)
    return label, prob


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LBP-motion MobileNet smoke detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset paths
    parser.add_argument(
        "--train_root", required=True,
        help="Path to training split root (must contain smoke/ and no_smoke/)",
    )
    parser.add_argument(
        "--val_root", default=None,
        help=(
            "Path to validation split root.  "
            "If omitted, --val_split fraction of training data is held out."
        ),
    )

    # Model
    parser.add_argument("--variant",    default="v3_small", choices=["v2", "v3_small"],
                        help="MobileNet variant (v2 = paper, v3_small = recommended)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights (recommended for fast convergence)")

    # Training hyper-parameters
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Fraction held out for val when --val_root is not given")

    # Temporal window
    parser.add_argument("--n_frames",   type=int,   default=2,
                        help=(
                            "Consecutive frames per LBP-motion sample. "
                            "2 = paper default (one pair); >2 averages N-1 pairwise images."
                        ))
    parser.add_argument("--frame_gap",  type=int,   default=1,
                        help="Stride between frames within each window (1 = adjacent)")

    # Cache
    parser.add_argument(
        "--preload_cache", action="store_true",
        help=(
            "Load all cached pair PNGs into RAM at startup.  Eliminates disk "
            "I/O after the first epoch.  Requires ~2.5 GB free RAM for the "
            "full train split.  Only effective when --cache_root is set."
        ),
    )
    parser.add_argument(
        "--cache_root", default=None,
        help=(
            "Path to pre-computed pairwise LBP cache (built by "
            "precompute_lbp_cache.py).  Highly recommended for repeated "
            "training runs or n_frames experiments — skips on-the-fly "
            "optical flow and LBP computation entirely."
        ),
    )

    # Runtime
    parser.add_argument("--save_dir",  default="checkpoints")
    parser.add_argument("--run_name",  default="best_model",
                        help="Filename stem for the saved checkpoint (default: best_model → best_model.pt)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (auto-set to 0 on Windows)")
    parser.add_argument("--device",      default=None,
                        help="e.g. 'cuda', 'cuda:0', 'cpu'. Auto-detected if omitted.")

    args = parser.parse_args()

    train(
        train_root=args.train_root,
        val_root=args.val_root,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        n_frames=args.n_frames,
        frame_gap=args.frame_gap,
        pretrained=args.pretrained,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        device=args.device,
        cache_root=args.cache_root,
        preload_cache=args.preload_cache,
        run_name=args.run_name,
    )
