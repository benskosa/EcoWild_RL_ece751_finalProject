"""
train.py

Training and evaluation loop for the LBP-motion + MobileNet smoke detector.

Reproduces the experimental setup from Shi et al. (2020):
  - 500 epochs
  - Adam optimizer, lr=0.001
  - BCE loss (BCEWithLogitsLoss for numerical stability)
  - Batch size 32
  - Random weight initialisation (pretrained=False to match paper;
    set pretrained=True for faster convergence in practice)
  - Reports accuracy, TPR, PPV, FPR at the end of each epoch

Usage:
    python train.py --data_root /path/to/dataset --variant v3_small

Directory structure expected under --data_root:
    data_root/
      smoke/
          video_001/  frame_0001.jpg  frame_0002.jpg ...
          video_002/  ...
      no_smoke/
          video_001/  ...
          ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import SmokeDataset, build_model, get_transforms


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
    accuracy = (tp + tn) / total          if total  > 0 else 0.0
    tpr      = tp / (tp + fn)            if (tp + fn) > 0 else 0.0   # recall
    ppv      = tp / (tp + fp)            if (tp + fp) > 0 else 0.0   # precision
    fpr      = fp / (fp + tn)            if (fp + tn) > 0 else 0.0   # false alarm

    return {"accuracy": accuracy, "tpr": tpr, "ppv": ppv, "fpr": fpr}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    data_root: str,
    variant: str = "v3_small",
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.2,
    frame_gap: int = 1,
    pretrained: bool = False,   # set False to match the paper exactly
    save_dir: str = "checkpoints",
    device: str | None = None,
):
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # --- Datasets -----------------------------------------------------------
    train_transform = get_transforms(train=True)
    val_transform   = get_transforms(train=False)

    # Build the full dataset with train transforms; we will override for val
    full_dataset = SmokeDataset(
        root=data_root,
        frame_gap=frame_gap,
        transform=train_transform,
    )

    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply val-specific transform to the validation split
    val_ds.dataset = SmokeDataset(
        root=data_root,
        frame_gap=frame_gap,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # --- Model, loss, optimiser ---------------------------------------------
    model = build_model(variant=variant, pretrained=pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history: list[dict] = []

    # --- Epoch loop ---------------------------------------------------------
    for epoch in range(1, epochs + 1):

        # ---- Train ---------------------------------------------------------
        model.train()
        train_logits, train_labels = [], []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)       # (B,)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_logits.append(logits.detach().cpu())
            train_labels.append(labels.cpu())

        train_metrics = compute_metrics(
            torch.cat(train_labels), torch.cat(train_logits)
        )

        # ---- Validate ------------------------------------------------------
        model.eval()
        val_logits, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
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

        # ---- Save best checkpoint ------------------------------------------
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_acc":    best_val_acc,
                    "variant":    variant,
                },
                Path(save_dir) / "best_model.pt",
            )

    # ---- Save training history as JSON ------------------------------------
    with open(Path(save_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {save_dir}/best_model.pt")
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
        description="Train LBP-motion MobileNet smoke detector"
    )
    parser.add_argument("--data_root",  required=True,     help="Path to dataset root")
    parser.add_argument("--variant",    default="v3_small", choices=["v2", "v3_small"])
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--val_split",  type=float, default=0.2)
    parser.add_argument("--frame_gap",  type=int,   default=1)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights (recommended)")
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument("--device",     default=None,
                        help="e.g. 'cuda', 'cuda:0', 'cpu'")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        frame_gap=args.frame_gap,
        pretrained=args.pretrained,
        save_dir=args.save_dir,
        device=args.device,
    )
