"""
simple_resnet.py
----------------
Fine-tunes a pretrained ResNet34 binary smoke classifier on the EcoWild
dataset.

Usage
-----
    python simple_resnet.py \\
        --train_root ../Dataset/train \\
        --val_root   ../Dataset/val \\
        --save_dir   checkpoints \\
        --run_name   resnet34_baseline

    # Fewer epochs, larger batch:
    python simple_resnet.py \\
        --train_root ../Dataset/train \\
        --val_root   ../Dataset/val \\
        --epochs 100 --batch_size 64

Directory structure expected under --train_root / --val_root:
    <root>/
      smoke/       <fire_id>/  *.jpg  ...
      no_smoke/    <fire_id>/  *.jpg  ...
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_num_workers(requested: int) -> int:
    return 0 if platform.system() == "Windows" else requested


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def compute_metrics(conf_matrix) -> dict[str, float]:
    """Derive accuracy, TPR, FPR, PPV from a 2x2 confusion matrix."""
    # ImageFolder sorts classes alphabetically: no_smoke=0, smoke=1
    tn, fp = int(conf_matrix[0, 0]), int(conf_matrix[0, 1])
    fn, tp = int(conf_matrix[1, 0]), int(conf_matrix[1, 1])

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total       if total       > 0 else 0.0
    tpr      = tp / (tp + fn)          if (tp + fn)   > 0 else 0.0
    fpr      = fp / (fp + tn)          if (fp + tn)   > 0 else 0.0
    ppv      = tp / (tp + fp)          if (tp + fp)   > 0 else 0.0
    f1       = 2*ppv*tpr / (ppv + tpr) if (ppv + tpr) > 0 else 0.0

    return {"accuracy": accuracy, "tpr": tpr, "fpr": fpr, "ppv": ppv, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    bar = tqdm(loader, desc=f"Epoch {epoch:4d}/{epochs} [train]",
               leave=False, unit="batch")
    for inputs, labels in bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return epoch_loss, compute_metrics(cm)


def val_epoch(model, loader, criterion, device, epoch, epochs):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader,
                                   desc=f"Epoch {epoch:4d}/{epochs} [val]  ",
                                   leave=False, unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return epoch_loss, compute_metrics(cm)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args) -> None:
    device  = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    workers = safe_num_workers(args.num_workers)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device  : {device}")
    print(f"Train root    : {args.train_root}")
    print(f"Val root      : {args.val_root}")

    # --- Transforms ---------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # --- Datasets -----------------------------------------------------------
    # ImageFolder reads class labels from subfolder names.
    # Alphabetical order → no_smoke=0, smoke=1  (matches SmokeDataset)
    train_ds = ImageFolder(root=args.train_root, transform=train_transform)
    val_ds   = ImageFolder(root=args.val_root,   transform=val_transform)

    print(f"Train samples : {len(train_ds)}  "
          f"(classes: {train_ds.class_to_idx})")
    print(f"Val samples   : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )

    # --- Model --------------------------------------------------------------
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)   # binary: no_smoke / smoke
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ------------------------------------------------------
    best_val_acc = 0.0
    best_val_tpr = 0.0
    early_stop_counter = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_m = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_m = val_epoch(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "train_acc":  round(train_m["accuracy"], 6),
            "val_acc":    round(val_m["accuracy"],   6),
            "val_tpr":    round(val_m["tpr"],        6),
            "val_fpr":    round(val_m["fpr"],        6),
            "val_ppv":    round(val_m["ppv"],        6),
            "val_f1":     round(val_m["f1"],         6),
        }
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_acc={train_m['accuracy']:.4f} | "
                f"val_acc={val_m['accuracy']:.4f} | "
                f"TPR={val_m['tpr']:.4f} | "
                f"FPR={val_m['fpr']:.4f} | "
                f"PPV={val_m['ppv']:.4f}"
            )
            # Checkpoint history every 10 epochs so it survives early kills
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        # Save best-accuracy checkpoint
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            early_stop_counter = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "val_acc": best_val_acc, "val_tpr": val_m["tpr"],
                 "model": "resnet34"},
                save_dir / f"{args.run_name}_best_acc.pt",
            )
        else:
            early_stop_counter += 1

        # Save best-TPR checkpoint
        if val_m["tpr"] > best_val_tpr:
            best_val_tpr = val_m["tpr"]
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "val_acc": val_m["accuracy"], "val_tpr": best_val_tpr,
                 "model": "resnet34"},
                save_dir / f"{args.run_name}_best_tpr.pt",
            )

        # Early stopping
        if early_stop_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no val_acc improvement for {args.patience} epochs).")
            break

    # --- Save history -------------------------------------------------------
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = fmt_time(time.time() - t_start)
    print(f"\nTraining complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}  "
          f"-> {save_dir}/{args.run_name}_best_acc.pt")
    print(f"  Best val TPR      : {best_val_tpr:.4f}  "
          f"-> {save_dir}/{args.run_name}_best_tpr.pt")
    print(f"  Total time        : {elapsed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train_root", required=True,
                        help="Training split root (smoke/ and no_smoke/ inside)")
    parser.add_argument("--val_root",   required=True,
                        help="Validation split root")
    parser.add_argument("--save_dir",   default="checkpoints",
                        help="Directory for checkpoints and history.json")
    parser.add_argument("--run_name",   default="resnet34_baseline",
                        help="Checkpoint filename stem")
    parser.add_argument("--epochs",     type=int,   default=1000)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.0001)
    parser.add_argument("--patience",   type=int,   default=50,
                        help="Early stopping patience (epochs without val_acc improvement)")
    parser.add_argument("--num_workers", type=int,  default=4)
    parser.add_argument("--device",     default=None,
                        help="e.g. 'cuda', 'cpu'. Auto-detected if omitted.")
    args = parser.parse_args()
    train(args)
