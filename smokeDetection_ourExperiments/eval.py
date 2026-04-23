"""
eval.py

Standalone evaluation script for the LBP-motion + MobileNet smoke detector.

Produces:
  - Confusion matrix at a given threshold
  - Full ROC curve + AUC  (saved as PNG)
  - TPR / FPR operating points for EcoWild config comparison
  - Two-stage pipeline effective metrics (Setup 2):
        TPR_sys = TPR_gate * TPR_ensemble
        FPR_sys = FPR_gate * FPR_ensemble
  - (Optional) Updated EcoWild config JSON files for Setup 1 and Setup 2

Usage
-----
# Basic evaluation (Setup 1 — standalone gate):
    python eval.py --checkpoint checkpoints/best_model.pt \\
                   --data_root  /path/to/test_data

# Two-stage evaluation (Setup 2 — gate + heavy ensemble):
    python eval.py --checkpoint checkpoints/best_model.pt \\
                   --data_root  /path/to/test_data \\
                   --mode       two_stage \\
                   --ensemble_tpr 0.90 \\
                   --ensemble_fpr 0.58

# Generate RL config JSONs for both setups:
    python eval.py --checkpoint checkpoints/best_model.pt \\
                   --data_root  /path/to/test_data \\
                   --gen_configs \\
                   --template_config /path/to/config_setup.json

Data layout expected under --data_root (same as train.py):
    data_root/
        smoke/
            video_001/   frame_0001.jpg  frame_0002.jpg  ...
            video_002/   ...
        no_smoke/
            video_001/   ...

    or, if using pre-computed LBP-motion images:
        data_root/
            smoke/     img_001.jpg  img_002.jpg  ...
            no_smoke/  img_001.jpg  ...
    (pass --precomputed in that case)

    For FIgLib data, use figlib_dataset.py instead of this loader.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SmokeDataset, build_model, get_transforms


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a DataLoader and collect raw probabilities + true labels.

    Returns
    -------
    probs  : np.ndarray  shape (N,)  sigmoid probabilities in [0, 1]
    labels : np.ndarray  shape (N,)  ground-truth {0, 1}
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Evaluating", unit="batch"):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            p = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(p)
            all_labels.append(lbls.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def threshold_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return accuracy / TPR / FPR / PPV at a specific threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total          if total       > 0 else 0.0
    tpr      = tp / (tp + fn)             if (tp + fn)   > 0 else 0.0
    fpr      = fp / (fp + tn)             if (fp + tn)   > 0 else 0.0
    ppv      = tp / (tp + fp)             if (tp + fp)   > 0 else 0.0
    f1       = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy":  accuracy,
        "tpr":       tpr,
        "fpr":       fpr,
        "ppv":       ppv,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def two_stage_metrics(
    gate_tpr: float,
    gate_fpr: float,
    ensemble_tpr: float,
    ensemble_fpr: float,
) -> dict[str, float]:
    """
    Compute effective TPR / FPR of the two-stage pipeline (Setup 2).

    Model:
        - Gate runs first.  If gate=0, final decision=0 (no smoke).
        - If gate=1, heavy ensemble runs; final decision = ensemble output.

    Therefore:
        TPR_sys = P(gate=1 | fire)   * P(ensemble=1 | fire)
                = gate_tpr           * ensemble_tpr

        FPR_sys = P(gate=1 | no-fire) * P(ensemble=1 | no-fire)
                = gate_fpr             * ensemble_fpr

    Also estimates the fraction of frames that reach the heavy ensemble
    (gate pass rate = gate_fpr on negatives, gate_tpr on positives —
    averaged assuming balanced classes as a conservative estimate).
    """
    sys_tpr = gate_tpr * ensemble_tpr
    sys_fpr = gate_fpr * ensemble_fpr

    # Gate pass rate (fraction of frames that trigger heavy ensemble)
    # Assuming class balance for estimation:
    gate_pass_rate_pos = gate_tpr   # positives that reach ensemble
    gate_pass_rate_neg = gate_fpr   # negatives that reach ensemble
    gate_pass_rate_avg = 0.5 * (gate_pass_rate_pos + gate_pass_rate_neg)

    return {
        "gate_tpr":          gate_tpr,
        "gate_fpr":          gate_fpr,
        "ensemble_tpr":      ensemble_tpr,
        "ensemble_fpr":      ensemble_fpr,
        "system_tpr":        sys_tpr,
        "system_fpr":        sys_fpr,
        "gate_pass_rate_pos": gate_pass_rate_pos,
        "gate_pass_rate_neg": gate_pass_rate_neg,
        "gate_pass_rate_avg": gate_pass_rate_avg,
        "ensemble_calls_saved_pct": (1 - gate_pass_rate_avg) * 100,
    }


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

def plot_roc(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    gate_threshold: float | None = None,
    ecowild_tpr: float = 0.90,
    ecowild_fpr: float = 0.58,
) -> float:
    """
    Plot ROC curve and save to disk.

    Optionally marks:
      - The chosen operating threshold on the curve
      - EcoWild's SD-time baseline point (TPR=0.90, FPR=0.58)

    Returns the AUC.
    """
    fpr_vals, tpr_vals, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr_vals, tpr_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_vals, tpr_vals, "b-", lw=2, label=f"LBP-Motion MobileNetV3 (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    # EcoWild SD-time baseline
    ax.scatter([ecowild_fpr], [ecowild_tpr], marker="*", s=200, color="red",
               zorder=5, label=f"EcoWild SD-time (TPR={ecowild_tpr:.2f}, FPR={ecowild_fpr:.2f})")

    # Operating threshold marker
    if gate_threshold is not None:
        m = threshold_metrics(probs, labels, gate_threshold)
        ax.scatter([m["fpr"]], [m["tpr"]], marker="o", s=120, color="navy",
                   zorder=6,
                   label=f"Gate @ thr={gate_threshold:.2f} (TPR={m['tpr']:.3f}, FPR={m['fpr']:.3f})")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
    ax.set_title("ROC Curve — LBP-Motion Smoke Detector", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved → {save_path}")
    return roc_auc


# ---------------------------------------------------------------------------
# Config generation for wildfire_env.py
# ---------------------------------------------------------------------------

def gen_rl_config(
    template_path: str,
    ml_tpr: float,
    ml_fpr: float,
    output_path: str,
    label: str = "",
) -> None:
    """
    Copy an EcoWild config JSON and overwrite ML_Performance with new values.

    Parameters
    ----------
    template_path : str
        Path to an existing config JSON (e.g. config_setup_*.json from rl_decisionTree/).
    ml_tpr / ml_fpr : float
        New TP_rate and FP_rate for the ML_Performance block.
    output_path : str
        Where to write the updated config.
    label : str
        Descriptive tag for the config (used only in the filename logic).
    """
    with open(template_path) as f:
        cfg = json.load(f)

    cfg["ML_Performance"]["TP_rate"] = round(ml_tpr, 6)
    cfg["ML_Performance"]["FP_rate"] = round(ml_fpr, 6)

    if label:
        cfg["file_name"] = label   # used by inference_main.py for output folder naming

    with open(output_path, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"Config written → {output_path}  (TP={ml_tpr:.4f}, FP={ml_fpr:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---------------------------------------------------------
    ckpt    = torch.load(args.checkpoint, map_location=device)
    variant = ckpt.get("variant", "v3_small")
    model   = build_model(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (variant={variant})")

    # --- Build test dataset -------------------------------------------------
    print("Building dataset...")
    if args.figlib:
        # Import FIgLib-specific loader (timestamp-aware pairing)
        from figlib_dataset import FIgLibDataset
        dataset = FIgLibDataset(
            root=args.data_root,
            transform=get_transforms(train=False),
            max_gap_minutes=args.figlib_max_gap,
        )
    else:
        dataset = SmokeDataset(
            root=args.data_root,
            n_frames=args.n_frames,
            frame_gap=args.frame_gap,
            transform=get_transforms(train=False),
            precomputed=args.precomputed,
            cache_root=args.cache_root,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Test samples: {len(dataset)}")

    # --- Collect predictions ------------------------------------------------
    print("Running inference...")
    probs, labels = collect_predictions(model, loader, device)
    print(f"Inference complete. Computing metrics...")

    # --- Metrics at chosen threshold ----------------------------------------
    m = threshold_metrics(probs, labels, args.threshold)
    print("\n=== Gate metrics (Setup 1 — standalone) ===")
    print(f"  Threshold : {m['threshold']:.3f}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  TPR       : {m['tpr']:.4f}   (EcoWild SD-time: 0.90)")
    print(f"  FPR       : {m['fpr']:.4f}   (EcoWild SD-time: 0.58)")
    print(f"  PPV       : {m['ppv']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  TP/TN/FP/FN: {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")

    # --- ROC curve ----------------------------------------------------------
    roc_path = str(out_dir / "roc_curve.png")
    roc_auc  = plot_roc(probs, labels, roc_path, gate_threshold=args.threshold)
    print(f"  AUC       : {roc_auc:.4f}")

    # Save metrics to JSON
    results: dict = {
        "setup": "1_standalone",
        "checkpoint": args.checkpoint,
        "gate_metrics": m,
        "auc": roc_auc,
    }

    # --- Two-stage pipeline (Setup 2) ---------------------------------------
    if args.mode == "two_stage":
        sys_m = two_stage_metrics(
            gate_tpr=m["tpr"],
            gate_fpr=m["fpr"],
            ensemble_tpr=args.ensemble_tpr,
            ensemble_fpr=args.ensemble_fpr,
        )
        print("\n=== Two-stage pipeline metrics (Setup 2) ===")
        print(f"  Gate:     TPR={sys_m['gate_tpr']:.4f}, FPR={sys_m['gate_fpr']:.4f}")
        print(f"  Ensemble: TPR={sys_m['ensemble_tpr']:.4f}, FPR={sys_m['ensemble_fpr']:.4f}")
        print(f"  System:   TPR={sys_m['system_tpr']:.4f}, FPR={sys_m['system_fpr']:.4f}")
        print(f"  Gate pass rate (positive frames): {sys_m['gate_pass_rate_pos']*100:.1f}%")
        print(f"  Gate pass rate (negative frames): {sys_m['gate_pass_rate_neg']*100:.1f}%")
        print(f"  Ensemble calls saved:             {sys_m['ensemble_calls_saved_pct']:.1f}%")

        results["setup"] = "1_and_2"
        results["two_stage_metrics"] = sys_m

    # --- Generate RL config JSONs -------------------------------------------
    if args.gen_configs:
        if not args.template_config:
            print("WARNING: --template_config not provided; skipping config generation.")
        else:
            # Setup 1 config
            cfg1_path = str(out_dir / f"config_setup1_{m['tpr']:.4f}TP_{m['fpr']:.4f}FP.json")
            gen_rl_config(
                template_path=args.template_config,
                ml_tpr=m["tpr"],
                ml_fpr=m["fpr"],
                output_path=cfg1_path,
                label=f"setup1_{m['tpr']:.4f}TP_{m['fpr']:.4f}FP",
            )
            results["config_setup1"] = cfg1_path

            # Setup 2 config (only if two_stage mode)
            if args.mode == "two_stage":
                sys_tpr = sys_m["system_tpr"]
                sys_fpr = sys_m["system_fpr"]
                cfg2_path = str(out_dir / f"config_setup2_{sys_tpr:.4f}TP_{sys_fpr:.4f}FP.json")
                gen_rl_config(
                    template_path=args.template_config,
                    ml_tpr=sys_tpr,
                    ml_fpr=sys_fpr,
                    output_path=cfg2_path,
                    label=f"setup2_{sys_tpr:.4f}TP_{sys_fpr:.4f}FP",
                )
                results["config_setup2"] = cfg2_path

    # --- Save all results ---------------------------------------------------
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved → {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LBP-motion MobileNet smoke detector (Setup 1 and/or Setup 2)"
    )
    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint from train.py")
    parser.add_argument("--data_root", required=True,
                        help="Test dataset root (smoke/ and no_smoke/ sub-dirs)")

    # Dataset options
    parser.add_argument("--figlib", action="store_true",
                        help="Use FIgLib-specific timestamp-aware loader (figlib_dataset.py)")
    parser.add_argument("--figlib_max_gap", type=float, default=5.0,
                        help="Max time gap (minutes) between paired FIgLib images (default: 5)")
    parser.add_argument("--n_frames", type=int, default=2,
                        help="Frames per sample window — must match the trained checkpoint (default: 2)")
    parser.add_argument("--frame_gap", type=int, default=1,
                        help="Frame gap for SmokeDataset — must match the trained checkpoint (default: 1)")
    parser.add_argument("--cache_root", default=None,
                        help="Path to pre-computed LBP cache dir for this frame_gap "
                             "(e.g. lbp_cache/gap_1). Highly recommended — avoids slow on-the-fly computation.")
    parser.add_argument("--precomputed", action="store_true",
                        help="Images are pre-computed LBP-motion images (skips feature extraction)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Evaluation options
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Decision threshold for gate (default: 0.4 — conservative for safety)")
    parser.add_argument("--mode", choices=["standalone", "two_stage"], default="standalone",
                        help="'standalone' = Setup 1 only; 'two_stage' = also compute Setup 2 metrics")

    # Two-stage options
    parser.add_argument("--ensemble_tpr", type=float, default=0.90,
                        help="EcoWild heavy ensemble TPR (default: 0.90 = SD-time)")
    parser.add_argument("--ensemble_fpr", type=float, default=0.58,
                        help="EcoWild heavy ensemble FPR (default: 0.58 = SD-time)")

    # Config generation
    parser.add_argument("--gen_configs", action="store_true",
                        help="Generate updated EcoWild config JSONs for Setup 1/2")
    parser.add_argument("--template_config", default=None,
                        help="Path to an existing EcoWild config JSON to use as template")

    # Output
    parser.add_argument("--out_dir", default="eval_output",
                        help="Directory for ROC plot, metrics JSON, and generated configs")
    parser.add_argument("--device", default=None,
                        help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.")

    args = parser.parse_args()
    evaluate(args)
