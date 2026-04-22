# ECE 751 Final Project — Wildfire Smoke Detection Pipeline

Reimplementation and extension of the smoke detection component from the
**EcoWild** wildfire-monitoring system, plus a lightweight LBP + optical-flow
gate designed to reduce computation (and energy) before calling the heavier
ensemble models.

> **Paper context:** Shi et al., *"Optimal Placement and Intelligent Smoke
> Detection Algorithm for Wildfire-Monitoring Cameras"*, IEEE Access 2020.
> DOI: [10.1109/ACCESS.2020.2987991](https://doi.org/10.1109/ACCESS.2020.2987991)

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Dataset](#2-dataset)
3. [Environment Setup](#3-environment-setup)
4. [Baseline Models](#4-baseline-models)
   - [ResNet34](#resnet34)
   - [YOLOv8](#yolov8)
5. [Our Method — LBP + Optical Flow → MobileNet](#5-our-method--lbp--optical-flow--mobilenet)
   - [Feature Extraction](#feature-extraction)
   - [Precomputing the LBP Cache](#precomputing-the-lbp-cache)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Grid Sweep](#grid-sweep)
6. [Visualization Tools](#6-visualization-tools)
7. [Results](#7-results)
8. [To-Do / Pending Experiments](#8-to-do--pending-experiments)
9. [References](#9-references)

---

## 1. Repository Layout

```
EcoWild_RL_ece751_finalProject/
│
├── README.md                          ← you are here
├── .gitignore
│
├── smokeDetection_baseline_ecoWild/   ← baseline models (ResNet34 + YOLOv8)
│   ├── Dataset/                       ← shared dataset (gitignored)
│   │   ├── train/  smoke/  <fire_id>/*.jpg
│   │   │          no_smoke/<fire_id>/*.jpg
│   │   ├── val/   smoke/  ...
│   │   └── test/  smoke/  ...
│   ├── Train/
│   │   ├── simple_resnet.py           ← ResNet34 fine-tuning script
│   │   └── yolov8_training.py         ← YOLOv8-cls training script
│   ├── reshuffle_dataset.py           ← reshuffles dataset to 70/15/15 by sequence
│   ├── accuracy_eval.py               ← original EcoWild accuracy evaluation
│   ├── energy_eval.py                 ← original EcoWild energy evaluation
│   └── environment.yml
│
├── smokeDetection_ourExperiments/     ← LBP + Farneback → MobileNet pipeline
│   ├── feature_extraction.py          ← LBP + optical flow feature builder
│   ├── model.py                       ← SmokeDataset + build_model (MobileNetV2/V3)
│   ├── train.py                       ← training loop with tqdm, dual checkpoints
│   ├── eval.py                        ← evaluation: ROC, AUC, confusion matrix
│   ├── gate.py                        ← LBPMotionGate inference wrapper
│   ├── precompute_lbp_cache.py        ← pre-bakes LBP-motion PNGs to disk
│   ├── reorganize_dataset.py          ← reshuffles flat images into fire_id/ subdirs
│   ├── grid_sweep.py                  ← automated sweep over n_frames × frame_gap
│   ├── plot_sweep.py                  ← heatmaps + training curves from sweep
│   ├── plot_history.py                ← training curves from a single history.json
│   ├── visualize_lbp.py               ← renders LBP-motion composite images
│   ├── figlib_dataset.py
│   ├── environment.yml                ← conda env spec (Python 3.10, CUDA 11.8)
│   └── README.md                      ← detailed per-script docs
│
└── rl_ecoWild/                        ← RL agent (TD3) for camera scheduling
    ├── wildfire_env.py
    ├── inference_main.py
    └── ...
```

---

## 2. Dataset

The shared dataset lives at `smokeDetection_baseline_ecoWild/Dataset/` (gitignored —
contains raw JPEGs from HPWREN wildfire cameras).

### Structure

```
Dataset/
  train/  smoke/<fire_id>/*.jpg    no_smoke/<fire_id>/*.jpg
  val/    smoke/<fire_id>/*.jpg    no_smoke/<fire_id>/*.jpg
  test/   smoke/<fire_id>/*.jpg    no_smoke/<fire_id>/*.jpg
```

Each `<fire_id>` directory contains chronologically ordered frames captured
roughly every 60 seconds from a single fixed camera during one fire event.

### Split

Split **by fire sequence** (no single fire event spans multiple splits)
using `reshuffle_dataset.py` with seed 42:

| Split | Sequences | Approx. frames |
|-------|-----------|---------------|
| train | ~70%      | —             |
| val   | ~15%      | —             |
| test  | ~15%      | —             |

To re-create or verify the split:

```bash
cd smokeDetection_baseline_ecoWild

# Preview without moving files:
python reshuffle_dataset.py --dry-run

# Apply (overwrites existing split):
python reshuffle_dataset.py
```

---

## 3. Environment Setup

Both sub-projects use the same conda environment (`ece751`).

```bash
cd smokeDetection_ourExperiments
conda env create -f environment.yml
conda activate ece751

# Needed for headless Linux servers (avoids X11 dependency):
pip uninstall opencv-python -y
pip install opencv-python-headless

# YOLOv8 (for baselines):
pip install ultralytics
```

> **Windows only:** If you see `OMP Error #15`, run:
> ```bash
> conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n ece751
> ```

---

## 4. Baseline Models

Both scripts write dual checkpoints (`_best_acc.pt` and `_best_tpr.pt`) plus
a `history.json` compatible with `plot_history.py`.

### ResNet34

Fine-tunes a pretrained ResNet34 on the dataset using `ImageFolder`
(alphabetical class order: `no_smoke=0`, `smoke=1`).

```bash
cd smokeDetection_baseline_ecoWild/Train

python simple_resnet.py \
    --train_root ../Dataset/train \
    --val_root   ../Dataset/val \
    --save_dir   checkpoints \
    --run_name   resnet34_baseline \
    --epochs     1000 \
    --batch_size 64 \
    --lr         0.0001 \
    --patience   50 \
    --num_workers 4
```

**Outputs:**
- `checkpoints/resnet34_baseline_best_acc.pt`
- `checkpoints/resnet34_baseline_best_tpr.pt`
- `checkpoints/history.json`

### YOLOv8

Trains YOLOv8-nano in classification mode (`yolov8n-cls.pt`).
YOLOv8 finds images recursively under each class folder.

```bash
cd smokeDetection_baseline_ecoWild/Train

python yolov8_training.py \
    --data_root ../Dataset \
    --model     yolov8n-cls.pt \
    --project   runs \
    --name      yolov8n_baseline \
    --epochs    200 \
    --batch     64 \
    --imgsz     224 \
    --patience  50 \
    --workers   4
```

**Outputs:** `runs/yolov8n_baseline/` — Ultralytics saves `results.csv`,
confusion matrix, `best.pt`, and `last.pt` automatically.

> **Tip:** Run both in separate `tmux` sessions so training survives SSH
> disconnects (`tmux new -s resnet` / `tmux new -s yolo`, detach with `Ctrl-b d`).

---

## 5. Our Method — LBP + Optical Flow → MobileNet

### Feature Extraction

Each sample is built from **N consecutive frames** spaced **frame_gap** apart.
For each adjacent pair `(frame_i, frame_{i+gap})`:

```
frame_1 → grayscale → LBP ──────────────────────────────► V  (texture / shape)
                │
                └→ Gaussian blur ─┐
                                   ├→ Farneback flow → angle    → H (direction)
frame_2 → grayscale → Gaussian blur┘               → magnitude → S (speed)

[H, S, V] → HSV→RGB → LBP-motion image → MobileNet binary classifier
```

When `n_frames > 2`, the N−1 pairwise LBP-motion images are pixel-averaged
into a single composite input. Larger `frame_gap` captures longer-range
motion; larger `n_frames` averages over more pairs (noise reduction).

### Precomputing the LBP Cache

Computing LBP + Farneback on-the-fly is slow (~10–20 min per epoch for the
full dataset). Pre-bake the cache once per `frame_gap` value:

```bash
cd smokeDetection_ourExperiments

python precompute_lbp_cache.py \
    --dataset_root ../smokeDetection_baseline_ecoWild/Dataset \
    --cache_root   ../smokeDetection_baseline_ecoWild/lbp_cache/gap_1 \
    --frame_gap    1 \
    --splits       train val
```

Cache layout: `lbp_cache/gap_{N}/train/smoke/<fire_id>/pair_NNNN.png`

> **After reshuffling the dataset**, the existing cache is stale — re-run
> precompute for every `frame_gap` you plan to use.

### Training

```bash
cd smokeDetection_ourExperiments

python train.py \
    --train_root  ../smokeDetection_baseline_ecoWild/Dataset/train \
    --val_root    ../smokeDetection_baseline_ecoWild/Dataset/val \
    --cache_root  ../smokeDetection_baseline_ecoWild/lbp_cache/gap_1 \
    --n_frames    2 \
    --epochs      500 \
    --batch_size  32 \
    --pretrained \
    --save_dir    checkpoints \
    --run_name    mobilenet_nf2_gap1 \
    --num_workers 4
```

Key flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--n_frames` | 2 | Frames per sample window |
| `--frame_gap` | 1 | Stride between frames (must match `--cache_root`) |
| `--pretrained` | off | Use ImageNet weights (recommended) |
| `--preload_cache` | off | Load all cached PNGs into RAM (~2.5 GB) for zero disk I/O |
| `--patience` | 50 | Early stopping epochs without val_acc improvement |

**Outputs:** `checkpoints/<run_name>_best_acc.pt`, `_best_tpr.pt`, `history.json`

### Evaluation

```bash
python eval.py \
    --checkpoint checkpoints/mobilenet_nf2_gap1_best_acc.pt \
    --data_root  ../smokeDetection_baseline_ecoWild/Dataset/val \
    --frame_gap  1 \
    --out_dir    eval_output/nf2_gap1
```

Outputs: ROC curve PNG, confusion matrix, `eval_results.json`
(includes AUC, accuracy, TPR, FPR, PPV at chosen threshold).

### Grid Sweep

Automatically sweeps all `(n_frames, frame_gap)` combinations,
running precompute → train → eval for each:

```bash
python grid_sweep.py \
    --dataset_root ../smokeDetection_baseline_ecoWild/Dataset \
    --cache_root   ../smokeDetection_baseline_ecoWild/lbp_cache \
    --out_dir      sweep_results \
    --n_frames_list  2 3 4 5 \
    --frame_gap_list 1 2 6 16 \
    --epochs         30 \
    --skip_existing_cache
```

Results are written to `sweep_results/sweep_summary.csv` after every run
(safe to interrupt and resume). Visualize with:

```bash
python plot_sweep.py --sweep_dir sweep_results --metric val_acc --smooth 5
```

Produces:
- `sweep_results/plots/curves_by_gap.png`
- `sweep_results/plots/curves_by_nframes.png`
- `sweep_results/plots/heatmap_acc.png`
- `sweep_results/plots/heatmap_tpr.png`
- `sweep_results/plots/heatmap_fpr.png`

---

## 6. Visualization Tools

### LBP-motion composite images

```bash
cd smokeDetection_ourExperiments

python visualize_lbp.py \
    --train_root ../smokeDetection_baseline_ecoWild/Dataset/train \
    --n_frames   3 \
    --frame_gap  2 \
    --n_examples 3 \
    --out_dir    lbp_visualizations
```

Renders side-by-side strips: `[frame_1 | ... | frame_N | LBP-motion | H | S | V]`

### Training curves (single run)

```bash
python plot_history.py \
    --history checkpoints/history.json \
    --out     training_curves.png \
    --smooth  5
```

### Training curves (compare multiple runs)

```bash
python plot_history.py \
    --history  checkpoints/run_a/history.json checkpoints/run_b/history.json \
    --labels   "n_frames=2" "n_frames=4" \
    --out      comparison.png
```

---

## 7. Results

> **TODO:** Fill in after training runs complete.

### Baseline models (val set)

| Model     | Accuracy | TPR  | FPR  | PPV  | F1   |
|-----------|----------|------|------|------|------|
| ResNet34  | —        | —    | —    | —    | —    |
| YOLOv8n   | —        | —    | —    | —    | —    |

### LBP + MobileNet sweep (best val metric per config)

| n_frames | frame_gap | Val Acc | Val TPR | Val FPR |
|----------|-----------|---------|---------|---------|
| 2        | 1         | —       | —       | —       |
| 2        | 2         | —       | —       | —       |
| 2        | 6         | —       | —       | —       |
| 2        | 16        | —       | —       | —       |
| 3        | 1         | —       | —       | —       |
| ...      | ...       | ...     | ...     | ...     |

### Table 2 recreation (EcoWild comparison)

> TODO — requires clarification on energy metrics (E_comm, E_total, min columns).

| Model                       | TP   | FP   | min  | E_comm | E_total |
|-----------------------------|------|------|------|--------|---------|
| EcoWild (original)          | 0.90 | 0.58 | —    | —      | —       |
| ResNet34                    | —    | —    | —    | —      | —       |
| YOLOv8n                     | —    | —    | —    | —      | —       |
| LBP + MobileNet (best)      | —    | —    | —    | —      | —       |
| LBP gate → ResNet ensemble  | —    | —    | —    | —      | —       |

---

## 8. To-Do / Pending Experiments

- [ ] Rebuild LBP cache after 70/15/15 reshuffle (old cache paths are stale)
- [ ] Run ResNet34 baseline training to completion
- [ ] Run YOLOv8n baseline training to completion
- [ ] Re-run grid sweep on new 70/15/15 split
- [ ] Write unified evaluator comparing all models on the test set
- [ ] Clarify energy metric definitions (E_comm, E_total, "min") from paper authors
- [ ] Implement LBP gate → ResNet/YOLOv8 ensemble pipeline
- [ ] Recreate Table 2 with final numbers

---

## 9. References

- Shi et al., "Optimal Placement and Intelligent Smoke Detection Algorithm for
  Wildfire-Monitoring Cameras," *IEEE Access*, 2020.
  DOI: [10.1109/ACCESS.2020.2987991](https://doi.org/10.1109/ACCESS.2020.2987991)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [HPWREN Camera Network](http://hpwren.ucsd.edu/) — source of wildfire imagery
- [EcoWild Project](https://ecowild.info/)
