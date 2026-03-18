# LBP-Motion + MobileNet Smoke Detector

Python reimplementation of the video smoke detection framework from:

> Shi et al., *"Optimal Placement and Intelligent Smoke Detection Algorithm
> for Wildfire-Monitoring Cameras"*, IEEE Access 2020.
> DOI: 10.1109/ACCESS.2020.2987991

Designed as a **lightweight first-stage gate** for the EcoWild wildfire
detection pipeline, sitting between the decision-tree risk gate and the
heavier ResNet34 + YOLOv8 ensemble.

---

## Files

| File | Purpose |
|---|---|
| `feature_extraction.py` | LBP + Farneback optical flow → LBP-motion image |
| `model.py` | `SmokeDataset` + `build_model` (MobileNetV2 or V3-Small) |
| `train.py` | Training loop, metrics (accuracy / TPR / PPV / FPR), checkpointing |
| `gate.py` | `LBPMotionGate` inference wrapper for EcoWild integration |
| `requirements.txt` | Python dependencies |

---

## How the LBP-motion image is built

```
frame₁ ──► grayscale ──► LBP          ──► V channel (texture / shape)
                │
                └──► Gaussian blur ──┐
                                     ├──► Farneback flow ──► angle     ──► H channel (direction)
frame₂ ──► grayscale ──► Gaussian blur┘                  └──► magnitude ──► S channel (speed)

[H, S, V]  ──► HSV→RGB ──► LBP-motion image  ──► MobileNet binary classifier
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```
data_root/
  smoke/
      video_001/   frame_0001.jpg  frame_0002.jpg  ...
      video_002/   ...
  no_smoke/
      video_001/   ...
```

```bash
python train.py \
    --data_root /path/to/dataset \
    --variant   v3_small \          # or v2 to match the paper exactly
    --epochs    500 \
    --batch_size 32 \
    --lr        0.001 \
    --pretrained \                  # recommended; omit to match paper exactly
    --save_dir  checkpoints
```

---

## EcoWild integration

```python
from gate import LBPMotionGate
import cv2

gate = LBPMotionGate("checkpoints/best_model.pt", threshold=0.4)

prev_frame = None
cap = cv2.VideoCapture(camera_source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if prev_frame is not None:
        label, prob, is_night = gate.check(prev_frame, frame)

        if label == 1:
            # Gate passed → invoke full ResNet34 + YOLOv8 ensemble
            run_heavy_smoke_detection(frame)
        # else: gate blocked → energy saved

    prev_frame = frame
```

### Threshold guidance

| threshold | effect |
|---|---|
| 0.5 | balanced (default for standalone use) |
| 0.4 | fewer false negatives — **recommended for EcoWild** (safety first) |
| 0.3 | very aggressive; will pass almost all frames |

### Nighttime behaviour

When mean frame luminance < 40 (configurable via `night_luminance_threshold`),
optical flow is unreliable. The gate **fails open** (returns label=1) so the
full ensemble is always invoked at night — preserving EcoWild's glow detection.

---

## Key differences from the paper

| Aspect | Paper (Shi et al.) | This implementation |
|---|---|---|
| CNN backbone | MobileNetV2 | MobileNetV3-Small (default) or V2 (`--variant v2`) |
| Pre-training | Random init | Random init by default; `--pretrained` for ImageNet |
| Frame gap | Adjacent (time-lapse 60×) | Configurable (`--frame_gap`) |
| Output activation | Sigmoid | BCEWithLogitsLoss (sigmoid fused into loss for stability) |
