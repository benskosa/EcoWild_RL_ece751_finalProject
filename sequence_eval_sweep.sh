#!/usr/bin/env bash
# sequence_eval_sweep.sh
# Runs sequence_eval.py for all 16 LBP+MobileNet sweep checkpoints on the test set.
# Each run evaluates MobileNet standalone only (no baseline checkpoints required).
#
# Optionally also evaluate baselines and the full gate pipeline:
#   ./sequence_eval_sweep.sh --with_baselines
#
# Usage:
#   ./sequence_eval_sweep.sh [--with_baselines] [--threshold 0.5]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_TEST="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Dataset/test"
CACHE_ROOT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/lbp_cache"
SWEEP_CKPTS="$SCRIPT_DIR/smokeDetection_ourExperiments/sweep_results/checkpoints"
OUT_ROOT="$SCRIPT_DIR/seq_eval_results"
SEQ_EVAL="$SCRIPT_DIR/sequence_eval.py"

RESNET_CKPT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt"
YOLO_CKPT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt"

# --- Parse arguments --------------------------------------------------------
THRESHOLD="0.5"
WITH_BASELINES=0
BEST_NF="2"
BEST_GAP="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold)     THRESHOLD="$2"; shift 2 ;;
        --with_baselines) WITH_BASELINES=1; shift ;;
        --best_nf)       BEST_NF="$2"; shift 2 ;;
        --best_gap)      BEST_GAP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Threshold        : $THRESHOLD"
echo "With baselines   : $WITH_BASELINES"
echo "Gate model       : nf${BEST_NF}_gap${BEST_GAP}"
echo "Output root      : $OUT_ROOT"
echo ""

# --- Sweep over all 16 MobileNet checkpoints --------------------------------
for nf in 2 3 4 5; do
  for gap in 1 2 6 16; do
    CKPT="$SWEEP_CKPTS/nf${nf}_gap${gap}/nf${nf}_gap${gap}_best_acc.pt"
    OUT="$OUT_ROOT/mobilenet_nf${nf}_gap${gap}"

    if [ ! -f "$CKPT" ]; then
        echo "WARNING: checkpoint not found, skipping: $CKPT"
        continue
    fi

    echo "============================================================"
    echo "  MobileNet  n_frames=${nf}  frame_gap=${gap}"
    echo "============================================================"

    python "$SEQ_EVAL" \
        --data_root      "$DATASET_TEST" \
        --mobilenet_ckpt "$CKPT" \
        --n_frames       "$nf" \
        --frame_gap      "$gap" \
        --cache_root     "$CACHE_ROOT/gap_${gap}" \
        --threshold      "$THRESHOLD" \
        --out_dir        "$OUT"

    echo ""
  done
done

# --- Baselines + gate pipeline ----------------------------------------------
if [ "$WITH_BASELINES" -eq 1 ]; then
    echo "============================================================"
    echo "  Baselines (ResNet34 + YOLOv8 + OR ensemble)"
    echo "============================================================"

    BASELINE_FLAGS=""
    [ -f "$RESNET_CKPT" ] && BASELINE_FLAGS="$BASELINE_FLAGS --resnet_ckpt $RESNET_CKPT"
    [ -f "$YOLO_CKPT"   ] && BASELINE_FLAGS="$BASELINE_FLAGS --yolo_ckpt   $YOLO_CKPT"

    python "$SEQ_EVAL" \
        --data_root "$DATASET_TEST" \
        $BASELINE_FLAGS \
        --threshold "$THRESHOLD" \
        --out_dir   "$OUT_ROOT/baselines"

    echo ""
    echo "============================================================"
    echo "  Gate pipeline: best MobileNet + OR ensemble"
    echo "  (uses nf2_gap1 as the gate — adjust if you find a better one)"
    echo "============================================================"

    BEST_CKPT="$SWEEP_CKPTS/nf${BEST_NF}_gap${BEST_GAP}/nf${BEST_NF}_gap${BEST_GAP}_best_acc.pt"

    if [ -f "$BEST_CKPT" ] && [ -f "$RESNET_CKPT" ] && [ -f "$YOLO_CKPT" ]; then
        python "$SEQ_EVAL" \
            --data_root      "$DATASET_TEST" \
            --mobilenet_ckpt "$BEST_CKPT" \
            --resnet_ckpt    "$RESNET_CKPT" \
            --yolo_ckpt      "$YOLO_CKPT" \
            --n_frames       "$BEST_NF" \
            --frame_gap      "$BEST_GAP" \
            --cache_root     "$CACHE_ROOT/gap_${BEST_GAP}" \
            --threshold      "$THRESHOLD" \
            --out_dir        "$OUT_ROOT/gate_nf${BEST_NF}_gap${BEST_GAP}"
    else
        echo "WARNING: one or more checkpoints missing for gate pipeline, skipping."
    fi
fi

echo ""
echo "All done. Results in: $OUT_ROOT"
