#!/usr/bin/env bash
# classifier_eval_final_comparison.sh
# Runs pipeline_classifier_eval.py on the test set for all three final
# comparison pipelines and writes classifier_metrics.json to out_dir.
#
# Usage:
#   ./classifier_eval_final_comparison.sh --best_nf 2 --best_gap 16 [--threshold 0.5]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_TEST="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Dataset/test"
CACHE_ROOT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/lbp_cache"
SWEEP_CKPTS="$SCRIPT_DIR/smokeDetection_ourExperiments/sweep_results/checkpoints"
OUT_DIR="$SCRIPT_DIR/pipeline_classifier_results"
EVAL_SCRIPT="$SCRIPT_DIR/pipeline_classifier_eval.py"

RESNET_CKPT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt"
YOLO_CKPT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt"

THRESHOLD="0.5"
BEST_NF=""
BEST_GAP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold)   THRESHOLD="$2"; shift 2 ;;
        --best_nf)     BEST_NF="$2";   shift 2 ;;
        --best_gap)    BEST_GAP="$2";  shift 2 ;;
        --resnet_ckpt) RESNET_CKPT="$2"; shift 2 ;;
        --yolo_ckpt)   YOLO_CKPT="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$BEST_NF" ] || [ -z "$BEST_GAP" ]; then
    echo "Error: --best_nf and --best_gap are required."
    exit 1
fi

MOB_CKPT="$SWEEP_CKPTS/nf${BEST_NF}_gap${BEST_GAP}/nf${BEST_NF}_gap${BEST_GAP}_best_acc.pt"

echo "Threshold   : $THRESHOLD"
echo "Best config : nf=${BEST_NF}  gap=${BEST_GAP}"
echo "Output dir  : $OUT_DIR"
echo ""

for f in "$MOB_CKPT" "$RESNET_CKPT" "$YOLO_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: checkpoint not found: $f"
        exit 1
    fi
done

python "$EVAL_SCRIPT" \
    --data_root      "$DATASET_TEST" \
    --mobilenet_ckpt "$MOB_CKPT" \
    --resnet_ckpt    "$RESNET_CKPT" \
    --yolo_ckpt      "$YOLO_CKPT" \
    --n_frames       "$BEST_NF" \
    --frame_gap      "$BEST_GAP" \
    --cache_root     "$CACHE_ROOT/gap_${BEST_GAP}" \
    --threshold      "$THRESHOLD" \
    --out_dir        "$OUT_DIR"

echo ""
echo "Results saved to: $OUT_DIR/classifier_metrics.json"
