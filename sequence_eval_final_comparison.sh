#!/usr/bin/env bash
# sequence_eval_final_comparison.sh
# Evaluates the three pipelines side by side on the test set:
#   1. ResNet34 + YOLOv8 OR ensemble (baseline)
#   2. LBP+MobileNet standalone (best sweep config)
#   3. LBP+MobileNet gate -> ResNet34 + YOLOv8 OR ensemble
#
# Run sequence_eval_mobilenet_sweep.sh first to identify the best config,
# then pass those params here via --best_nf and --best_gap.
#
# Usage:
#   ./sequence_eval_final_comparison.sh --best_nf 2 --best_gap 1 [--threshold 0.5]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_TEST="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Dataset/test"
CACHE_ROOT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/lbp_cache"
SWEEP_CKPTS="$SCRIPT_DIR/smokeDetection_ourExperiments/sweep_results/checkpoints"
OUT_ROOT="$SCRIPT_DIR/seq_eval_results"
SEQ_EVAL="$SCRIPT_DIR/sequence_eval.py"

RESNET_CKPT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Train/checkpoints/resnet34_baseline_best_acc.pt"
YOLO_CKPT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Train/runs/yolov8n_baseline/weights/best.pt"

THRESHOLD="0.5"
BEST_NF=""
BEST_GAP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold)   THRESHOLD="$2";   shift 2 ;;
        --best_nf)     BEST_NF="$2";     shift 2 ;;
        --best_gap)    BEST_GAP="$2";    shift 2 ;;
        --resnet_ckpt) RESNET_CKPT="$2"; shift 2 ;;
        --yolo_ckpt)   YOLO_CKPT="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$BEST_NF" ] || [ -z "$BEST_GAP" ]; then
    echo "Error: --best_nf and --best_gap are required."
    echo "Run sequence_eval_mobilenet_sweep.sh first, then check:"
    echo "  python smokeDetection_ourExperiments/summarize_sequence_eval.py --eval_dir seq_eval_results --no_plots"
    exit 1
fi

MOB_CKPT="$SWEEP_CKPTS/nf${BEST_NF}_gap${BEST_GAP}/nf${BEST_NF}_gap${BEST_GAP}_best_acc.pt"

echo "Threshold   : $THRESHOLD"
echo "Best config : nf=${BEST_NF}  gap=${BEST_GAP}"
echo "Output root : $OUT_ROOT"
echo ""

# Verify checkpoints exist
for f in "$MOB_CKPT" "$RESNET_CKPT" "$YOLO_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: checkpoint not found: $f"
        exit 1
    fi
done

# --- 1. Baseline ensemble (ResNet34 + YOLOv8) --------------------------------
echo "============================================================"
echo "  Baseline: ResNet34 + YOLOv8 OR ensemble"
echo "============================================================"

python "$SEQ_EVAL" \
    --data_root   "$DATASET_TEST" \
    --resnet_ckpt "$RESNET_CKPT" \
    --yolo_ckpt   "$YOLO_CKPT" \
    --threshold   "$THRESHOLD" \
    --out_dir     "$OUT_ROOT/baseline_ensemble"

echo ""

# --- 2. Best MobileNet standalone --------------------------------------------
echo "============================================================"
echo "  Best MobileNet standalone  (nf=${BEST_NF}, gap=${BEST_GAP})"
echo "============================================================"

python "$SEQ_EVAL" \
    --data_root      "$DATASET_TEST" \
    --mobilenet_ckpt "$MOB_CKPT" \
    --n_frames       "$BEST_NF" \
    --frame_gap      "$BEST_GAP" \
    --cache_root     "$CACHE_ROOT/gap_${BEST_GAP}" \
    --threshold      "$THRESHOLD" \
    --out_dir        "$OUT_ROOT/best_mobilenet_nf${BEST_NF}_gap${BEST_GAP}"

echo ""

# --- 3. Gate pipeline --------------------------------------------------------
echo "============================================================"
echo "  Gate pipeline: MobileNet (nf=${BEST_NF}, gap=${BEST_GAP}) -> OR ensemble"
echo "============================================================"

python "$SEQ_EVAL" \
    --data_root      "$DATASET_TEST" \
    --mobilenet_ckpt "$MOB_CKPT" \
    --resnet_ckpt    "$RESNET_CKPT" \
    --yolo_ckpt      "$YOLO_CKPT" \
    --n_frames       "$BEST_NF" \
    --frame_gap      "$BEST_GAP" \
    --cache_root     "$CACHE_ROOT/gap_${BEST_GAP}" \
    --threshold      "$THRESHOLD" \
    --out_dir        "$OUT_ROOT/gate_nf${BEST_NF}_gap${BEST_GAP}"

echo ""
echo "All done. Results in: $OUT_ROOT"
echo ""
echo "Compare the three pipelines:"
echo "  baseline_ensemble/sequence_summary.json"
echo "  best_mobilenet_nf${BEST_NF}_gap${BEST_GAP}/sequence_summary.json"
echo "  gate_nf${BEST_NF}_gap${BEST_GAP}/sequence_summary.json"
