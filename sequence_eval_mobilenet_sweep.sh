#!/usr/bin/env bash
# sequence_eval_mobilenet_sweep.sh
# Runs sequence_eval.py for all 16 LBP+MobileNet sweep checkpoints on the
# test set. Use summarize_sequence_eval.py afterwards to find the best config.
#
# Usage:
#   ./sequence_eval_mobilenet_sweep.sh [--threshold 0.5]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_TEST="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Dataset/test"
CACHE_ROOT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/lbp_cache"
SWEEP_CKPTS="$SCRIPT_DIR/smokeDetection_ourExperiments/sweep_results/checkpoints"
OUT_ROOT="$SCRIPT_DIR/seq_eval_results"
SEQ_EVAL="$SCRIPT_DIR/sequence_eval.py"

THRESHOLD="0.5"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold) THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Threshold  : $THRESHOLD"
echo "Output root: $OUT_ROOT"
echo ""

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

echo "All done. Results in: $OUT_ROOT"
echo ""
echo "Next: find the best config with:"
echo "  python smokeDetection_ourExperiments/summarize_sequence_eval.py --eval_dir $OUT_ROOT --no_plots"
