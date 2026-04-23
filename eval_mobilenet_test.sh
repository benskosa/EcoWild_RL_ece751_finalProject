#!/usr/bin/env bash
# eval_mobilenet_test.sh
# Evaluates all 16 LBP+MobileNet sweep checkpoints on the test set.
# Run precompute_test_cache.sh first if the test cache doesn't exist yet.

set -e  # stop on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_TEST="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Dataset/test"
CACHE_ROOT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/lbp_cache"
EVAL="$SCRIPT_DIR/smokeDetection_ourExperiments/eval.py"
SWEEP_CKPTS="$SCRIPT_DIR/smokeDetection_ourExperiments/sweep_results/checkpoints"
OUT_ROOT="$SCRIPT_DIR/smokeDetection_ourExperiments/sweep_results/eval_test"

THRESHOLD="${1:-0.5}"   # pass threshold as first argument, default 0.5
                        # e.g.: ./eval_mobilenet_test.sh 0.75

echo "Threshold: $THRESHOLD"
echo ""

for nf in 2 3 4 5; do
  for gap in 1 2 6 16; do
    CKPT="$SWEEP_CKPTS/nf${nf}_gap${gap}/nf${nf}_gap${gap}_best_acc.pt"
    OUT="$OUT_ROOT/nf${nf}_gap${gap}"

    echo "============================================================"
    echo "  Evaluating  n_frames=${nf}  frame_gap=${gap}"
    echo "============================================================"

    python "$EVAL" \
        --checkpoint  "$CKPT" \
        --data_root   "$DATASET_TEST" \
        --n_frames    "$nf" \
        --frame_gap   "$gap" \
        --cache_root  "$CACHE_ROOT/gap_${gap}" \
        --threshold   "$THRESHOLD" \
        --out_dir     "$OUT"

    echo ""
  done
done

echo "All evaluations complete. Results in: $OUT_ROOT"
