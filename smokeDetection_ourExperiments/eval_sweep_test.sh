#!/bin/bash
# eval_sweep_test.sh
# Evaluates all n_frames x frame_gap sweep checkpoints on the test set.
# Run precompute_test_cache.sh first if the test cache doesn't exist yet.
#
# Usage:
#   bash eval_sweep_test.sh
#   bash eval_sweep_test.sh --threshold 0.75
#   bash eval_sweep_test.sh --n_frames "2 3" --gaps "1 6" --threshold 0.5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_TEST="$SCRIPT_DIR/../smokeDetection_baseline_ecoWild/Dataset/test"
CACHE_ROOT="$SCRIPT_DIR/../smokeDetection_baseline_ecoWild/lbp_cache"
CKPT_ROOT="$SCRIPT_DIR/sweep_results/checkpoints"
OUT_ROOT="$SCRIPT_DIR/sweep_results/eval_test"
N_FRAMES="2 3 4 5"
GAPS="1 2 6 16"
THRESHOLD="0.5"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --n_frames)  N_FRAMES="$2";  shift 2 ;;
        --gaps)      GAPS="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Sweep evaluation on test set"
echo "  Test data  : $DATASET_TEST"
echo "  Cache root : $CACHE_ROOT"
echo "  Checkpoints: $CKPT_ROOT"
echo "  n_frames   : $N_FRAMES"
echo "  frame_gaps : $GAPS"
echo "  Threshold  : $THRESHOLD"
echo "  Output     : $OUT_ROOT"
echo "============================================================"

n_ok=0
n_fail=0

for nf in $N_FRAMES; do
  for gap in $GAPS; do
    run="nf${nf}_gap${gap}"
    ckpt="$CKPT_ROOT/$run/${run}_best_acc.pt"
    out="$OUT_ROOT/$run"

    echo ""
    echo "--- $run ---"

    if [ ! -f "$ckpt" ]; then
        echo "  [SKIP] Checkpoint not found: $ckpt"
        n_fail=$((n_fail + 1))
        continue
    fi

    python "$SCRIPT_DIR/eval.py" \
        --checkpoint "$ckpt" \
        --data_root  "$DATASET_TEST" \
        --n_frames   "$nf" \
        --frame_gap  "$gap" \
        --cache_root "$CACHE_ROOT/gap_${gap}" \
        --threshold  "$THRESHOLD" \
        --out_dir    "$out"

    n_ok=$((n_ok + 1))
  done
done

echo ""
echo "============================================================"
echo "  Eval complete.  Succeeded: $n_ok   Skipped/failed: $n_fail"
echo "  Results in: $OUT_ROOT"
echo "============================================================"
