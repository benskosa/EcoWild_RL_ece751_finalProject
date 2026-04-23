#!/usr/bin/env bash
# precompute_test_cache.sh
# Precomputes LBP cache for the test split for all frame_gap values used in the sweep.

set -e  # stop on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/Dataset"
CACHE_ROOT="$SCRIPT_DIR/smokeDetection_baseline_ecoWild/lbp_cache"
PRECOMPUTE="$SCRIPT_DIR/smokeDetection_ourExperiments/precompute_lbp_cache.py"

for gap in 1 2 6 16; do
    echo "============================================================"
    echo "  Precomputing test cache  frame_gap=${gap}"
    echo "============================================================"
    python "$PRECOMPUTE" \
        --dataset_root "$DATASET_ROOT" \
        --cache_root   "$CACHE_ROOT/gap_${gap}" \
        --frame_gap    "$gap" \
        --splits       test
done

echo ""
echo "All test caches precomputed."
