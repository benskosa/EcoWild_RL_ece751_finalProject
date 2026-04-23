#!/bin/bash
# precompute_test_cache.sh
# Precomputes the LBP cache for the test split for each frame_gap value.
# Run this once before eval_sweep_test.sh.
#
# Usage:
#   bash precompute_test_cache.sh
#   bash precompute_test_cache.sh --gaps "1 2 6 16"

set -e  # exit immediately on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$SCRIPT_DIR/../smokeDetection_baseline_ecoWild/Dataset"
CACHE_ROOT="$SCRIPT_DIR/../smokeDetection_baseline_ecoWild/lbp_cache"
GAPS="1 2 6 16"

# Parse optional --gaps override
while [[ $# -gt 0 ]]; do
    case $1 in
        --gaps) GAPS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Precomputing LBP test cache"
echo "  Dataset : $DATASET_ROOT"
echo "  Cache   : $CACHE_ROOT"
echo "  Gaps    : $GAPS"
echo "============================================================"

for gap in $GAPS; do
    echo ""
    echo "--- frame_gap=$gap ---"
    python "$SCRIPT_DIR/precompute_lbp_cache.py" \
        --dataset_root "$DATASET_ROOT" \
        --cache_root   "$CACHE_ROOT/gap_${gap}" \
        --frame_gap    "$gap" \
        --splits       test
done

echo ""
echo "============================================================"
echo "  Test cache precomputation complete."
echo "============================================================"
