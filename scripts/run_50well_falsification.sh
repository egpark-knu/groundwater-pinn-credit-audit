#!/bin/bash
# Run Whittaker-vs-ODE falsification on 50 selected wells, 3 seeds
# Estimated: 450 runs × ~10s = ~75 min

set -e
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

# Read well stems from selection CSV (skip header, column 1)
WELLS=$(tail -n +2 results/well_selection/selected_50_wells.csv | cut -d',' -f1)

# Convert to space-separated list for argparse
WELL_ARGS=""
for w in $WELLS; do
    WELL_ARGS="$WELL_ARGS $w"
done

echo "Running Whittaker-vs-ODE falsification on $(echo $WELLS | wc -w | tr -d ' ') wells..."
echo "Seeds: 7 42 99, Lambda: 0.1, Window: 30, Horizon: 7"
echo "Output: results/whittaker_vs_ode_50well/"
echo ""

$PYTHON scripts/run_whittaker_vs_ode.py \
    --wells $WELL_ARGS \
    --seeds 7 42 99 \
    --lambda-value 0.1 \
    --window 30 \
    --forecast-horizon 7 \
    --epochs 80 \
    --patience 15 \
    --clean-head-outliers \
    --output-dir results/whittaker_vs_ode_50well

echo "Done! Check results/whittaker_vs_ode_50well/"
