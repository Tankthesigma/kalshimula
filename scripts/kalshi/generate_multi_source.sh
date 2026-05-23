#!/bin/bash
# Generate multi-source predictions for May 1-22 2026.
set -e

RUN_DIR=data/runs/may2024_apr2026_10city_openmeteo_sources_2yr
mkdir -p outputs/multi_source

for day in {1..22}; do
  DATE=$(printf "2026-05-%02d" $day)
  OUT=outputs/multi_source/predictions_${DATE}.json
  if [ -f "$OUT" ]; then
    echo "skip $DATE (exists)"
    continue
  fi
  echo "=== $DATE ==="
  python3 -m src.predict_batch_cli \
    --multi-source-mode blend_equal \
    --cities nyc,chicago,miami,austin,la,denver,philadelphia,phoenix,boston \
    --date "$DATE" \
    --model-run-dir "$RUN_DIR" \
    --selected-sources "$RUN_DIR/source_selection/recommended_sources.csv" \
    --bias-table "$RUN_DIR/model_policy/bias_table.csv" \
    --interval-table "$RUN_DIR/model_policy/interval_table.csv" \
    --threshold-residuals "$RUN_DIR/probability_calibration/threshold_residuals.csv" \
    --threshold-recalibration-table "$RUN_DIR/probability_calibration/threshold_recalibration_table.csv" \
    --threshold-offsets=-4,-2,0,2,4 \
    --out "$OUT" 2>&1 | grep -E "Fetch|Error|error" | tail -3
done
echo "done"
