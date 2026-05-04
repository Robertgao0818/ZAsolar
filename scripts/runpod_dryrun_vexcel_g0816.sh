#!/usr/bin/env bash
# RunPod dry-run: V3-C inference on Vexcel JHB G0816, evaluate vs Li GT.
# Run this on the pod (NOT locally).
#
# Tiles must already be present at $SOLAR_TILES_ROOT/G0816/G0816_{col}_{row}_geo.tif
# (typically /dev/shm/tiles/G0816/ — see scripts/runpod_pod.sh init or batch script).

set -euo pipefail

REPO=/workspace/ZAsolar
SOLAR_TILES_ROOT="${SOLAR_TILES_ROOT:-/dev/shm/tiles}"
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/G0816_vexcel_dryrun.log"

if [ ! -d "$SOLAR_TILES_ROOT/G0816" ]; then
  echo "FATAL: $SOLAR_TILES_ROOT/G0816 not found. Stage tiles first." >&2
  exit 1
fi

cd "$REPO"
SOLAR_TILES_ROOT="$SOLAR_TILES_ROOT" python3 detect_and_evaluate.py \
  --grid-id G0816 \
  --region johannesburg \
  --imagery-layer vexcel_2024 \
  --model-run v3c_vexcel_2024 \
  --model-path "$REPO/checkpoints/exp003_C_targeted_hn/best_model.pth" \
  --postproc-config "$REPO/configs/postproc/v4_canonical.json" \
  --evaluation-profile installation \
  --force \
  2>&1 | tee "$LOG"

# Quick health-check
RESULTS_DIR="$REPO/results/johannesburg/v3c_vexcel_2024/G0816"
echo
echo "=== summary ==="
ls -la "$RESULTS_DIR" 2>/dev/null | head
test -s "$RESULTS_DIR/predictions_metric.gpkg" && \
  python3 -c "
import geopandas as gpd
g = gpd.read_file('$RESULTS_DIR/predictions_metric.gpkg')
print(f'predictions: {len(g)} polygons in CRS {g.crs}')
print(f'mean area: {g.geometry.area.mean():.1f} m²')
"
test -s "$RESULTS_DIR/presence_metrics.csv" && \
  cat "$RESULTS_DIR/presence_metrics.csv"
