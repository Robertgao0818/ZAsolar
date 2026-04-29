#!/usr/bin/env bash
# RunPod template: direct_maskrcnn_v1 inference + finalize + eval per grid.
#
# Replaces the legacy bash launcher pattern (PARALLEL=4 × hardcoded BATCH=4)
# with a saturation-friendly single-process model: PARALLEL=2 default, but
# each detect_direct.py uses --batch-size 16 + --num-workers 10 internally.
#
# Tiles must be staged where regions.yaml expects them (or symlinked there).
# Per .claude/rules/05-runpod-inference.md, /dev/shm fast-path is ideal:
#   mkdir -p /dev/shm/tiles && cp -r /workspace/tiles/<grid> /dev/shm/tiles/
#   export SOLAR_TILES_ROOT=/dev/shm/tiles
#
# Outputs go to results/analysis/direct_maskrcnn_v1/<region>/<model_run>/<grid>/.

set -euo pipefail

REPO="${REPO:-/workspace/ZAsolar}"
MODEL="${MODEL:-$REPO/checkpoints/exp003_C_targeted_hn/best_model.pth}"
POSTPROC="${POSTPROC:-$REPO/configs/postproc/v4_canonical.json}"
REGION="${REGION:-johannesburg}"
IMAGERY_LAYER="${IMAGERY_LAYER:-vexcel_2024}"
MODEL_RUN="${MODEL_RUN:-v3c_vexcel_2024_direct}"
PARALLEL="${PARALLEL:-2}"     # detect_direct.py is GPU-saturating; 2 ≈ pipeline overlap
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-10}"
LOG_DIR="${LOG_DIR:-/workspace/logs/direct_maskrcnn_v1}"
mkdir -p "$LOG_DIR"

GRIDS=(
  "${@:-G0816}"   # default to single-grid dryrun; pass grids as args to override
)

run_grid() {
  local g=$1
  local grid_log="$LOG_DIR/${g}.log"
  local out_dir="$REPO/results/analysis/direct_maskrcnn_v1/${REGION}/${MODEL_RUN}/${g}"

  echo "[$g] detect_direct.py" >> "$grid_log"
  python3 -u "$REPO/detect_direct.py" \
    --grid-id "$g" \
    --region "$REGION" \
    --imagery-layer "$IMAGERY_LAYER" \
    --model-run "$MODEL_RUN" \
    --model-path "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor 2 \
    --chip-size 400 \
    --overlap 0.25 \
    --detector-score-threshold 0.05 \
    --detections-per-img 300 \
    --mask-threshold 0.3 \
    --raw-mask-storage crop \
    --device cuda \
    --profile \
    >> "$grid_log" 2>&1

  echo "[$g] finalize.py (merge-mode=pixel-or, geoai-equivalent merge)" >> "$grid_log"
  python3 -u "$REPO/finalize.py" \
    --input "$out_dir/raw_detections.pkl" \
    --output-dir "$out_dir" \
    --postproc-config "$POSTPROC" \
    --merge-mode pixel-or \
    >> "$grid_log" 2>&1

  echo "[$g] evaluate_predictions.py" >> "$grid_log"
  python3 -u "$REPO/evaluate_predictions.py" \
    --predictions-gpkg "$out_dir/predictions_metric.gpkg" \
    --region "$REGION" \
    --grid-id "$g" \
    --imagery-layer "$IMAGERY_LAYER" \
    --model-run "$MODEL_RUN" \
    --output-dir "$out_dir" \
    --evaluation-profile installation \
    >> "$grid_log" 2>&1
}

cd "$REPO"
echo "[batch] PARALLEL=$PARALLEL, ${#GRIDS[@]} grid(s), region=$REGION, layer=$IMAGERY_LAYER"
SECONDS=0
running=0
for g in "${GRIDS[@]}"; do
  echo "[$SECONDS s] launch $g"
  run_grid "$g" &
  running=$((running + 1))
  if [ $running -ge $PARALLEL ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "[batch] all ${#GRIDS[@]} grid(s) finished in ${SECONDS}s"

# Summary
SUMMARY="$LOG_DIR/_summary.csv"
echo "grid,gt_count,pred_count,tp,fp,fn,precision,recall,f1" > "$SUMMARY"
for g in "${GRIDS[@]}"; do
  csv="$REPO/results/analysis/direct_maskrcnn_v1/${REGION}/${MODEL_RUN}/${g}/presence_metrics.csv"
  if [ -s "$csv" ]; then
    awk -F, 'NR==2 {print $0}' "$csv" >> "$SUMMARY" 2>/dev/null || true
  else
    echo "$g,MISSING,,,,,,," >> "$SUMMARY"
  fi
done
echo "=== summary written to $SUMMARY ==="
column -s, -t "$SUMMARY" | head -40
