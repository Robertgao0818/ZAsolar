#!/usr/bin/env bash
# RunPod template: direct_maskrcnn_v1 inference + finalize + SAM-refine + eval per grid.
#
# Phase A (parallel): detect_direct.py + finalize.py per grid.
# Phase B (serial)  : sam_refine_maskbox.py over all grids (single SAM model load).
#                     Skipped if SAM_REFINE=0.
# Phase C (parallel): evaluate_predictions.py per grid on the SAM-refined output
#                     (or the raw finalize output if SAM_REFINE=0).
#
# Tiles must be staged where regions.yaml expects them (or symlinked there).
# Per .claude/rules/05-runpod-inference.md, /dev/shm fast-path is ideal:
#   mkdir -p /dev/shm/tiles && cp -r /workspace/tiles/<grid> /dev/shm/tiles/
#   export SOLAR_TILES_ROOT=/dev/shm/tiles
#
# Raw outputs go to results/analysis/direct_maskrcnn_v1/<region>/<model_run>/<grid>/.
# SAM-refined outputs go to results/analysis/direct_maskrcnn_v1/<region>/<model_run>_sam_maskbox/<grid>/.

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
SAM_REFINE="${SAM_REFINE:-1}"           # 1=run SAM mask+box refinement, 0=skip
SAM_PROMPT_MODE="${SAM_PROMPT_MODE:-mask_box}"
SAM_BATCH_SIZE="${SAM_BATCH_SIZE:-8}"   # polygons per SAM forward; laptop 8GB ≈ 8, RTX 5090 32GB ≈ 16-32
LOG_DIR="${LOG_DIR:-/workspace/logs/direct_maskrcnn_v1}"
mkdir -p "$LOG_DIR"

GRIDS=(
  "${@:-G0816}"   # default to single-grid dryrun; pass grids as args to override
)

RAW_RUN_ROOT="$REPO/results/analysis/direct_maskrcnn_v1/${REGION}/${MODEL_RUN}"
SAM_RUN_ROOT="$REPO/results/analysis/direct_maskrcnn_v1/${REGION}/${MODEL_RUN}_sam_maskbox"

# ── Phase A: detect_direct + finalize ────────────────────────────────────
phaseA_grid() {
  local g=$1
  local grid_log="$LOG_DIR/${g}.log"
  local out_dir="$RAW_RUN_ROOT/$g"

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
}

# ── Phase C: evaluate ────────────────────────────────────────────────────
phaseC_grid() {
  local g=$1
  local eval_root=$2          # RAW_RUN_ROOT or SAM_RUN_ROOT
  local grid_log="$LOG_DIR/${g}.log"
  local out_dir="$eval_root/$g"

  if [ ! -s "$out_dir/predictions_metric.gpkg" ]; then
    echo "[$g] SKIP eval — no predictions_metric.gpkg at $out_dir" >> "$grid_log"
    return
  fi
  echo "[$g] evaluate_predictions.py ($eval_root)" >> "$grid_log"
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
echo "[batch] PARALLEL=$PARALLEL, ${#GRIDS[@]} grid(s), region=$REGION, layer=$IMAGERY_LAYER, sam_refine=$SAM_REFINE"
SECONDS=0

# ── Phase A: detect + finalize, parallel ─────────────────────────────────
echo "[phase A] detect_direct + finalize × ${#GRIDS[@]} grid(s) (PARALLEL=$PARALLEL)"
running=0
for g in "${GRIDS[@]}"; do
  echo "[$SECONDS s] phaseA launch $g"
  phaseA_grid "$g" &
  running=$((running + 1))
  if [ $running -ge $PARALLEL ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "[phase A] done in ${SECONDS}s"

# ── Phase B: SAM refine, serial (single model load) ──────────────────────
if [ "$SAM_REFINE" = "1" ]; then
  echo "[phase B] sam_refine_maskbox.py × ${#GRIDS[@]} grid(s) → $SAM_RUN_ROOT"
  sam_log="$LOG_DIR/_sam_refine.log"
  python3 -u "$REPO/scripts/analysis/sam_refine_maskbox.py" \
    --region "$REGION" \
    --grids "${GRIDS[@]}" \
    --src-results-root "$RAW_RUN_ROOT" \
    --imagery-layer "$IMAGERY_LAYER" \
    --output-root "$SAM_RUN_ROOT" \
    --prompt-mode "$SAM_PROMPT_MODE" \
    --sam-batch-size "$SAM_BATCH_SIZE" \
    --label "${MODEL_RUN}_sam_maskbox" \
    > "$sam_log" 2>&1 || {
      echo "[phase B] FAILED — see $sam_log" >&2
      tail -20 "$sam_log" >&2
      SAM_REFINE=0   # fall back to raw output for eval
    }
  echo "[phase B] done at ${SECONDS}s"
fi

# ── Phase C: evaluate, parallel ──────────────────────────────────────────
if [ "$SAM_REFINE" = "1" ]; then
  EVAL_ROOT="$SAM_RUN_ROOT"
  EVAL_LABEL="SAM-refined"
else
  EVAL_ROOT="$RAW_RUN_ROOT"
  EVAL_LABEL="raw"
fi
echo "[phase C] evaluate_predictions × ${#GRIDS[@]} grid(s) on $EVAL_LABEL output (PARALLEL=$PARALLEL)"
running=0
for g in "${GRIDS[@]}"; do
  phaseC_grid "$g" "$EVAL_ROOT" &
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
  csv="$EVAL_ROOT/$g/presence_metrics.csv"
  if [ -s "$csv" ]; then
    awk -F, 'NR==2 {print $0}' "$csv" >> "$SUMMARY" 2>/dev/null || true
  else
    echo "$g,MISSING,,,,,,," >> "$SUMMARY"
  fi
done
echo "=== summary written to $SUMMARY ($EVAL_LABEL) ==="
column -s, -t "$SUMMARY" | head -40
