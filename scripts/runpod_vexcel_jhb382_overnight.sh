#!/usr/bin/env bash
# Overnight unified_reviewall_A + per-detection finalize + SAM mask+box on
# all 382 JNB Vexcel grids. Production decision 2026-05-14 per
# docs/experiments/2026-05-14-jhb-cbd25-3model-sam.md:
#   - unified_A per-det+SAM has -33 % halo, oversize p95 -3.0 vs train20
#   - 6 pp better gt match rate vs V3-C+SAM
#   - σ_Bw 0.157 (train20 0.148 marginally tighter on bulk calibration only)
# Polygon-conf c=0.925 is applied at consumption time on predictions_metric.gpkg,
# NOT in the pipeline (keep low-conf polygons available for downstream sweeps).
#
# Pipeline per batch:
#   1. stage  : copy /workspace tiles -> /dev/shm/tiles/<JNB.../>
#   2. phaseA : detect_direct.py + finalize.py (per-detection, v4_canonical) parallel × PARALLEL
#   3. phaseB : sam_refine_maskbox.py (one batch call covering the whole batch)
#   4. clear  : rm /dev/shm staging for this batch
#
# Resume semantics:
#   - SAM output present  -> grid marked done, fully skipped
#   - raw finalize present -> phaseA skipped, phaseB still runs
#   - tile dir absent      -> grid logged to tiles_missing.txt, not retried
#
# Status (consumed by overnight supervisor agent):
#   $STATUS_DIR/queue.txt          one JNB id per to-do grid (built once)
#   $STATUS_DIR/done.txt           appended when SAM output appears
#   $STATUS_DIR/running.txt        appended when phaseA starts
#   $STATUS_DIR/failed.txt         "<grid> <phase> rc=<n>"
#   $STATUS_DIR/tiles_missing.txt  grids skipped for missing tiles
#   $STATUS_DIR/heartbeat.txt      ISO-8601 line per milestone (stage/phaseA/phaseB/clear/done)
#
# Override via env vars; defaults assume RunPod /workspace layout.

set -uo pipefail
# Intentionally NOT set -e: a single grid crash must not kill the whole night.

REPO="${REPO:-/workspace/ZAsolar}"
REGION="${REGION:-johannesburg}"
IMAGERY_LAYER="${IMAGERY_LAYER:-vexcel_2024}"
MODEL_PATH="${MODEL_PATH:-$REPO/checkpoints/exp_unified_reviewall_A/best_model.pth}"
MODEL_RUN_ID="${MODEL_RUN_ID:-unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382}"
POSTPROC="${POSTPROC:-$REPO/configs/postproc/v4_canonical.json}"

TILES_SRC="${TILES_SRC:-/workspace/tiles/johannesburg/vexcel_2024}"
SHM_TILES="${SHM_TILES:-/dev/shm/tiles}"

RESULTS_RAW_ROOT="${RESULTS_RAW_ROOT:-$REPO/results/johannesburg/${MODEL_RUN_ID}_raw}"
RESULTS_SAM_ROOT="${RESULTS_SAM_ROOT:-$REPO/results/johannesburg/${MODEL_RUN_ID}_sam_maskbox}"

GRID_LIST_FILE="${GRID_LIST_FILE:-$REPO/configs/jnb_vexcel_full382_grids.txt}"
STAMP="$(date +%Y%m%d_%H%M)"
LOG_ROOT="${LOG_ROOT:-/workspace/logs/vexcel_jhb382_${STAMP}}"
STATUS_DIR="${STATUS_DIR:-/workspace/status/vexcel_jhb382_${STAMP}}"

BATCH_SIZE="${BATCH_SIZE:-150}"
PARALLEL="${PARALLEL:-6}"
DETECT_BATCH_SIZE="${DETECT_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SAM_BATCH_SIZE="${SAM_BATCH_SIZE:-16}"
SAM_PROMPT_MODE="${SAM_PROMPT_MODE:-mask_box}"

mkdir -p "$LOG_ROOT" "$STATUS_DIR" "$SHM_TILES" "$RESULTS_RAW_ROOT" "$RESULTS_SAM_ROOT"

QUEUE_FILE="$STATUS_DIR/queue.txt"
DONE_FILE="$STATUS_DIR/done.txt"
RUNNING_FILE="$STATUS_DIR/running.txt"
FAILED_FILE="$STATUS_DIR/failed.txt"
TILES_MISSING_FILE="$STATUS_DIR/tiles_missing.txt"
HEARTBEAT_FILE="$STATUS_DIR/heartbeat.txt"
META_FILE="$STATUS_DIR/meta.json"

: > "$QUEUE_FILE"
: > "$DONE_FILE"
: > "$RUNNING_FILE"
: > "$FAILED_FILE"
: > "$TILES_MISSING_FILE"
: > "$HEARTBEAT_FILE"

heartbeat() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$HEARTBEAT_FILE"
}

atomic_append() {
  local file=$1 line=$2
  ( flock -x 200; echo "$line" >> "$file" ) 200>"${file}.lock"
}

[ ! -f "$GRID_LIST_FILE" ] && { echo "[fatal] grid list missing: $GRID_LIST_FILE" >&2; exit 2; }
[ ! -f "$MODEL_PATH" ]    && { echo "[fatal] checkpoint missing: $MODEL_PATH"    >&2; exit 2; }
[ ! -f "$POSTPROC" ]      && { echo "[fatal] postproc missing: $POSTPROC"        >&2; exit 2; }
[ ! -d "$TILES_SRC" ]     && { echo "[fatal] tiles source missing: $TILES_SRC"   >&2; exit 2; }

cd "$REPO"

cat > "$META_FILE" <<JSON
{
  "stamp": "$STAMP",
  "model_run_id": "$MODEL_RUN_ID",
  "model_path": "$MODEL_PATH",
  "postproc": "$POSTPROC",
  "region": "$REGION",
  "imagery_layer": "$IMAGERY_LAYER",
  "tiles_src": "$TILES_SRC",
  "results_raw_root": "$RESULTS_RAW_ROOT",
  "results_sam_root": "$RESULTS_SAM_ROOT",
  "grid_list_file": "$GRID_LIST_FILE",
  "batch_size": $BATCH_SIZE,
  "parallel": $PARALLEL
}
JSON

# Build queue: classify each grid as done / missing-tiles / to-do
while IFS= read -r raw; do
  g=$(echo "$raw" | tr -d '[:space:]')
  [ -z "$g" ] && continue
  if [ -f "$RESULTS_SAM_ROOT/$g/predictions_metric.gpkg" ]; then
    atomic_append "$DONE_FILE" "$g"
    continue
  fi
  if [ ! -d "$TILES_SRC/$g" ] || ! ls "$TILES_SRC/$g"/*_geo.tif >/dev/null 2>&1; then
    atomic_append "$TILES_MISSING_FILE" "$g"
    continue
  fi
  atomic_append "$QUEUE_FILE" "$g"
done < "$GRID_LIST_FILE"

heartbeat "queue built: to_do=$(wc -l < "$QUEUE_FILE") already_done=$(wc -l < "$DONE_FILE") tiles_missing=$(wc -l < "$TILES_MISSING_FILE")"

run_phaseA_grid() {
  local g=$1
  local raw_out="$RESULTS_RAW_ROOT/$g"
  local grid_log="$LOG_ROOT/phaseA/${g}.log"
  mkdir -p "$raw_out" "$(dirname "$grid_log")"

  if [ -f "$raw_out/predictions_metric.gpkg" ]; then
    echo "[$g] phaseA cached" >> "$grid_log"
    return 0
  fi

  atomic_append "$RUNNING_FILE" "$g"
  echo "[$g] $(date -u +%H:%M:%SZ) detect_direct" >> "$grid_log"
  python3 -u "$REPO/detect_direct.py" \
    --grid-id "$g" \
    --region "$REGION" \
    --imagery-layer "$IMAGERY_LAYER" \
    --model-run "$MODEL_RUN_ID" \
    --model-path "$MODEL_PATH" \
    --output-dir "$raw_out" \
    --batch-size "$DETECT_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --prefetch-factor 2 \
    --chip-size 400 \
    --overlap 0.25 \
    --detector-score-threshold 0.05 \
    --detections-per-img 300 \
    --mask-threshold 0.3 \
    --raw-mask-storage crop \
    --device cuda \
    >> "$grid_log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    atomic_append "$FAILED_FILE" "$g phaseA_detect rc=$rc"
    return $rc
  fi

  echo "[$g] $(date -u +%H:%M:%SZ) finalize" >> "$grid_log"
  python3 -u "$REPO/finalize.py" \
    --input "$raw_out/raw_detections.pkl" \
    --output-dir "$raw_out" \
    --postproc-config "$POSTPROC" \
    --merge-mode per-detection \
    >> "$grid_log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    atomic_append "$FAILED_FILE" "$g phaseA_finalize rc=$rc"
    return $rc
  fi
  return 0
}

run_phaseB_batch() {
  local batch_label=$1
  shift
  local grids=("$@")
  local sam_log="$LOG_ROOT/phaseB/${batch_label}.log"
  mkdir -p "$(dirname "$sam_log")"

  echo "[$(date -u +%H:%M:%SZ)] phaseB ${batch_label} (${#grids[@]} grids)" > "$sam_log"
  python3 -u "$REPO/scripts/analysis/sam_refine_maskbox.py" \
    --region "$REGION" \
    --grids "${grids[@]}" \
    --src-results-root "$RESULTS_RAW_ROOT" \
    --imagery-layer "$IMAGERY_LAYER" \
    --output-root "$RESULTS_SAM_ROOT" \
    --prompt-mode "$SAM_PROMPT_MODE" \
    --sam-batch-size "$SAM_BATCH_SIZE" \
    --label "${MODEL_RUN_ID}_sam_maskbox" \
    >> "$sam_log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    atomic_append "$FAILED_FILE" "$batch_label phaseB_sam rc=$rc"
  fi
  return $rc
}

stage_batch_to_shm() {
  local grids=("$@")
  for g in "${grids[@]}"; do
    [ -d "$SHM_TILES/$g" ] && continue
    cp -r "$TILES_SRC/$g" "$SHM_TILES/$g" &
  done
  wait
}

clear_shm_batch() {
  local grids=("$@")
  for g in "${grids[@]}"; do
    rm -rf "${SHM_TILES:?}/$g"
  done
}

mapfile -t ALL_GRIDS < "$QUEUE_FILE"
TOTAL=${#ALL_GRIDS[@]}
echo "[main] queue=$TOTAL batch=$BATCH_SIZE parallel=$PARALLEL"
heartbeat "main start: total=$TOTAL"

SECONDS=0
batch_idx=0
i=0
while [ $i -lt $TOTAL ]; do
  batch_idx=$((batch_idx + 1))
  batch_label="batch$(printf '%03d' $batch_idx)"
  batch_end=$((i + BATCH_SIZE))
  [ $batch_end -gt $TOTAL ] && batch_end=$TOTAL
  batch_grids=("${ALL_GRIDS[@]:$i:$((batch_end - i))}")

  echo
  echo "============================================================"
  echo "## $batch_label grids[$i..$((batch_end - 1))] (${#batch_grids[@]}), elapsed=${SECONDS}s"
  echo "============================================================"
  heartbeat "$batch_label begin: n=${#batch_grids[@]}"

  echo "[$batch_label] stage tiles -> $SHM_TILES"
  stage_batch_to_shm "${batch_grids[@]}"
  heartbeat "$batch_label staged: shm_use=$(du -sh "$SHM_TILES" 2>/dev/null | cut -f1)"

  echo "[$batch_label] phaseA detect+finalize (parallel=$PARALLEL)"
  running=0
  for g in "${batch_grids[@]}"; do
    ( SOLAR_TILES_ROOT="$SHM_TILES" run_phaseA_grid "$g" ) &
    running=$((running + 1))
    if [ $running -ge $PARALLEL ]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
  heartbeat "$batch_label phaseA done"

  echo "[$batch_label] phaseB SAM refine"
  ( SOLAR_TILES_ROOT="$SHM_TILES" run_phaseB_batch "$batch_label" "${batch_grids[@]}" )
  heartbeat "$batch_label phaseB done"

  for g in "${batch_grids[@]}"; do
    if [ -f "$RESULTS_SAM_ROOT/$g/predictions_metric.gpkg" ]; then
      atomic_append "$DONE_FILE" "$g"
    fi
  done

  echo "[$batch_label] clear shm"
  clear_shm_batch "${batch_grids[@]}"
  heartbeat "$batch_label cleared shm"

  i=$batch_end
done

DONE_N=$(wc -l < "$DONE_FILE")
FAIL_N=$(wc -l < "$FAILED_FILE")
MISS_N=$(wc -l < "$TILES_MISSING_FILE")
heartbeat "main end: done=$DONE_N failed=$FAIL_N tiles_missing=$MISS_N elapsed=${SECONDS}s"

echo
echo "============================================================"
echo "## FINAL done=$DONE_N failed=$FAIL_N tiles_missing=$MISS_N elapsed=${SECONDS}s"
echo "## status   $STATUS_DIR"
echo "## raw      $RESULTS_RAW_ROOT"
echo "## sam      $RESULTS_SAM_ROOT"
echo "============================================================"
