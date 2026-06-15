#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# CT census overnight orchestrator — Phases B/C/D. RUN INSIDE tmux on the pod:
#     tmux new -s census 'bash scripts/ct_census_run.sh; bash'
#
# Addressing: CPT census ids end-to-end (geometry guaranteed by
#   data/task_grid_cpt.gpkg). download_tiles writes tiles under the source dir
#   that resolve_tiles_dir/detect/classifier read; results are CPT-keyed. No
#   CPT<->G bookkeeping in this script.
#
# Phase B  pipeline: a background WMS downloader stays AHEAD of a batched
#          inferencer. Tiles read straight off local NVMe (community local disk
#          is fast — the /dev/shm rule targets the slow MFS network volume, not
#          here). Per batch: detect (PARALLEL=6), back up essentials to Dropbox.
# Phase C  after ALL inference: cls v2 (classify-all) over every grid in one
#          pass, then merge -> single CPT census inventory.
# Phase D  final backup + VERIFY (remote vs local), write BACKUP_VERIFIED.
#          Does NOT stop the pod — the operator confirms then stops.
#
# Fully resumable: per-grid .ok/.fail markers in $STATE; re-running skips done.
# Threshold policy: detector at MAX RECALL (v4_canonical, no extra polygon-conf
#   cut) + cls recall-calibrated to 0.95 PV-recall. cls recovers precision.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail   # deliberately NOT -e: one bad grid must not kill the run

# ── config (env-overridable) ────────────────────────────────────────────────
ZAS=${ZAS:-/root/ZAsolar}
CLS=${CLS:-/root/solar_cls}
RUN=${RUN:-unifiedA_census_perdet}
DETECTOR_CKPT=${DETECTOR_CKPT:-$ZAS/checkpoints/exp_unified_reviewall_A/best_model.pth}
CLS_CKPT=${CLS_CKPT:-$CLS/checkpoints/cls_pv_thermal_v2_dinov2_vits14/best_cls.pth}
POSTPROC=${POSTPROC:-$ZAS/configs/postproc/v4_canonical.json}
TILES_DISK=${TILES_DISK:-/root/tiles_disk}     # persistent local tile store (kept for cls)
PARALLEL=${PARALLEL:-6}                         # detect procs (RTX 5090)
DL_WORKERS=${DL_WORKERS:-4}                     # concurrent WMS grid downloads
BATCH=${BATCH:-100}                            # backup granularity
DROPBOX_DEST=${DROPBOX_DEST:-dropbox:RA_Solar/Gao/ct_census}
STATE=${STATE:-/root/census_state}
LOGS=${LOGS:-/root/census_logs}
GLIST="$STATE/glist.txt"

mkdir -p "$STATE" "$LOGS" "$TILES_DISK"
[ -f "$GLIST" ] || { echo "no $GLIST — run scripts/ct_census_setup.sh first"; exit 1; }
[ -f "$DETECTOR_CKPT" ] || { echo "no detector ckpt $DETECTOR_CKPT"; exit 1; }
[ -f "$CLS_CKPT" ] || { echo "no cls ckpt $CLS_CKPT"; exit 1; }

cd "$ZAS" && source scripts/activate_env.sh
RESULTS_DIR=$(python -c "from core.grid_utils import get_results_root;print(get_results_root('ct',model_run='$RUN'))")
TOTAL=$(wc -l < "$GLIST")
echo "results dir: $RESULTS_DIR | grids: $TOTAL | batch: $BATCH | parallel: $PARALLEL | tiles: $TILES_DISK"

now(){ date +%s; }
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
write_status(){
  local dl inf bk dlf inff
  dl=$(ls "$STATE"/dl_*.ok 2>/dev/null | wc -l)
  inf=$(ls "$STATE"/infer_*.ok 2>/dev/null | wc -l)
  bk=$(ls "$STATE"/bk_*.ok 2>/dev/null | wc -l)
  dlf=$(ls "$STATE"/dl_*.fail 2>/dev/null | wc -l)
  inff=$(ls "$STATE"/infer_*.fail 2>/dev/null | wc -l)
  { echo "phase=$(cat "$STATE/PHASE" 2>/dev/null)"
    echo "total=$TOTAL download=$dl/$TOTAL(fail $dlf) infer=$inf/$TOTAL(fail $inff) backup=$bk"
    echo "run=$RUN results=$RESULTS_DIR"
    echo "last_update=$(now) ts=$(date '+%F %T')"
  } > "$STATE/STATUS.txt"
}

# ── helpers exported for xargs subshells ────────────────────────────────────
export ZAS CLS RUN DETECTOR_CKPT POSTPROC TILES_DISK STATE LOGS RESULTS_DIR

dl_one(){
  local G=$1
  [ -f "$STATE/dl_$G.ok" ] && return 0
  SOLAR_TILES_ROOT="$TILES_DISK" python "$ZAS/scripts/imagery/download_tiles.py" \
    --grid-id "$G" --region ct > "$LOGS/dl_$G.log" 2>&1
  if grep -q "errors=0" "$LOGS/dl_$G.log"; then     # download_grid prints errors=N on completion
    rm -f "$STATE/dl_$G.fail"; touch "$STATE/dl_$G.ok"
  else
    touch "$STATE/dl_$G.fail"
  fi
}
infer_one(){
  local G=$1
  [ -f "$STATE/infer_$G.ok" ] && return 0
  SOLAR_TILES_ROOT="$TILES_DISK" python "$ZAS/detect_and_evaluate.py" \
    --grid-id "$G" --region ct --imagery-layer aerial_2025 \
    --model-run "$RUN" --model-path "$DETECTOR_CKPT" \
    --postproc-config "$POSTPROC" --data-scope full_grid --force \
    > "$LOGS/infer_$G.log" 2>&1
  if [ -f "$RESULTS_DIR/$G/predictions_metric.gpkg" ]; then
    rm -f "$STATE/infer_$G.fail"; touch "$STATE/infer_$G.ok"
  else
    touch "$STATE/infer_$G.fail"
  fi
}
export -f dl_one infer_one

# ── Phase B: background downloader (stays ahead of inference) ───────────────
echo "phaseB-download+infer" > "$STATE/PHASE"; write_status
log "starting background downloader (DL_WORKERS=$DL_WORKERS)"
( xargs -P "$DL_WORKERS" -I{} bash -c 'dl_one "$@"' _ {} < "$GLIST" ) >"$LOGS/downloader.log" 2>&1 &
DL_PID=$!

# ── Phase B: batched inferencer (consumes downloaded grids) ─────────────────
mapfile -t GRIDS < "$GLIST"
for ((i=0; i<${#GRIDS[@]}; i+=BATCH)); do
  BATCH_GRIDS=("${GRIDS[@]:i:BATCH}")
  bn=$(( i/BATCH + 1 ))
  log "batch $bn: ${#BATCH_GRIDS[@]} grids (${BATCH_GRIDS[0]}..${BATCH_GRIDS[-1]})"

  # wait for this batch's tiles to be downloaded (download stays ahead)
  for G in "${BATCH_GRIDS[@]}"; do
    while [ ! -f "$STATE/dl_$G.ok" ] && [ ! -f "$STATE/dl_$G.fail" ]; do
      sleep 5; write_status
      kill -0 "$DL_PID" 2>/dev/null || log "WARN downloader exited early"
    done
  done

  # detect (parallel; reads tiles from local disk via resolve_tiles_dir fast-path)
  printf '%s\n' "${BATCH_GRIDS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'infer_one "$@"' _ {}

  # per-batch backup of essentials -> Dropbox (masks/ excluded; small + safe)
  for G in "${BATCH_GRIDS[@]}"; do
    [ -f "$STATE/infer_$G.ok" ] || continue
    [ -f "$STATE/bk_$G.ok" ] && continue
    if rclone copy "$RESULTS_DIR/$G" "$DROPBOX_DEST/results/$RUN/$G" \
         --include 'predictions_metric.gpkg' --include 'predictions.geojson' \
         --include 'config.json' >>"$LOGS/backup.log" 2>&1; then
      touch "$STATE/bk_$G.ok"
    fi
  done

  write_status
  log "batch $bn done: infer=$(ls "$STATE"/infer_*.ok 2>/dev/null|wc -l)/$TOTAL backup=$(ls "$STATE"/bk_*.ok 2>/dev/null|wc -l)"
done

wait "$DL_PID" 2>/dev/null
log "Phase B complete. infer ok=$(ls "$STATE"/infer_*.ok 2>/dev/null|wc -l) fail=$(ls "$STATE"/infer_*.fail 2>/dev/null|wc -l)"

# ── Phase C: cls v2 over ALL grids (one pass) + merge ───────────────────────
echo "phaseC-cls+merge" > "$STATE/PHASE"; write_status
OK_GRIDS=(); for G in "${GRIDS[@]}"; do [ -f "$STATE/infer_$G.ok" ] && OK_GRIDS+=("$G"); done
log "cls classify-all over ${#OK_GRIDS[@]} grids"
( cd "$CLS" && source scripts/activate_env.sh
  SOLAR_TILES_ROOT="$TILES_DISK" ZASOLAR_ROOT="$ZAS" python scripts/classifier/classify_predictions.py \
    --grid-ids "${OK_GRIDS[@]}" --region ct --imagery-layer aerial_2025 \
    --model-path "$CLS_CKPT" --classify-all --results-dir "$RESULTS_DIR" --batch-size 64 \
) > "$LOGS/cls_finalize.log" 2>&1 || log "WARN cls finalize returned nonzero — check $LOGS/cls_finalize.log"

log "merging -> single CPT census inventory"
MERGED="$STATE/ct_census_inventory_cpt.gpkg"
python "$ZAS/scripts/ct_census_merge.py" \
  --results-dir "$RESULTS_DIR" --glist "$GLIST" --state "$STATE" \
  --run "$RUN" --out "$MERGED" > "$LOGS/merge.log" 2>&1 || log "WARN merge nonzero — see $LOGS/merge.log"

# ── Phase D: final backup + VERIFY ──────────────────────────────────────────
echo "phaseD-verify-backup" > "$STATE/PHASE"; write_status
log "backing up cls outputs + merged inventory -> Dropbox"
rclone copy "$RESULTS_DIR" "$DROPBOX_DEST/results/$RUN" \
  --include '*/predictions_metric_cls_filtered.gpkg' \
  --include '*/predictions_metric.gpkg' --include '*/config.json' \
  >>"$LOGS/backup.log" 2>&1
rclone copy "$MERGED" "$DROPBOX_DEST/" >>"$LOGS/backup.log" 2>&1

# verify: merged file present remotely at matching size; per-grid filtered counts agree
loc_filt=$(ls "$RESULTS_DIR"/*/predictions_metric_cls_filtered.gpkg 2>/dev/null | wc -l)
rem_filt=$(rclone lsf -R "$DROPBOX_DEST/results/$RUN" 2>/dev/null | grep -c 'predictions_metric_cls_filtered.gpkg')
loc_sz=$(stat -c%s "$MERGED" 2>/dev/null || echo 0)
rem_sz=$(rclone size "$DROPBOX_DEST/$(basename "$MERGED")" --json 2>/dev/null | grep -o '"bytes":[0-9]*' | grep -o '[0-9]*' || echo -1)
log "verify: merged_local=${loc_sz}B merged_remote=${rem_sz}B | filtered local=$loc_filt remote=$rem_filt"
if [ "$loc_sz" -gt 0 ] && [ "$loc_sz" = "$rem_sz" ] && [ "$loc_filt" = "$rem_filt" ] && [ "$loc_filt" -gt 0 ]; then
  echo "verified=$(date '+%F %T') merged=${loc_sz}B grids_filtered=$loc_filt" > "$STATE/BACKUP_VERIFIED"
  echo "done" > "$STATE/PHASE"; write_status
  log "✅ BACKUP VERIFIED. Safe to stop the pod (operator confirms):  bash scripts/runpod_pod.sh stop"
else
  log "❌ BACKUP VERIFY FAILED — DO NOT STOP THE POD. Inspect $LOGS/backup.log"
fi
