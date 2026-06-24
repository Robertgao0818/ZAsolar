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
ENGINE_ID=${ENGINE_ID:-detect_direct_finalize_perdet_v4canonical_cls_adaptive_v1}
DETECTOR_CKPT=${DETECTOR_CKPT:-$ZAS/checkpoints/exp_unified_reviewall_A/best_model.pth}
# cls = the LOCKED CT winner: adaptive_v1 checkpoint + adaptive thresholds table
# (cov50 0.829/σ_Bw 0.260, project_cls_chip_adaptive_upgrade). classify_predictions
# now ALWAYS extracts adaptive_v1 chips, so the non-adaptive fixed-400 checkpoint
# (cls_pv_thermal_v2_dinov2_vits14) is a train/inference chip-geometry mismatch and
# MUST NOT be used here. The matched threshold table (aerial_2025 dinov2 = 0.7168,
# top-level chip_spec_version=adaptive_v1) must be passed explicitly — the DEFAULT
# thresholds_v2.json is the stale fixed-400 table (0.5396) whose chip_spec_version
# guard is inert (no top-level key), so it would silently mis-calibrate FP suppression.
CLS_CKPT=${CLS_CKPT:-$CLS/checkpoints/cls_pv_thermal_v2_dinov2_vits14_adaptive/best_cls.pth}
CLS_THRESHOLDS=${CLS_THRESHOLDS:-$CLS/configs/classifier/thresholds_v2_adaptive.json}
POSTPROC=${POSTPROC:-$ZAS/configs/postproc/v4_canonical.json}
TILES_DISK=${TILES_DISK:-/root/tiles_disk}     # persistent local tile store (kept for cls)
PARALLEL=${PARALLEL:-6}                         # detect procs (RTX 5090)
DL_WORKERS=${DL_WORKERS:-4}                     # concurrent WMS grid downloads
BATCH=${BATCH:-100}                            # backup granularity
DROPBOX_DEST=${DROPBOX_DEST:-dropbox:RA_Solar/Gao/ct_census}
STATE=${STATE:-/root/census_state}
LOGS=${LOGS:-/root/census_logs}
GLIST="$STATE/glist.txt"

# Fragmentation guard: N parallel detect_direct procs each load their own Mask R-CNN
# on one 5090 (32GB). expandable_segments keeps a late dense grid from OOM-ing a
# long run (feedback_vexcel_inference_parallel_oom). Keep PARALLEL≤6 for CT WMS tiles;
# do NOT raise to 10 without first reading peak-VRAM off the first batch's log.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
[ "$PARALLEL" -gt 6 ] && echo "WARN: PARALLEL=$PARALLEL > 6 — OOM risk on a 32GB 5090; verify peak VRAM before trusting this." >&2

mkdir -p "$STATE" "$LOGS" "$TILES_DISK"
[ -f "$GLIST" ] || { echo "no $GLIST — run scripts/ct_census_setup.sh first"; exit 1; }
[ -f "$DETECTOR_CKPT" ] || { echo "no detector ckpt $DETECTOR_CKPT"; exit 1; }
[ -f "$CLS_CKPT" ] || { echo "no cls ckpt $CLS_CKPT"; exit 1; }
[ -f "$CLS_THRESHOLDS" ] || { echo "no cls thresholds table $CLS_THRESHOLDS"; exit 1; }

cd "$ZAS" && source scripts/activate_env.sh
RESULTS_DIR=$(python -c "from core.grid_utils import get_results_root;print(get_results_root('ct',model_run='$RUN'))")
TOTAL=$(wc -l < "$GLIST")
echo "results dir: $RESULTS_DIR | grids: $TOTAL | batch: $BATCH | parallel: $PARALLEL | tiles: $TILES_DISK"

STATE_ENGINE="$STATE/CENSUS_ENGINE"
EXPECTED_ENGINE="run=$RUN results=$RESULTS_DIR engine=$ENGINE_ID"
terminal_markers=$(ls "$STATE"/infer_*.ok "$STATE"/infer_*.empty "$STATE"/infer_*.fail 2>/dev/null | wc -l)
if [ -f "$STATE_ENGINE" ]; then
  current_engine=$(cat "$STATE_ENGINE")
  if [ "$current_engine" != "$EXPECTED_ENGINE" ]; then
    if [ "$terminal_markers" -gt 0 ]; then
      echo "STATE engine mismatch: $STATE_ENGINE says '$current_engine', expected '$EXPECTED_ENGINE'."
      echo "Clear old infer_*.{ok,empty,fail} and bk_*.ok markers before restarting this census caliber."
      exit 1
    fi
    echo "$EXPECTED_ENGINE" > "$STATE_ENGINE"
  fi
elif [ "$terminal_markers" -gt 0 ]; then
  echo "STATE has $terminal_markers existing infer markers but no direct-engine stamp."
  echo "This is likely the aborted geoai-caliber run. Clear infer_*.{ok,empty,fail} and bk_*.ok before restart."
  exit 1
else
  echo "$EXPECTED_ENGINE" > "$STATE_ENGINE"
fi

now(){ date +%s; }
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
write_status(){
  local dl inf emp bk dlf inff
  dl=$(ls "$STATE"/dl_*.ok 2>/dev/null | wc -l)
  inf=$(ls "$STATE"/infer_*.ok 2>/dev/null | wc -l)
  emp=$(ls "$STATE"/infer_*.empty 2>/dev/null | wc -l)
  bk=$(ls "$STATE"/bk_*.ok 2>/dev/null | wc -l)
  dlf=$(ls "$STATE"/dl_*.fail 2>/dev/null | wc -l)
  inff=$(ls "$STATE"/infer_*.fail 2>/dev/null | wc -l)
  { echo "phase=$(cat "$STATE/PHASE" 2>/dev/null)"
    echo "total=$TOTAL download=$dl/$TOTAL(fail $dlf) infer=$inf empty=$emp fail=$inff done=$((inf+emp))/$TOTAL backup=$bk"
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
  # terminal states needing no recompute: detected (.ok) OR surveyed-empty (.empty)
  { [ -f "$STATE/infer_$G.ok" ] || [ -f "$STATE/infer_$G.empty" ]; } && return 0
  local OUT="$RESULTS_DIR/$G"
  mkdir -p "$OUT"
  # Start each non-terminal attempt from a clean direct-chain artifact set. This
  # prevents an interrupted/failed retry from finalizing an old raw_detections.pkl
  # or preserving a stale gpkg from the previous geoai-caliber run.
  rm -f "$OUT/raw_detections.pkl" "$OUT/predictions_metric.gpkg" \
        "$OUT/predictions.geojson" "$OUT/config.json"
  rm -rf "$OUT/masks" "$OUT/vectors"
  # ENGINE = direct Mask R-CNN (NO geoai). Two stages, matching the LOCKED CT
  # baseline `unifiedA_li_perdet` (regions.yaml): detect_direct.py -> finalize.py
  # --merge-mode per-detection, v4_canonical. detect_direct feeds the GPU via a
  # DataLoader (workers/prefetch); geoai's path used a num_workers=0 loop that
  # starved the GPU AND wrote per_detection_geoai output the cls calib never saw.
  # Params MIRROR run_benchmark.py's direct pipeline that BUILT the baseline.
  # detect_direct's DIRECT-mode default for detections-per-img is 300 (NOT 100 —
  # the 100 value only applies under --parity-mode geoai, which we never set).
  # run_benchmark.py also omits the flag and so also ran at 300, so 300 IS the
  # baseline. We pass it EXPLICITLY (as every other orchestrator does) so nobody
  # can "fix" the script toward 100 and silently halve the proposal cap off the
  # cls calibration. score-thresh 0.05 / raw-mask-storage crop come from defaults
  # and match the baseline. batch/workers/prefetch are throughput-only (no numeric
  # effect). detect_direct resolves tiles by SOURCE grid id (CPT→G) since this fix.
  SOLAR_TILES_ROOT="$TILES_DISK" python "$ZAS/detect_direct.py" \
    --grid-id "$G" --region ct --imagery-layer aerial_2025 \
    --model-run "$RUN" --model-path "$DETECTOR_CKPT" --output-dir "$OUT" \
    --detections-per-img 300 \
    --chip-size 400 --overlap 0.25 --mask-threshold 0.3 \
    --batch-size 4 --num-workers 2 --prefetch-factor 2 --device cuda \
    > "$LOGS/infer_$G.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    rm -f "$STATE/infer_$G.ok" "$STATE/infer_$G.empty"
    touch "$STATE/infer_$G.fail"
    rm -rf "$OUT/raw_detections.pkl" "$OUT/masks" "$OUT/vectors"
    return 0
  fi
  SOLAR_TILES_ROOT="$TILES_DISK" python "$ZAS/finalize.py" \
    --input "$OUT/raw_detections.pkl" --output-dir "$OUT" \
    --postproc-config "$POSTPROC" --merge-mode per-detection \
    --allow-overwrite-canonical \
    >> "$LOGS/infer_$G.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    rm -f "$STATE/infer_$G.ok" "$STATE/infer_$G.empty"
    touch "$STATE/infer_$G.fail"
    rm -f "$OUT/predictions_metric.gpkg" "$OUT/predictions.geojson" "$OUT/config.json"
    rm -rf "$OUT/raw_detections.pkl" "$OUT/masks" "$OUT/vectors"
    return 0
  fi
  # State machine. ORDER MATTERS: under the direct chain a surveyed-empty grid
  # (0 detections) STILL writes an EMPTY predictions_metric.gpkg (finalize prints
  # "no polygons after vectorization — writing empty outputs"), so the empty
  # signal MUST be tested BEFORE gpkg-exists — unlike the old geoai chain, which
  # wrote no gpkg on zero detections and was checked the other way round.
  if grep -q 'no polygons after vectorization' "$LOGS/infer_$G.log"; then
    # surveyed, count=0: a VALID census zero (surveyed-empty), NOT a failure.
    rm -f "$STATE/infer_$G.fail"; touch "$STATE/infer_$G.empty"
  elif [ -s "$OUT/predictions_metric.gpkg" ]; then
    rm -f "$STATE/infer_$G.fail" "$STATE/infer_$G.empty"; touch "$STATE/infer_$G.ok"
  else
    # no empty-signal AND no gpkg = real failure (detect/finalize crash, OOM, no
    # tiles). Keep as .fail: retried on resume, flagged 'infer_failed' in manifest.
    touch "$STATE/infer_$G.fail"
  fi
  # Reap heavy intermediates at TERMINAL state. The direct chain's disk hogs are
  # raw_detections.pkl (mask crops) + finalize's masks/ + vectors/ dirs; nothing
  # downstream reads them (inventory = predictions_metric.gpkg; cls reads
  # gpkg+tiles; backup/merge read gpkg).
  rm -rf "$OUT/raw_detections.pkl" "$OUT/masks" "$OUT/vectors"
}
export -f dl_one infer_one

# ── Phase B: background downloader (stays ahead of inference) ───────────────
echo "phaseB-download+infer" > "$STATE/PHASE"; write_status
log "starting background downloader (DL_WORKERS=$DL_WORKERS)"
( xargs -P "$DL_WORKERS" -I{} bash -c 'dl_one "$@"' _ {} < "$GLIST" ) >"$LOGS/downloader.log" 2>&1 &
DL_PID=$!

# ── Phase B: batched inferencer (consumes downloaded grids) ─────────────────
mapfile -t GRIDS < "$GLIST"
BK_PIDS=()                                     # background per-batch backup PIDs
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

  # per-batch backup of essentials -> Dropbox, in the BACKGROUND so the next
  # batch's detect (GPU) overlaps this upload (network I/O — disjoint resources).
  # Backups never pile up: one upload (~min) << one detect batch (~tens of min).
  # Phase D's full recursive rclone sweep is the completeness backstop, so an
  # interrupted bg backup is harmless; bk_*.ok markers make it resumable too.
  ( for G in "${BATCH_GRIDS[@]}"; do
      [ -f "$STATE/infer_$G.ok" ] || continue
      [ -f "$STATE/bk_$G.ok" ] && continue
      if rclone copy "$RESULTS_DIR/$G" "$DROPBOX_DEST/results/$RUN/$G" \
           --include 'predictions_metric.gpkg' --include 'predictions.geojson' \
           --include 'config.json' >>"$LOGS/backup.log" 2>&1; then
        touch "$STATE/bk_$G.ok"
      fi
    done ) &
  BK_PIDS+=($!)

  write_status
  log "batch $bn detect done: infer=$(ls "$STATE"/infer_*.ok 2>/dev/null|wc -l)/$TOTAL backup=$(ls "$STATE"/bk_*.ok 2>/dev/null|wc -l) (bg backup running)"
done

# drain any in-flight background per-batch backups before the final phases
log "Phase B inference done; draining ${#BK_PIDS[@]} background backups..."
for p in "${BK_PIDS[@]}"; do wait "$p" 2>/dev/null; done
wait "$DL_PID" 2>/dev/null
log "Phase B complete. infer ok=$(ls "$STATE"/infer_*.ok 2>/dev/null|wc -l) empty=$(ls "$STATE"/infer_*.empty 2>/dev/null|wc -l) fail=$(ls "$STATE"/infer_*.fail 2>/dev/null|wc -l)"

# ── Phase C: cls v2 over ALL grids (one pass) + merge ───────────────────────
echo "phaseC-cls+merge" > "$STATE/PHASE"; write_status
OK_GRIDS=(); for G in "${GRIDS[@]}"; do [ -f "$STATE/infer_$G.ok" ] && OK_GRIDS+=("$G"); done
log "cls classify-all over ${#OK_GRIDS[@]} grids"
( cd "$CLS" && source scripts/activate_env.sh
  SOLAR_TILES_ROOT="$TILES_DISK" ZASOLAR_ROOT="$ZAS" python scripts/classifier/classify_predictions.py \
    --grid-ids "${OK_GRIDS[@]}" --region ct --imagery-layer aerial_2025 \
    --model-path "$CLS_CKPT" --thresholds-v2 "$CLS_THRESHOLDS" \
    --classify-all --results-dir "$RESULTS_DIR" --batch-size 64 \
) > "$LOGS/cls_finalize.log" 2>&1 || log "WARN cls finalize returned nonzero — check $LOGS/cls_finalize.log"

log "merging -> single CPT census inventory"
MERGED="$STATE/ct_census_inventory_cpt.gpkg"
python "$ZAS/scripts/ct_census_merge.py" \
  --results-dir "$RESULTS_DIR" --glist "$GLIST" --state "$STATE" \
  --run "$RUN" --out "$MERGED" > "$LOGS/merge.log" 2>&1 || log "WARN merge nonzero — see $LOGS/merge.log"

# per-grid coverage manifest: every CPT cell tagged ok / empty / infer_failed /
# download_failed / not_reached (+census_count). The grid-level deliverable that
# the polygon inventory cannot express — surveyed-empty cells (count=0) included.
log "building per-grid coverage manifest"
MANIFEST="$STATE/ct_census_coverage_cpt"
python "$ZAS/scripts/ct_census_coverage_manifest.py" \
  --glist "$GLIST" --state "$STATE" --results-dir "$RESULTS_DIR" --logs "$LOGS" \
  --run "$RUN" --task-grid "$ZAS/data/task_grid_cpt.gpkg" \
  --out "$MANIFEST" > "$LOGS/manifest.log" 2>&1 || log "WARN manifest nonzero — see $LOGS/manifest.log"
cat "$LOGS/manifest.log" | tail -3

# ── Phase D: final backup + VERIFY ──────────────────────────────────────────
echo "phaseD-verify-backup" > "$STATE/PHASE"; write_status
log "backing up cls outputs + merged inventory -> Dropbox"
rclone copy "$RESULTS_DIR" "$DROPBOX_DEST/results/$RUN" \
  --include '*/predictions_metric_cls_filtered.gpkg' \
  --include '*/predictions_metric.gpkg' --include '*/config.json' \
  >>"$LOGS/backup.log" 2>&1
rclone copy "$MERGED" "$DROPBOX_DEST/" >>"$LOGS/backup.log" 2>&1
# coverage manifest (.csv always, .gpkg if task-grid join succeeded) — the
# grid-level census denominator; ship it so surveyed-empty evidence survives teardown
for ext in csv gpkg; do
  [ -f "$MANIFEST.$ext" ] && rclone copy "$MANIFEST.$ext" "$DROPBOX_DEST/" >>"$LOGS/backup.log" 2>&1
done

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
