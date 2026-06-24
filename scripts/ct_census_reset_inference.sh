#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# CT census — reset INFERENCE state before re-running on a new engine/caliber.
#
# Why: the census was previously run on the geoai engine (detect_and_evaluate.py,
# pixel-OR caliber). The pipeline is now the direct chain (detect_direct ->
# finalize per-detection). Stale per-grid markers from the geoai run would make
# ct_census_run.sh infer_one SKIP those grids (infer_<G>.ok/.empty early-return),
# silently shipping geoai-caliber polygons into the merge. This clears the
# inference + backup state so every grid is recomputed on the new engine.
#
# KEEPS (so tiles are reused, not re-downloaded):
#   - dl_<G>.ok / dl_<G>.fail  (download markers)
#   - $TILES_DISK               (the downloaded WMS tiles)
# CLEARS:
#   - infer_<G>.{ok,empty,fail}  (inference state)
#   - bk_<G>.ok                  (per-grid Dropbox backup markers)
#   - BACKUP_VERIFIED            (stale verification flag)
#   - CENSUS_ENGINE              (engine/caliber stamp)
#   - optionally: per-grid result dirs under $RESULTS (the geoai-caliber gpkgs)
#
# SAFE BY DEFAULT: prints what it WOULD remove and exits. Set APPLY=1 to delete.
# Result dirs are only removed when WIPE_RESULTS=1 (also requires APPLY=1).
#
# Usage:
#   bash scripts/ct_census_reset_inference.sh                 # dry-run (markers)
#   APPLY=1 bash scripts/ct_census_reset_inference.sh         # clear markers only
#   APPLY=1 WIPE_RESULTS=1 bash scripts/ct_census_reset_inference.sh  # + result dirs
# Env: STATE, RESULTS (per-grid result root), GLIST.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
STATE=${STATE:-/root/census_state}
RESULTS=${RESULTS:-/root/ZAsolar/results}
TILES_DISK=${TILES_DISK:-/root/tiles_disk}
GLIST=${GLIST:-$STATE/glist.txt}
APPLY=${APPLY:-0}
WIPE_RESULTS=${WIPE_RESULTS:-0}

[ -d "$STATE" ] || { echo "no STATE dir: $STATE"; exit 1; }

n_ok=$(ls "$STATE"/infer_*.ok    2>/dev/null | wc -l)
n_em=$(ls "$STATE"/infer_*.empty 2>/dev/null | wc -l)
n_fa=$(ls "$STATE"/infer_*.fail  2>/dev/null | wc -l)
n_bk=$(ls "$STATE"/bk_*.ok       2>/dev/null | wc -l)
n_dl=$(ls "$STATE"/dl_*.ok       2>/dev/null | wc -l)
echo "STATE=$STATE  RESULTS=$RESULTS"
echo "would clear: infer.ok=$n_ok infer.empty=$n_em infer.fail=$n_fa bk.ok=$n_bk  (KEEP dl.ok=$n_dl + tiles)"

# result dirs corresponding to inference (only those with an infer marker / the
# glist grids), to avoid nuking unrelated dirs that share the bare results root.
res_dirs=()
if [ "$WIPE_RESULTS" = "1" ] && [ -f "$GLIST" ]; then
  while IFS= read -r g; do
    [ -n "$g" ] || continue
    [ -d "$RESULTS/$g" ] && res_dirs+=("$RESULTS/$g")
  done < "$GLIST"
  echo "would remove ${#res_dirs[@]} per-grid result dirs under $RESULTS (glist-scoped)"
fi

if [ "$APPLY" != "1" ]; then
  echo
  echo "DRY-RUN. Nothing deleted. Re-run with APPLY=1 (add WIPE_RESULTS=1 to also drop result dirs)."
  exit 0
fi

rm -f "$STATE"/infer_*.ok "$STATE"/infer_*.empty "$STATE"/infer_*.fail "$STATE"/bk_*.ok "$STATE"/BACKUP_VERIFIED "$STATE"/CENSUS_ENGINE
echo "cleared inference + backup markers."
if [ "$WIPE_RESULTS" = "1" ]; then
  for d in "${res_dirs[@]}"; do rm -rf "$d"; done
  echo "removed ${#res_dirs[@]} result dirs."
fi
echo "done. dl_*.ok markers and $TILES_DISK tiles preserved (re-download avoided)."
