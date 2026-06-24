#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# CT census intermediate-artifact janitor (safety-net reaper for the overnight run).
#
# The detect_direct -> finalize chain writes heavy per-grid intermediates BEFORE
# the final inventory. Nothing downstream reads them:
#   - raw_detections.pkl   (detect_direct: cropped masks + boxes; the disk hog)
#   - masks/ , vectors/    (finalize scratch dirs)
# Downstream consumers only touch predictions_metric.gpkg:
#   - the census inventory lives in predictions_metric.gpkg
#   - cls (classify_predictions.py) reads gpkg + source tiles
#   - the Dropbox backup ships gpkg/geojson/config only
#   - ct_census_merge.py reads gpkg only
# Left to accumulate, raw_detections.pkl + masks/ will fill the disk.
#
# A grid's intermediates are SAFE to delete once the grid reaches a TERMINAL state:
#   infer_<G>.ok    -> predictions_metric.gpkg written with detections.
#   infer_<G>.empty -> surveyed zero-detection cell (finalize wrote an EMPTY gpkg).
#   infer_<G>.fail  -> detect/finalize crashed (retried on resume).
# Grids with NO marker yet are in-flight; their intermediates are kept (a detect
# proc may still be writing them).
#
# The orchestrator (scripts/ct_census_run.sh, infer_one) now also drops these
# inline at terminal state, so for fresh runs this janitor is redundant. It is
# kept as a safety net / one-shot reclaim tool for runs already in flight.
#
# Usage:
#   bash scripts/ct_census_mask_janitor.sh            # loop forever (default)
#   ONESHOT=1 bash scripts/ct_census_mask_janitor.sh  # sweep once and exit
# Env overrides: STATE, RESULTS, INTERVAL (seconds, default 120).
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
STATE=${STATE:-/workspace/census_state}
RESULTS=${RESULTS:-/workspace/ZAsolar/results}
INTERVAL=${INTERVAL:-120}
ONESHOT=${ONESHOT:-0}

sweep(){
  local reaped=0 freed_mb=0 d g sz
  for d in "$RESULTS"/*/; do
    [ -d "$d" ] || continue
    g=$(basename "$d")
    # terminal iff any marker exists; keep in-flight (no marker) intermediates.
    if [ -f "$STATE/infer_$g.ok" ] || [ -f "$STATE/infer_$g.empty" ] || [ -f "$STATE/infer_$g.fail" ]; then
      for art in "$d/raw_detections.pkl" "$d/masks" "$d/vectors"; do
        [ -e "$art" ] || continue
        sz=$(du -sm "$art" 2>/dev/null | cut -f1); sz=${sz:-0}
        rm -rf "$art" && { reaped=$((reaped+1)); freed_mb=$((freed_mb+sz)); }
      done
    fi
  done
  printf '[%s] janitor swept: reaped=%d freed=%dMB\n' "$(date '+%F %T')" "$reaped" "$freed_mb"
}

if [ "$ONESHOT" = "1" ]; then
  sweep
else
  while true; do sweep; sleep "$INTERVAL"; done
fi
