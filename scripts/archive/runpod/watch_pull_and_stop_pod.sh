#!/bin/bash
# watch_pull_and_stop_pod.sh
# Session helper: poll for all CT-census Li prediction gpkgs to land locally,
# then stop the RunPod pod (billing pause). If results never complete within the
# timeout, EXIT WITHOUT STOPPING — never kill a pod with work still in flight.
#
# Condition for stop: all 16 Li grids x 2 models have predictions_metric.gpkg
# under results/cape_town/<model>_li_perdet/<grid>/.
set -u

PROJECT_DIR="/home/gaosh/projects/ZAsolar"
cd "$PROJECT_DIR" || exit 2

GRIDS="L1842 L1843 L1844 L1846 L1896 L1897 L1898 L1899 L1900 L1901 L1902 L1950 L1951 L1952 L1953 L1954"
MODELS="v3c_li_perdet unifiedA_li_perdet"
EXPECTED=32
POLL_SECONDS=60
MAX_ITERS=240          # ~4h safety ceiling
LOG="$PROJECT_DIR/results/analysis/_watch_stop_pod.log"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

count_present() {
  local n=0 m g f
  for m in $MODELS; do
    for g in $GRIDS; do
      f="$PROJECT_DIR/results/cape_town/$m/$g/predictions_metric.gpkg"
      [ -s "$f" ] && n=$((n+1))
    done
  done
  echo "$n"
}

log "watcher started; waiting for $EXPECTED gpkgs (16 grids x 2 models)"
i=0
while [ "$i" -lt "$MAX_ITERS" ]; do
  present=$(count_present)
  log "progress: $present/$EXPECTED present"
  if [ "$present" -ge "$EXPECTED" ]; then
    log "all $EXPECTED predictions present locally -> stopping pod"
    set -a; source "$PROJECT_DIR/.env" 2>/dev/null; set +a
    export RUNPOD_API_KEY
    if bash "$PROJECT_DIR/scripts/runpod_pod.sh" stop >>"$LOG" 2>&1; then
      log "pod stop command issued OK (RUNPOD_POD_ID=$RUNPOD_POD_ID)"
    else
      log "WARNING: pod stop command FAILED — stop the pod manually!"
    fi
    exit 0
  fi
  i=$((i+1))
  sleep "$POLL_SECONDS"
done

log "TIMEOUT after ~$((MAX_ITERS*POLL_SECONDS/60)) min with only $(count_present)/$EXPECTED present — NOT stopping pod (results incomplete). Investigate."
exit 1
