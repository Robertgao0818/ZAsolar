#!/usr/bin/env bash
# Compact one-shot status digest for the CT census run. This is the SINGLE
# command a monitoring sub-agent SSHes in to run, so the main agent only ever
# holds this digest (not raw logs) — context economy by design.
#   ssh "$RUNPOD_SSH_HOST" -p "$RUNPOD_SSH_PORT" 'bash /root/ZAsolar/scripts/ct_census_status.sh'
set -uo pipefail
STATE=${STATE:-/root/census_state}
LOGS=${LOGS:-/root/census_logs}
TILES_DISK=${TILES_DISK:-/root/tiles_disk}
GLIST="$STATE/glist.txt"

TOTAL=$( [ -f "$GLIST" ] && wc -l < "$GLIST" || echo "?")
dl=$(ls "$STATE"/dl_*.ok 2>/dev/null | wc -l)
inf=$(ls "$STATE"/infer_*.ok 2>/dev/null | wc -l)
bk=$(ls "$STATE"/bk_*.ok 2>/dev/null | wc -l)
dlf=$(ls "$STATE"/dl_*.fail 2>/dev/null | wc -l)
inff=$(ls "$STATE"/infer_*.fail 2>/dev/null | wc -l)

echo "PHASE   : $(cat "$STATE/PHASE" 2>/dev/null || echo '?')"
echo "download: $dl/$TOTAL  (fail $dlf)"
echo "infer   : $inf/$TOTAL  (fail $inff)"
echo "backup  : $bk/$TOTAL"
echo "verified: $([ -f "$STATE/BACKUP_VERIFIED" ] && cat "$STATE/BACKUP_VERIFIED" || echo no)"

# heartbeat staleness (STATUS.txt last_update vs now)
if [ -f "$STATE/STATUS.txt" ]; then
  lu=$(grep -o 'last_update=[0-9]*' "$STATE/STATUS.txt" | cut -d= -f2)
  [ -n "$lu" ] && echo "heartbeat: $(( ($(date +%s) - lu) ))s since last status update"
fi

echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
echo "--- disk ---"
df -h "$TILES_DISK" /dev/shm / 2>/dev/null | awk 'NR==1 || /tiles|shm|\/$/'

# surface recent errors only (not full logs)
echo "--- recent infer errors (last 5) ---"
grep -lim1 -e Traceback -e Error "$LOGS"/infer_*.log 2>/dev/null | tail -5 | while read -r f; do
  echo "  $(basename "$f"): $(grep -m1 -e Traceback -e Error "$f" | cut -c1-100)"
done
