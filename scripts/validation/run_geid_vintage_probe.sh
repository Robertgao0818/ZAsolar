#!/usr/bin/env bash
# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/scripts/validation/run_geid_vintage_probe.sh
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
# Drive Allmapsoft GEID 6.48 downloader.exe headlessly from WSL over the
# vintage-probe task CSV. Each row → one CLI invocation with the [date]
# positional arg (historical-imagery flow).
#
# CLI signature (per https://www.allmapsoft.com/geid/commandline.htm):
#   downloader.exe task zfrom zto L R T B savepath [date]
#
# Output naming (mirrors GEID GUI behaviour):
#   <save_to>/<task>/<z>/<x>/gesh_<x>_<y>_<z>.jpg
#
# Skips a task if its year folder already contains >=1 .jpg under
# <save_to>/<task>/. Logs each invocation as a JSONL row to --log.

set -euo pipefail

TASKS_CSV="${TASKS_CSV:-/home/gaosh/projects/ZAsolar/data/geid_vintage_probe/probe_tasks.csv}"
DOWNLOADER="${DOWNLOADER:-/mnt/c/allmapsoft/geid/downloader.exe}"
LOG_DIR="${LOG_DIR:-/home/gaosh/projects/ZAsolar/data/geid_vintage_probe/run_logs}"
PER_TASK_TIMEOUT="${PER_TASK_TIMEOUT:-180}"
WSL_DISTRO_FOR_UNC="${WSL_DISTRO_FOR_UNC:-Ubuntu}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).jsonl"
echo "log: $LOG_FILE"

# Skip header, count tasks
N_TOTAL=$(($(wc -l < "$TASKS_CSV") - 1))
echo "tasks: $N_TOTAL"

idx=0
ok=0
skip=0
fail=0

# Read CSV (no embedded commas in our data — anchor IDs are ASCII, paths use \).
while IFS=, read -r grid_id task_name save_to map_type date zoom_from zoom_to L R T B; do
    if [[ "$grid_id" == "grid_id" ]]; then continue; fi
    idx=$((idx + 1))

    # save_to is canonical WSL/POSIX by default.  Windows drive paths are still
    # supported for explicit staging.  When invoking downloader.exe, POSIX paths
    # are converted to a Windows UNC path so outputs still land in WSL storage.
    if [[ "$save_to" =~ ^[A-Za-z]: ]]; then
        wsl_save=$(echo "$save_to" | sed -E 's|^([A-Za-z]):|/mnt/\L\1|; s|\\|/|g')
        win_save="$save_to"
    elif [[ "$save_to" == \\\\* ]]; then
        wsl_save=$(echo "$save_to" | sed -E "s|^\\\\\\\\wsl.localhost\\\\${WSL_DISTRO_FOR_UNC}||; s|\\\\|/|g")
        win_save="$save_to"
    else
        wsl_save="${save_to%/}"
        win_save="\\\\wsl.localhost\\${WSL_DISTRO_FOR_UNC}$(echo "$wsl_save" | sed 's|/|\\|g')"
    fi
    wsl_save_task="$wsl_save/$task_name"

    # Skip if task folder already has any jpg
    if [[ -d "$wsl_save_task" ]] && find "$wsl_save_task" -maxdepth 4 -name '*.jpg' -print -quit 2>/dev/null | grep -q .; then
        skip=$((skip + 1))
        printf '[%4d/%d] SKIP %s (already has jpg)\n' "$idx" "$N_TOTAL" "$task_name"
        printf '{"idx":%d,"task":"%s","date":"%s","status":"skip","rc":null,"elapsed_s":null}\n' \
            "$idx" "$task_name" "$date" >> "$LOG_FILE"
        continue
    fi

    mkdir -p "$wsl_save"
    started=$(date +%s)
    set +e
    timeout "$PER_TASK_TIMEOUT" "$DOWNLOADER" \
        "$task_name" "$zoom_from" "$zoom_to" \
        "$L" "$R" "$T" "$B" \
        "$win_save" "$date" </dev/null >/dev/null 2>&1
    rc=$?
    set -e
    elapsed=$(($(date +%s) - started))

    n_jpg=0
    if [[ -d "$wsl_save_task" ]]; then
        n_jpg=$(find "$wsl_save_task" -name '*.jpg' 2>/dev/null | wc -l)
    fi

    if [[ "$rc" -eq 0 && "$n_jpg" -gt 0 ]]; then
        ok=$((ok + 1))
        status=ok
    else
        fail=$((fail + 1))
        status=fail
    fi
    printf '[%4d/%d] %s rc=%d %2ds %d jpgs  %s (%s)\n' \
        "$idx" "$N_TOTAL" "$status" "$rc" "$elapsed" "$n_jpg" "$task_name" "$date"
    printf '{"idx":%d,"task":"%s","date":"%s","status":"%s","rc":%d,"elapsed_s":%d,"n_jpg":%d}\n' \
        "$idx" "$task_name" "$date" "$status" "$rc" "$elapsed" "$n_jpg" >> "$LOG_FILE"
done < "$TASKS_CSV"

echo
echo "=== summary ==="
echo "total : $N_TOTAL"
echo "ok    : $ok"
echo "skip  : $skip"
echo "fail  : $fail"
echo "log   : $LOG_FILE"
