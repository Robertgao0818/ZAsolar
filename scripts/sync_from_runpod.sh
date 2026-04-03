#!/bin/bash
# Sync results and/or tiles from RunPod pod to local D drive.
#
# Usage:
#   bash scripts/sync_from_runpod.sh results <grid_list_file>   — download results only
#   bash scripts/sync_from_runpod.sh tiles   <grid_list_file>   — download tiles only
#   bash scripts/sync_from_runpod.sh all     <grid_list_file>   — download both
#   bash scripts/sync_from_runpod.sh results                    — all grids with results
#
# Environment variables (or set in .env):
#   RUNPOD_SSH_HOST   — e.g. root@213.173.103.184
#   RUNPOD_SSH_PORT   — e.g. 29416
#   RUNPOD_SSH_KEY    — e.g. ~/.ssh/id_ed25519  (default)
#
# Examples:
#   # Sync batch 004 results
#   RUNPOD_SSH_HOST=root@1.2.3.4 RUNPOD_SSH_PORT=29416 \
#     bash scripts/sync_from_runpod.sh results /workspace/download_grids_batch004.txt
#
#   # Sync specific grids (inline)
#   echo -e "G1855\nG1856" > /tmp/grids.txt
#   bash scripts/sync_from_runpod.sh tiles /tmp/grids.txt

set -euo pipefail

# --- Config ---
[ -f .env ] && source .env
[ -f scripts/.env ] && source scripts/.env

SSH_HOST="${RUNPOD_SSH_HOST:?Set RUNPOD_SSH_HOST (e.g. root@1.2.3.4)}"
SSH_PORT="${RUNPOD_SSH_PORT:?Set RUNPOD_SSH_PORT}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

REMOTE_WORKSPACE="/workspace/ZAsolar"
REMOTE_TILES="/workspace/tiles"
LOCAL_RESULTS="/mnt/d/ZAsolar/results"
LOCAL_TILES="/mnt/d/ZAsolar/tiles"

MODE="${1:?Usage: $0 <results|tiles|all> [grid_list_file]}"
GRID_LIST_FILE="${2:-}"

# --- Helpers ---
ssh_cmd() { ssh $SSH_OPTS "$SSH_HOST" "$@"; }

get_grids() {
    if [ -n "$GRID_LIST_FILE" ]; then
        # Grid list can be local file or remote file
        if [ -f "$GRID_LIST_FILE" ]; then
            cat "$GRID_LIST_FILE"
        else
            # Try as remote path
            ssh_cmd "cat $GRID_LIST_FILE" 2>/dev/null
        fi
    else
        # Auto-discover: all grids with results on remote
        ssh_cmd "ls -d $REMOTE_WORKSPACE/results/G* 2>/dev/null | xargs -n1 basename"
    fi
}

sync_results() {
    local grids=("$@")
    local total=${#grids[@]}
    local done=0
    local skipped=0

    echo ""
    echo "=== Syncing Results ($total grids) ==="
    mkdir -p "$LOCAL_RESULTS"

    for g in "${grids[@]}"; do
        done=$((done + 1))
        local remote="$REMOTE_WORKSPACE/results/$g/"
        local local_dir="$LOCAL_RESULTS/$g/"

        # Check remote exists
        if ! ssh_cmd "[ -d $remote ]" 2>/dev/null; then
            echo "[$done/$total] $g — no results on remote, skipping"
            skipped=$((skipped + 1))
            continue
        fi

        echo -n "[$done/$total] $g — "
        rsync -az --info=progress2 \
            -e "ssh $SSH_OPTS" \
            "$SSH_HOST:$remote" "$local_dir" 2>&1 | tail -1
    done

    echo ""
    echo "Results sync done: $((done - skipped))/$total downloaded, $skipped skipped"
}

sync_tiles() {
    local grids=("$@")
    local total=${#grids[@]}
    local done=0
    local skipped=0
    local existed=0

    echo ""
    echo "=== Syncing Tiles ($total grids) ==="
    mkdir -p "$LOCAL_TILES"

    for g in "${grids[@]}"; do
        done=$((done + 1))
        local local_dir="$LOCAL_TILES/$g/"

        # Skip if already have tiles locally
        if [ -d "$local_dir" ] && [ "$(ls "$local_dir"/*.tif 2>/dev/null | head -1)" ]; then
            local n=$(ls "$local_dir"/*.tif 2>/dev/null | wc -l)
            echo "[$done/$total] $g — already have $n tiles locally, skipping"
            existed=$((existed + 1))
            continue
        fi

        local remote="$REMOTE_TILES/$g/"
        if ! ssh_cmd "[ -d $remote ]" 2>/dev/null; then
            echo "[$done/$total] $g — no tiles on remote, skipping"
            skipped=$((skipped + 1))
            continue
        fi

        echo -n "[$done/$total] $g — "
        rsync -az --info=progress2 \
            -e "ssh $SSH_OPTS" \
            "$SSH_HOST:$remote" "$local_dir" 2>&1 | tail -1
    done

    echo ""
    echo "Tiles sync done: $((done - existed - skipped)) downloaded, $existed already local, $skipped missing"
}

# --- Main ---
mapfile -t GRIDS < <(get_grids | grep -E '^G[0-9]+' | sort -u)

if [ ${#GRIDS[@]} -eq 0 ]; then
    echo "ERROR: No grids found."
    exit 1
fi

echo "RunPod Sync: $MODE"
echo "  Host: $SSH_HOST:$SSH_PORT"
echo "  Grids: ${#GRIDS[@]}"
echo "  Results → $LOCAL_RESULTS"
echo "  Tiles   → $LOCAL_TILES"

case "$MODE" in
    results)
        sync_results "${GRIDS[@]}"
        ;;
    tiles)
        sync_tiles "${GRIDS[@]}"
        ;;
    all)
        sync_results "${GRIDS[@]}"
        sync_tiles "${GRIDS[@]}"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Use: results, tiles, or all"
        exit 1
        ;;
esac

echo ""
echo "=== Sync Complete ==="
