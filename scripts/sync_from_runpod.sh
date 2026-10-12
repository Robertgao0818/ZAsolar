#!/bin/bash
# Sync results and/or tiles from a RunPod pod to the local machine.
#
# Grid filtering, results roots and tile roots are resolved through the region
# registry (configs/datasets/regions.yaml via core.region_registry /
# core.grid_utils), NOT a hardcoded namespace prefix — per .claude/rules/06.
# Every registered grid_id_pattern is accepted (CPT#### / G#### / L#### /
# JNB#### / JHB## / …), so the CT CPT regrid (ADR-0002) and Li L-grids sync too.
#
# Usage:
#   bash scripts/sync_from_runpod.sh <results|tiles|all> [grid_list_file] [flags]
#
# Positional:
#   MODE            results | tiles | all
#   grid_list_file  optional; local OR remote path, one grid id per line.
#                   Omit to auto-discover grids on the remote.
#
# Flags (all optional, additive — legacy invocations keep working unchanged):
#   --region <ct|jhb|cape_town|johannesburg>
#                   region for path resolution. REQUIRED to reach the canonical
#                   nested layout (results/<region>/<model_run>/, tiles/<region>/
#                   <imagery_layer>/). Region NEVER inferred from grid id.
#   --model-run <id>
#                   registered model_run; routes results to its results_path.
#   --imagery-layer <id>
#                   imagery layer for tile resolution (else region default).
#   --dry-run       print filtered grids + resolved remote/local roots, then exit
#                   (no transfer). Fully offline / creds-free ONLY with a
#                   grid_list_file; without one, grid auto-discovery still SSHes
#                   the pod to list remote grids.
#
# Without --region/--model-run the script keeps the pre-2026-04-19 flat layout
# ($REMOTE_WORKSPACE/results/<grid>/, $REMOTE_TILES/<grid>/) verbatim — only the
# G-only grid filter is lifted. Nested runs (CPT census, current JNB) are only
# reachable with the flags.
#
# Environment variables (or set in .env):
#   RUNPOD_SSH_HOST   — e.g. root@213.173.103.184
#   RUNPOD_SSH_PORT   — e.g. 29416
#   RUNPOD_SSH_KEY    — e.g. ~/.ssh/id_ed25519  (default)
#   REMOTE_WORKSPACE  — remote checkout root (default /workspace/ZAsolar;
#                       Community-Cloud census pods use /root/ZAsolar)
#   REMOTE_TILES      — remote tiles root (default /workspace/tiles)
#   LOCAL_RESULTS     — legacy local results root (default ~/zasolar_data/results)
#   LOCAL_TILES       — legacy local tiles root (default ~/zasolar_data/tiles)
#
# Examples:
#   # Dry-run the completed CT census results sync (lists CPT grids, no transfer)
#   REMOTE_WORKSPACE=/root/ZAsolar \
#     bash scripts/sync_from_runpod.sh results \
#       --region ct --model-run unifiedA_census_perdet --dry-run
#
#   # Legacy flat batch-004 results (pre-restructure JHB tree, unchanged)
#   RUNPOD_SSH_HOST=root@1.2.3.4 RUNPOD_SSH_PORT=29416 \
#     bash scripts/sync_from_runpod.sh results /workspace/download_grids_batch004.txt
#
#   # Sync specific grids (inline)
#   echo -e "CPT1240\nCPT1241" > /tmp/grids.txt
#   bash scripts/sync_from_runpod.sh tiles /tmp/grids.txt --region ct --imagery-layer aerial_2025

set -euo pipefail

# --- Config ---
[ -f .env ] && source .env
[ -f scripts/.env ] && source scripts/.env

# Remote/local roots — overridable (Community-Cloud pods put the checkout at
# /root/ZAsolar and tiles on local NVMe, not /workspace).
REMOTE_WORKSPACE="${REMOTE_WORKSPACE:-/workspace/ZAsolar}"
REMOTE_TILES="${REMOTE_TILES:-/workspace/tiles}"
LOCAL_RESULTS="${LOCAL_RESULTS:-/home/gao/zasolar_data/results}"
LOCAL_TILES="${LOCAL_TILES:-/home/gao/zasolar_data/tiles}"

# --- Parse args: pull flags out, keep positionals (MODE, grid_list_file) ---
REGION=""
MODEL_RUN=""
IMAGERY_LAYER=""
DRY_RUN=0
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --region)        REGION="${2:?--region needs a value}"; shift 2 ;;
        --model-run)     MODEL_RUN="${2:?--model-run needs a value}"; shift 2 ;;
        --imagery-layer) IMAGERY_LAYER="${2:?--imagery-layer needs a value}"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --) shift; while [ $# -gt 0 ]; do POSITIONAL+=("$1"); shift; done ;;
        -*) echo "ERROR: unknown flag '$1'" >&2; exit 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
MODE="${POSITIONAL[0]:?Usage: $0 <results|tiles|all> [grid_list_file] [flags]}"
GRID_LIST_FILE="${POSITIONAL[1]:-}"

if [ -n "$MODEL_RUN" ] && [ -z "$REGION" ]; then
    echo "ERROR: --model-run requires --region (grid namespaces overlap between regions; region is never inferred)." >&2
    exit 2
fi

# Results live under results/<region>/<model_run>/<grid>/; a bare region root
# holds model_run dirs, not grid dirs. So results/all with --region but no
# --model-run would scope to a container of run dirs and silently sync nothing.
if { [ "$MODE" = results ] || [ "$MODE" = all ]; } && [ -n "$REGION" ] && [ -z "$MODEL_RUN" ]; then
    echo "ERROR: MODE=$MODE with --region requires --model-run (results are nested results/<region>/<model_run>/<grid>/)." >&2
    exit 2
fi

# SSH creds: hard-required for a real transfer, soft for --dry-run.
if [ "$DRY_RUN" = 1 ]; then
    SSH_HOST="${RUNPOD_SSH_HOST:-<unset:dry-run>}"
    SSH_PORT="${RUNPOD_SSH_PORT:-0}"
else
    SSH_HOST="${RUNPOD_SSH_HOST:?Set RUNPOD_SSH_HOST (e.g. root@1.2.3.4)}"
    SSH_PORT="${RUNPOD_SSH_PORT:?Set RUNPOD_SSH_PORT}"
fi
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

# --- Registry-backed resolver (regions.yaml only via core.*, never grep) ---
# Run locally: grid filtering and path derivation are self-locating via BASE_DIR,
# so no per-grid SSH round trip (CT census is ~2083 grids).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# grid_id matching MUST go through core.region_registry.matches_any_region_pattern
# (Python re.fullmatch). Do NOT re-derive a grep -E/-P regex from the yaml
# patterns: they use \d{4}, which GNU grep -E does not interpret (silent no-op).
PYRESOLVE=$(cat <<'PY'
import os, sys
os.environ.pop("SOLAR_TILES_ROOT", None)  # force canonical registry layout, not flat env fast-path
sub = sys.argv[1]
from core import region_registry as R

if sub == "filter":
    for line in sys.stdin:
        g = line.strip().upper()
        if g and R.matches_any_region_pattern(g):
            print(g)

elif sub == "results":
    from core.grid_utils import get_results_root
    region = sys.argv[2] or None
    model_run = sys.argv[3] or None
    remote_ws = sys.argv[4].rstrip("/")
    local_root = get_results_root(region, model_run=model_run)
    # results_path is repo-root-relative by construction; derive the remote path
    # by re-rooting onto the remote checkout (no SSH needed).
    try:
        remote_root = f"{remote_ws}/{local_root.relative_to(R.BASE_DIR)}"
    except ValueError:
        remote_root = str(local_root)  # absolute elsewhere; assume same path on pod
    print(f"{local_root}\t{remote_root}")

elif sub == "tiles":
    from core.grid_utils import resolve_tiles_dir
    region = sys.argv[2]
    imagery_layer = sys.argv[3] or None
    remote_tiles = sys.argv[4].rstrip("/")
    rkey = R.normalize_region_key(region) or region
    # <region>/<imagery_layer>/<grid> lives under the region-agnostic tiles base.
    tiles_base = R.get_tiles_path(rkey).parent
    for g in sys.argv[5:]:
        d = resolve_tiles_dir(g, region=region, imagery_layer=imagery_layer)
        layout = "mosaic" if d.name.endswith(".tif") else "chunked"
        try:
            remote = f"{remote_tiles}/{d.relative_to(tiles_base)}"
        except ValueError:
            remote = f"{remote_tiles}/{d.name}"
        print(f"{g}\t{d}\t{remote}\t{layout}")
PY
)
py_resolve() { "$LOCAL_PY" -c "$PYRESOLVE" "$@"; }

# --- Helpers ---
ssh_cmd() { ssh $SSH_OPTS "$SSH_HOST" "$@"; }

# Resolve results roots once (not per grid). Sets REMOTE_RESULTS_DIR /
# LOCAL_RESULTS_DIR / RESULTS_RESOLVED.
resolve_results_roots() {
    if [ -n "$REGION" ] || [ -n "$MODEL_RUN" ]; then
        local line
        line=$(py_resolve results "$REGION" "$MODEL_RUN" "$REMOTE_WORKSPACE") \
            || { echo "ERROR: could not resolve results root (region='$REGION' model_run='$MODEL_RUN')." >&2; exit 1; }
        LOCAL_RESULTS_DIR=$(printf '%s' "$line" | cut -f1)
        REMOTE_RESULTS_DIR=$(printf '%s' "$line" | cut -f2)
        RESULTS_RESOLVED=1
    else
        REMOTE_RESULTS_DIR="$REMOTE_WORKSPACE/results"
        LOCAL_RESULTS_DIR="$LOCAL_RESULTS"
        RESULTS_RESOLVED=0
        echo "WARN: no --region/--model-run — using legacy flat results layout ($REMOTE_RESULTS_DIR/<grid>/)." >&2
        echo "      Nested runs (CPT census, current JNB) need --region + --model-run to be reachable." >&2
    fi
}

get_grids() {
    if [ -n "$GRID_LIST_FILE" ]; then
        # Grid list can be a local file or a remote path.
        if [ -f "$GRID_LIST_FILE" ]; then
            cat "$GRID_LIST_FILE"
        else
            ssh_cmd "cat '$GRID_LIST_FILE'" 2>/dev/null
        fi
    elif [ "$RESULTS_RESOLVED" = 1 ]; then
        # Scoped discovery: grid dirs inside the resolved run directory.
        ssh_cmd "ls -1 '$REMOTE_RESULTS_DIR' 2>/dev/null"
    else
        # Global discovery: recurse the results tree so every namespace shows up
        # at any depth — flat legacy at depth 1, region/model_run/grid at depth 3.
        ssh_cmd "find '$REMOTE_WORKSPACE/results' -mindepth 1 -maxdepth 3 -type d -printf '%f\n' 2>/dev/null"
    fi
}

sync_results() {
    local grids=("$@")
    local total=${#grids[@]}
    local done=0 skipped=0

    echo ""
    echo "=== Syncing Results ($total grids) ==="
    echo "  remote: $REMOTE_RESULTS_DIR/<grid>/"
    echo "  local:  $LOCAL_RESULTS_DIR/<grid>/"
    mkdir -p "$LOCAL_RESULTS_DIR"

    # One SSH lists every grid dir present under the run root (doubles as the
    # per-grid existence gate — no per-grid round trip).
    local present
    present=$(ssh_cmd "ls -1 '$REMOTE_RESULTS_DIR' 2>/dev/null" || true)

    for g in "${grids[@]}"; do
        done=$((done + 1))
        if ! grep -qxF "$g" <<< "$present"; then
            echo "[$done/$total] $g — no results on remote, skipping"
            skipped=$((skipped + 1))
            continue
        fi
        echo -n "[$done/$total] $g — "
        rsync -az --info=progress2 \
            -e "ssh $SSH_OPTS" \
            "$SSH_HOST:$REMOTE_RESULTS_DIR/$g/" "$LOCAL_RESULTS_DIR/$g/" 2>&1 | tail -1
    done

    echo ""
    echo "Results sync done: $((done - skipped))/$total downloaded, $skipped skipped"
}

sync_tiles() {
    local grids=("$@")
    local total=${#grids[@]}
    local done=0 skipped=0 existed=0

    echo ""
    echo "=== Syncing Tiles ($total grids) ==="

    # Resolve every grid's (local_dir, remote_dir, layout) in ONE python call.
    local resolved
    if [ -n "$REGION" ]; then
        resolved=$(py_resolve tiles "$REGION" "$IMAGERY_LAYER" "$REMOTE_TILES" "${grids[@]}") \
            || { echo "ERROR: could not resolve tile paths (region='$REGION' imagery_layer='$IMAGERY_LAYER')." >&2; exit 1; }
        echo "  canonical layout: $REMOTE_TILES/<region>/<imagery_layer>/<grid>/"
    else
        # Legacy flat layout, verbatim.
        resolved=$(for g in "${grids[@]}"; do
            printf '%s\t%s\t%s\tchunked\n' "$g" "$LOCAL_TILES/$g" "$REMOTE_TILES/$g"
        done)
        echo "WARN: no --region — using legacy flat tiles layout ($REMOTE_TILES/<grid>/)." >&2
        echo "      Pass --region [--imagery-layer] for the canonical tiles/<region>/<imagery_layer>/ layout." >&2
    fi

    while IFS=$'\t' read -r g local_dir remote_dir layout; do
        [ -z "$g" ] && continue
        done=$((done + 1))

        if [ "$layout" = "mosaic" ]; then
            if [ -e "$local_dir" ]; then
                echo "[$done/$total] $g — mosaic already local, skipping"
                existed=$((existed + 1)); continue
            fi
            if ! ssh_cmd "[ -e '$remote_dir' ]" 2>/dev/null; then
                echo "[$done/$total] $g — no tiles on remote, skipping"
                skipped=$((skipped + 1)); continue
            fi
            mkdir -p "$(dirname "$local_dir")"
            echo -n "[$done/$total] $g — "
            rsync -az --info=progress2 -e "ssh $SSH_OPTS" \
                "$SSH_HOST:$remote_dir" "$local_dir" 2>&1 | tail -1
        else
            if [ -d "$local_dir" ] && [ -n "$(ls "$local_dir"/*.tif 2>/dev/null | head -1)" ]; then
                local n; n=$(ls "$local_dir"/*.tif 2>/dev/null | wc -l)
                echo "[$done/$total] $g — already have $n tiles locally, skipping"
                existed=$((existed + 1)); continue
            fi
            if ! ssh_cmd "[ -d '$remote_dir' ]" 2>/dev/null; then
                echo "[$done/$total] $g — no tiles on remote, skipping"
                skipped=$((skipped + 1)); continue
            fi
            mkdir -p "$local_dir"
            echo -n "[$done/$total] $g — "
            rsync -az --info=progress2 -e "ssh $SSH_OPTS" \
                "$SSH_HOST:$remote_dir/" "$local_dir/" 2>&1 | tail -1
        fi
    done <<< "$resolved"

    echo ""
    echo "Tiles sync done: $((done - existed - skipped)) downloaded, $existed already local, $skipped missing"
}

# --- Main ---
resolve_results_roots

mapfile -t GRIDS < <(get_grids | py_resolve filter | sort -u)

if [ ${#GRIDS[@]} -eq 0 ]; then
    echo "ERROR: No grids found (after registry pattern filter)."
    exit 1
fi

echo "RunPod Sync: $MODE"
echo "  Host:   $SSH_HOST:$SSH_PORT"
echo "  Region: ${REGION:-<none>}   Model-run: ${MODEL_RUN:-<none>}   Imagery-layer: ${IMAGERY_LAYER:-<default>}"
echo "  Grids:  ${#GRIDS[@]}"

if [ "$DRY_RUN" = 1 ]; then
    echo ""
    echo "=== DRY RUN (no transfer) ==="
    echo "Matched grids:"
    printf '  %s\n' "${GRIDS[@]:0:20}"
    [ ${#GRIDS[@]} -gt 20 ] && echo "  ... (+$(( ${#GRIDS[@]} - 20 )) more)"
    if [ "$MODE" = "results" ] || [ "$MODE" = "all" ]; then
        echo "Results roots:"
        echo "  remote: $REMOTE_RESULTS_DIR/<grid>/"
        echo "  local:  $LOCAL_RESULTS_DIR/<grid>/"
    fi
    if [ "$MODE" = "tiles" ] || [ "$MODE" = "all" ]; then
        echo "Tiles roots:"
        if [ -n "$REGION" ]; then
            # Resolve one sample grid to show the canonical parent.
            local_sample=$(py_resolve tiles "$REGION" "$IMAGERY_LAYER" "$REMOTE_TILES" "${GRIDS[0]}" 2>/dev/null || true)
            if [ -n "$local_sample" ]; then
                IFS=$'\t' read -r _ s_local s_remote _ <<< "$local_sample"
                echo "  remote: $(dirname "$s_remote")/<grid>/"
                echo "  local:  $(dirname "$s_local")/<grid>/"
            else
                echo "  (could not resolve sample tile path for '${GRIDS[0]}')"
            fi
        else
            echo "  remote: $REMOTE_TILES/<grid>/  (legacy flat)"
            echo "  local:  $LOCAL_TILES/<grid>/  (legacy flat)"
        fi
    fi
    echo ""
    echo "=== Dry run complete (nothing transferred) ==="
    exit 0
fi

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
