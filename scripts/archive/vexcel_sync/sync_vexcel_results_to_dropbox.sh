#!/usr/bin/env bash
# Mirror results/johannesburg/v3c_vexcel_2024/ → Windows Dropbox so the
# cloud client uploads it. Excludes heavy intermediate artefacts (masks/,
# vectors/) — only the small evaluation outputs travel.
#
# Usage:
#   ./scripts/sync_vexcel_results_to_dropbox.sh           # silent
#   ./scripts/sync_vexcel_results_to_dropbox.sh --verbose # show transfers
#   ./scripts/sync_vexcel_results_to_dropbox.sh --dry-run # preview only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/results/johannesburg/v3c_vexcel_2024/"
DEST="/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/inference_results/v3c_vexcel_2024/"

FLAGS=(-a --delete \
       --exclude='masks/' \
       --exclude='vectors/' \
       --exclude='__pycache__/' \
       --exclude='*.tmp')
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=1; shift ;;
        -n|--dry-run) FLAGS+=(--dry-run); VERBOSE=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ $VERBOSE -eq 1 ]] && FLAGS+=(-v --stats)

if [[ ! -d "$SRC" ]]; then
    echo "[sync_vexcel→dropbox] source missing: $SRC" >&2
    echo "  pull from pod first: ./scripts/sync_vexcel_results_from_pod.sh" >&2
    exit 1
fi

if [[ ! -d "/mnt/c/Users/gaosh/Dropbox" ]]; then
    echo "[sync_vexcel→dropbox] Dropbox folder not mounted: /mnt/c/Users/gaosh/Dropbox" >&2
    exit 1
fi

mkdir -p "$DEST"

[[ $VERBOSE -eq 1 ]] && echo "[sync_vexcel→dropbox] $SRC → $DEST"

rsync "${FLAGS[@]}" "$SRC" "$DEST"

[[ $VERBOSE -eq 1 ]] && echo "[sync_vexcel→dropbox] Done."
