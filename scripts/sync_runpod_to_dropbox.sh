#!/usr/bin/env bash
# Cloud-to-cloud sync: RunPod S3 network volume → Dropbox.
#
# Replaces RunPod's built-in cloud sync (too-low concurrency).
#
# IMPORTANT: despite the "cloud-to-cloud" framing, rclone has no server-side
# cross-provider copy. Bytes flow RunPod S3 → THIS MACHINE → Dropbox, so the
# local uplink and downlink are the real bottleneck, and the machine must stay
# online for the full duration. For large runs (>10G) prefer running this on
# the RunPod pod itself — the pod's backbone gets 50-100 MB/s vs. home ~5 Mbps.
#
# Prereqs (already set up 2026-04-23):
#   - rclone installed
#   - ~/.config/rclone/rclone.conf has [runpod-s3] + [dropbox] remotes
#   - RunPod volume id = k5r31jwc9k (EU-RO-1, 500GB)
#   - Dropbox destination root = ZAsolar/
#
# Usage:
#   ./scripts/sync_runpod_to_dropbox.sh tiles            # CT aerial tiles (~376G)
#   ./scripts/sync_runpod_to_dropbox.sh tiles/johannesburg/aerial_2023     # JHB tiles (~16G)
#   ./scripts/sync_runpod_to_dropbox.sh base-maps        # both tiles dirs
#   ./scripts/sync_runpod_to_dropbox.sh checkpoints      # model weights (~1.4G)
#   ./scripts/sync_runpod_to_dropbox.sh <PATH>           # any /workspace/<PATH>
#   ./scripts/sync_runpod_to_dropbox.sh base-maps --dry-run
#
# Env overrides:
#   TRANSFERS=16 CHECKERS=32 CHUNK=128M   # parallelism + Dropbox chunk size
#
# Resumability: `rclone copy` skips files already present with matching size.
# Safe to ctrl-C and re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUCKET="k5r31jwc9k"
DBX_ROOT="ZAsolar"
LOG_DIR="$REPO_ROOT/logs/rclone"
mkdir -p "$LOG_DIR"

TRANSFERS="${TRANSFERS:-16}"
CHECKERS="${CHECKERS:-32}"
CHUNK="${CHUNK:-128M}"

[[ $# -ge 1 ]] || { grep '^# ' "$0" | sed 's/^# \?//'; exit 1; }

TARGET="$1"; shift || true
EXTRA_FLAGS=("$@")

# Expand alias → list of paths
case "$TARGET" in
    base-maps) PATHS=(tiles tiles/johannesburg/aerial_2023) ;;
    *)         PATHS=("$TARGET") ;;
esac

COMMON_FLAGS=(
    --transfers "$TRANSFERS"
    --checkers "$CHECKERS"
    --dropbox-chunk-size "$CHUNK"
    --s3-no-check-bucket
    --exclude '__pycache__/**'
    --exclude '.s3compat_uploads/**'
    --exclude '*.tmp'
    --stats 30s
    --stats-one-line
)

ts="$(date +%Y%m%d-%H%M%S)"
for p in "${PATHS[@]}"; do
    # Trailing slashes required: RunPod S3 gateway returns 400 on HeadObject
    # for bare prefixes, which rclone uses to distinguish file vs directory.
    src="runpod-s3:${BUCKET}/${p}/"
    dst="dropbox:${DBX_ROOT}/${p}/"
    log="$LOG_DIR/sync_${p//\//_}_${ts}.log"

    echo "===> $src  →  $dst"
    echo "     log: $log"
    rclone copy "$src" "$dst" \
        "${COMMON_FLAGS[@]}" \
        --log-file "$log" --log-level INFO \
        "${EXTRA_FLAGS[@]}"
done
