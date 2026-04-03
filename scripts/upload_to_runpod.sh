#!/bin/bash
# Upload split tar parts to RunPod S3 with progress and resume support
# Usage: bash scripts/upload_to_runpod.sh
#
# Required env vars (set in .env or export before running):
#   RUNPOD_S3_KEY_ID      — RunPod S3 access key ID
#   RUNPOD_S3_SECRET      — RunPod S3 secret access key
#   RUNPOD_S3_ENDPOINT    — (optional) default: https://s3api-eu-ro-1.runpod.io
#   RUNPOD_S3_BUCKET      — (optional) default: s3://k5r31jwc9k
set -e

# Load .env if present
[ -f .env ] && source .env
[ -f scripts/.env ] && source scripts/.env

if [ -z "$RUNPOD_S3_KEY_ID" ] || [ -z "$RUNPOD_S3_SECRET" ]; then
    echo "ERROR: RUNPOD_S3_KEY_ID and RUNPOD_S3_SECRET must be set."
    echo "  Export them or create a .env file. See .env.example."
    exit 1
fi

export AWS_ACCESS_KEY_ID="$RUNPOD_S3_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$RUNPOD_S3_SECRET"
ENDPOINT="${RUNPOD_S3_ENDPOINT:-https://s3api-eu-ro-1.runpod.io}"
REGION="eu-ro-1"
BUCKET="${RUNPOD_S3_BUCKET:-s3://k5r31jwc9k}"
AWS="$HOME/.local/bin/aws"
UPLOAD_DIR="/mnt/d/ZAsolar/upload_tmp"

upload_parts() {
    local prefix="$1"    # e.g. "coco_part_" or "tiles_part_"
    local s3_dir="$2"    # e.g. "coco_parts" or "tiles_parts"
    local files=("$UPLOAD_DIR"/${prefix}*)

    if [ ${#files[@]} -eq 0 ]; then
        echo "[SKIP] No files matching $UPLOAD_DIR/${prefix}*"
        return
    fi

    local total=${#files[@]}
    local done=0
    local failed=0

    echo ""
    echo "========================================"
    echo " Uploading: $s3_dir ($total parts)"
    echo "========================================"

    for f in "${files[@]}"; do
        local name=$(basename "$f")
        local size=$(du -h "$f" | cut -f1)
        done=$((done + 1))

        # Check if already uploaded (resume support)
        local remote_size=$($AWS s3api head-object \
            --bucket k5r31jwc9k --key "$s3_dir/$name" \
            --region $REGION --endpoint-url $ENDPOINT \
            --query 'ContentLength' --output text 2>/dev/null || echo "0")
        local local_size=$(stat -c%s "$f")

        if [ "$remote_size" = "$local_size" ]; then
            echo "[$done/$total] $name ($size) — already uploaded, skipping"
            continue
        fi

        echo "[$done/$total] $name ($size) — uploading..."
        local start=$(date +%s)

        if $AWS s3 cp "$f" "$BUCKET/$s3_dir/$name" \
            --region $REGION --endpoint-url $ENDPOINT 2>&1; then
            local elapsed=$(( $(date +%s) - start ))
            local speed=$(echo "scale=1; $local_size / 1048576 / $elapsed" | bc 2>/dev/null || echo "?")
            echo "  ✓ Done in ${elapsed}s (~${speed} MB/s)"
        else
            echo "  ✗ FAILED: $name"
            failed=$((failed + 1))
        fi
    done

    echo ""
    echo "[$s3_dir] Complete: $((done - failed))/$total uploaded, $failed failed"
}

echo "RunPod S3 Upload Script"
echo "======================="
echo "Endpoint: $ENDPOINT"
echo "Upload dir: $UPLOAD_DIR"
echo ""

# List what we have
echo "Available split files:"
ls -lh "$UPLOAD_DIR"/coco_part_* 2>/dev/null | awk '{print "  COCO: " $NF " (" $5 ")"}'
ls -lh "$UPLOAD_DIR"/tiles_part_* 2>/dev/null | awk '{print "  TILES: " $NF " (" $5 ")"}'
echo ""

# Upload COCO parts
upload_parts "coco_part_" "coco_parts"

# Upload tiles parts
upload_parts "tiles_part_" "tiles_parts"

echo ""
echo "========================================"
echo " ALL DONE"
echo "========================================"
echo ""
echo "To reassemble on RunPod Pod:"
echo "  cd /workspace"
echo "  cat coco_parts/coco_part_* | tar -xf - -C ZAsolar/"
echo "  cat tiles_parts/tiles_part_* | tar -xf - -C ZAsolar/"
