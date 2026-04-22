#!/bin/bash
# One-shot: tar + split + upload coco_v4_2_jhb_ft to RunPod S3.
set -eo pipefail

[ -f .env ] && source .env
if [ -z "$RUNPOD_S3_KEY_ID" ] || [ -z "$RUNPOD_S3_SECRET" ]; then
    echo "ERROR: RUNPOD_S3_KEY_ID and RUNPOD_S3_SECRET must be set in .env"; exit 1
fi

export AWS_ACCESS_KEY_ID="$RUNPOD_S3_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$RUNPOD_S3_SECRET"
ENDPOINT="${RUNPOD_S3_ENDPOINT:-https://s3api-eu-ro-1.runpod.io}"
REGION="eu-ro-1"
BUCKET="${RUNPOD_S3_BUCKET:-s3://k5r31jwc9k}"
AWS="$HOME/.local/bin/aws"

SRC_PARENT="/mnt/d/ZAsolar"
SRC_NAME="coco_v4_2_jhb_ft"
UPLOAD_DIR="/mnt/d/ZAsolar/upload_tmp"
PART_PREFIX="coco_v4_2_jhb_ft_part_"
S3_DIR="coco_v4_2_jhb_ft_parts"
PART_SIZE="1G"

mkdir -p "$UPLOAD_DIR"

echo "=== tar + split ==="
echo "Source: $SRC_PARENT/$SRC_NAME  (size: $(du -sh "$SRC_PARENT/$SRC_NAME" | cut -f1))"
echo "Output: $UPLOAD_DIR/${PART_PREFIX}*"
echo "Part size: $PART_SIZE"
rm -f "$UPLOAD_DIR/${PART_PREFIX}"*
cd "$SRC_PARENT"
tar -cf - "$SRC_NAME" | split -b "$PART_SIZE" -d -a 2 - "$UPLOAD_DIR/$PART_PREFIX"
cd - >/dev/null

ls -lh "$UPLOAD_DIR/${PART_PREFIX}"*
TOTAL_PARTS=$(ls "$UPLOAD_DIR/${PART_PREFIX}"* | wc -l)
echo "Created $TOTAL_PARTS parts."

echo ""
echo "=== upload to $BUCKET/$S3_DIR/ ==="
done=0
for f in "$UPLOAD_DIR/${PART_PREFIX}"*; do
    name=$(basename "$f")
    size=$(du -h "$f" | cut -f1)
    done=$((done + 1))
    remote_size=$($AWS s3api head-object --bucket k5r31jwc9k --key "$S3_DIR/$name" \
        --region $REGION --endpoint-url $ENDPOINT --query 'ContentLength' \
        --output text 2>/dev/null || echo "0")
    local_size=$(stat -c%s "$f")
    if [ "$remote_size" = "$local_size" ]; then
        echo "[$done/$TOTAL_PARTS] $name ($size) — already uploaded, skipping"
        continue
    fi
    echo "[$done/$TOTAL_PARTS] $name ($size) — uploading..."
    start=$(date +%s)
    $AWS s3 cp "$f" "$BUCKET/$S3_DIR/$name" \
        --region $REGION --endpoint-url $ENDPOINT
    elapsed=$(( $(date +%s) - start ))
    speed=$(echo "scale=1; $local_size / 1048576 / $elapsed" | bc 2>/dev/null || echo "?")
    echo "  done in ${elapsed}s (~${speed} MB/s)"
done

echo ""
echo "=== ALL DONE ==="
echo ""
echo "To reassemble on RunPod pod:"
echo "  cd /workspace"
echo "  mkdir -p coco_v4_2_jhb_ft_parts && cd coco_v4_2_jhb_ft_parts"
echo "  aws s3 sync s3://k5r31jwc9k/$S3_DIR/ ./ --endpoint-url $ENDPOINT --region $REGION"
echo "  cd /workspace && cat coco_v4_2_jhb_ft_parts/${PART_PREFIX}* | tar -xf -"
echo "  # Should produce /workspace/coco_v4_2_jhb_ft/"
