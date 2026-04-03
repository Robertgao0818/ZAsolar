#!/bin/bash
# RunPod training launcher — stages COCO data to local SSD for fast I/O
set -e

WORKSPACE="${WORKSPACE:-/workspace/ZAsolar}"
LOCAL_SSD="/tmp/ZAsolar"
COCO_SRC="$WORKSPACE/data/coco_cleaned"
COCO_DST="$LOCAL_SSD/coco_cleaned"

# --- Stage COCO to local NVMe SSD ---
if [ -d "$COCO_DST/train" ] && [ "$(ls $COCO_DST/train/*.tif 2>/dev/null | head -1)" ]; then
    echo "[STAGE] COCO already staged at $COCO_DST, skipping copy"
else
    echo "[STAGE] Copying COCO dataset to local SSD..."
    mkdir -p "$LOCAL_SSD"
    rsync -a --info=progress2 "$COCO_SRC/" "$COCO_DST/"
    echo "[STAGE] Done. $(ls $COCO_DST/train/ | wc -l) train + $(ls $COCO_DST/val/ | wc -l) val chips"
fi

# --- Launch training ---
cd "$WORKSPACE"
echo "[TRAIN] Starting training from $WORKSPACE"
echo "[TRAIN] COCO dir: $COCO_DST"
echo "[TRAIN] Output: $WORKSPACE/checkpoints_cleaned"

python3 train.py \
    --coco-dir "$COCO_DST" \
    --output-dir "$WORKSPACE/checkpoints_cleaned" \
    --batch-size 4 \
    --num-workers 4 \
    "$@"
