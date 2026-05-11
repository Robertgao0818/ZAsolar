#!/usr/bin/env bash
# Rebuild train20_val5_hn predictions with pixel-or finalize + SAM mask+box + v4_agg conf>=0.65.
# Mirrors V3-C+SAM postproc chain so we can isolate model contribution from postproc artifact.
#
# Inputs:
#   results/analysis/v3c_failed_weight_compare/perdet/train20_val5_hn/G*/raw_detections.pkl
# Outputs (under results/analysis/v3c_failed_weight_compare/pixelor/):
#   train20_val5_hn_pixelor/                 — pixel-or finalized
#   train20_val5_hn_pixelor_sam_maskbox/     — SAM mask+box refined
#   train20_val5_hn_pixelor_sam_maskbox_v4agg/ — v4_agg conf>=0.65 filtered (final)

set -euo pipefail
cd /home/gaosh/projects/ZAsolar
source scripts/activate_env.sh

GRIDS=(G0772 G0773 G0774 G0775 G0776 G0814 G0815 G0816 G0817 G0818 \
       G0853 G0854 G0855 G0856 G0857 G0888 G0889 G0890 G0891 G0892 \
       G0922 G0923 G0924 G0925 G0926)

RAW_ROOT=results/analysis/v3c_failed_weight_compare/perdet/train20_val5_hn
OUT_BASE=results/analysis/v3c_failed_weight_compare/pixelor
PIXOR_DIR=$OUT_BASE/train20_val5_hn_pixelor
SAM_DIR=$OUT_BASE/train20_val5_hn_pixelor_sam_maskbox
V4AGG_DIR=$OUT_BASE/train20_val5_hn_pixelor_sam_maskbox_v4agg
TILES=/home/gaosh/zasolar_data/tiles/johannesburg/vexcel_2024
LOGDIR=$OUT_BASE/_logs
mkdir -p "$PIXOR_DIR" "$SAM_DIR" "$V4AGG_DIR" "$LOGDIR"

echo "[$(date +%T)] Stage 1: pixel-or finalize ($((${#GRIDS[@]})) grids)"
for g in "${GRIDS[@]}"; do
    raw=$RAW_ROOT/$g/raw_detections.pkl
    out=$PIXOR_DIR/$g
    if [[ ! -f $raw ]]; then
        echo "  [skip] $g: no raw_detections.pkl"
        continue
    fi
    mkdir -p "$out"
    python3 finalize.py \
        --input "$raw" \
        --output-dir "$out" \
        --postproc-config configs/postproc/v4_canonical.json \
        --merge-mode pixel-or \
        --allow-overwrite-canonical \
        > "$LOGDIR/${g}_finalize.log" 2>&1 \
        && echo "  [ok] $g finalize" \
        || { echo "  [FAIL] $g finalize"; tail -5 "$LOGDIR/${g}_finalize.log"; }
done

echo "[$(date +%T)] Stage 2: SAM mask+box refine"
python3 scripts/analysis/sam_refine_v4_2_maskprompt.py \
    --src-results-root "$PIXOR_DIR" \
    --tiles-root "$TILES" \
    --output-root "$SAM_DIR" \
    --prompt-mode mask_box \
    --grids "${GRIDS[@]}" \
    > "$LOGDIR/sam_refine.log" 2>&1 \
    && echo "  [ok] SAM refine done" \
    || { echo "  [FAIL] SAM refine"; tail -20 "$LOGDIR/sam_refine.log"; exit 1; }

echo "[$(date +%T)] Stage 3: v4_agg conf>=0.65 filter"
python3 scripts/analysis/filter_sam_inventory.py \
    --src-root "$SAM_DIR" \
    --config configs/postproc/v4_agg.json \
    --output-root "$V4AGG_DIR" \
    --force \
    > "$LOGDIR/filter_v4agg.log" 2>&1 \
    && echo "  [ok] v4_agg filter done" \
    || { echo "  [FAIL] v4_agg filter"; tail -20 "$LOGDIR/filter_v4agg.log"; exit 1; }

echo "[$(date +%T)] All stages complete. Final output: $V4AGG_DIR"
