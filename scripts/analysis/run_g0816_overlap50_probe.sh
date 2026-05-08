#!/usr/bin/env bash
# G0816 overlap=0.5 probe: run V3-C inference with doubled overlap (stride 200 px),
# then finalize with canonical post-proc. Output to results/diag/v3c_overlap50_G0816/.
# This isolates whether chip-edge cutoffs (cat 2) close up with denser sliding window.
set -euo pipefail

cd /home/gaosh/projects/ZAsolar
source scripts/activate_env.sh

GRID=G0816
RAW_OUT=results/diag/v3c_overlap50_G0816_raw
FINAL_OUT=results/diag/v3c_overlap50_G0816
MODEL=checkpoints/exp003_C_targeted_hn/best_model.pth

mkdir -p "$RAW_OUT" "$FINAL_OUT"

echo "[$(date +%T)] === detect_direct (overlap=0.5) ==="
python3 detect_direct.py \
    --grid-id "$GRID" \
    --region jhb \
    --imagery-layer vexcel_2024 \
    --model-run "v3c_overlap50_probe" \
    --model-path "$MODEL" \
    --chip-size 400 \
    --overlap 0.5 \
    --batch-size 4 \
    --output-dir "$RAW_OUT"

echo "[$(date +%T)] === finalize (pixel-or canonical) ==="
python3 finalize.py \
    --input "$RAW_OUT/raw_detections.pkl" \
    --output-dir "$FINAL_OUT" \
    --postproc-config configs/postproc/v4_canonical.json

echo "[$(date +%T)] === done ==="
ls -lh "$FINAL_OUT"
