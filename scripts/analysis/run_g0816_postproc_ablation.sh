#!/usr/bin/env bash
# Re-finalize G0816 overlap=0.5 raw_detections.pkl under three postproc variants.
# All from the SAME raw detections — only post-proc differs. CPU-only.
set -euo pipefail

cd /home/gaosh/projects/ZAsolar
source scripts/activate_env.sh

RAW=results/diag/v3c_overlap50_G0816_raw/raw_detections.pkl
OUT_BASE=results/diag/g0816_postproc_ablation

for VARIANT in canonical tier_gentle tier_aggressive hysteresis; do
    if [[ "$VARIANT" == "canonical" ]]; then
        CFG=configs/postproc/v4_canonical.json
    else
        CFG=configs/postproc/v4_canonical_${VARIANT}.json
    fi
    OUT="$OUT_BASE/$VARIANT"
    mkdir -p "$OUT"
    echo "[$(date +%T)] === finalize: $VARIANT ==="
    python3 finalize.py \
        --input "$RAW" \
        --output-dir "$OUT" \
        --postproc-config "$CFG" \
        --allow-overwrite-canonical \
        2>&1 | tail -8
done

echo
echo "Polygon counts:"
for V in canonical tier_gentle tier_aggressive hysteresis; do
    if [[ -f "$OUT_BASE/$V/predictions_metric.gpkg" ]]; then
        N=$(python3 -c "import geopandas as gpd; print(len(gpd.read_file('$OUT_BASE/$V/predictions_metric.gpkg')))")
        echo "  $V: $N polygons"
    fi
done
