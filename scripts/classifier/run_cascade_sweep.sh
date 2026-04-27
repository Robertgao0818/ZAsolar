#!/usr/bin/env bash
# Cascade evaluation: V4.2 GEID predictions → classifier filter → installation eval
# Sweeps PV-threshold over {0.3, 0.4, 0.5, 0.6, 0.7} on the 25 JHB CBD GEID grids.
#
# Usage:
#   bash scripts/classifier/run_cascade_sweep.sh \
#     --classifier checkpoints/cls_pv_thermal_v1_effb0/best_cls.pth \
#     --tag effb0 \
#     [--thresholds "0.3 0.4 0.5 0.6 0.7"]
#
# Outputs:
#   results/johannesburg/v4_2_geid_2024_02/<grid>/predictions_metric_cls_filtered_<tag>_thr<thr>.gpkg
#   results/analysis/cls_cascade_sweep/<tag>/<thr>/eval_summary.csv

set -euo pipefail

THRESHOLDS="0.3 0.4 0.5 0.6 0.7"
TAG=""
CLASSIFIER=""
RESULTS_DIR="results/johannesburg/v4_2_geid_2024_02"
GRIDS="G0772 G0773 G0774 G0775 G0776 G0814 G0815 G0816 G0817 G0818 G0853 G0854 G0855 G0856 G0857 G0888 G0889 G0890 G0891 G0892 G0922 G0923 G0924 G0925 G0926"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --classifier) CLASSIFIER="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --thresholds) THRESHOLDS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$CLASSIFIER" || -z "$TAG" ]] && { echo "ERROR: --classifier and --tag required"; exit 1; }

OUT_ROOT="results/analysis/cls_cascade_sweep/$TAG"
mkdir -p "$OUT_ROOT"

for thr in $THRESHOLDS; do
  echo ""
  echo "===== Threshold: $thr ====="
  thr_dir="$OUT_ROOT/thr$thr"
  mkdir -p "$thr_dir"

  echo "[1/2] Classifying predictions..."
  python3 scripts/classifier/classify_predictions.py \
    --grid-ids $GRIDS \
    --model-path "$CLASSIFIER" \
    --results-dir "$RESULTS_DIR" \
    --pv-threshold "$thr" \
    --area-cutoff 30 \
    > "$thr_dir/classify.log" 2>&1

  echo "[2/2] Eval per grid (installation profile)..."
  : > "$thr_dir/eval_summary.csv"
  for g in $GRIDS; do
    filtered_gpkg="$RESULTS_DIR/$g/predictions_metric_cls_filtered.gpkg"
    [[ ! -f "$filtered_gpkg" ]] && { echo "MISS $g"; continue; }
    python3 detect_and_evaluate.py \
      --grid-id "$g" --region johannesburg --imagery-layer geid_2024_02 \
      --classifier-filtered-gpkg "$filtered_gpkg" \
      --postproc-config configs/postproc/v4_canonical.json \
      > "$thr_dir/eval_$g.log" 2>&1 || echo "EVAL_FAIL $g"
  done
  echo "Done thr=$thr → $thr_dir"
done

echo ""
echo "===== Sweep complete: $OUT_ROOT ====="
