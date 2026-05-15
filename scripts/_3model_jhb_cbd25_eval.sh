#!/usr/bin/env bash
# Local-side driver for the 3-model JHB CBD 25-grid comparison.
# Run AFTER the pod-side scripts/runpod_3model_jhb_cbd25_compare.sh has
# produced six result dirs and they've been pulled back to:
#   results/analysis/direct_maskrcnn_v1/johannesburg/<run_id>/<grid>/
#
# Assumes the six model_run entries below are already registered in
# configs/datasets/regions.yaml (johannesburg/model_runs). If not, exit
# with a message so the operator can patch the registry first.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

RUNS=(
  v3c_vexcel_2024_direct
  v3c_vexcel_2024_direct_sam_maskbox
  train20_val5_vexcel_2024_direct
  train20_val5_vexcel_2024_direct_sam_maskbox
  unified_reviewall_A_vexcel_2024_direct
  unified_reviewall_A_vexcel_2024_direct_sam_maskbox
)

# Sanity check: all 6 runs must be registered.
for r in "${RUNS[@]}"; do
  if ! grep -q "^      ${r}:" configs/datasets/regions.yaml; then
    echo "[fatal] model_run '${r}' is not registered in configs/datasets/regions.yaml"
    echo "        Patch the johannesburg/model_runs block before running this script."
    exit 2
  fi
done

OUTDIR="results/analysis/jhb_cbd25_3model_20260514"
mkdir -p "$OUTDIR"

# Tier-1 area-aggregate eval against the locked clean_gt (per
# .claude memory: feedback_eval_gt_lock_clean.md).
python3 scripts/analysis/area_aggregate_eval.py \
  --region johannesburg \
  --run "${RUNS[@]}" \
  --gt-root data/annotations_channel2_clean \
  --gt-pattern "{grid}/{grid}_clean_gt.gpkg" \
  --output-dir "$OUTDIR"

echo
echo "=== outputs ==="
ls -la "$OUTDIR"
echo
echo "Primary winner table (from per_run_summary.csv):"
python3 - <<PY
import csv
from pathlib import Path
path = Path("$OUTDIR/per_run_summary.csv")
with path.open() as f:
    rows = list(csv.DictReader(f))
# Per feedback_tier1_metric_system.md: rank by σ_Bw + RMSE, gate bulk ∈ [0.5, 2.0]
def num(r, k):
    v = r.get(k)
    try: return float(v) if v not in (None, "", "None") else None
    except Exception: return None
print(f"{'model_run':<48} {'n':>3} {'F1':>6} {'pgF1':>6} {'bulk':>6} "
      f"{'σ_Bw':>6} {'log-σ':>6} {'RMSE':>8} {'thru0_β':>8} {'R²':>6}")
print("-" * 110)
def keyfn(r):
    sigma = num(r, "std_ratio_Bw")
    rmse = num(r, "rmse_m2")
    bulk = num(r, "bulk_pred_gt_ratio")
    in_gate = bulk is not None and 0.5 <= bulk <= 2.0
    sigma_v = sigma if sigma is not None else 9.99
    rmse_v = rmse if rmse is not None else 9e9
    return (not in_gate, sigma_v + rmse_v / 1e5)
for r in sorted(rows, key=keyfn):
    def f(k, w=6, p=3):
        v = num(r, k)
        return f"{'-':>{w}}" if v is None else f"{v:>{w}.{p}f}"
    print(f"{r['model_run']:<48} {r['n_grids']:>3} "
          f"{f('agg_area_F1')} {f('mean_per_grid_F1')} {f('bulk_pred_gt_ratio')} "
          f"{f('std_ratio_Bw')} {f('std_logratio')} {f('rmse_m2', 8, 1)} "
          f"{f('thru0_slope', 8, 3)} {f('ols_r2')}")
PY
