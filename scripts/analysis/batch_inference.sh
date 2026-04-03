#!/bin/bash
# Parallel batch inference with proper error handling and validation.
# Usage: bash batch_inference.sh <model_path> <grid_list> [jobs=4]
#
# Example:
#   bash scripts/analysis/batch_inference.sh \
#     /workspace/checkpoints/exp003_C_targeted_hn/best_model.pth \
#     "G1682 G1683 G1685" 4

set -u
WORKSPACE="${WORKSPACE:-/workspace/ZAsolar}"
cd "$WORKSPACE"

MODEL="$1"
GRIDS="$2"
JOBS="${3:-4}"
LABEL=$(basename "$(dirname "$MODEL")")
LOGDIR="/workspace/inference_logs/${LABEL}"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Batch inference: $LABEL"
echo "  Model: $MODEL"
echo "  Grids: $(echo $GRIDS | wc -w) grids, $JOBS parallel"
echo "  Logs:  $LOGDIR/"
echo "============================================"

# Clear old results for these grids
for g in $GRIDS; do rm -rf "results/$g"; done

# ── Parallel dispatch ─────────────────────────────────────────────
run_grid() {
    local grid=$1
    local log="$LOGDIR/${grid}.log"

    python3 detect_and_evaluate.py \
        --grid-id "$grid" \
        --model-path "$MODEL" \
        --evaluation-profile installation \
        --force \
        > "$log" 2>&1
    local rc=$?

    # Validate: presence_metrics.csv must exist and have data rows
    local csv="results/${grid}/presence_metrics.csv"
    if [ $rc -ne 0 ]; then
        echo "FAIL $grid  (exit=$rc, see $log)"
        return 1
    elif [ ! -f "$csv" ]; then
        echo "FAIL $grid  (no metrics CSV, see $log)"
        return 1
    elif [ "$(wc -l < "$csv")" -lt 2 ]; then
        echo "FAIL $grid  (empty metrics CSV)"
        return 1
    else
        # Extract key numbers for live feedback
        local line=$(tail -1 "$csv")
        echo "OK   $grid  $(echo "$line" | awk -F, '{printf "P=%.1f%% R=%.1f%% F1=%.1f%% TP=%s FP=%s FN=%s", $7*100, $8*100, $9*100, $4, $5, $6}')"
        return 0
    fi
}
export -f run_grid
export MODEL LOGDIR

running=0
pids=()
for g in $GRIDS; do
    run_grid "$g" &
    pids+=($!)
    running=$((running + 1))
    if [ $running -ge $JOBS ]; then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done
wait

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Summary: $LABEL"
echo "============================================"

python3 << 'PYEOF'
import csv, os, sys

grids = os.environ.get("GRIDS_ENV", "").split()
if not grids:
    # fallback: scan results dir
    grids = sorted(d for d in os.listdir("results") if os.path.isfile(f"results/{d}/presence_metrics.csv"))

print(f"{'grid':<8} {'prec':>6} {'recall':>6} {'f1':>6} {'gt':>5} {'pred':>5} {'tp':>5} {'fp':>5} {'fn':>5}")
print("-" * 62)

total_tp = total_fp = total_fn = 0
ok = fail = 0

for g in grids:
    path = f"results/{g}/presence_metrics.csv"
    if not os.path.exists(path):
        print(f"{g:<8} {'MISS':>6}")
        fail += 1
        continue
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{g:<8} {'EMPTY':>6}")
        fail += 1
        continue
    r = rows[0]
    tp = int(float(r.get("tp", 0)))
    fp = int(float(r.get("fp", 0)))
    fn = int(float(r.get("fn", 0)))
    n_gt = int(float(r.get("gt_count", r.get("n_gt", 0))))
    n_pred = int(float(r.get("pred_count", r.get("n_pred", 0))))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0
    total_tp += tp
    total_fp += fp
    total_fn += fn
    ok += 1
    print(f"{g:<8} {p:>5.1%} {rc:>5.1%} {f1:>5.1%} {n_gt:>5} {n_pred:>5} {tp:>5} {fp:>5} {fn:>5}")

print("-" * 62)
p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
rc = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0
print(f"{'TOTAL':<8} {p:>5.1%} {rc:>5.1%} {f1:>5.1%} {total_tp+total_fn:>5} {total_tp+total_fp:>5} {total_tp:>5} {total_fp:>5} {total_fn:>5}")
print(f"\n{ok} OK, {fail} FAIL out of {ok+fail} grids")
PYEOF
