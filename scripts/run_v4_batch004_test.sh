#!/bin/bash
# V4 batch 004 快速测试：拷到 /dev/shm → 并行推理 → 汇总结果
set -euo pipefail

MODEL="/workspace/checkpoints/exp004_v4_hn/best_model.pth"
POSTPROC="/workspace/ZAsolar/configs/postproc/v4_canonical.json"
TILES_SRC="/workspace/tiles"
TILES_SHM="/dev/shm/tiles"
WORK="/workspace/ZAsolar"
LOG_DIR="/workspace/inference_v4_batch004"
PARALLEL=6  # 5090 32GB, ~3-4GB per process

# Batch 004 grids (all with GT annotations)
GRIDS=(
  G1855 G1856 G1862 G1863 G1864
  G1909 G1910 G1911 G1917 G1919 G1920 G1921
  G1966 G1970 G1971 G1972 G1973 G1974 G1975 G1976 G1981
  G2025 G2026 G2027 G2028 G2029 G2030 G2031 G2032 G2037 G2038
)

mkdir -p "$LOG_DIR"

echo "=== Step 1: Copy tiles to /dev/shm ==="
mkdir -p "$TILES_SHM"
for g in "${GRIDS[@]}"; do
  if [ ! -d "$TILES_SHM/$g" ]; then
    cp -r "$TILES_SRC/$g" "$TILES_SHM/$g" &
  fi
done
wait
echo "Copied ${#GRIDS[@]} grids to /dev/shm"
du -sh "$TILES_SHM"

echo ""
echo "=== Step 2: Parallel inference (${PARALLEL}x) ==="
run_grid() {
  local g=$1
  cd "$WORK"
  SOLAR_TILES_ROOT="$TILES_SHM" python3 detect_and_evaluate.py \
    --grid-id "$g" \
    --model-path "$MODEL" \
    --postproc-config "$POSTPROC" \
    --force \
    > "$LOG_DIR/${g}.log" 2>&1
  # Extract key metrics
  local metrics="$WORK/results/${g}/presence_metrics.csv"
  if [ -f "$metrics" ]; then
    echo "OK $g $(tail -1 "$metrics")"
  else
    echo "FAIL $g (no metrics)"
  fi
}

# Run in parallel batches
running=0
for g in "${GRIDS[@]}"; do
  run_grid "$g" &
  running=$((running + 1))
  if [ $running -ge $PARALLEL ]; then
    wait -n  # Wait for any one job to finish
    running=$((running - 1))
  fi
done
wait
echo ""
echo "=== Step 3: Summary ==="

# Collect results
cd "$WORK"
python3 -u -c "
import csv, glob, json
from pathlib import Path

grids = '$( IFS=,; echo "${GRIDS[*]}" )'.split(',')
results = []

for g in grids:
    metrics_path = Path(f'results/{g}/presence_metrics.csv')
    if not metrics_path.exists():
        print(f'  {g}: NO RESULTS')
        continue
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'grid': g,
                'threshold': row.get('iou_threshold', '?'),
                'precision': float(row.get('precision', 0)),
                'recall': float(row.get('recall', 0)),
                'f1': float(row.get('f1', 0)),
                'tp': int(row.get('tp', 0)),
                'fp': int(row.get('fp', 0)),
                'fn': int(row.get('fn', 0)),
            })

if not results:
    print('No results found!')
    exit(1)

# Filter IoU=0.5 rows
r05 = [r for r in results if r['threshold'] in ('0.5', '0.50')]
if not r05:
    r05 = results  # fallback

print(f'\\n=== V4 Batch 004 Test Results (IoU=0.5) ===')
print(f'{\"Grid\":<8} {\"P\":>6} {\"R\":>6} {\"F1\":>6} {\"TP\":>5} {\"FP\":>5} {\"FN\":>5}')
print('-' * 50)

total_tp = total_fp = total_fn = 0
for r in sorted(r05, key=lambda x: x['grid']):
    print(f'{r[\"grid\"]:<8} {r[\"precision\"]:>5.1%} {r[\"recall\"]:>5.1%} {r[\"f1\"]:>5.1%} {r[\"tp\"]:>5} {r[\"fp\"]:>5} {r[\"fn\"]:>5}')
    total_tp += r['tp']
    total_fp += r['fp']
    total_fn += r['fn']

# Micro-average
p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
print('-' * 50)
print(f'{\"TOTAL\":<8} {p:>5.1%} {r:>5.1%} {f1:>5.1%} {total_tp:>5} {total_fp:>5} {total_fn:>5}')
print(f'\\n{len(r05)} grids evaluated')
"

echo ""
echo "=== Done ==="
echo "Logs: $LOG_DIR/"
echo "Results: $WORK/results/G*/presence_metrics.csv"
