#!/usr/bin/env bash
# RunPod batch: V4.1 inference on Vexcel JHB CBD 25 grids.
# Tiles must be staged at $SOLAR_TILES_ROOT/{grid}/ (default /dev/shm/tiles).
# PARALLEL=6 (RTX 4090, ~3-4GB VRAM/proc).
# Output goes to results/johannesburg/v4_1_vexcel_2024/ (symlinked to /root).

set -euo pipefail

REPO=/workspace/ZAsolar
MODEL="$REPO/checkpoints/exp005_v4_1_hn/best_model.pth"
POSTPROC="$REPO/configs/postproc/v4_canonical.json"
SOLAR_TILES_ROOT="${SOLAR_TILES_ROOT:-/dev/shm/tiles}"
PARALLEL="${PARALLEL:-6}"
LOG_DIR=/workspace/logs/vexcel_jhb_cbd25_v4_1
mkdir -p "$LOG_DIR"

GRIDS=(
  G0772 G0773 G0774 G0775 G0776
  G0814 G0815 G0816 G0817 G0818
  G0853 G0854 G0855 G0856 G0857
  G0888 G0889 G0890 G0891 G0892
  G0922 G0923 G0924 G0925 G0926
)

missing=()
for g in "${GRIDS[@]}"; do
  [ -d "$SOLAR_TILES_ROOT/$g" ] || missing+=("$g")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "FATAL: missing tiles in $SOLAR_TILES_ROOT for: ${missing[*]}" >&2
  exit 1
fi

run_grid() {
  local g=$1
  SOLAR_TILES_ROOT="$SOLAR_TILES_ROOT" python3 -u "$REPO/detect_and_evaluate.py" \
    --grid-id "$g" \
    --region johannesburg \
    --imagery-layer vexcel_2024 \
    --model-run v4_1_vexcel_2024 \
    --model-path "$MODEL" \
    --postproc-config "$POSTPROC" \
    --evaluation-profile installation \
    --force \
    > "$LOG_DIR/${g}.log" 2>&1
}

cd "$REPO"
echo "[v4_1 batch] PARALLEL=$PARALLEL, SOLAR_TILES_ROOT=$SOLAR_TILES_ROOT, ${#GRIDS[@]} grids @ $(date)"
SECONDS=0
running=0
for g in "${GRIDS[@]}"; do
  echo "[$SECONDS s] launch $g"
  run_grid "$g" &
  running=$((running + 1))
  if [ $running -ge $PARALLEL ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "[v4_1 batch] all ${#GRIDS[@]} grids finished in ${SECONDS}s @ $(date)"
