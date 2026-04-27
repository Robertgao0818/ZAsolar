#!/usr/bin/env bash
# Train cls_pv_thermal_v2 with three backbones sequentially.
# Outputs go to checkpoints/cls_pv_thermal_v2_{arch}/ and logs/cls_v2_{arch}.log
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/activate_env.sh

DATA_DIR=data/cls_pv_thermal_v2
LOG_DIR=logs
mkdir -p "$LOG_DIR"

run_one() {
  local arch=$1
  local config=$2
  local bs=$3
  local out_dir="checkpoints/cls_pv_thermal_v2_${arch}"
  local log="${LOG_DIR}/cls_v2_${arch}.log"

  echo "============================================================"
  echo "[$(date +%H:%M:%S)] Training $arch (bs=$bs) → $out_dir"
  echo "============================================================"

  python3 -u scripts/classifier/train_cls.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$out_dir" \
    --config "$config" \
    --batch-size "$bs" \
    --workers 4 \
    2>&1 | tee "$log"

  echo "[$(date +%H:%M:%S)] Done $arch"
}

# 4070 Laptop 8GB VRAM — conservative batch sizes
run_one efficientnet_b0 configs/classifier/efficientnet_b0.json 32
run_one convnext_tiny   configs/classifier/convnext_tiny.json   24
run_one dinov2_vits14   configs/classifier/dinov2_vits14.json   16

echo "All three backbones trained. Checkpoints under checkpoints/cls_pv_thermal_v2_*"
