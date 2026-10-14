#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/activate_env.sh

python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this shell. Run this from your normal WSL terminal.")

print(f"device0={torch.cuda.get_device_name(0)}")
PY

echo
echo "[1/2] 3-grid baseline with current default parameters"
python scripts/analysis/multi_grid_baseline.py \
  --grid-ids G1238 G1189 G1190 \
  --output-subdir baseline_default \
  --force

echo
echo "[2/2] Generalization check using best G1238 params"
python scripts/analysis/multi_grid_baseline.py \
  --grid-ids G1238 G1189 G1190 \
  --best-from-summary G1238 \
  --force

echo
echo "Done."
