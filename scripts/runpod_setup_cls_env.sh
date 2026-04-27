#!/usr/bin/env bash
set -euo pipefail

# Lightweight RunPod classifier environment setup.
#
# This intentionally reuses the CUDA PyTorch stack provided by the RunPod
# image via --system-site-packages instead of rebuilding the full local .venv.
# It only installs the missing classifier/build dependencies and exposes the
# venv as <project>/.venv for existing scripts.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ZAsolar}"
VENV_PATH="${VENV_PATH:-/root/venvs/ZAsolar_cls}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PIP_PACKAGES=(
  "numpy==2.3.5"
  "scipy==1.17.1"
  "scikit-learn==1.8.0"
  "timm==1.0.25"
)

mkdir -p "$PROJECT_ROOT" "$(dirname "$VENV_PATH")"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_PATH"
fi

"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_PATH/bin/pip" install --no-cache-dir "${PIP_PACKAGES[@]}"

ln -sfn "$VENV_PATH" "$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import importlib
import sys

print("python", sys.executable)
for name in ["numpy", "torch", "torchvision", "timm", "sklearn", "scipy"]:
    mod = importlib.import_module(name)
    print(f"{name} {getattr(mod, '__version__', '?')}")

import torch
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))

from scripts.classifier import train_cls
for arch in ["efficientnet_b0", "resnet18", "convnext_tiny"]:
    train_cls.build_model(arch, pretrained=False)
print("classifier_model_smoke ok")
PY

echo
echo "Environment ready:"
echo "  source $PROJECT_ROOT/.venv/bin/activate"
echo "  python scripts/classifier/train_cls.py --data-dir /root/cls_data/cls_pv_thermal_v1 --output-dir /workspace/ZAsolar/checkpoints/cls_pv_thermal_v1_effb0 --config configs/classifier/efficientnet_b0.json"
