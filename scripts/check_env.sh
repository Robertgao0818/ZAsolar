#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

unset VIRTUAL_ENV
export PATH="${HOME}/.local/bin:${PATH}"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
export XDG_DATA_HOME="$PROJECT_ROOT/.local/share"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export JOBLIB_TEMP_FOLDER="$PROJECT_ROOT/.tmp/joblib"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" \
  "$MPLCONFIGDIR" \
  "$JOBLIB_TEMP_FOLDER"

python3 - <<'PY'
import importlib
import os
import sys

modules = [
    "geopandas",
    "pandas",
    "shapely",
    "rasterio",
    "rasterstats",
    "matplotlib",
    "seaborn",
    "sklearn",
    "geoai",
    "osmnx",
    "requests",
    "torch",
    "torchvision",
    "cv2",
    "huggingface_hub",
    "pycocotools",
]

print(f"python={sys.executable}")
print(f"version={sys.version.split()[0]}")
print(f"mplconfigdir={os.environ.get('MPLCONFIGDIR')}")
print(f"joblib_temp={os.environ.get('JOBLIB_TEMP_FOLDER')}")

for name in modules:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}={version}")

import torch
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
PY
