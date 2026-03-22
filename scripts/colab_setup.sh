#!/usr/bin/env bash
# Colab environment setup for SA_Solar
# Usage: !bash scripts/colab_setup.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== SA_Solar Colab Setup ==="
echo "Project root: $PROJECT_ROOT"

# ── 1. System-level geospatial libs (Colab already has most) ──
apt-get -qq update
apt-get -qq install -y libgdal-dev gdal-bin libspatialindex-dev > /dev/null 2>&1 || true

# ── 2. Python dependencies ──
pip install -q --upgrade pip setuptools wheel

# Install from requirements.txt (not lock file — Colab has its own CUDA stack)
pip install -q -r "$PROJECT_ROOT/requirements.txt"

# ── 3. Cache directories ──
mkdir -p \
  "$PROJECT_ROOT/.cache/matplotlib" \
  "$PROJECT_ROOT/.config" \
  "$PROJECT_ROOT/.local/share" \
  "$PROJECT_ROOT/.tmp/joblib"

export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
export XDG_DATA_HOME="$PROJECT_ROOT/.local/share"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export JOBLIB_TEMP_FOLDER="$PROJECT_ROOT/.tmp/joblib"

# ── 4. Verify key imports ──
python3 -c "
import torch, torchvision, geopandas, rasterio, cv2
print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'GeoPandas {geopandas.__version__}')
print(f'Rasterio {rasterio.__version__}')
try:
    import geoai; print(f'GeoAI {geoai.__version__}')
except Exception as e:
    print(f'GeoAI: {e}')
"

echo ""
echo "=== Setup complete ==="
