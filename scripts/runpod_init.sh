#!/bin/bash
# RunPod pod 初始化脚本
# 基于 runpod/pytorch template (torch + CUDA 预装)，只需补装 GIS 依赖
# 用法: ssh 连上后执行 bash /workspace/ZAsolar/scripts/runpod_init.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace/ZAsolar}"

echo "=== RunPod Environment Init ==="
echo "Workspace: $WORKSPACE"

# 1. 补装 GIS 和项目依赖（系统 Python，不用 venv）
echo "[1/3] Installing project dependencies..."
pip install --break-system-packages --ignore-installed blinker \
    geopandas rasterio rasterstats shapely pycocotools \
    huggingface_hub opencv-python-headless geoai-py seaborn \
    2>&1 | tail -3

# 2. 验证关键 import
echo "[2/3] Verifying imports..."
python3 -c "
import torch, geopandas, rasterio, rasterstats, seaborn
print(f'  torch={torch.__version__}, CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU={torch.cuda.get_device_name(0)}')
print('  All imports OK')
"

# 3. 验证推理脚本可加载
echo "[3/3] Verifying detect_and_evaluate.py..."
cd "$WORKSPACE"
python3 detect_and_evaluate.py --help > /dev/null 2>&1 && echo "  detect_and_evaluate.py OK" || echo "  WARN: detect_and_evaluate.py failed to load"

echo ""
echo "=== Init Complete ==="
echo "Tips:"
echo "  - Tiles (network volume): /workspace/tiles/"
echo "  - Copy to RAM:  cp -r /workspace/tiles/GXXXX /dev/shm/tiles/"
echo "  - Checkpoints:  /workspace/checkpoints/"
echo "  - V3-C weights: /workspace/checkpoints/exp003_C_targeted_hn/best_model.pth"
