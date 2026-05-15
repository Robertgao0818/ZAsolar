#!/bin/bash
# RunPod pod 初始化脚本
# Base image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   已预装 torch 2.8.0+cu128 / torchvision 0.23.0+cu128 / triton 3.4 / numpy 2.1 / pillow 11
#   + 完整 nvidia-cuda-12.8 toolchain
# 本脚本只补装项目热路径需要的 GIS 栈，绝不让 pip 重装 torch
# （重装会从 PyPI 默认源抓 cu126 wheel 直接搞坏 Blackwell sm_120，参见
#  feedback_pod_runtime.md §6）
#
# 用法: ssh 连上后执行 bash /workspace/ZAsolar/scripts/runpod_init.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace/ZAsolar}"

# pip wheel cache 走 /workspace（mfs 持久卷，跨 pod 保留）；只缓存 wheel 文件
# 几十 MB，不撞 mfs inode quota。第二次新 pod init 时直接从 cache 装。
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.pip_cache}"
mkdir -p "$PIP_CACHE_DIR"

echo "=== RunPod Environment Init ==="
echo "Workspace:      $WORKSPACE"
echo "PIP_CACHE_DIR:  $PIP_CACHE_DIR"

# 1. 装项目热路径依赖到系统 python（base image 的 torch 也在系统 python 上）
#    显式列出非 torch 包；pip resolver 看到 torch 已装满足约束就不会重装。
#    geoai-py / pytest / osmnx 不装：主路径 detect_direct.py + finalize.py(direct mode)
#    都不需要，要跑 legacy postprocess_ablation 时再 `pip install --no-deps geoai-py`。
echo "[1/3] Installing project dependencies (system python)..."
pip install --break-system-packages \
    geopandas shapely rasterio pyogrio rasterstats \
    opencv-python-headless pycocotools \
    scikit-learn scipy matplotlib seaborn \
    timm huggingface_hub \
    "transformers>=4.45" \
    2>&1 | tail -5

# 2. 验证关键 import + GPU
echo "[2/3] Verifying imports..."
python3 -c "
import torch, geopandas, rasterio, rasterstats, shapely, cv2, pycocotools, timm, transformers
print(f'  torch={torch.__version__}, CUDA={torch.cuda.is_available()}')
print(f'  transformers={transformers.__version__}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'  GPU={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}')
print('  All imports OK')
"

# 3. 验证当前主推理脚本可加载
echo "[3/3] Verifying detect_direct.py..."
cd "$WORKSPACE"
python3 detect_direct.py --help > /dev/null 2>&1 && echo "  detect_direct.py OK" || echo "  WARN: detect_direct.py failed to load"

echo ""
echo "=== Init Complete ==="
echo "Tips:"
echo "  - Tiles (network volume): /workspace/tiles/"
echo "  - Copy to RAM:  cp -r /workspace/tiles/GXXXX /dev/shm/tiles/"
echo "  - Checkpoints:  /workspace/checkpoints/"
echo "  - V3-C weights: /workspace/checkpoints/exp003_C_targeted_hn/best_model.pth"
echo "  - 需要 geoai-py 时（legacy postprocess_ablation 等）:"
echo "      pip install --break-system-packages --no-deps geoai-py"
