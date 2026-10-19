#!/usr/bin/env bash
# Pod setup for ct_city_infer_v2.py (RunPod RTX 5090, PyTorch 2.8.0 + CUDA 12.9 image). Keeps the image's torch.
set -euo pipefail
python3 -m pip install --no-input "transformers==5.16.1" "timm==1.0.25" "rasterio>=1.4" "scipy" "pandas" "geopandas" "shapely>=2" "pyogrio" "affine"
python3 - <<'PY'
import torch, transformers, timm, rasterio
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'timm', timm.__version__, 'rasterio', rasterio.__version__, 'gdal', rasterio.__gdal_version__)
assert torch.cuda.is_available(), 'no CUDA'
cap = torch.cuda.get_device_capability(0); arch = f'sm_{cap[0]}{cap[1]}'
print('device', torch.cuda.get_device_name(0), arch, 'compiled archs', torch.cuda.get_arch_list())
assert arch in torch.cuda.get_arch_list(), f'{arch} not in this torch build'
x = torch.randn(2, 3, 640, 640, device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    y = torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3, device='cuda'))
torch.cuda.synchronize(); print('bf16 conv ok', tuple(y.shape))
import rasterio.env
with rasterio.env.Env() as e: assert 'COG' in e.drivers(), 'GDAL has no COG driver'
print('COG driver ok')
PY
