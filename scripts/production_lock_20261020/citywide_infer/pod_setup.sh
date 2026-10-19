#!/usr/bin/env bash
# Pod env for ct_city_infer_v2.py: venv at /workspace/venv, versions pinned to rog (2026-10-20).
# RTX 5090 = sm_120 needs a cu128 torch wheel. Usage: bash pod_setup.sh ; then source /workspace/venv/bin/activate
set -euo pipefail
V=${VENV:-/workspace/venv}; PY=${PYVER:-3.12}
command -v uv >/dev/null || { python3 -m pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -x "$V/bin/python" ] || uv venv --python "$PY" "$V"
uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"
uv pip install --python "$V/bin/python" "timm==1.0.25" "transformers==5.16.1" "segmentation_models_pytorch==0.5.0" \
  "rasterio==1.5.1" "scipy==1.18.1" "pandas==3.0.5" "geopandas==1.1.2" "shapely==2.1.2" "pyogrio==0.12.1" "numpy==2.5.2" affine pytest
"$V/bin/python" - <<'PY'
import torch, transformers, timm, rasterio, rasterio.env
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'timm', timm.__version__,
      'rasterio', rasterio.__version__, 'gdal', rasterio.__gdal_version__)
assert torch.cuda.is_available(), 'no CUDA'
cap = torch.cuda.get_device_capability(0); print('device', torch.cuda.get_device_name(0), cap, torch.cuda.get_arch_list())
assert cap == (12, 0), f'expected sm_120, got {cap}'
assert 'sm_120' in torch.cuda.get_arch_list(), 'torch build lacks sm_120'
a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16); b = a @ a; torch.cuda.synchronize()
print('bf16 matmul ok', float(b.float().abs().mean()))
with rasterio.env.Env() as e: assert 'COG' in e.drivers(), 'GDAL has no COG driver'
print('COG driver ok; SETUP OK')
PY
