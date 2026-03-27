"""
WMS 瓦片下载 + 地理配准
Download and georeference aerial tiles from Cape Town WMS for a given grid.

数据源: Cape Town Open Data — Aerial Imagery 2025Jan
WMS:    https://cityimg.capetown.gov.za/erdas-iws/ogc/wms/GeoSpatial Datasets
Layer:  Aerial Imagery_Aerial Imagery 2025Jan

Usage:
  python scripts/download_tiles.py --grid-id G1189
  python scripts/download_tiles.py --grid-id G1190
  python scripts/download_tiles.py --grid-id G1189 --dry   # 只打印 tile 数
"""

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import requests
from PIL import Image

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import get_grid_spec, get_tile_bounds, normalize_grid_id

# ── WMS 配置（来自 QGIS 项目原始存档） ──────────────────────────────────
WMS_URL = "https://cityimg.capetown.gov.za/erdas-iws/ogc/wms/GeoSpatial Datasets"
WMS_LAYER = "Aerial Imagery_Aerial Imagery 2025Jan"
WMS_FORMAT = "image/jpeg"
DEFAULT_TIMEOUT = 300


def download_tile(spec, col, row, out_path: Path, timeout=DEFAULT_TIMEOUT):
    """下载单个 WMS 瓦片并保存为带地理参考的 GeoTIFF (EPSG:4326)。"""
    txmin, tymin, txmax, tymax = get_tile_bounds(spec, col, row)

    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": WMS_LAYER,
        "srs": "EPSG:4326",
        "bbox": f"{txmin},{tymin},{txmax},{tymax}",
        "width": spec.pixel_size,
        "height": spec.pixel_size,
        "format": WMS_FORMAT,
        "styles": "",
    }

    response = requests.get(WMS_URL, params=params, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "xml" in content_type.lower() or "html" in content_type.lower():
        # 尝试提取 WMS 错误信息
        error_detail = ""
        try:
            import re
            match = re.search(r"<ServiceException[^>]*>(.*?)</ServiceException>",
                              response.text, re.DOTALL)
            if match:
                error_detail = f" — {match.group(1).strip()}"
        except Exception:
            pass
        raise RuntimeError(f"WMS 返回非影像响应: {content_type}{error_detail}")

    # 解码 JPEG → numpy array
    img = Image.open(BytesIO(response.content))
    data = np.array(img)

    # 写入带地理参考的 GeoTIFF
    transform = from_bounds(txmin, tymin, txmax, tymax, spec.pixel_size, spec.pixel_size)

    if data.ndim == 3:
        bands, h, w = data.shape[2], data.shape[0], data.shape[1]
        data = np.transpose(data, (2, 0, 1))  # HWC → CHW
    else:
        bands, h, w = 1, data.shape[0], data.shape[1]
        data = data[np.newaxis, :, :]

    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": w,
        "height": h,
        "count": bands,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(data)


def download_grid(grid_id: str, dry_run: bool = False, tile_mask: set | None = None):
    """Download tiles for a grid.

    Args:
        grid_id: Grid identifier.
        dry_run: Only print info, don't download.
        tile_mask: Optional set of (col, row) tuples to download.
                   If None, download all tiles.
    """
    grid_id = normalize_grid_id(grid_id)
    spec = get_grid_spec(grid_id)

    from core.grid_utils import TILES_ROOT
    tiles_dir = TILES_ROOT / grid_id
    tiles_dir.mkdir(parents=True, exist_ok=True)

    total_all = spec.n_cols * spec.n_rows
    if tile_mask is not None:
        tiles_to_download = sorted(tile_mask)
        total = len(tiles_to_download)
        mode = f"masked ({total}/{total_all} tiles)"
    else:
        tiles_to_download = [(col, row) for col in range(spec.n_cols) for row in range(spec.n_rows)]
        total = total_all
        mode = f"all ({total} tiles)"

    print(f"Grid: {grid_id}")
    print(f"  Bounds: ({spec.xmin:.6f}, {spec.ymin:.6f}) → ({spec.xmax:.6f}, {spec.ymax:.6f})")
    print(f"  Tiles: {spec.n_cols} cols × {spec.n_rows} rows = {total_all}")
    print(f"  Download: {mode}")
    print(f"  WMS Layer: {WMS_LAYER}")
    print(f"  Output: {tiles_dir}")

    if dry_run:
        return

    downloaded = 0
    skipped = 0
    errors = 0

    for col, row in tiles_to_download:
        tile_name = f"{grid_id}_{col}_{row}_geo.tif"
        out_path = tiles_dir / tile_name

        if out_path.exists():
            skipped += 1
            continue

        try:
            download_tile(spec, col, row, out_path)
            downloaded += 1
            print(f"  [{downloaded + skipped}/{total}] {tile_name}")
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {tile_name}: {e}")
            time.sleep(3)
            try:
                download_tile(spec, col, row, out_path, timeout=600)
                downloaded += 1
                errors -= 1
                print(f"  [RETRY OK] {tile_name}")
            except Exception as e2:
                print(f"  [RETRY FAIL] {tile_name}: {e2}")

    print(f"\n[DONE] {grid_id}: downloaded={downloaded}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 Cape Town WMS 瓦片")
    parser.add_argument("--grid-id", required=True, help="目标 grid ID (e.g. G1189)")
    parser.add_argument("--dry", action="store_true", help="只打印 tile 信息")
    args = parser.parse_args()
    download_grid(args.grid_id, dry_run=args.dry)
