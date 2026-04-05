"""
ArcGIS ImageServer 瓦片下载 + 地理配准 (并发版)
Download Johannesburg aerial tiles from CoJ 2023 ArcGIS ImageServer.

数据源: City of Johannesburg — Aerial Photography 2023
ImageServer: https://ags.joburg.org.za/server/rest/services/AerialPhotography/2023/ImageServer
分辨率: 0.15m/px (15cm GSD)

Usage:
  # 单 grid
  python scripts/imagery/download_jhb_tiles.py --grid-id G0772
  python scripts/imagery/download_jhb_tiles.py --grid-id G0772 --dry

  # 整批 (100 grids, 并发下载)
  python scripts/imagery/download_jhb_tiles.py --batch batch1 --workers 6
  python scripts/imagery/download_jhb_tiles.py --batch batch1 --dry

  # RunPod 用法 (tiles 落到 /workspace/tiles_joburg/)
  SOLAR_TILES_ROOT=/workspace/tiles_joburg python scripts/imagery/download_jhb_tiles.py --batch batch1 --workers 6

注意: Joburg tiles 必须和 Cape Town tiles 分开存放 (GridID 体系不同但编号可能重叠)
  - Cape Town: SOLAR_TILES_ROOT=/workspace/tiles (默认)
  - Joburg:    SOLAR_TILES_ROOT=/workspace/tiles_joburg
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import rasterio
from rasterio.transform import from_bounds

# ── 项目导入 ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import normalize_grid_id

# Read Joburg grid directly — do NOT use get_grid_record() which merges
# Cape Town + Joburg grids and has 46 overlapping IDs.
import geopandas as gpd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_JHB_GRID = gpd.read_file(_PROJECT_ROOT / "data" / "jhb_task_grid.gpkg")


def _get_jhb_record(grid_id: str):
    grid_id = normalize_grid_id(grid_id)
    matches = _JHB_GRID.loc[_JHB_GRID["gridcell_id"] == grid_id]
    if len(matches) == 0:
        raise KeyError(f"Grid {grid_id} not found in jhb_task_grid.gpkg")
    return matches.iloc[0]

TILES_ROOT = Path(os.environ.get("SOLAR_TILES_ROOT", Path(__file__).resolve().parents[2] / "tiles"))

# Joburg grids are ~0.010 x 0.009 deg. Equal-split into N_COLS x N_ROWS
# so every tile has the same pixel dimensions (no thin edge strips).
JHB_N_COLS = 6
JHB_N_ROWS = 6

# ── ImageServer 配置 ──────────────────────────────────────────────────
IMAGE_SERVER_URL = (
    "https://ags.joburg.org.za/server/rest/services/"
    "AerialPhotography/2023/ImageServer/exportImage"
)
NATIVE_PIXEL_SIZE = 0.15  # meters in EPSG:3857
USER_AGENT = "Mozilla/5.0 (ZAsolar pipeline)"
DEFAULT_TIMEOUT = 300
MAX_RETRIES = 3

# ── 进度跟踪 (线程安全) ─────────────────────────────────────────────
_lock = threading.Lock()
_progress = {"downloaded": 0, "skipped": 0, "errors": 0, "total": 0}


def _report(msg: str) -> None:
    with _lock:
        print(msg, flush=True)


def _tick(field: str) -> int:
    with _lock:
        _progress[field] += 1
        return _progress["downloaded"] + _progress["skipped"]


# ── Batch definitions ────────────────────────────────────────────────
BATCHES = {
    "batch1": [
        # CBD (Joburg)
        "G0772", "G0773", "G0774", "G0775", "G0776",
        "G0814", "G0815", "G0816", "G0817", "G0818",
        "G0853", "G0854", "G0855", "G0856", "G0857",
        "G0888", "G0889", "G0890", "G0891", "G0892",
        "G0922", "G0923", "G0924", "G0925", "G0926",
        # Residential Rich (Sandton)
        "G1110", "G1111", "G1112", "G1113", "G1114",
        "G1144", "G1145", "G1146", "G1147", "G1148",
        "G1179", "G1180", "G1181", "G1182", "G1183",
        "G1214", "G1215", "G1216", "G1217", "G1218",
        "G1250", "G1251", "G1252", "G1253", "G1254",
        # Residential Poor (Alexandra)
        "G1151", "G1152", "G1153", "G1154", "G1155",
        "G1186", "G1187", "G1188", "G1189", "G1190",
        "G1221", "G1222", "G1223", "G1224", "G1225",
        "G1257", "G1258", "G1259", "G1260", "G1261",
        "G1293", "G1294", "G1295", "G1296", "G1297",
        # Industrial (Midrand)
        "G1512", "G1513", "G1514", "G1515", "G1516",
        "G1541", "G1542", "G1543", "G1544", "G1545",
        "G1570", "G1571", "G1572", "G1573", "G1574",
        "G1600", "G1601", "G1602", "G1603", "G1604",
        "G1630", "G1631", "G1632", "G1633", "G1634",
    ],
}


# ── 核心下载逻辑 ─────────────────────────────────────────────────────

def lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


@dataclass(frozen=True)
class JhbGridSpec:
    grid_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    n_cols: int
    n_rows: int

    @property
    def col_width(self) -> float:
        return (self.xmax - self.xmin) / self.n_cols

    @property
    def row_height(self) -> float:
        return (self.ymax - self.ymin) / self.n_rows


def get_jhb_grid_spec(grid_id: str) -> JhbGridSpec:
    grid_id = normalize_grid_id(grid_id)
    record = _get_jhb_record(grid_id)
    xmin, ymin, xmax, ymax = record.geometry.bounds
    return JhbGridSpec(
        grid_id=grid_id,
        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
        n_cols=JHB_N_COLS, n_rows=JHB_N_ROWS,
    )


def get_jhb_tile_bounds(spec: JhbGridSpec, col: int, row: int) -> tuple[float, float, float, float]:
    txmin = spec.xmin + col * spec.col_width
    txmax = spec.xmin + (col + 1) * spec.col_width
    tymax = spec.ymax - row * spec.row_height
    tymin = spec.ymax - (row + 1) * spec.row_height
    return txmin, tymin, txmax, tymax


def download_one_tile(
    spec: JhbGridSpec, col: int, row: int, out_path: Path, timeout: int = DEFAULT_TIMEOUT
) -> None:
    """Download a single tile from ArcGIS ImageServer and save as GeoTIFF (EPSG:4326)."""
    txmin, tymin, txmax, tymax = get_jhb_tile_bounds(spec, col, row)

    mx_min, my_min = lonlat_to_3857(txmin, tymin)
    mx_max, my_max = lonlat_to_3857(txmax, tymax)

    width_px = max(1, round((mx_max - mx_min) / NATIVE_PIXEL_SIZE))
    height_px = max(1, round((my_max - my_min) / NATIVE_PIXEL_SIZE))

    params = {
        "bbox": f"{mx_min},{my_min},{mx_max},{my_max}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    url = f"{IMAGE_SERVER_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    if data[:4] not in (b"II*\x00", b"MM\x00*"):
        preview = data[:200].decode("utf-8", "ignore")
        raise RuntimeError(f"Expected TIFF, got: {preview[:100]}")

    with rasterio.open(BytesIO(data)) as src:
        img_data = src.read()

    transform = from_bounds(txmin, tymin, txmax, tymax, width_px, height_px)
    profile = {
        "driver": "GTiff",
        "dtype": img_data.dtype,
        "width": width_px,
        "height": height_px,
        "count": img_data.shape[0],
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(img_data)


def _download_tile_task(spec, col: int, row: int, out_path: Path, grid_id: str) -> str:
    """Worker function for thread pool. Returns status string."""
    tile_name = out_path.name

    if out_path.exists() and out_path.stat().st_size > 1000:
        _tick("skipped")
        return "skipped"

    for attempt in range(MAX_RETRIES):
        try:
            t = DEFAULT_TIMEOUT if attempt == 0 else 600
            download_one_tile(spec, col, row, out_path, timeout=t)
            done = _tick("downloaded")
            total = _progress["total"]
            if done % 20 == 0 or done == 1:
                _report(f"  [{done}/{total}] {grid_id}/{tile_name}")
            return "ok"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(3 * (attempt + 1))
            else:
                _tick("errors")
                _report(f"  [ERROR] {grid_id}/{tile_name}: {e}")
                return "error"
    return "error"


def download_grid(grid_id: str, workers: int = 1, dry_run: bool = False) -> dict:
    grid_id = normalize_grid_id(grid_id)
    spec = get_jhb_grid_spec(grid_id)

    tiles_dir = TILES_ROOT / grid_id
    tiles_dir.mkdir(parents=True, exist_ok=True)

    total = spec.n_cols * spec.n_rows

    print(f"Grid: {grid_id}")
    print(f"  Bounds: ({spec.xmin:.6f}, {spec.ymin:.6f}) -> ({spec.xmax:.6f}, {spec.ymax:.6f})")
    print(f"  Tiles: {spec.n_cols} cols x {spec.n_rows} rows = {total}")
    print(f"  Source: CoJ 2023 Aerial (0.15m/px)")
    print(f"  Output: {tiles_dir}")

    if dry_run:
        return {"grid_id": grid_id, "total": total, "downloaded": 0, "skipped": 0, "errors": 0}

    # Build tile list
    tasks = []
    for col in range(spec.n_cols):
        for row in range(spec.n_rows):
            tile_name = f"{grid_id}_{col}_{row}_geo.tif"
            out_path = tiles_dir / tile_name
            tasks.append((spec, col, row, out_path, grid_id))

    # Download (single grid uses sequential for cleaner output)
    downloaded = 0
    skipped = 0
    errors = 0

    for spec_, col, row, out_path, gid in tasks:
        result = _download_tile_task(spec_, col, row, out_path, gid)
        if result == "ok":
            downloaded += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1

    print(f"  => downloaded={downloaded}, skipped={skipped}, errors={errors}")
    return {"grid_id": grid_id, "total": total, "downloaded": downloaded, "skipped": skipped, "errors": errors}


def download_batch(grid_ids: list[str], workers: int = 4, dry_run: bool = False) -> list[dict]:
    """Download multiple grids with tile-level concurrency across all grids."""

    # Build global task list across all grids
    all_tasks = []
    grid_specs = {}
    for grid_id in grid_ids:
        grid_id = normalize_grid_id(grid_id)
        spec = get_jhb_grid_spec(grid_id)
        grid_specs[grid_id] = spec
        tiles_dir = TILES_ROOT / grid_id
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for col in range(spec.n_cols):
            for row in range(spec.n_rows):
                out_path = tiles_dir / f"{grid_id}_{col}_{row}_geo.tif"
                all_tasks.append((spec, col, row, out_path, grid_id))

    # Reset progress
    with _lock:
        _progress["downloaded"] = 0
        _progress["skipped"] = 0
        _progress["errors"] = 0
        _progress["total"] = len(all_tasks)

    print(f"Total: {len(grid_ids)} grids, {len(all_tasks)} tiles, {workers} workers")

    if dry_run:
        for gid in grid_ids:
            spec = grid_specs[normalize_grid_id(gid)]
            total = spec.n_cols * spec.n_rows
            print(f"  {gid}: {total} tiles")
        return [{"grid_id": gid, "total": grid_specs[normalize_grid_id(gid)].n_cols * grid_specs[normalize_grid_id(gid)].n_rows,
                 "downloaded": 0, "skipped": 0, "errors": 0} for gid in grid_ids]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_tile_task, s, c, r, p, g): g
            for s, c, r, p, g in all_tasks
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions if any

    elapsed = time.time() - t0

    with _lock:
        dl = _progress["downloaded"]
        sk = _progress["skipped"]
        er = _progress["errors"]

    print(f"\n=== Done in {elapsed:.0f}s ({elapsed/60:.1f}min) ===")
    print(f"Downloaded: {dl}, Skipped: {sk}, Errors: {er}")
    rate = dl / elapsed if elapsed > 0 else 0
    print(f"Rate: {rate:.1f} tiles/s")

    return [{"grid_id": gid, "total": 0, "downloaded": dl, "skipped": sk, "errors": er}]


def main():
    parser = argparse.ArgumentParser(description="Download Joburg 2023 aerial tiles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--grid-id", help="Single grid ID (e.g. G0772)")
    group.add_argument("--batch", choices=sorted(BATCHES), help="Named batch of grids")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download threads (default: 4)")
    parser.add_argument("--dry", action="store_true", help="Dry run only")
    args = parser.parse_args()

    if args.grid_id:
        grid_ids = [args.grid_id]
    else:
        grid_ids = BATCHES[args.batch]

    print(f"=== Joburg 2023 Aerial Download ===")
    print(f"Grids: {len(grid_ids)}, Workers: {args.workers}")
    print(f"Tiles root: {TILES_ROOT}\n")

    if len(grid_ids) == 1:
        download_grid(grid_ids[0], workers=args.workers, dry_run=args.dry)
    else:
        download_batch(grid_ids, workers=args.workers, dry_run=args.dry)


if __name__ == "__main__":
    main()
