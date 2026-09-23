"""Enumerate the chips Mask R-CNN actually sees.

The window rule is ``SlidingWindowDataset`` with the production arguments
(chip 400, overlap 0.25, anchored origin). This module does not reimplement
``_iter_window_origins``.

Step 0 has no municipal imagery. Each tile is a geometry carrier: a
2000×2000 GeoTIFF whose affine is ``rasterio.transform.from_bounds`` on
``get_tile_bounds``, which is the same pair of calls ``download_tiles.py``
uses when it writes a real tile. Step 1 must rebuild this table from the
downloaded GeoTIFFs and check that it matches.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.geometry import box

from core.grid_utils import get_grid_spec, get_tile_bounds
from core.inference.tile_dataset import SlidingWindowDataset

from scripts.gate_pilot.common import CHIP_SIZE, GEOMETRY_CARRIER_DIR, OVERLAP, TILE_PIXELS

EXPECTED_ORIGINS_2000 = (0, 300, 600, 900, 1200, 1500, 1600)


def production_tile_names(grid_id: str) -> list[str]:
    """Filenames in the order ``detect_direct.py`` passes to the dataset.

    ``detect_direct`` globs ``{grid_id}_*_*_geo.tif`` and sorts the paths.
    For the Cape Town census cells that is the same order as column-major
    generation with single-digit indexes. The sort is what gets returned.
    """
    spec = get_grid_spec(grid_id, region="ct")
    names = [
        f"{grid_id}_{col}_{row}_geo.tif"
        for col in range(spec.n_cols)
        for row in range(spec.n_rows)
    ]
    return sorted(names)


def _write_zero_geotiff(path: Path, transform) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": TILE_PIXELS,
        "height": TILE_PIXELS,
        "count": 3,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((3, TILE_PIXELS, TILE_PIXELS), dtype=np.uint8))


def _bounds_match(actual, expected, tol: float = 1e-8) -> bool:
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(actual, expected))


def _carrier_is_current(path: Path, expected_bounds: tuple[float, float, float, float]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with rasterio.open(path) as src:
        return (
            src.width == TILE_PIXELS
            and src.height == TILE_PIXELS
            and src.crs is not None
            and src.crs.to_epsg() == 4326
            and _bounds_match(src.bounds, expected_bounds)
        )


def ensure_geometry_carriers(grid_id: str, carrier_root: Path | None = None) -> list[Path]:
    """Write one zero GeoTIFF per tile and return paths in production order."""
    spec = get_grid_spec(grid_id, region="ct")
    if spec.pixel_size != TILE_PIXELS:
        raise ValueError(f"{grid_id} pixel_size={spec.pixel_size}, expected {TILE_PIXELS}")
    root = (carrier_root or GEOMETRY_CARRIER_DIR) / grid_id
    root.mkdir(parents=True, exist_ok=True)
    names = production_tile_names(grid_id)
    template: Path | None = None
    for col in range(spec.n_cols):
        for row in range(spec.n_rows):
            name = f"{grid_id}_{col}_{row}_geo.tif"
            path = root / name
            west, south, east, north = get_tile_bounds(spec, col, row)
            expected = (west, south, east, north)
            transform = from_bounds(west, south, east, north, TILE_PIXELS, TILE_PIXELS)
            if _carrier_is_current(path, expected):
                template = template or path
                continue
            if template is None or not template.exists():
                _write_zero_geotiff(path, transform)
                template = path
            else:
                shutil.copyfile(template, path)
                with rasterio.open(path, "r+") as dst:
                    dst.transform = transform
            if not _carrier_is_current(path, expected):
                raise RuntimeError(f"{path} affine does not match get_tile_bounds")
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{grid_id} missing geometry carriers: {missing[:5]}")
    return [root / name for name in names]


def chip_windows_from_tifs(tif_paths: list[Path], grid_id: str) -> gpd.GeoDataFrame:
    """Call SlidingWindowDataset and return each chip's valid-pixel footprint.

    The footprint is the geographic rectangle of ``valid_window`` in the tile
    CRS (EPSG:4326). Zero-padding is not part of the rectangle. On a 2000 px
    tile every window is a full 400 px, so the valid rectangle is the whole chip.
    """
    if not tif_paths:
        raise ValueError("tif_paths is empty")
    dataset = SlidingWindowDataset(
        tif_paths,
        chip_size=CHIP_SIZE,
        overlap=OVERLAP,
        window_origin_mode="anchored",
    )
    try:
        if dataset.stride != 300:
            raise RuntimeError(f"expected stride 300, got {dataset.stride}")
        rows: list[dict] = []
        for index, (tif_index, window) in enumerate(dataset._items):
            meta = dataset._tif_meta[tif_index]
            transform = Affine(*meta["transform"])
            col_off = int(window.col_off)
            row_off = int(window.row_off)
            width = int(window.width)
            height = int(window.height)
            left, bottom, right, top = rasterio.windows.bounds(
                Window(col_off, row_off, width, height),
                transform,
            )
            rows.append(
                {
                    "grid_id": grid_id,
                    "chip_index": index,
                    "source_tile": Path(meta["path"]).name,
                    "col_off": col_off,
                    "row_off": row_off,
                    "valid_w": width,
                    "valid_h": height,
                    "minx": left,
                    "miny": bottom,
                    "maxx": right,
                    "maxy": top,
                    "geometry": box(left, bottom, right, top),
                }
            )
        spot = (0, len(dataset) - 1) if len(dataset) > 1 else (0,)
        for index in spot:
            _, chip_meta = dataset[index]
            row = rows[index]
            if chip_meta.chip_index != index:
                raise RuntimeError("chip_index drifted from SlidingWindowDataset")
            if chip_meta.valid_window != (row["col_off"], row["row_off"], row["valid_w"], row["valid_h"]):
                raise RuntimeError(
                    f"valid_window mismatch at {index}: "
                    f"{chip_meta.valid_window} vs row"
                )
    finally:
        dataset.close()

    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    frame = frame.sort_values("chip_index").reset_index(drop=True)
    return frame


def build_grid_chip_windows(grid_id: str, carrier_root: Path | None = None) -> gpd.GeoDataFrame:
    paths = ensure_geometry_carriers(grid_id, carrier_root=carrier_root)
    frame = chip_windows_from_tifs(paths, grid_id)
    _assert_census_tile_expectations(grid_id, frame)
    return frame


def _assert_census_tile_expectations(grid_id: str, frame: pd.DataFrame) -> None:
    spec = get_grid_spec(grid_id, region="ct")
    n_tiles = spec.n_cols * spec.n_rows
    per_tile = len(EXPECTED_ORIGINS_2000) ** 2
    expected = n_tiles * per_tile
    if len(frame) != expected:
        raise RuntimeError(f"{grid_id}: {len(frame)} chips, expected {expected}")
    if not (frame["valid_w"].eq(CHIP_SIZE) & frame["valid_h"].eq(CHIP_SIZE)).all():
        raise RuntimeError(f"{grid_id}: a 2000 px tile produced a partial chip")
    first_tile = frame["source_tile"].iloc[0]
    origins = set(
        zip(
            frame.loc[frame["source_tile"] == first_tile, "col_off"],
            frame.loc[frame["source_tile"] == first_tile, "row_off"],
        )
    )
    expected_origins = {
        (col, row) for row in EXPECTED_ORIGINS_2000 for col in EXPECTED_ORIGINS_2000
    }
    if origins != expected_origins:
        raise RuntimeError(f"{grid_id}: first-tile origins {sorted(origins)[:8]} != anchored 2000 px rule")
