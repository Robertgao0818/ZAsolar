from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import geopandas as gpd

BASE_DIR = Path(__file__).parent.parent
TILES_ROOT = Path(os.environ.get("SOLAR_TILES_ROOT", BASE_DIR / "tiles"))
TASK_GRID_GPKG = BASE_DIR / "data" / "task_grid.gpkg"
JHB_TASK_GRID_GPKG = BASE_DIR / "data" / "jhb_task_grid.gpkg"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
COMBINED_ANNOTATION_GPKG = ANNOTATIONS_DIR / "solarpanel_g0001_g1190.gpkg"

DEFAULT_GRID_ID = "G1238"
DEFAULT_TILE_SIZE_DEG = 0.0016
DEFAULT_PIXEL_SIZE = 2000


@dataclass(frozen=True)
class GridPaths:
    grid_id: str
    tiles_dir: Path
    output_dir: Path
    gt_gpkg: Path
    gt_geojson: Path


@dataclass(frozen=True)
class GridSpec:
    grid_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    tile_size_deg: float
    pixel_size: int
    n_cols: int
    n_rows: int


def normalize_grid_id(grid_id: str) -> str:
    return str(grid_id).strip().upper()


CLEANED_DIR = ANNOTATIONS_DIR / "cleaned"


def _resolve_gt_gpkg(grid_id: str) -> Path:
    """Return the best available GT file for a grid.

    Search order: cleaned/*_SAM2_*.gpkg → annotations/{grid_id}.gpkg
    """
    # Auto-discover SAM2 files in cleaned/ directory
    if CLEANED_DIR.exists():
        matches = sorted(CLEANED_DIR.glob(f"{grid_id}_SAM2_*.gpkg"))
        if matches:
            return matches[-1]  # latest by filename
    # Legacy: direct annotation file
    legacy = ANNOTATIONS_DIR / f"{grid_id}.gpkg"
    if legacy.exists():
        return legacy
    return legacy  # return path even if missing (caller handles error)


def get_grid_paths(grid_id: str, output_subdir: str | None = None) -> GridPaths:
    grid_id = normalize_grid_id(grid_id)
    output_dir = BASE_DIR / "results" / grid_id
    if output_subdir:
        output_dir = output_dir / output_subdir
    return GridPaths(
        grid_id=grid_id,
        tiles_dir=TILES_ROOT / grid_id,
        output_dir=output_dir,
        gt_gpkg=_resolve_gt_gpkg(grid_id),
        gt_geojson=ANNOTATIONS_DIR / f"{grid_id.lower()}.geojson",
    )


def get_task_grid() -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    if TASK_GRID_GPKG.exists():
        frames.append(gpd.read_file(TASK_GRID_GPKG))
    if JHB_TASK_GRID_GPKG.exists():
        frames.append(gpd.read_file(JHB_TASK_GRID_GPKG))
    if not frames:
        raise FileNotFoundError(f"task grid not found: {TASK_GRID_GPKG}")
    if len(frames) == 1:
        return frames[0]
    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )


def get_grid_record(grid_id: str):
    grid_id = normalize_grid_id(grid_id)
    task_grid = get_task_grid()
    matches = task_grid.loc[task_grid["gridcell_id"].astype(str) == grid_id]
    if len(matches) == 0:
        raise KeyError(f"grid_id not found in task_grid.gpkg: {grid_id}")
    return matches.iloc[0]


def get_grid_spec(
    grid_id: str,
    tile_size_deg: float = DEFAULT_TILE_SIZE_DEG,
    pixel_size: int = DEFAULT_PIXEL_SIZE,
) -> GridSpec:
    record = get_grid_record(grid_id)
    xmin, ymin, xmax, ymax = record.geometry.bounds
    width = xmax - xmin
    height = ymax - ymin
    n_cols = math.ceil(width / tile_size_deg)
    n_rows = math.ceil(height / tile_size_deg)
    return GridSpec(
        grid_id=normalize_grid_id(grid_id),
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        tile_size_deg=tile_size_deg,
        pixel_size=pixel_size,
        n_cols=n_cols,
        n_rows=n_rows,
    )


def get_metric_crs(grid_id: str) -> str:
    """Return a suitable UTM CRS for the given grid based on its centroid."""
    record = get_grid_record(grid_id)
    centroid = record.geometry.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    zone = int((lon + 180) // 6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return f"EPSG:{epsg}"


def get_tile_bounds(spec: GridSpec, col: int, row: int) -> tuple[float, float, float, float]:
    txmin = spec.xmin + col * spec.tile_size_deg
    txmax = min(txmin + spec.tile_size_deg, spec.xmax)
    tymax = spec.ymax - row * spec.tile_size_deg
    tymin = max(tymax - spec.tile_size_deg, spec.ymin)
    return txmin, tymin, txmax, tymax
