from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import geopandas as gpd

from core import region_registry

BASE_DIR = Path(__file__).parent.parent
TILES_ROOT = Path(os.environ.get("SOLAR_TILES_ROOT", BASE_DIR / "tiles"))
RESULTS_ROOT = BASE_DIR / "results"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
CAPETOWN_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "Capetown"
JOBURG_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "Joburg"
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


def normalize_region(region: str | None) -> str | None:
    """Normalize region alias to short form (ct/jhb).

    For the canonical regions.yaml key, use
    ``region_registry.normalize_region_key()`` instead.
    """
    if region is None:
        return None
    value = str(region).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"jhb", "joburg", "johannesburg"}:
        return "jhb"
    if value in {"ct", "cape_town", "capetown", "cape"}:
        return "ct"
    return value or None


# Short alias → canonical regions.yaml key
_ALIAS_TO_KEY = {"jhb": "johannesburg", "ct": "cape_town"}


def _region_key(region: str | None) -> str | None:
    """Convert a short alias (ct/jhb) to a canonical regions.yaml key."""
    if region is None:
        return None
    return _ALIAS_TO_KEY.get(region, region)


CLEANED_DIR = ANNOTATIONS_DIR / "cleaned"


def _find_latest_gpkg(directory: Path, patterns: list[str]) -> Path | None:
    if not directory.exists():
        return None
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _resolve_gt_gpkg(grid_id: str, *, region: str | None = None) -> Path:
    """Return the best available GT file for a grid.

    Tries region_registry first (annotation_source from regions.yaml),
    then falls back to directory-based search.
    """
    grid_id = normalize_grid_id(grid_id)
    region = normalize_region(region)

    # --- Try registry lookup first ---
    rkey = _region_key(region)
    if rkey is None:
        rkey = region_registry.lookup_region(grid_id)
    if rkey:
        try:
            source = region_registry.get_annotation_source(grid_id, rkey)
            if source.exists():
                return source
        except KeyError:
            pass
        # Scheme-aware auto-discovery: route grid_id (e.g. L-prefix Li grids)
        # to the annotations dir of the scheme that owns it before falling back
        # to the region's default Capetown/Joburg dirs below.
        try:
            scheme_dir = region_registry.get_annotations_dir_for_grid(rkey, grid_id)
        except KeyError:
            scheme_dir = None
        if scheme_dir is not None:
            match = _find_latest_gpkg(
                scheme_dir,
                [f"{grid_id}_SAM2_*.gpkg", f"{grid_id}.gpkg", f"{grid_id}_*.gpkg"],
            )
            if match is not None:
                return match

    # --- Fallback: directory-based search ---
    search_roots: list[tuple[Path, list[str]]] = []
    if region == "jhb":
        search_roots.append((JOBURG_ANNOTATIONS_DIR, [f"{grid_id}_*.gpkg", f"{grid_id}.gpkg"]))
    elif region == "ct":
        search_roots.append((CAPETOWN_ANNOTATIONS_DIR, [f"{grid_id}_SAM2_*.gpkg", f"{grid_id}.gpkg", f"{grid_id}_*.gpkg"]))
    else:
        search_roots.extend([
            (CAPETOWN_ANNOTATIONS_DIR, [f"{grid_id}_SAM2_*.gpkg", f"{grid_id}.gpkg", f"{grid_id}_*.gpkg"]),
            (JOBURG_ANNOTATIONS_DIR, [f"{grid_id}_*.gpkg", f"{grid_id}.gpkg"]),
        ])

    for directory, patterns in search_roots:
        match = _find_latest_gpkg(directory, patterns)
        if match is not None:
            return match

    # Auto-discover SAM2 files in cleaned/ directory
    if CLEANED_DIR.exists():
        matches = sorted(CLEANED_DIR.glob(f"{grid_id}_SAM2_*.gpkg"))
        if matches:
            return matches[-1]  # latest by filename
    # Legacy: direct annotation file
    legacy = ANNOTATIONS_DIR / f"{grid_id}.gpkg"
    return legacy  # return path even if missing (caller handles error)


def _preferred_tiles_root(region: str | None) -> Path:
    region = normalize_region(region)
    env_root = os.environ.get("SOLAR_TILES_ROOT")
    rkey = _region_key(region)
    if rkey:
        try:
            registry_path = region_registry.get_tiles_path(rkey)
            if registry_path.exists():
                return registry_path
        except KeyError:
            pass
    return Path(env_root) if env_root else TILES_ROOT


def get_results_root(
    region: str | None = None,
    *,
    model_run: str | None = None,
) -> Path:
    """Return the results root for (region, model_run).

    When ``model_run`` is given, routes to the registered results_path for
    that run. Otherwise returns the region's legacy results_root.
    """
    region = normalize_region(region)
    rkey = _region_key(region)

    if model_run is not None and rkey is not None:
        try:
            return region_registry.get_model_run_path(rkey, model_run)
        except KeyError:
            pass

    if rkey:
        try:
            return region_registry.get_results_path(rkey)
        except KeyError:
            pass
    return RESULTS_ROOT


def resolve_tiles_dir(
    grid_id: str,
    *,
    region: str | None = None,
    imagery_layer: str | None = None,
) -> Path:
    """Return the source tile Path for (grid_id, region, imagery_layer).

    Behavior by layer file_layout:
      - ``chunked``: returns directory `<layer>/<grid_id>/` (caller globs chips)
      - ``mosaic``:  returns the single file `<layer>/{grid_id}_mosaic.tif`

    Resolution order for ``imagery_layer``:
      1. Explicit ``imagery_layer=`` argument.
      2. Region's default_imagery_layer.
      3. First registered layer whose coverage_grids contains the grid.

    ``SOLAR_TILES_ROOT`` env var overrides **only** when the resulting
    ``<env>/<grid_id>/`` directory actually exists (for RunPod /dev/shm).

    Tile paths are keyed by the **source** grid ID
    (``region_registry.resolve_source_grid_id``): the CPT regrid (ADR-0002 §5)
    renamed Cape Town census cells G#### -> CPT#### digit-preserving but left
    every on-disk tile under its source Gao ID, so a logical ``CPT1240`` must
    resolve to ``aerial_2025/G1240/`` (not a non-existent ``tiles_root/CPT1240``).
    Non-renamed IDs map to themselves.
    """
    grid_id = normalize_grid_id(grid_id)
    region = normalize_region(region)
    rkey = _region_key(region)

    if rkey is None:
        rkey = region_registry.lookup_region(grid_id)

    # On-disk tiles stay keyed under the source grid ID (CPT1240 -> G1240).
    source_id = grid_id
    if rkey is not None:
        source_id = region_registry.resolve_source_grid_id(grid_id, rkey)

    # RunPod /dev/shm fast path: only if the legacy layout exists there.
    env_root = os.environ.get("SOLAR_TILES_ROOT")
    if env_root:
        env_path = Path(env_root)
        candidate = env_path / source_id
        if candidate.exists():
            return candidate
        mosaic_candidate = env_path / f"{source_id}_mosaic.tif"
        if mosaic_candidate.exists():
            return mosaic_candidate

    if rkey is not None:
        try:
            layer_id = imagery_layer or region_registry.resolve_imagery_layer_for_grid(
                grid_id, rkey
            )
            layer = region_registry.get_imagery_layer(rkey, layer_id)
            layer_path = region_registry.get_imagery_layer_path(rkey, layer_id)
            if layer.file_layout == "mosaic":
                return layer_path / f"{source_id}_mosaic.tif"
            return layer_path / source_id
        except KeyError:
            pass

    # Fallback: legacy behavior (may return non-existent path; caller errors)
    return _preferred_tiles_root(region) / source_id


def get_grid_paths(
    grid_id: str,
    output_subdir: str | None = None,
    *,
    region: str | None = None,
    imagery_layer: str | None = None,
    model_run: str | None = None,
) -> GridPaths:
    grid_id = normalize_grid_id(grid_id)
    region = normalize_region(region)
    output_dir = get_results_root(region, model_run=model_run) / grid_id
    if output_subdir:
        output_dir = output_dir / output_subdir
    return GridPaths(
        grid_id=grid_id,
        tiles_dir=resolve_tiles_dir(grid_id, region=region, imagery_layer=imagery_layer),
        output_dir=output_dir,
        gt_gpkg=_resolve_gt_gpkg(grid_id, region=region),
        gt_geojson=ANNOTATIONS_DIR / f"{grid_id.lower()}.geojson",
    )


def get_task_grid() -> gpd.GeoDataFrame:
    """Load and concatenate task grids from all registered regions."""
    frames: list[gpd.GeoDataFrame] = []
    for rkey in region_registry.list_regions():
        try:
            tg_path = region_registry.get_task_grid_path(rkey)
            if tg_path.exists():
                frames.append(gpd.read_file(tg_path))
        except KeyError:
            continue

    # Fallback to hardcoded paths if registry found nothing
    if not frames:
        ct_tg = BASE_DIR / "data" / "task_grid.gpkg"
        jhb_tg = BASE_DIR / "data" / "jhb_task_grid.gpkg"
        if ct_tg.exists():
            frames.append(gpd.read_file(ct_tg))
        if jhb_tg.exists():
            frames.append(gpd.read_file(jhb_tg))

    if not frames:
        raise FileNotFoundError("No task grid files found")
    if len(frames) == 1:
        return frames[0]
    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )


def get_grid_record(grid_id: str, *, region: str | None = None):
    """Lookup grid record. Use region='jhb' to prefer Johannesburg task grid
    when grid IDs overlap with Cape Town."""
    grid_id = normalize_grid_id(grid_id)
    region = normalize_region(region)

    # Try region-specific task grid first.
    rkey = _region_key(region)
    # ADR-0002 / CPT regrid: when no region is passed, infer one so the
    # region-specific task grid AND its annotation-scheme fallback below get
    # consulted. This is what keeps a RETIRED bare G-ID resolving to its source
    # cell geometry: CT's primary task_grid is now the CPT census grid (no
    # G-cells), but the gao scheme's data/task_grid.gpkg still carries them.
    # It also fixes the overlap trap — lookup_region('G1189') returns cape_town
    # (regions.yaml order; ADR-0002), so a bare overlapping G-ID resolves to the
    # CAPE TOWN cell (lon<18.7), never JHB's same-named-but-different cell. JHB
    # legacy flows must still pass region='jhb' explicitly (rule 06-multi-city).
    if rkey is None:
        rkey = region_registry.lookup_region(grid_id)
    if rkey:
        try:
            tg_path = region_registry.get_task_grid_path(rkey)
            if tg_path.exists():
                tg = gpd.read_file(tg_path)
                matches = tg.loc[tg["gridcell_id"].astype(str) == grid_id]
                if len(matches) > 0:
                    return matches.iloc[0]
        except KeyError:
            pass

        # Fall back to the matching annotation-scheme task grid. L-prefix Li
        # grids live in the `li` scheme's data/task_grid_li.gpkg, and RETIRED
        # G-cells live in the `gao` scheme's data/task_grid.gpkg — neither is the
        # region's primary (CPT census) task_grid, so get_metric_crs/get_grid_spec
        # must consult the scheme grid before the aggregate fallback.
        try:
            scheme = region_registry.resolve_annotation_scheme(rkey, grid_id)
            if scheme is not None and scheme.task_grid:
                stg_path = BASE_DIR / scheme.task_grid
                if stg_path.exists():
                    stg = gpd.read_file(stg_path)
                    matches = stg.loc[stg["gridcell_id"].astype(str) == grid_id]
                    if len(matches) > 0:
                        return matches.iloc[0]
        except (KeyError, AttributeError):
            pass

    task_grid = get_task_grid()
    matches = task_grid.loc[task_grid["gridcell_id"].astype(str) == grid_id]
    if len(matches) == 0:
        raise KeyError(f"grid_id not found in task_grid: {grid_id}")
    if region == "jhb" and len(matches) > 1:
        return matches.iloc[-1]
    return matches.iloc[0]


def get_grid_spec(
    grid_id: str,
    tile_size_deg: float = DEFAULT_TILE_SIZE_DEG,
    pixel_size: int = DEFAULT_PIXEL_SIZE,
    region: str | None = None,
) -> GridSpec:
    region = normalize_region(region)
    record = get_grid_record(grid_id, region=region)
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


def get_metric_crs(grid_id: str, *, region: str | None = None) -> str:
    """Return a suitable UTM CRS for the given grid based on its centroid."""
    region = normalize_region(region)
    record = get_grid_record(grid_id, region=region)
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
