"""Region registry — loads configs/datasets/regions.yaml and provides
region/grid lookup functions.

This module is the single programmatic interface to the region configuration.
``core.grid_utils`` delegates to it internally; callers that only need path
resolution should continue using grid_utils.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
REGIONS_YAML = BASE_DIR / "configs" / "datasets" / "regions.yaml"


@dataclass(frozen=True)
class RegionPaths:
    tiles_root: str
    results_root: str
    annotations_dir: str
    task_grid: str


@dataclass(frozen=True)
class ImageryLayerConfig:
    region_key: str
    layer_id: str
    path: str                  # project-relative, may be legacy during transition
    source: str                # aerial | geid | satellite | ...
    vintage: str
    file_layout: str           # chunked | mosaic
    file_pattern: str | None
    crs: str
    coverage_grids: tuple[str, ...]
    provenance: str | None = None
    # ISO YYYY-MM-DD mid-date of this imagery layer's capture window. Consumed by
    # the install-date back-dating subrepo (solar_backdating) as the upper bound
    # for status=done_installed_during_census. May be None for layers added before
    # the field was introduced.
    census_imagery_mid_date: str | None = None


@dataclass(frozen=True)
class ModelRunConfig:
    region_key: str
    run_id: str
    model_version: str
    imagery_layer: str
    results_path: str
    inference_date: str | None = None
    grid_count: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class RegionConfig:
    key: str
    description: str
    crs_metric: str
    crs_exchange: str
    paths: RegionPaths
    grid_id_pattern: str
    grids: dict[str, dict[str, Any]] = field(default_factory=dict)
    imagery_layers: dict[str, ImageryLayerConfig] = field(default_factory=dict)
    model_runs: dict[str, ModelRunConfig] = field(default_factory=dict)
    default_imagery_layer: str | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_registry() -> dict[str, RegionConfig]:
    """Parse regions.yaml into RegionConfig objects.  Cached after first call."""
    with open(REGIONS_YAML) as f:
        raw = yaml.safe_load(f)

    regions: dict[str, RegionConfig] = {}
    for key, data in raw.get("regions", {}).items():
        paths_data = data.get("paths", {})
        paths = RegionPaths(
            tiles_root=paths_data.get("tiles_root", "tiles"),
            results_root=paths_data.get("results_root", "results"),
            annotations_dir=paths_data.get("annotations_dir", f"data/annotations/{key}"),
            task_grid=paths_data.get("task_grid", f"data/{key}_task_grid.gpkg"),
        )

        imagery_layers: dict[str, ImageryLayerConfig] = {}
        for layer_id, layer_data in (data.get("imagery_layers") or {}).items():
            imagery_layers[layer_id] = ImageryLayerConfig(
                region_key=key,
                layer_id=layer_id,
                path=layer_data["path"],
                source=layer_data["source"],
                vintage=str(layer_data["vintage"]),
                file_layout=layer_data["file_layout"],
                file_pattern=layer_data.get("file_pattern"),
                crs=layer_data.get("crs", "EPSG:4326"),
                coverage_grids=tuple(layer_data.get("coverage_grids", [])),
                provenance=layer_data.get("provenance"),
                census_imagery_mid_date=layer_data.get("census_imagery_mid_date"),
            )

        model_runs: dict[str, ModelRunConfig] = {}
        for run_id, run_data in (data.get("model_runs") or {}).items():
            model_runs[run_id] = ModelRunConfig(
                region_key=key,
                run_id=run_id,
                model_version=run_data["model_version"],
                imagery_layer=run_data["imagery_layer"],
                results_path=run_data["results_path"],
                inference_date=run_data.get("inference_date"),
                grid_count=run_data.get("grid_count"),
                notes=run_data.get("notes"),
            )

        regions[key] = RegionConfig(
            key=key,
            description=data.get("description", ""),
            crs_metric=data.get("crs_metric", ""),
            crs_exchange=data.get("crs_exchange", "EPSG:4326"),
            paths=paths,
            grid_id_pattern=data.get("grid_id_pattern", ""),
            grids=data.get("grids", {}),
            imagery_layers=imagery_layers,
            model_runs=model_runs,
            default_imagery_layer=data.get("default_imagery_layer"),
        )
    return regions


def _get_registry() -> dict[str, RegionConfig]:
    return _load_registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_regions() -> list[str]:
    """Return all registered region keys."""
    return list(_get_registry().keys())


def get_region_config(region_key: str) -> RegionConfig:
    """Return config for a region key (e.g. 'cape_town', 'johannesburg').

    Raises KeyError if the region is not registered.
    """
    registry = _get_registry()
    if region_key not in registry:
        raise KeyError(
            f"Region '{region_key}' not found in {REGIONS_YAML}. "
            f"Available: {list(registry.keys())}"
        )
    return registry[region_key]


def lookup_region(grid_id: str) -> str | None:
    """Given a grid ID, find which region it belongs to.

    WARNING: unreliable when grid IDs overlap between regions (e.g. G1189
    exists in both cape_town and johannesburg task grids). Returns the
    first match. Prefer lookup_regions() for multi-region awareness.
    """
    hits = lookup_regions(grid_id)
    return hits[0] if hits else None


def lookup_regions(grid_id: str) -> list[str]:
    """Return ALL region keys whose imagery_layers cover this grid ID.

    Falls back to the legacy `grids` section if no imagery_layers declare
    coverage. An empty list means the grid is not registered anywhere.
    """
    grid_id = grid_id.strip().upper()
    hits: list[str] = []
    for key, config in _get_registry().items():
        covered = any(
            grid_id in layer.coverage_grids for layer in config.imagery_layers.values()
        )
        if covered or grid_id in config.grids or grid_id in _task_grid_ids(key):
            hits.append(key)
    return hits


def list_grids(region_key: str) -> list[str]:
    """Return all grid IDs registered under a region."""
    return list(get_region_config(region_key).grids.keys())


def get_annotation_source(grid_id: str, region_key: str | None = None) -> Path:
    """Return the annotation source path for a grid.

    If region_key is not given, looks up the region via lookup_region().
    Returns an absolute path resolved against the project root.
    """
    grid_id = grid_id.strip().upper()
    if region_key is None:
        region_key = lookup_region(grid_id)
    if region_key is None:
        raise KeyError(f"Grid '{grid_id}' not found in any region")

    config = get_region_config(region_key)
    grid_data = config.grids.get(grid_id, {})
    source = grid_data.get("annotation_source")
    if source:
        return BASE_DIR / source
    # Fallback: search annotations_dir
    ann_dir = BASE_DIR / config.paths.annotations_dir
    if ann_dir.exists():
        matches = sorted(ann_dir.glob(f"{grid_id}*.gpkg"))
        if matches:
            return matches[-1]
    return ann_dir / f"{grid_id}.gpkg"


def get_results_path(region_key: str) -> Path:
    """Return the absolute results root for a region."""
    config = get_region_config(region_key)
    return BASE_DIR / config.paths.results_root


def get_tiles_path(region_key: str) -> Path:
    """Return the project-relative tiles root for a region."""
    config = get_region_config(region_key)
    return BASE_DIR / config.paths.tiles_root


def get_task_grid_path(region_key: str) -> Path:
    """Return the task grid GPKG path for a region."""
    config = get_region_config(region_key)
    return BASE_DIR / config.paths.task_grid


@lru_cache(maxsize=None)
def _task_grid_ids(region_key: str) -> frozenset[str]:
    """Return grid IDs from a region task_grid, if the file exists.

    New Vexcel regions can have thousands of task cells, so their imagery layer
    coverage is represented by the task grid itself instead of a long YAML list.
    """
    try:
        path = get_task_grid_path(region_key)
    except KeyError:
        return frozenset()
    if not path.exists():
        return frozenset()

    try:
        import geopandas as gpd

        gdf = gpd.read_file(path, columns=["gridcell_id"])
    except Exception:
        return frozenset()

    if "gridcell_id" not in gdf.columns:
        return frozenset()
    return frozenset(str(item).strip().upper() for item in gdf["gridcell_id"].dropna())


def get_all_grid_id_patterns() -> list[str]:
    """Return compiled grid_id_pattern strings for all regions."""
    return [c.grid_id_pattern for c in _get_registry().values() if c.grid_id_pattern]


def matches_any_region_pattern(grid_id: str) -> bool:
    """Check if a grid ID matches any registered region's pattern."""
    grid_id = grid_id.strip().upper()
    for config in _get_registry().values():
        if config.grid_id_pattern and re.fullmatch(config.grid_id_pattern, grid_id):
            return True
    return False


# ---------------------------------------------------------------------------
# Normalization helpers (canonical region key from aliases)
# ---------------------------------------------------------------------------

_REGION_ALIASES: dict[str, str] = {
    "jhb": "johannesburg",
    "jnb": "johannesburg",
    "joburg": "johannesburg",
    "ct": "cape_town",
    "capetown": "cape_town",
    "cape": "cape_town",
}


def normalize_region_key(alias: str | None) -> str | None:
    """Convert a region alias to its canonical regions.yaml key.

    Returns None if input is None/empty.  Returns the input unchanged
    if it's already a valid key or not a recognized alias.
    """
    if not alias:
        return None
    value = alias.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _REGION_ALIASES.get(value, value)
    # Validate
    registry = _get_registry()
    if canonical in registry:
        return canonical
    return value  # return as-is; caller decides what to do


# ---------------------------------------------------------------------------
# Imagery layers (added 2026-04-19 for tiles/results restructure)
# ---------------------------------------------------------------------------

def list_imagery_layers(region_key: str) -> list[str]:
    """Return all imagery layer IDs registered under a region."""
    return list(get_region_config(region_key).imagery_layers.keys())


def get_imagery_layer(region_key: str, layer_id: str) -> ImageryLayerConfig:
    """Return the ImageryLayerConfig for a (region, layer_id) pair."""
    config = get_region_config(region_key)
    if layer_id not in config.imagery_layers:
        raise KeyError(
            f"Imagery layer '{layer_id}' not found in region '{region_key}'. "
            f"Available: {list(config.imagery_layers.keys())}"
        )
    return config.imagery_layers[layer_id]


def get_default_imagery_layer(region_key: str) -> str | None:
    """Return the default imagery_layer for a region (may be None)."""
    return get_region_config(region_key).default_imagery_layer


def get_imagery_layer_path(region_key: str, layer_id: str) -> Path:
    """Return the absolute filesystem path for an imagery layer."""
    layer = get_imagery_layer(region_key, layer_id)
    return BASE_DIR / layer.path


def resolve_imagery_layer_for_grid(
    grid_id: str,
    region_key: str,
    *,
    prefer: str | None = None,
) -> str:
    """Pick an imagery layer for (grid_id, region).

    Resolution order:
      1. `prefer` if passed and the grid is covered by it.
      2. region's default_imagery_layer if it covers the grid.
      3. first imagery layer whose coverage_grids contains grid_id.
    Raises KeyError if no layer covers the grid.
    """
    grid_id = grid_id.strip().upper()
    config = get_region_config(region_key)

    if prefer is not None:
        layer = get_imagery_layer(region_key, prefer)
        if grid_id in layer.coverage_grids:
            return prefer

    default_id = config.default_imagery_layer
    if default_id is not None and default_id in config.imagery_layers:
        default_layer = config.imagery_layers[default_id]
        if (
            grid_id in default_layer.coverage_grids
            or (not default_layer.coverage_grids and grid_id in _task_grid_ids(region_key))
        ):
            return default_id

    for layer_id, layer in config.imagery_layers.items():
        if grid_id in layer.coverage_grids or (
            not layer.coverage_grids and grid_id in _task_grid_ids(region_key)
        ):
            return layer_id

    raise KeyError(
        f"No imagery layer covers grid '{grid_id}' in region '{region_key}'. "
        f"Available layers: {list(config.imagery_layers.keys())}"
    )


# ---------------------------------------------------------------------------
# Model runs
# ---------------------------------------------------------------------------

def list_model_runs(region_key: str) -> list[str]:
    """Return all model_run IDs registered under a region."""
    return list(get_region_config(region_key).model_runs.keys())


def get_model_run(region_key: str, run_id: str) -> ModelRunConfig:
    """Return the ModelRunConfig for a (region, run_id) pair."""
    config = get_region_config(region_key)
    if run_id not in config.model_runs:
        raise KeyError(
            f"Model run '{run_id}' not found in region '{region_key}'. "
            f"Available: {list(config.model_runs.keys())}"
        )
    return config.model_runs[run_id]


def get_model_run_path(region_key: str, run_id: str) -> Path:
    """Return the absolute filesystem path where a model_run writes results."""
    run = get_model_run(region_key, run_id)
    return BASE_DIR / run.results_path
