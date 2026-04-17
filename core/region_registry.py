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
class RegionConfig:
    key: str
    description: str
    crs_metric: str
    crs_exchange: str
    paths: RegionPaths
    grid_id_pattern: str
    grids: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        regions[key] = RegionConfig(
            key=key,
            description=data.get("description", ""),
            crs_metric=data.get("crs_metric", ""),
            crs_exchange=data.get("crs_exchange", "EPSG:4326"),
            paths=paths,
            grid_id_pattern=data.get("grid_id_pattern", ""),
            grids=data.get("grids", {}),
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

    Returns the region key (e.g. 'cape_town') or None if not found.
    Searches all registered regions' grid lists.
    """
    grid_id = grid_id.strip().upper()
    for key, config in _get_registry().items():
        if grid_id in config.grids:
            return key
    return None


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
