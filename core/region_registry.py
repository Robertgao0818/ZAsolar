"""Region registry — loads configs/datasets/regions.yaml and provides
region/grid lookup functions.

This module is the single programmatic interface to the region configuration.
``core.grid_utils`` delegates to it internally; callers that only need path
resolution should continue using grid_utils.
"""

from __future__ import annotations

import re
import warnings
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
    # Real capture-window bounds (ISO YYYY-MM-DD) when known from the provider's
    # collection metadata (e.g. Vexcel /ortho/dates). census_imagery_mid_date is a
    # single-value fallback; for per-grid install-date present-side clamping prefer
    # the real per-grid flight date. capture_date_range_end is the safe global
    # upper bound (no detection can be present-dated later than the last flight).
    capture_date_range_start: str | None = None
    capture_date_range_end: str | None = None
    # Comma-separated distinct flight dates observed across the collection footprint.
    capture_flight_dates: str | None = None


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
    deprecated: bool = False


@dataclass(frozen=True)
class AnnotationSchemeConfig:
    """One annotation scheme within a region.

    A region can host more than one independent grid scheme that share the
    same imagery (e.g. Cape Town's primary ``gao`` scheme and ``li``, Li Yang's
    independently-gridded SAM2 GT on the eastern Cape Flats). Schemes do NOT
    share a grid namespace — Li's G1895 != Gao's G1895 — so each scheme can
    carry its own ``annotations_dir``, ``task_grid``, ``task_kml`` and an
    optional scheme-level ``grid_id_pattern`` used to route a grid_id to the
    right annotations directory.
    """
    scheme_id: str
    region_key: str
    annotations_dir: str
    grid_id_pattern: str | None = None
    task_grid: str | None = None
    task_kml: str | None = None
    grid_count: int | None = None
    grid_id_range: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class RegionConfig:
    key: str
    description: str
    crs_metric: str
    crs_exchange: str
    paths: RegionPaths
    grid_id_pattern: str
    # Retired grid namespaces (regex, fullmatch): IDs the region can still
    # resolve for historical artifacts but no longer claims in default
    # ambiguous lookup. lookup_regions() returns active-namespace hits first
    # and falls back to retired ones only when no active hit exists (ADR-0002).
    retired_grid_id_patterns: tuple[str, ...] = ()
    grids: dict[str, dict[str, Any]] = field(default_factory=dict)
    imagery_layers: dict[str, ImageryLayerConfig] = field(default_factory=dict)
    model_runs: dict[str, ModelRunConfig] = field(default_factory=dict)
    annotation_schemes: dict[str, AnnotationSchemeConfig] = field(default_factory=dict)
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
                capture_date_range_start=layer_data.get("capture_date_range_start"),
                capture_date_range_end=layer_data.get("capture_date_range_end"),
                capture_flight_dates=layer_data.get("capture_flight_dates"),
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
                deprecated=bool(run_data.get("deprecated", False)),
            )

        annotation_schemes: dict[str, AnnotationSchemeConfig] = {}
        for scheme_id, scheme_data in (data.get("annotation_schemes") or {}).items():
            annotation_schemes[scheme_id] = AnnotationSchemeConfig(
                scheme_id=scheme_id,
                region_key=key,
                annotations_dir=scheme_data.get(
                    "annotations_dir", paths.annotations_dir
                ),
                grid_id_pattern=scheme_data.get("grid_id_pattern"),
                task_grid=scheme_data.get("task_grid"),
                task_kml=scheme_data.get("task_kml"),
                grid_count=scheme_data.get("grid_count"),
                grid_id_range=scheme_data.get("grid_id_range"),
                notes=scheme_data.get("notes"),
            )

        regions[key] = RegionConfig(
            key=key,
            description=data.get("description", ""),
            crs_metric=data.get("crs_metric", ""),
            crs_exchange=data.get("crs_exchange", "EPSG:4326"),
            paths=paths,
            grid_id_pattern=data.get("grid_id_pattern", ""),
            retired_grid_id_patterns=tuple(data.get("retired_grid_id_patterns", [])),
            grids=data.get("grids", {}),
            imagery_layers=imagery_layers,
            model_runs=model_runs,
            annotation_schemes=annotation_schemes,
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

    Resolution is namespace-aware (ADR-0002): regions whose
    ``retired_grid_id_patterns`` match the ID only count when no
    active-namespace region claims it. If multiple active regions still
    claim the same ID (a registry invariant violation), returns the first
    match — prefer lookup_regions() and disambiguate explicitly.
    """
    hits = lookup_regions(grid_id)
    return hits[0] if hits else None


@lru_cache(maxsize=None)
def _retired_patterns(region_key: str) -> tuple[re.Pattern[str], ...]:
    config = _get_registry().get(region_key)
    if config is None:
        return ()
    return tuple(re.compile(p) for p in config.retired_grid_id_patterns)


def _is_retired_grid(region_key: str, grid_id: str) -> bool:
    """True if grid_id belongs to one of the region's retired namespaces."""
    return any(p.fullmatch(grid_id) for p in _retired_patterns(region_key))


def lookup_regions(grid_id: str, include_retired: bool = False) -> list[str]:
    """Return region keys that claim this grid ID, active namespaces first.

    A region claims an ID via imagery_layer coverage, the legacy ``grids``
    section, or its task grids. Hits are split into two tiers by the
    region's ``retired_grid_id_patterns`` (ADR-0002):

    - default: return active-namespace hits; fall back to retired-namespace
      hits only when no active region claims the ID (keeps historical IDs
      like JHB's G0816 resolvable while killing CT/JHB G-overlap ambiguity).
    - ``include_retired=True``: return both tiers (active first) — the
      pre-ADR-0002 behavior, for tools that need every possible owner.

    An empty list means the grid is not registered anywhere. Multiple
    active hits violate the active-namespace disjointness invariant and
    emit a UserWarning.
    """
    grid_id = grid_id.strip().upper()
    active: list[str] = []
    retired: list[str] = []
    for key, config in _get_registry().items():
        covered = any(
            grid_id in layer.coverage_grids for layer in config.imagery_layers.values()
        )
        if (
            covered
            or grid_id in config.grids
            or grid_id in _task_grid_ids(key)
            or grid_id in _scheme_task_grid_ids(key)
        ):
            (retired if _is_retired_grid(key, grid_id) else active).append(key)
    if len(active) > 1:
        warnings.warn(
            f"Grid '{grid_id}' is claimed by multiple ACTIVE namespaces {active}; "
            f"active grid namespaces must be disjoint (ADR-0002) — fix "
            f"regions.yaml (retire one side) or pass region explicitly.",
            stacklevel=2,
        )
    if include_retired:
        return active + retired
    return active if active else retired


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


@lru_cache(maxsize=None)
def _scheme_task_grid_ids(region_key: str) -> frozenset[str]:
    """Return grid IDs from all annotation_scheme task_grids in a region.

    Lets independently-gridded schemes (e.g. Cape Town's ``li`` L-prefix GT,
    whose cells live in ``data/task_grid_li.gpkg`` rather than the region's
    primary task_grid) be discovered by ``lookup_regions``.
    """
    try:
        config = get_region_config(region_key)
    except KeyError:
        return frozenset()
    ids: set[str] = set()
    for scheme in config.annotation_schemes.values():
        if not scheme.task_grid:
            continue
        path = BASE_DIR / scheme.task_grid
        if not path.exists():
            continue
        try:
            import geopandas as gpd

            gdf = gpd.read_file(path, columns=["gridcell_id"])
        except Exception:
            continue
        if "gridcell_id" not in gdf.columns:
            continue
        ids.update(str(item).strip().upper() for item in gdf["gridcell_id"].dropna())
    return frozenset(ids)


@lru_cache(maxsize=None)
def _legacy_source_grid_map(region_key: str) -> dict[str, str]:
    """Map a region's *logical* grid IDs to the *on-disk source* grid IDs.

    The CPT regrid (ADR-0002 §5, 2026-06-12) renamed Cape Town's census cells
    G#### -> CPT#### **digit-preserving** but did NOT move any on-disk artifact:
    tiles, results and annotation gpkgs all stay keyed under the source G-ID
    (``aerial_2025/G1240/``). The region's primary task grid records that back-
    reference in a ``legacy_gao_id`` column (CPT1240 -> G1240). Imagery/coverage
    resolution must follow that column so a logical CPT id reaches its real
    on-disk tiles instead of a non-existent ``tiles_root/CPT1240``.

    Returns ``{logical_id: source_id}`` (both upper-cased) for every cell whose
    ``legacy_gao_id`` differs from its own ``gridcell_id``. Empty for regions
    whose primary task grid lacks a ``legacy_gao_id`` column (no rename in play).
    """
    try:
        path = get_task_grid_path(region_key)
    except KeyError:
        return {}
    if not path.exists():
        return {}
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path, columns=["gridcell_id", "legacy_gao_id"])
    except Exception:
        return {}
    if "gridcell_id" not in gdf.columns or "legacy_gao_id" not in gdf.columns:
        return {}
    mapping: dict[str, str] = {}
    for _, row in gdf[["gridcell_id", "legacy_gao_id"]].dropna().iterrows():
        logical = str(row["gridcell_id"]).strip().upper()
        source = str(row["legacy_gao_id"]).strip().upper()
        if logical and source and logical != source:
            mapping[logical] = source
    return mapping


def resolve_source_grid_id(grid_id: str, region_key: str) -> str:
    """Map a logical grid ID to the grid ID its imagery/tiles are keyed under.

    For the Cape Town CPT census scheme this returns the digit-preserving source
    Gao ID (CPT1240 -> G1240) recorded in the primary task grid's
    ``legacy_gao_id`` column; on-disk tiles, imagery coverage_grids and results
    all live under that source ID. For every other ID (already-source G-IDs, Li
    L-IDs, JHB grids, Vexcel CPT-native cells, …) there is no rename, so the
    input is returned unchanged.
    """
    grid_id = grid_id.strip().upper()
    return _legacy_source_grid_map(region_key).get(grid_id, grid_id)


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

    Coverage is matched against the **source** grid ID
    (``resolve_source_grid_id``) so a logical CPT census ID resolves through its
    digit-preserving Gao source (CPT1240 -> G1240), which is what the imagery
    ``coverage_grids`` and on-disk tiles are keyed under (ADR-0002 §5). When the
    grid is not present in any coverage list, an imagery layer whose
    ``coverage_grids`` is empty is still claimed if the source ID is in the
    region's primary task grid (Vexcel-style task-grid-as-coverage regions).
    """
    grid_id = grid_id.strip().upper()
    config = get_region_config(region_key)
    source_id = resolve_source_grid_id(grid_id, region_key)

    if prefer is not None:
        layer = get_imagery_layer(region_key, prefer)
        if source_id in layer.coverage_grids:
            return prefer

    default_id = config.default_imagery_layer
    if default_id is not None and default_id in config.imagery_layers:
        default_layer = config.imagery_layers[default_id]
        if (
            source_id in default_layer.coverage_grids
            or (not default_layer.coverage_grids and source_id in _task_grid_ids(region_key))
        ):
            return default_id

    for layer_id, layer in config.imagery_layers.items():
        if source_id in layer.coverage_grids or (
            not layer.coverage_grids and source_id in _task_grid_ids(region_key)
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


# ---------------------------------------------------------------------------
# Annotation schemes (added 2026-06-04 for Li CT GT independent grid scheme)
# ---------------------------------------------------------------------------

def list_annotation_schemes(region_key: str) -> list[str]:
    """Return all annotation scheme IDs registered under a region."""
    return list(get_region_config(region_key).annotation_schemes.keys())


def get_annotation_scheme(region_key: str, scheme_id: str) -> AnnotationSchemeConfig:
    """Return the AnnotationSchemeConfig for a (region, scheme_id) pair."""
    config = get_region_config(region_key)
    if scheme_id not in config.annotation_schemes:
        raise KeyError(
            f"Annotation scheme '{scheme_id}' not found in region '{region_key}'. "
            f"Available: {list(config.annotation_schemes.keys())}"
        )
    return config.annotation_schemes[scheme_id]


def resolve_annotation_scheme(region_key: str, grid_id: str) -> AnnotationSchemeConfig | None:
    """Pick the annotation scheme that owns a grid_id.

    Resolution: a scheme with its own ``grid_id_pattern`` claims a grid_id
    whose normalized form fully matches that pattern. If exactly one scheme
    declares a pattern and matches, return it. If no scheme declares a pattern
    (or none match), fall back to the scheme whose ``annotations_dir`` equals
    the region default (the "primary" scheme), else None.

    This lets independently-gridded schemes (e.g. Cape Town's ``li`` L-prefix
    GT) route grid IDs to their own ``annotations_dir`` without colliding with
    the primary scheme.
    """
    grid_id = grid_id.strip().upper()
    config = get_region_config(region_key)
    schemes = config.annotation_schemes
    if not schemes:
        return None

    # 1. Schemes that declare a grid_id_pattern and match this grid.
    for scheme in schemes.values():
        pat = scheme.grid_id_pattern
        if pat and re.fullmatch(pat, grid_id):
            return scheme

    # 2. Fall back to the scheme pointing at the region's default annotations_dir.
    default_dir = config.paths.annotations_dir
    for scheme in schemes.values():
        if scheme.annotations_dir == default_dir:
            return scheme
    return None


def get_annotations_dir_for_grid(region_key: str, grid_id: str) -> Path | None:
    """Return the absolute annotations directory that should hold a grid's GT.

    Routes via ``resolve_annotation_scheme``; falls back to the region's
    default ``annotations_dir`` when no scheme matches. Used by GT auto-
    discovery so L-prefix Li grids resolve to ``data/annotations/Capetown_Li``
    instead of the primary ``data/annotations/Capetown``.
    """
    scheme = resolve_annotation_scheme(region_key, grid_id)
    if scheme is not None:
        return BASE_DIR / scheme.annotations_dir
    try:
        config = get_region_config(region_key)
    except KeyError:
        return None
    return BASE_DIR / config.paths.annotations_dir
