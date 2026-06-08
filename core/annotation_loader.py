"""Unified annotation discovery and loading across all registered regions.

This module is the single entry point for answering "what annotations are
available and how to load them."  It normalizes schema differences between
Cape Town SAM2, Joburg V4-reviewed, and legacy annotation files.

Source-of-truth policy
~~~~~~~~~~~~~~~~~~~~~~
``configs/datasets/regions.yaml`` (via ``core.region_registry``) is the
**primary authority** for grid registration and annotation paths.

**Temporary compatibility fallback**: because only ~3 of ~100 Cape Town
grids are currently registered in ``regions.yaml``, discovery also scans
each region's ``annotations_dir`` for .gpkg files not yet in the registry.
Entries from the fallback are marked ``registered=False`` in their
``AnnotationEntry`` so downstream code (manifests, logs, validation) can
distinguish them.  This fallback is a transitional measure — the target
state is 100% registry coverage, at which point the fallback scan can be
removed.

Public API
----------
- ``discover_annotations(regions, exclude_grids)`` — find annotation files
  (registry-first, with temporary directory-scan fallback).
- ``load_annotation_gdf(entry)`` — load one annotation file into a
  normalized GeoDataFrame.
- ``resolve_gt_path(grid_id, region)`` — public GT path resolution
  (canonical replacement for private ``grid_utils._resolve_gt_gpkg``).

Usage::

    from core.annotation_loader import discover_annotations, load_annotation_gdf

    entries = discover_annotations(regions=["cape_town", "johannesburg"])
    for grid_id, entry in entries.items():
        gdf = load_annotation_gdf(entry)
        print(f"{grid_id}: {len(gdf)} polygons, registered={entry.registered}")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from core import region_registry


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnotationEntry:
    """Metadata for one grid's annotation source."""
    grid_id: str
    region_key: str          # canonical key from regions.yaml, e.g. "cape_town"
    path: Path               # absolute path to the .gpkg file
    schema_type: str         # "sam2" | "v4_reviewed" | "legacy_li" | "legacy_ct"
    annotation_count: int | None   # from regions.yaml if known
    annotation_layer: str | None = None  # named layer hint from regions.yaml
    registered: bool = True  # True if from regions.yaml, False if from fallback scan


# ---------------------------------------------------------------------------
# Schema classification (reporting only — does not affect loading logic)
# ---------------------------------------------------------------------------

def _classify_schema(path: Path, grid_id: str) -> str:
    """Classify annotation file schema by filename pattern.

    This is for provenance/reporting.  Loading logic handles all
    variants uniformly (read first layer, drop invalids, ensure 4326).
    """
    name = path.name
    if "_SAM2_" in name:
        return "sam2"
    if "_V4_" in name:
        return "v4_reviewed"
    if grid_id.startswith("JHB"):
        return "legacy_li"
    # Bare G*.gpkg files in Capetown/ are legacy CT annotations
    return "legacy_ct"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_annotations(
    regions: list[str] | None = None,
    exclude_grids: set[str] | None = None,
) -> dict[str, AnnotationEntry]:
    """Discover all available annotation files across regions.

    **Primary path**: iterate registered grids in ``regions.yaml`` and
    resolve their ``annotation_source``.

    **Fallback path**: scan each region's ``annotations_dir`` for .gpkg
    files not registered in ``regions.yaml``.  This handles the current
    state where only 3 of 94 Cape Town grids are registered.  Unregistered
    files are logged with a warning to encourage registration.

    Args:
        regions: List of region keys to scan (e.g. ``["cape_town"]``).
            If None, scans all registered regions.
        exclude_grids: Grid IDs to skip.

    Returns:
        Dict mapping grid_id → AnnotationEntry for grids with existing
        annotation files.
    """
    if regions is None:
        regions = region_registry.list_regions()

    exclude = exclude_grids or set()
    entries: dict[str, AnnotationEntry] = {}

    base_dir = Path(__file__).resolve().parent.parent

    for rkey in regions:
        try:
            config = region_registry.get_region_config(rkey)
        except KeyError:
            print(f"[WARN] Region '{rkey}' not found in registry, skipping")
            continue

        # --- Primary: registered grids ---
        for grid_id, grid_data in config.grids.items():
            if grid_id in exclude:
                continue

            try:
                ann_path = region_registry.get_annotation_source(grid_id, rkey)
            except KeyError:
                continue

            if not ann_path.exists():
                continue

            ann_layer = grid_data.get("annotation_layer") if isinstance(grid_data, dict) else None
            ann_count = grid_data.get("annotation_count") if isinstance(grid_data, dict) else None

            entries[grid_id] = AnnotationEntry(
                grid_id=grid_id,
                region_key=rkey,
                path=ann_path,
                schema_type=_classify_schema(ann_path, grid_id),
                annotation_count=ann_count,
                annotation_layer=ann_layer,
            )

        # --- Fallback: scan annotations dirs for unregistered files ---
        # Scan the region's primary annotations_dir plus every annotation_scheme
        # dir (e.g. Cape Town's li scheme -> Capetown_Li), so independently-
        # gridded schemes (L-prefix Li grids) are discovered, not just Gao.
        scan_dirs: list[Path] = [base_dir / config.paths.annotations_dir]
        for scheme in config.annotation_schemes.values():
            sdir = base_dir / scheme.annotations_dir
            if sdir not in scan_dirs:
                scan_dirs.append(sdir)

        for ann_dir in scan_dirs:
            if not ann_dir.exists():
                continue

            unregistered_count = 0
            for gpkg in sorted(ann_dir.glob("*.gpkg")):
                # Extract grid_id from filename
                grid_id = _extract_grid_id(gpkg.name, config.grid_id_pattern)
                if grid_id is None:
                    continue
                if grid_id in entries or grid_id in exclude:
                    continue

                entries[grid_id] = AnnotationEntry(
                    grid_id=grid_id,
                    region_key=rkey,
                    path=gpkg,
                    schema_type=_classify_schema(gpkg, grid_id),
                    annotation_count=None,
                    annotation_layer=None,
                    registered=False,
                )
                unregistered_count += 1

            if unregistered_count > 0:
                print(
                    f"[WARN] {rkey}: {unregistered_count} annotation files found "
                    f"in {ann_dir.name}/ but not registered in regions.yaml. "
                    f"Consider adding them for full provenance tracking."
                )

    return entries


def _extract_grid_id(filename: str, pattern: str) -> str | None:
    """Extract grid ID from an annotation filename using the region's pattern.

    Examples:
        "G1238_SAM2_260320.gpkg" with pattern "G\\d{4}" → "G1238"
        "G0922_V4_260407.gpkg"   with pattern "(JHB\\d{2}|G\\d{4})" → "G0922"
        "JHB01.gpkg"             with pattern "(JHB\\d{2}|G\\d{4})" → "JHB01"
    """
    import re
    stem = filename.replace(".gpkg", "")
    # Try to match the pattern at the start of the filename
    m = re.match(f"({pattern})", stem)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _list_layers(path: str) -> list[str]:
    """List layer names in a geopackage."""
    try:
        import pyogrio
        layers = pyogrio.list_layers(path)
        return [row[0] for row in layers]
    except ImportError:
        pass
    try:
        import fiona
        return fiona.listlayers(path)
    except ImportError:
        pass
    return []


def load_annotation_gdf(entry: AnnotationEntry) -> gpd.GeoDataFrame:
    """Load one annotation file into a normalized GeoDataFrame.

    - Reads the named layer (from ``annotation_layer``) or the first
      layer if not specified.
    - Drops invalid, empty, and null geometries.
    - Ensures CRS is EPSG:4326.
    - Returns a GeoDataFrame with at minimum a ``geometry`` column.
      Other columns are preserved but not required by downstream code.
    """
    path_str = str(entry.path)

    # Resolve layer name: try hint from regions.yaml, verify it exists,
    # fall back to first available layer.
    layer_name = None
    available_layers = _list_layers(path_str)

    if entry.annotation_layer and entry.annotation_layer in available_layers:
        layer_name = entry.annotation_layer
    elif available_layers:
        layer_name = available_layers[0]

    read_kwargs: dict = {"filename": path_str}
    if layer_name:
        read_kwargs["layer"] = layer_name

    gdf = gpd.read_file(**read_kwargs)

    # Drop invalid / empty / null geometries
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    gdf = gdf.reset_index(drop=True)

    # Ensure EPSG:4326
    if gdf.crs is None:
        # Heuristic: if coordinates look like UTM (x > 1000), assume
        # the region's metric CRS and reproject.
        if len(gdf) > 0:
            sample_x = gdf.iloc[0].geometry.bounds[0]
            if sample_x > 1000:
                try:
                    config = region_registry.get_region_config(entry.region_key)
                    gdf = gdf.set_crs(config.crs_metric)
                except KeyError:
                    gdf = gdf.set_crs(epsg=32734)  # fallback
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


# ---------------------------------------------------------------------------
# Public GT path resolution
# ---------------------------------------------------------------------------

def resolve_gt_path(grid_id: str, region: str | None = None) -> Path:
    """Return the best available GT annotation path for a grid.

    This is the canonical public API for GT path resolution.  It
    delegates to ``region_registry.get_annotation_source()`` with
    fallback to directory-based search in ``grid_utils``.

    Args:
        grid_id: The grid identifier (e.g. "G1238", "JHB01").
        region: Optional region key.  If None, auto-detected via
            ``region_registry.lookup_region()``.

    Returns:
        Path to the annotation .gpkg file (may not exist if no
        annotation is registered for this grid).
    """
    from core.grid_utils import _resolve_gt_gpkg, normalize_grid_id, normalize_region

    grid_id = normalize_grid_id(grid_id)
    region = normalize_region(region)
    return _resolve_gt_gpkg(grid_id, region=region)
