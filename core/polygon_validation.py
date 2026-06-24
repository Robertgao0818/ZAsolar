"""Polygon geometry-validity + area-cap filtering — the single canonical
definition of "valid polygon" for the V1.4 inventory pipeline.

Extracted from ``scripts/analysis/area_aggregate_eval.py`` on 2026-06-19
(architecture-deepening track, ADR-0001). Before this, the same
"load gpkg → drop invalid/non-finite geometry → reproject to metric CRS →
cap implausibly large polygons" pipeline was independently re-derived across
8+ analysis scripts, with five different names for the ``20 000 m²`` cap and
subtle behavioural drift between copies. A reviewer's first question about any
inventory number is *"exactly which polygons were included or excluded, and by
what rule?"* — this module is the one auditable, testable, citable answer.

Scope (load-bearing, see handoff 2026-06-19-polygon-validation-extraction.md):

  IN  — geometry *validity* filtering only: notna+valid, finite bounds,
        reproject to a caller-supplied metric CRS, area cap in metric m².
  OUT — *policy / score* filters (confidence min, SAM-score min, tunable area
        windows) and the Tier-1 statistic formulas (σ_Bw / RMSE / OLS / CI,
        which live in ``core.area_metrics``). Callers layer their own policy
        on top of canonical validity; this module owns validity only.

CRS policy (rule 06 + ``project_crs_policy`` memory): the metric CRS flows in
from ``core.grid_utils.get_metric_crs(grid_id, region=)``. This module takes
``metric_crs`` as a parameter — it never resolves or hardcodes an EPSG.

Byte-equivalence contract (ADR-0001 全程铁律): the two path-level readers
reproduce the original ``_sum_area_m2`` / ``_read_polys_geom`` exactly,
including their *different* summation kernels —

  * the 4-tuple path (``with_union=False``, was ``_sum_area_m2``) sums areas
    with pandas (``Series.sum()`` / ``.max()``, numpy pairwise reduction) and
    **keeps** zero-area polygons;
  * the 5-tuple path (``with_union=True``, was ``_read_polys_geom``) sums areas
    with Python (``sum(g.area for g in geoms)``), **drops** zero-area polygons,
    and also returns the ``unary_union`` geometry.

The zero-area split is exposed explicitly via the required ``drop_zero_area``
keyword so no caller can silently inherit the wrong behaviour.

``area_aggregate_eval`` re-exports ``_geometry_finite`` / ``_sum_area_m2`` /
``_read_polys_geom`` / ``_MAX_PLAUSIBLE_POLY_M2`` from here, so existing
``from scripts.analysis.area_aggregate_eval import _read_polys_geom`` (and the
bare ``from area_aggregate_eval import ...`` in ``poly_conf_sweep``) keep
resolving unchanged (D3: move + shim, never a second implementation).
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pyogrio
from shapely.ops import unary_union

# Any single solar installation polygon larger than this in metric area is
# almost certainly a corrupted geometry (broken coord, grid-tile outline, etc.)
# and will distort aggregate sums. Residential installs are <~200 m²,
# commercial <~5000 m². 20 000 m² is a generous upper bound.
MAX_PLAUSIBLE_POLY_M2 = 20_000.0


def geometry_finite(geom) -> bool:
    """Reject polygons with non-finite coordinates (NaN, inf, denormals).

    Canonical boundary: a coordinate is rejected only when ``abs(coord) > 1e18``
    (so exactly ``1e18`` is *kept*). A geometry whose ``.bounds`` raises is
    rejected. This matches ``area_aggregate_eval._geometry_finite`` exactly.
    """
    try:
        minx, miny, maxx, maxy = geom.bounds
    except Exception:
        return False
    for v in (minx, miny, maxx, maxy):
        if not math.isfinite(v) or abs(v) > 1e18:
            return False
    return True


def clean_metric_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    metric_crs: str,
    drop_zero_area: bool,
    max_area_m2: float = MAX_PLAUSIBLE_POLY_M2,
) -> tuple[gpd.GeoDataFrame, int]:
    """Canonical validity pipeline on an already-loaded GeoDataFrame.

    Order (must stay identical to the original area_aggregate_eval funcs):
      1. ``notna() & is_valid``
      2. ``geometry_finite`` (NaN/inf/|coord|>1e18)
      3. reproject **only if** ``crs is None or str(crs) != metric_crs``
      4. area cap ``area <= max_area_m2``
      5. if ``drop_zero_area``: additionally drop ``area == 0`` polygons

    Returns ``(cleaned_gdf, n_dropped)`` where ``n_dropped`` counts **only**
    the area-cap exceedances (step 4) — matching the original. Zero-area drops
    (step 5) are intentionally *not* counted in ``n_dropped``, preserving the
    ``_read_polys_geom`` semantics where ``n_kept + n_dropped`` need not equal
    the input length.
    """
    if gdf.empty:
        return gdf, 0
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.apply(geometry_finite)]
    if gdf.empty:
        return gdf, 0
    if gdf.crs is None or str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    areas = gdf.geometry.area
    keep_mask = areas <= max_area_m2
    n_dropped = int((~keep_mask).sum())
    if drop_zero_area:
        keep_mask = keep_mask & (areas > 0)
    gdf = gdf[keep_mask]
    return gdf, n_dropped


def _load_first_layer(gpkg_path, layer: str | None) -> gpd.GeoDataFrame:
    """Read ``gpkg_path``; use ``layer`` if present else fall back to the first
    layer. Mirrors the canonical layer-selection in area_aggregate_eval."""
    available = [row[0] for row in pyogrio.list_layers(gpkg_path)]
    chosen: str | None = layer if layer and layer in available else None
    if chosen is None and available:
        chosen = available[0]
    read_kwargs: dict[str, object] = {}
    if chosen:
        read_kwargs["layer"] = chosen
    return gpd.read_file(gpkg_path, **read_kwargs)


def read_polygons(
    gpkg_path,
    *,
    metric_crs: str,
    drop_zero_area: bool,
    with_union: bool = False,
    layer: str | None = None,
    max_area_m2: float = MAX_PLAUSIBLE_POLY_M2,
):
    """Load a gpkg and return its validity-filtered area summary.

    With ``with_union=False`` returns the 4-tuple
    ``(n_kept, total_area_m2, max_poly_m2, n_dropped)`` (was ``_sum_area_m2``);
    with ``with_union=True`` returns the 5-tuple
    ``(n_kept, sum_area_m2, max_poly_m2, n_dropped, union_geom_or_None)``
    (was ``_read_polys_geom``).

    The two paths deliberately use different summation kernels to stay
    byte-identical to the originals (see module docstring).
    """
    gdf = _load_first_layer(gpkg_path, layer)
    empty_ret = (0, 0.0, 0.0, 0, None) if with_union else (0, 0.0, 0.0, 0)
    if gdf.empty:
        return empty_ret
    cleaned, n_dropped = clean_metric_gdf(
        gdf, metric_crs=metric_crs, drop_zero_area=drop_zero_area,
        max_area_m2=max_area_m2,
    )
    if with_union:
        kept_geoms = list(cleaned.geometry)
        if not kept_geoms:
            return 0, 0.0, 0.0, n_dropped, None
        sum_area = float(sum(g.area for g in kept_geoms))
        max_area = float(max(g.area for g in kept_geoms))
        u = unary_union(kept_geoms)
        return len(kept_geoms), sum_area, max_area, n_dropped, u
    areas = cleaned.geometry.area
    if areas.empty:
        return 0, 0.0, 0.0, n_dropped
    return len(areas), float(areas.sum()), float(areas.max()), n_dropped


# --- Backward-compatibility aliases (D3 shim surface) ----------------------
# These preserve the exact names + positional signatures the existing callers
# import. Do not change their semantics; they are the byte-equivalence anchor.

_MAX_PLAUSIBLE_POLY_M2 = MAX_PLAUSIBLE_POLY_M2

_geometry_finite = geometry_finite


def _sum_area_m2(
    gpkg_path: Path, metric_crs: str, layer: str | None
) -> tuple[int, float, float, int]:
    """Backward-compat wrapper: keeps zero-area polygons, pandas summation.

    Returns ``(n_features_kept, total_area_m2, max_poly_m2, n_dropped)``.
    """
    return read_polygons(
        gpkg_path, metric_crs=metric_crs, layer=layer,
        drop_zero_area=False, with_union=False,
    )


def _read_polys_geom(gpkg_path: Path, metric_crs: str, layer: str | None):
    """Backward-compat wrapper: drops zero-area polygons, Python summation,
    returns the ``unary_union`` geometry for set-theoretic R/P/F1 / IoU.

    Returns ``(n_kept, sum_area_m2, max_poly_m2, n_dropped, union_geom_or_None)``.
    """
    return read_polygons(
        gpkg_path, metric_crs=metric_crs, layer=layer,
        drop_zero_area=True, with_union=True,
    )
