"""Shared post-processing primitives for the direct Mask R-CNN pipeline.

Pure functions, no GPU, no torchvision. Extracted from the inline logic in
`detect_and_evaluate.py` so `finalize.py` and any future re-postprocessing
script can share a single implementation.

Operation parity with the legacy inline logic is enforced via
`tests/postproc/test_parity_against_old.py`. Numerical parity with geoai
output is NOT a goal; only operations parity.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Defaults (matching detect_and_evaluate.py constants for backward compat)
# ─────────────────────────────────────────────────────────────────────────
DEFAULT_POST_CONF_THRESHOLD = 0.85
DEFAULT_MIN_OBJECT_AREA = 5.0
DEFAULT_MAX_ELONGATION = 8.0
DEFAULT_MIN_SOLIDITY = 0.0
DEFAULT_SHADOW_RGB_THRESH = 60
DEFAULT_OVER_BRIGHT_THRESH = 250
DEFAULT_MASK_THRESHOLD = 0.3
DEFAULT_DETECTOR_SCORE_THRESHOLD = 0.05
DEFAULT_CONFIDENCE_THRESHOLD = 0.3  # legacy key; maps to pre_vector_score_threshold

DEFAULT_ELONGATION_TIERED: list[tuple[float, float]] = [
    (100.0, 15.0),
    (0.0, 8.0),
]
DEFAULT_CONF_TIERED: list[tuple[float, float]] = [
    (200.0, 0.70),
    (100.0, 0.65),
    (0.0, 0.85),
]


# ─────────────────────────────────────────────────────────────────────────
# Config parser (corrected superset, NOT a verbatim port)
# ─────────────────────────────────────────────────────────────────────────
_KNOWN_KEYS = {
    "confidence_threshold",       # legacy → pre_vector_score_threshold
    "detector_score_threshold",
    "pre_vector_score_threshold",
    "mask_threshold",
    "post_conf_threshold",
    "min_object_area",
    "max_elongation",
    "min_solidity",
    "shadow_rgb_thresh",
    "over_bright_thresh",
    "elongation_tiered",
    "conf_tiered",
}


def load_postproc_config(config_path: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Parse a postproc config JSON.

    Corrected superset of the legacy `detect_and_evaluate.py:223-242` parser,
    which silently dropped `confidence_threshold`, `mask_threshold`,
    `min_solidity`, and `shadow_rgb_thresh`. The new parser accepts all keys
    currently present in `configs/postproc/v4_canonical.json` plus the new
    direct-pipeline keys.

    Legacy `confidence_threshold` is mapped to `pre_vector_score_threshold`
    (V1.4 plan A: applied finalize-side; raw artifact still preserves all
    `score >= detector_score_threshold` detections).

    Unknown keys log a warning by default; `strict=True` raises ValueError.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"postproc config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if k == "_meta":
            continue
        if k in _KNOWN_KEYS:
            out[k] = v
        else:
            msg = f"unknown postproc config key: {k!r} (in {path.name})"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

    # Legacy mapping: confidence_threshold → pre_vector_score_threshold
    if "confidence_threshold" in out and "pre_vector_score_threshold" not in out:
        out["pre_vector_score_threshold"] = out["confidence_threshold"]

    return out


# ─────────────────────────────────────────────────────────────────────────
# Spatial NMS — extracted from detect_and_evaluate.py:414-451
# ─────────────────────────────────────────────────────────────────────────
def spatial_nms(gdf, iou_threshold: float = 0.5):
    """Grid-level spatial NMS: drop polygons whose IoU with a kept polygon
    exceeds the threshold; keep the larger polygon.

    Matches `detect_and_evaluate.py:414-451`. Uses a spatial index for
    candidate lookup. Returns a copy of the surviving rows.
    """
    if len(gdf) <= 1:
        return gdf

    keep = [True] * len(gdf)
    sindex = gdf.sindex

    for i in range(len(gdf)):
        if not keep[i]:
            continue
        geom_i = gdf.iloc[i].geometry
        candidates = list(sindex.intersection(geom_i.bounds))
        for j in candidates:
            if j <= i or not keep[j]:
                continue
            geom_j = gdf.iloc[j].geometry
            try:
                inter = geom_i.intersection(geom_j).area
                union = geom_i.area + geom_j.area - inter
                if union > 0 and (inter / union) > iou_threshold:
                    if geom_i.area >= geom_j.area:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
            except Exception:
                continue

    return gdf[keep].copy()


# ─────────────────────────────────────────────────────────────────────────
# Geometric properties — replaces geoai.add_geometric_properties
# ─────────────────────────────────────────────────────────────────────────
def compute_geometric_properties(gdf):
    """Add area_m2, elongation, solidity columns (computed in metric CRS).

    Caller must ensure the gdf is already in a metric CRS (e.g. via
    `core.grid_utils.get_metric_crs`); this function does NOT reproject.
    """
    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf["elongation"] = gdf.geometry.apply(_elongation_of)
    gdf["solidity"] = gdf.geometry.apply(_solidity_of)
    return gdf


def _elongation_of(geom) -> float:
    """Elongation = major / minor of minimum rotated rectangle.

    Returns 1.0 for empty/null/degenerate geometry. Matches the convention
    used by the existing tiered elongation filter (residential ≤ 8,
    commercial ≤ 15).
    """
    if geom is None or geom.is_empty:
        return 1.0
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        if len(coords) < 5:
            return 1.0
        # Minimum rotated rectangle has 5 points (closed ring): 4 corners + duplicate.
        side_a = _dist(coords[0], coords[1])
        side_b = _dist(coords[1], coords[2])
        major = max(side_a, side_b)
        minor = min(side_a, side_b)
        return float(major / minor) if minor > 0 else 1.0
    except Exception:
        return 1.0


def _solidity_of(geom) -> float:
    """Solidity = area / convex_hull.area. 1.0 for empty/null geometry."""
    if geom is None or geom.is_empty:
        return 1.0
    try:
        hull = geom.convex_hull
        if hull.area <= 0:
            return 1.0
        return float(geom.area / hull.area)
    except Exception:
        return 1.0


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ─────────────────────────────────────────────────────────────────────────
# RGB zonal means — replicates detect_and_evaluate.py:663-693
# ─────────────────────────────────────────────────────────────────────────
def compute_rgb_zonal_means(gdf, raster_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-polygon mean of bands 1/2/3 from `raster_path`.

    Parity behavior with the legacy inline `_zonal_rgb_means`:
      - Initialize means to 0.0.
      - Empty / null / non-overlapping mask → mean stays (0.0, 0.0, 0.0).
        (Downstream shadow filter then drops these rows. Intentional —
        matches legacy behavior at `detect_and_evaluate.py:705-707`.)
      - For non-empty masks: per-band, exclude `vals == 0` (nodata=0
        convention), then average.

    Implementation: per-polygon, derive a windowed read using the polygon's
    bounds clipped to the raster extent. This avoids allocating an
    H×W boolean mask per polygon when polygons are small relative to the
    source raster (the typical case for chunked tile layouts where a chunk
    is e.g. 7500×7500 but a panel is ~50×50).

    Caller (in `finalize.py`) is responsible for grouping the GDF by
    `source_tif` and calling this once per group; this function does NOT
    cross source files.
    """
    from rasterio import open as _rio_open
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds

    n = len(gdf)
    means = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return means[:, 0], means[:, 1], means[:, 2]

    geoms = list(gdf.geometry.values)
    with _rio_open(str(raster_path)) as src:
        H, W = src.height, src.width
        for j, geom in enumerate(geoms):
            if geom is None or geom.is_empty:
                continue
            try:
                # Polygon bounds → window in source CRS → integer pixel window
                minx, miny, maxx, maxy = geom.bounds
                win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                # Clip to raster extent + round to int.
                col_off = int(max(0, np.floor(win.col_off)))
                row_off = int(max(0, np.floor(win.row_off)))
                col_end = int(min(W, np.ceil(win.col_off + win.width)))
                row_end = int(min(H, np.ceil(win.row_off + win.height)))
                w_w = col_end - col_off
                w_h = row_end - row_off
                if w_w <= 0 or w_h <= 0:
                    continue
                clipped = Window(col_off, row_off, w_w, w_h)
                rgb_local = src.read([1, 2, 3], window=clipped)  # (3, w_h, w_w)
                tr_local = src.window_transform(clipped)
                m = geometry_mask(
                    [geom], out_shape=(w_h, w_w), transform=tr_local,
                    invert=True, all_touched=False,
                )
            except Exception:
                continue
            if not m.any():
                continue
            for b in range(3):
                vals = rgb_local[b][m]
                vals = vals[vals != 0]  # nodata=0 parity
                if vals.size:
                    means[j, b] = float(vals.mean())

    return means[:, 0], means[:, 1], means[:, 2]


# ─────────────────────────────────────────────────────────────────────────
# Mask mean confidence — operates in chip pixel space
# ─────────────────────────────────────────────────────────────────────────
def compute_mask_mean_confidence(
    detection_indices: Sequence[int],
    masks_by_index: dict[int, tuple[np.ndarray, tuple[int, int]]],
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
) -> np.ndarray:
    """Mean of soft mask values inside the thresholded polygon, per detection.

    `masks_by_index[i]` returns (`mask_crop_uint8`, `(offset_x, offset_y)`).
    The mean is computed over pixels where the soft mask is above
    `mask_threshold * 255`. Returns an array of length `len(detection_indices)`,
    with NaN for missing entries.
    """
    out = np.full(len(detection_indices), np.nan, dtype=np.float64)
    cutoff = int(round(mask_threshold * 255))
    for i, idx in enumerate(detection_indices):
        if idx not in masks_by_index:
            continue
        mask_crop, _offset = masks_by_index[idx]
        if mask_crop.size == 0:
            continue
        kept = mask_crop[mask_crop >= cutoff]
        if kept.size == 0:
            continue
        out[i] = float(kept.mean()) / 255.0
    return out


# ─────────────────────────────────────────────────────────────────────────
# Vectorize a single detection's mask crop → polygon(s) in source CRS
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VectorizeResult:
    """Output of vectorize_chip_mask. `geoms` is in source_crs."""
    geoms: list  # list[shapely.geometry.Polygon]
    n_components_dropped: int


@dataclass(frozen=True)
class PaintedPolygon:
    """One connected component from pixel-OR vectorization.

    `score` = max detection score whose mask painted any pixel inside this
    component (uint8 stored as float in [0, 1]).
    `mask_mean_confidence` = mean soft-mask probability over the
    above-threshold pixels inside the polygon.
    `contributing_detection_indices` = global detection indices that
    painted at least one pixel inside this component (for traceback).
    """
    geom: object   # shapely.geometry.Polygon
    score: float
    mask_mean_confidence: float
    label: int
    contributing_detection_count: int


def vectorize_chip_mask(
    mask_crop_uint8: np.ndarray,
    mask_crop_offset: tuple[int, int],
    *,
    threshold: float,
    window_transform,
    source_crs,
    multi_component: str = "largest",
    simplify_tolerance_pixels: float = 0.0,
) -> VectorizeResult:
    """Paste a cropped uint8 mask back at offset, threshold, vectorize.

    `window_transform` is the rasterio Affine for the *chip* (whole padded
    chip in source TIF coordinates). `mask_crop_offset = (x, y)` is the
    integer pixel offset of the crop within the chip.

    Returns polygons in `source_crs`. `multi_component`:
      - "largest" (default): keep the largest by area.
      - "union": dissolve into a single MultiPolygon.
      - "explode": return all polygons.

    `simplify_tolerance_pixels` runs `shapely.simplify` in pixel space (the
    pixel size in source CRS units is derived from `window_transform`). 0.0
    means no simplification.
    """
    from rasterio.features import shapes as _shapes
    from shapely.geometry import shape as _shape

    if mask_crop_uint8.size == 0:
        return VectorizeResult(geoms=[], n_components_dropped=0)

    cutoff = int(round(threshold * 255))
    binary = (mask_crop_uint8 >= cutoff).astype(np.uint8)
    if not binary.any():
        return VectorizeResult(geoms=[], n_components_dropped=0)

    # Build a per-crop affine: chip_transform * translation(offset_x, offset_y).
    # rasterio.features.shapes uses a transform that maps (col, row) → CRS.
    from affine import Affine
    crop_transform = window_transform * Affine.translation(mask_crop_offset[0], mask_crop_offset[1])

    polys = []
    for geom_dict, val in _shapes(binary, mask=binary.astype(bool), transform=crop_transform):
        if val == 0:
            continue
        polys.append(_shape(geom_dict))

    if not polys:
        return VectorizeResult(geoms=[], n_components_dropped=0)

    n_dropped = 0
    if multi_component == "largest":
        polys.sort(key=lambda g: g.area, reverse=True)
        n_dropped = len(polys) - 1
        out_polys = [polys[0]]
    elif multi_component == "union":
        from shapely.ops import unary_union
        merged = unary_union(polys)
        # unary_union returns Polygon or MultiPolygon
        out_polys = [merged]
    elif multi_component == "explode":
        out_polys = polys
    else:
        raise ValueError(f"unknown multi_component policy: {multi_component!r}")

    if simplify_tolerance_pixels > 0:
        # Pixel size in source CRS units along x:
        px_size_x = abs(window_transform.a)
        tol = simplify_tolerance_pixels * px_size_x
        out_polys = [g.simplify(tol, preserve_topology=True) for g in out_polys]

    return VectorizeResult(geoms=out_polys, n_components_dropped=n_dropped)


# ─────────────────────────────────────────────────────────────────────────
# Pixel-OR vectorization (geoai-equivalent merge semantics)
# ─────────────────────────────────────────────────────────────────────────
def paint_and_vectorize_pixel_or(
    detections: list[dict],
    *,
    raster_height: int,
    raster_width: int,
    source_transform,            # affine.Affine for the source raster
    source_crs,                  # str or rasterio CRS
    mask_threshold: float,
    multi_component: str = "explode",
    simplify_tolerance_pixels: float = 0.0,
) -> list[PaintedPolygon]:
    """Geoai-equivalent merge: paint every detection's soft mask onto a
    chunk-sized raster (max-merging on overlap), threshold, vectorize.

    Each `detections` item is a dict with at least:
      - mask_crop_uint8: np.ndarray (h, w) uint8
      - source_offset: tuple[int, int]   # (col, row) in source-pixel space
      - score: float
      - label: int

    Returns a list of `PaintedPolygon` in `source_crs`. One connected
    component = one polygon. Components touched by multiple detections
    naturally fuse (this is the geoai semantic).
    """
    from affine import Affine
    from rasterio.features import shapes as _shapes, geometry_mask
    from rasterio.windows import Window, transform as window_transform
    from shapely.geometry import shape as _shape

    if raster_height <= 0 or raster_width <= 0:
        return []
    if not detections:
        return []

    # Per-pixel max of soft mask values; per-pixel max of detection scores.
    soft_raster = np.zeros((raster_height, raster_width), dtype=np.uint8)
    score_raster = np.zeros((raster_height, raster_width), dtype=np.uint8)
    cutoff = int(round(mask_threshold * 255))

    n_painted = 0
    for det in detections:
        crop = det["mask_crop_uint8"]
        if crop is None or crop.size == 0:
            continue
        x0_src, y0_src = det["source_offset"]
        h_c, w_c = crop.shape
        x1_src = x0_src + w_c
        y1_src = y0_src + h_c

        # Clip to raster bounds (detections near edges may overhang)
        cx0 = max(0, int(x0_src)); cy0 = max(0, int(y0_src))
        cx1 = min(raster_width, int(x1_src)); cy1 = min(raster_height, int(y1_src))
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        crop_x0 = cx0 - int(x0_src); crop_y0 = cy0 - int(y0_src)
        crop_x1 = crop_x0 + (cx1 - cx0); crop_y1 = crop_y0 + (cy1 - cy0)
        crop_slice = crop[crop_y0:crop_y1, crop_x0:crop_x1]

        # Soft mask: max-merge on overlap (parity with geoai's raster paint)
        soft_view = soft_raster[cy0:cy1, cx0:cx1]
        np.maximum(soft_view, crop_slice, out=soft_view)

        # Score: paint det.score into the binary footprint of this detection
        det_score_uint8 = int(round(float(det["score"]) * 255))
        binary_crop = (crop_slice >= cutoff).astype(np.uint8)
        if binary_crop.any():
            score_paint = (binary_crop * det_score_uint8).astype(np.uint8)
            score_view = score_raster[cy0:cy1, cx0:cx1]
            np.maximum(score_view, score_paint, out=score_view)
            n_painted += 1

    if n_painted == 0:
        return []

    binary = (soft_raster >= cutoff).astype(np.uint8)
    if not binary.any():
        return []

    # Vectorize the OR-merged mask
    polys = []
    for geom_dict, val in _shapes(
        binary, mask=binary.astype(bool), transform=source_transform,
    ):
        if val == 0:
            continue
        polys.append(_shape(geom_dict))

    if not polys:
        return []

    # Apply multi-component policy. In pixel-OR mode "largest" is rarely the
    # right default (it would drop disconnected installations on the same
    # roof); we recommend "explode". Caller chooses.
    if multi_component == "largest":
        polys.sort(key=lambda g: g.area, reverse=True)
        polys = polys[:1]
    elif multi_component == "union":
        from shapely.ops import unary_union
        polys = [unary_union(polys)]
    # "explode" = leave as-is

    # Simplify in pixel space (matches V1.4 decision #20).
    if simplify_tolerance_pixels > 0:
        px_size = abs(source_transform.a)
        tol = simplify_tolerance_pixels * px_size
        polys = [g.simplify(tol, preserve_topology=True) for g in polys]

    # For each polygon: clipped windowed read of soft / score raster → stats.
    out: list[PaintedPolygon] = []
    for poly in polys:
        if poly is None or poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        # Convert to pixel window
        col_off, row_off = ~source_transform * (minx, maxy)
        col_end, row_end = ~source_transform * (maxx, miny)
        col_off = int(max(0, np.floor(min(col_off, col_end))))
        row_off = int(max(0, np.floor(min(row_off, row_end))))
        col_end = int(min(raster_width, np.ceil(max(col_off, col_end))))
        row_end = int(min(raster_height, np.ceil(max(row_off, row_end))))
        ww = col_end - col_off
        wh = row_end - row_off
        if ww <= 0 or wh <= 0:
            out.append(PaintedPolygon(
                geom=poly, score=0.0, mask_mean_confidence=0.0,
                label=1, contributing_detection_count=0,
            ))
            continue
        local_tr = window_transform(
            Window(col_off, row_off, ww, wh), source_transform,
        )
        try:
            m = geometry_mask(
                [poly], out_shape=(wh, ww), transform=local_tr,
                invert=True, all_touched=False,
            )
        except Exception:
            out.append(PaintedPolygon(
                geom=poly, score=0.0, mask_mean_confidence=0.0,
                label=1, contributing_detection_count=0,
            ))
            continue
        soft_local = soft_raster[row_off:row_off + wh, col_off:col_off + ww]
        score_local = score_raster[row_off:row_off + wh, col_off:col_off + ww]
        if not m.any():
            out.append(PaintedPolygon(
                geom=poly, score=0.0, mask_mean_confidence=0.0,
                label=1, contributing_detection_count=0,
            ))
            continue
        soft_inside = soft_local[m]
        soft_above = soft_inside[soft_inside >= cutoff]
        mmc = float(soft_above.mean()) / 255.0 if soft_above.size else 0.0
        score_inside = score_local[m]
        score_above = score_inside[score_inside > 0]
        sc_max = float(score_above.max()) / 255.0 if score_above.size else 0.0
        out.append(PaintedPolygon(
            geom=poly, score=sc_max, mask_mean_confidence=mmc,
            label=1, contributing_detection_count=0,  # not tracked in this fast path
        ))

    return out


# ─────────────────────────────────────────────────────────────────────────
# Postproc filter pipeline — area / tiered elong / tiered conf / RGB
# ─────────────────────────────────────────────────────────────────────────
def apply_postproc_filters(gdf, config: dict[str, Any]):
    """Run the full filter chain.

    Required columns:
      area_m2, elongation, confidence, mean_r, mean_g, mean_b
    Returns (filtered_gdf, stats_dict) where stats_dict has counts per stage.
    """
    stats = {"input": len(gdf)}
    if len(gdf) == 0:
        return gdf, stats

    # 1) area filter
    min_area = float(config.get("min_object_area", DEFAULT_MIN_OBJECT_AREA))
    if "area_m2" in gdf.columns:
        gdf = gdf[gdf["area_m2"] >= min_area].copy()
    stats["after_area"] = len(gdf)

    # 2) tiered elongation
    if "elongation" in gdf.columns and "area_m2" in gdf.columns:
        tiers = config.get("elongation_tiered", DEFAULT_ELONGATION_TIERED)
        gdf = _apply_tiered_keep(gdf, "elongation", tiers, op="<=")
    elif "elongation" in gdf.columns:
        max_elong = float(config.get("max_elongation", DEFAULT_MAX_ELONGATION))
        if max_elong < 999:
            gdf = gdf[gdf["elongation"] <= max_elong].copy()
    stats["after_elongation"] = len(gdf)

    # 3) RGB shadow filter (RGB all < threshold)
    shadow_thresh = float(config.get("shadow_rgb_thresh", DEFAULT_SHADOW_RGB_THRESH))
    over_bright_thresh = float(config.get("over_bright_thresh", DEFAULT_OVER_BRIGHT_THRESH))
    if all(c in gdf.columns for c in ("mean_r", "mean_g", "mean_b")):
        is_shadow = (
            (gdf["mean_r"] < shadow_thresh)
            & (gdf["mean_g"] < shadow_thresh)
            & (gdf["mean_b"] < shadow_thresh)
        )
        is_too_bright = (
            (gdf["mean_r"] > over_bright_thresh)
            & (gdf["mean_g"] > over_bright_thresh)
            & (gdf["mean_b"] > over_bright_thresh)
        )
        gdf = gdf[~(is_shadow | is_too_bright)].copy()
    stats["after_rgb"] = len(gdf)

    # 4) tiered confidence
    if "confidence" in gdf.columns and "area_m2" in gdf.columns:
        tiers = config.get("conf_tiered", DEFAULT_CONF_TIERED)
        gdf = _apply_tiered_keep(gdf, "confidence", tiers, op=">=")
    elif "confidence" in gdf.columns:
        post_conf = float(config.get("post_conf_threshold", DEFAULT_POST_CONF_THRESHOLD))
        gdf = gdf[gdf["confidence"] >= post_conf].copy()
    stats["after_confidence"] = len(gdf)

    return gdf, stats


def _apply_tiered_keep(gdf, value_col: str, tiers: Iterable[tuple[float, float]], *, op: str):
    """Apply tiered filter: first matching tier (by min_area) wins.

    `tiers` is ordered (largest min_area first). For each row, find the first
    tier whose `min_area` is satisfied, then check `value_col op threshold`.
    """
    import pandas as pd
    keep = pd.Series(False, index=gdf.index)
    matched = pd.Series(False, index=gdf.index)
    for min_area, threshold in tiers:
        tier_rows = (gdf["area_m2"] >= min_area) & ~matched
        if op == "<=":
            keep |= tier_rows & (gdf[value_col] <= threshold)
        elif op == ">=":
            keep |= tier_rows & (gdf[value_col] >= threshold)
        else:
            raise ValueError(f"unknown op: {op!r}")
        matched |= tier_rows
    return gdf[keep].copy()
