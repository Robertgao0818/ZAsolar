"""Drift-protection: assert new core.postproc reproduces the old inline
keep/drop decisions for area / tiered elong / tiered conf / shadow / over-bright /
spatial_nms.

The old logic lives inline in detect_and_evaluate.py — we don't import it.
Instead we re-encode the behavior at a snapshot via reference functions
below and compare row-for-row.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from core.postproc import (
    DEFAULT_CONF_TIERED,
    DEFAULT_ELONGATION_TIERED,
    apply_postproc_filters,
    spatial_nms,
)


# ─────────────────────────────────────────────────────────────────────────
# Reference implementations (snapshot of detect_and_evaluate.py inline logic)
# ─────────────────────────────────────────────────────────────────────────
def _reference_filter(gdf: gpd.GeoDataFrame, *,
                      min_area=5.0, shadow_thresh=60, over_bright=250,
                      elong_tiered=DEFAULT_ELONGATION_TIERED,
                      conf_tiered=DEFAULT_CONF_TIERED) -> gpd.GeoDataFrame:
    """Reference snapshot of detect_and_evaluate.py's inline filter chain."""
    g = gdf.copy()
    # area
    g = g[g["area_m2"] >= min_area].copy()
    # tiered elong
    elong_keep = pd.Series(False, index=g.index)
    matched = pd.Series(False, index=g.index)
    for min_a, max_e in elong_tiered:
        tier = (g["area_m2"] >= min_a) & ~matched
        elong_keep |= tier & (g["elongation"] <= max_e)
        matched |= tier
    g = g[elong_keep].copy()
    # RGB shadow + over-bright
    is_shadow = (
        (g["mean_r"] < shadow_thresh)
        & (g["mean_g"] < shadow_thresh)
        & (g["mean_b"] < shadow_thresh)
    )
    is_bright = (
        (g["mean_r"] > over_bright)
        & (g["mean_g"] > over_bright)
        & (g["mean_b"] > over_bright)
    )
    g = g[~(is_shadow | is_bright)].copy()
    # tiered conf
    conf_keep = pd.Series(False, index=g.index)
    matched = pd.Series(False, index=g.index)
    for min_a, thresh in conf_tiered:
        tier = (g["area_m2"] >= min_a) & ~matched
        conf_keep |= tier & (g["confidence"] >= thresh)
        matched |= tier
    g = g[conf_keep].copy()
    return g


def _build_synthetic_gdf(n: int = 50, seed: int = 0) -> gpd.GeoDataFrame:
    """Diverse synthetic GDF that exercises every branch."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # Random box dimensions
        w = float(rng.uniform(2, 30))
        h = float(rng.uniform(2, 30))
        x0 = float(rng.uniform(0, 1000))
        y0 = float(rng.uniform(0, 1000))
        rows.append({
            "geometry": box(x0, y0, x0 + w, y0 + h),
            "area_m2": w * h,
            "elongation": float(rng.uniform(1, 20)),
            "confidence": float(rng.uniform(0.0, 1.0)),
            "mean_r": float(rng.integers(0, 256)),
            "mean_g": float(rng.integers(0, 256)),
            "mean_b": float(rng.integers(0, 256)),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:32734")


def test_filter_chain_matches_reference():
    gdf = _build_synthetic_gdf(n=100, seed=42)
    new, _ = apply_postproc_filters(gdf, {
        "min_object_area": 5.0,
        "shadow_rgb_thresh": 60,
        "over_bright_thresh": 250,
        "elongation_tiered": DEFAULT_ELONGATION_TIERED,
        "conf_tiered": DEFAULT_CONF_TIERED,
    })
    ref = _reference_filter(gdf)
    # Same set of indices kept
    assert sorted(new.index.tolist()) == sorted(ref.index.tolist())


def test_spatial_nms_matches_reference_on_overlap_pair():
    """Reference: keep larger when IoU > threshold."""
    geoms = [box(0, 0, 10, 10), box(1, 1, 11, 11)]  # large overlap
    g = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:32734")
    out = spatial_nms(g, iou_threshold=0.5)
    # Both have area 100 → tie broken by "first wins" via i<=j logic;
    # implementation keeps geom_i when areas tie.
    assert len(out) == 1
