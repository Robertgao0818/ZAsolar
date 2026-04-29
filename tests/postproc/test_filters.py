"""Unit tests for area / tiered elongation / tiered confidence / RGB filters."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from core.postproc import (
    DEFAULT_CONF_TIERED,
    DEFAULT_ELONGATION_TIERED,
    apply_postproc_filters,
    compute_geometric_properties,
)


def _gdf_from_rows(rows):
    """Build a GeoDataFrame in EPSG:32734 (metric) from a list of dicts."""
    geoms = [box(*r.pop("bbox")) for r in rows]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:32734")


def test_area_filter_drops_below_min():
    gdf = _gdf_from_rows([
        {"bbox": (0, 0, 1, 1), "elongation": 1.0, "confidence": 0.9,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},   # 1 m²
        {"bbox": (0, 0, 5, 5), "elongation": 1.0, "confidence": 0.9,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},   # 25 m²
    ])
    gdf = compute_geometric_properties(gdf)
    out, stats = apply_postproc_filters(gdf, {"min_object_area": 5.0})
    assert len(out) == 1
    assert out.iloc[0]["area_m2"] == 25.0
    assert stats["after_area"] == 1


def test_tiered_elongation_residential_vs_commercial():
    """Residential (<100 m²) capped at 8; commercial (>=100 m²) capped at 15."""
    gdf = _gdf_from_rows([
        # 50 m² residential, elongation 9 → DROP
        {"bbox": (0, 0, 25, 2), "elongation": 9.0, "confidence": 0.9,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},
        # 200 m² commercial, elongation 12 → KEEP (≥100m² tier allows ≤15)
        {"bbox": (0, 0, 50, 4), "elongation": 12.0, "confidence": 0.9,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},
        # 200 m² commercial, elongation 16 → DROP
        {"bbox": (0, 0, 100, 2), "elongation": 16.0, "confidence": 0.9,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},
    ])
    # Override computed elongation with our test values
    gdf = compute_geometric_properties(gdf)
    gdf["elongation"] = [9.0, 12.0, 16.0]
    out, _ = apply_postproc_filters(gdf, {
        "min_object_area": 5.0,
        "elongation_tiered": DEFAULT_ELONGATION_TIERED,
    })
    # Only the 200 m²/12 elongation row survives
    areas = sorted(out["area_m2"].tolist())
    assert areas == [200.0]


def test_tiered_confidence_residential_strict_commercial_lenient():
    """Residential needs 0.85; commercial 0.65 (or 0.70 ≥200m²).

    Use square boxes so elongation=1 doesn't interfere with the conf test.
    """
    gdf = _gdf_from_rows([
        # Residential 49 m² (7×7), conf 0.7 → DROP (<0.85)
        {"bbox": (0, 0, 7, 7), "confidence": 0.70,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},
        # Commercial 144 m² (12×12), conf 0.7 → KEEP (≥100m² → 0.65 cutoff)
        {"bbox": (0, 0, 12, 12), "confidence": 0.70,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},
        # Big commercial 256 m² (16×16), conf 0.65 → DROP (≥200m² → 0.70 cutoff)
        {"bbox": (0, 0, 16, 16), "confidence": 0.65,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},
    ])
    gdf = compute_geometric_properties(gdf)
    out, _ = apply_postproc_filters(gdf, {
        "min_object_area": 5.0,
        "elongation_tiered": DEFAULT_ELONGATION_TIERED,
        "conf_tiered": DEFAULT_CONF_TIERED,
    })
    confs = sorted(out["confidence"].tolist())
    assert confs == [0.70]


def test_rgb_shadow_filter():
    """All three channels < shadow_rgb_thresh → drop."""
    gdf = _gdf_from_rows([
        # All dark — DROP
        {"bbox": (0, 0, 5, 5), "elongation": 1.0, "confidence": 0.9,
         "mean_r": 30, "mean_g": 30, "mean_b": 30},
        # Only R is below threshold — KEEP (need ALL three)
        {"bbox": (0, 0, 5, 5), "elongation": 1.0, "confidence": 0.9,
         "mean_r": 30, "mean_g": 100, "mean_b": 100},
    ])
    gdf = compute_geometric_properties(gdf)
    out, _ = apply_postproc_filters(gdf, {
        "min_object_area": 5.0, "shadow_rgb_thresh": 60,
        "elongation_tiered": DEFAULT_ELONGATION_TIERED,
        "conf_tiered": DEFAULT_CONF_TIERED,
    })
    assert len(out) == 1
    assert out.iloc[0]["mean_r"] == 30
    assert out.iloc[0]["mean_g"] == 100


def test_rgb_over_bright_filter():
    """All three channels > 250 → drop (over-exposed)."""
    gdf = _gdf_from_rows([
        # All blown out — DROP
        {"bbox": (0, 0, 5, 5), "elongation": 1.0, "confidence": 0.9,
         "mean_r": 252, "mean_g": 253, "mean_b": 254},
        # Normal — KEEP
        {"bbox": (0, 0, 5, 5), "elongation": 1.0, "confidence": 0.9,
         "mean_r": 100, "mean_g": 100, "mean_b": 100},
    ])
    gdf = compute_geometric_properties(gdf)
    out, _ = apply_postproc_filters(gdf, {
        "min_object_area": 5.0,
        "elongation_tiered": DEFAULT_ELONGATION_TIERED,
        "conf_tiered": DEFAULT_CONF_TIERED,
    })
    assert len(out) == 1
    assert out.iloc[0]["mean_r"] == 100


def test_geometric_properties_basic_square():
    """5×5 square → area=25, elongation≈1, solidity=1."""
    gdf = gpd.GeoDataFrame(
        {"x": [1]},
        geometry=[box(0, 0, 5, 5)],
        crs="EPSG:32734",
    )
    out = compute_geometric_properties(gdf)
    assert out.iloc[0]["area_m2"] == 25.0
    assert abs(out.iloc[0]["elongation"] - 1.0) < 1e-6
    assert abs(out.iloc[0]["solidity"] - 1.0) < 1e-6


def test_geometric_properties_long_rectangle():
    """1×10 rectangle → elongation=10."""
    gdf = gpd.GeoDataFrame(
        {"x": [1]},
        geometry=[box(0, 0, 10, 1)],
        crs="EPSG:32734",
    )
    out = compute_geometric_properties(gdf)
    assert abs(out.iloc[0]["elongation"] - 10.0) < 1e-3
