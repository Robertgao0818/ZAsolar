"""Tests for paint_and_vectorize_pixel_or.

Verifies the geoai-equivalent merge semantics: detections with adjacent /
overlapping mask regions fuse into a single connected-component polygon.
"""
from __future__ import annotations

import numpy as np
from affine import Affine

from core.postproc import paint_and_vectorize_pixel_or


def _identity_transform() -> Affine:
    """1 unit per pixel, origin at (0, 0). Lets us treat polygon coords
    as pixel coords for assertions."""
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def test_two_adjacent_detections_fuse_into_one():
    """Two detections side-by-side with edge-touching binary masks → ONE polygon."""
    # Detection A: 10×10 square at (0, 0); B: 10×10 square at (10, 0); they share an edge.
    crop = np.full((10, 10), 255, dtype=np.uint8)
    dets = [
        {"mask_crop_uint8": crop, "source_offset": (0, 0), "score": 0.9, "label": 1},
        {"mask_crop_uint8": crop, "source_offset": (10, 0), "score": 0.8, "label": 1},
    ]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20, raster_width=30,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert len(out) == 1, f"adjacent detections should fuse; got {len(out)} polygons"
    # Polygon area should be roughly 20 × 10 = 200 pixels² (the union of the two squares).
    assert 180 <= out[0].geom.area <= 220
    # Score = max of contributing detections
    assert abs(out[0].score - 0.9) < 0.005


def test_two_disjoint_detections_stay_separate():
    """Two detections far apart → TWO polygons."""
    crop = np.full((10, 10), 255, dtype=np.uint8)
    dets = [
        {"mask_crop_uint8": crop, "source_offset": (0, 0), "score": 0.9, "label": 1},
        {"mask_crop_uint8": crop, "source_offset": (50, 50), "score": 0.7, "label": 1},
    ]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=80, raster_width=80,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert len(out) == 2


def test_overlapping_detections_take_max_score():
    """Two heavily overlapping detections → ONE polygon, score = max."""
    crop = np.full((10, 10), 255, dtype=np.uint8)
    dets = [
        {"mask_crop_uint8": crop, "source_offset": (0, 0), "score": 0.6, "label": 1},
        {"mask_crop_uint8": crop, "source_offset": (5, 5), "score": 0.95, "label": 1},
    ]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=30, raster_width=30,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert len(out) == 1
    assert abs(out[0].score - 0.95) < 0.005


def test_below_threshold_is_dropped():
    """Soft mask values all below threshold → no polygon."""
    crop = np.full((10, 10), 50, dtype=np.uint8)  # 50/255 = 0.196 < 0.3 default
    dets = [{"mask_crop_uint8": crop, "source_offset": (0, 0), "score": 0.9, "label": 1}]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20, raster_width=20,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert out == []


def test_three_in_a_row_fuse_into_one():
    """Three adjacent detections along a row (typical big commercial panel) → 1 polygon."""
    crop = np.full((10, 10), 255, dtype=np.uint8)
    dets = [
        {"mask_crop_uint8": crop, "source_offset": (0, 0),  "score": 0.7, "label": 1},
        {"mask_crop_uint8": crop, "source_offset": (10, 0), "score": 0.8, "label": 1},
        {"mask_crop_uint8": crop, "source_offset": (20, 0), "score": 0.9, "label": 1},
    ]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20, raster_width=40,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert len(out) == 1
    # Combined area ≈ 30 × 10 = 300
    assert 270 <= out[0].geom.area <= 330
    # Score = max
    assert abs(out[0].score - 0.9) < 0.005


def test_partial_overlap_with_gap_stays_separate():
    """Two detections separated by a 1-pixel gap → still two components.

    Validates that pixel-OR does NOT fall into 'all detections within bbox-near
    each other merge' — only physically touching pixels merge."""
    crop = np.full((10, 10), 255, dtype=np.uint8)
    dets = [
        # A: pixels [0..9]
        {"mask_crop_uint8": crop, "source_offset": (0, 0),  "score": 0.9, "label": 1},
        # B: pixels [11..20] — gap at pixel 10
        {"mask_crop_uint8": crop, "source_offset": (11, 0), "score": 0.8, "label": 1},
    ]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20, raster_width=30,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    # 1-pixel gap → still two components (rasterio.features.shapes uses 4-conn by default)
    assert len(out) == 2


def test_empty_detections_returns_empty():
    out = paint_and_vectorize_pixel_or(
        [],
        raster_height=20, raster_width=20,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert out == []


def test_mask_mean_confidence_computed():
    """mask_mean_confidence is mean of soft-mask values above threshold."""
    crop = np.full((10, 10), 200, dtype=np.uint8)  # 200/255 ≈ 0.784
    dets = [{"mask_crop_uint8": crop, "source_offset": (0, 0), "score": 0.9, "label": 1}]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20, raster_width=20,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert len(out) == 1
    # All inside pixels are 200 → mean / 255 ≈ 0.784
    assert abs(out[0].mask_mean_confidence - 200 / 255) < 1e-3


def test_overhanging_detection_is_clipped():
    """Detection mask extends past raster bounds → safely clipped, no crash."""
    crop = np.full((10, 10), 255, dtype=np.uint8)
    # Place detection so half overhangs the right edge
    dets = [{"mask_crop_uint8": crop, "source_offset": (15, 0), "score": 0.9, "label": 1}]
    out = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20, raster_width=20,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    assert len(out) == 1
    # Visible area is 5 × 10 = 50
    assert 40 <= out[0].geom.area <= 60
