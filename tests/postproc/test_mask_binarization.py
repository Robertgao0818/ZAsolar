"""Tests for adaptive and hysteresis mask binarization."""
from __future__ import annotations

import numpy as np
from affine import Affine

from core.postproc import (
    affine_pixel_area,
    binarize_mask_uint8,
    paint_and_vectorize_pixel_or,
    parse_threshold_area_tiers,
    vectorize_chip_mask,
)


def _identity_transform() -> Affine:
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def test_adaptive_threshold_tightens_large_low_confidence_mask():
    mask = np.full((20, 20), 100, dtype=np.uint8)  # 0.392

    out = binarize_mask_uint8(
        mask,
        threshold=0.3,
        threshold_area_tiers=[(300, 0.5)],
        threshold_area_scale=1.0,
    )

    assert out.effective_threshold == 0.5
    assert out.low_area_units == 400
    assert out.binary.sum() == 0


def test_adaptive_threshold_keeps_small_low_confidence_mask_at_base_threshold():
    mask = np.full((10, 10), 100, dtype=np.uint8)  # 0.392

    out = binarize_mask_uint8(
        mask,
        threshold=0.3,
        threshold_area_tiers=[(300, 0.5)],
        threshold_area_scale=1.0,
    )

    assert out.effective_threshold == 0.3
    assert out.low_area_units == 100
    assert out.binary.sum() == 100


def test_hysteresis_keeps_only_components_touching_high_confidence_core():
    mask = np.zeros((12, 25), dtype=np.uint8)
    mask[1:11, 1:11] = 90
    mask[4:7, 4:7] = 220
    mask[1:11, 14:24] = 90

    out = binarize_mask_uint8(
        mask,
        threshold=0.3,
        hysteresis_high_threshold=0.7,
        hysteresis_min_core_area_px=1,
    )

    assert out.core_pixel_count == 9
    assert out.binary[:, :12].sum() == 100
    assert out.binary[:, 12:].sum() == 0


def test_hysteresis_drops_mask_when_no_high_confidence_core():
    mask = np.full((10, 10), 90, dtype=np.uint8)

    out = binarize_mask_uint8(
        mask,
        threshold=0.3,
        hysteresis_high_threshold=0.7,
        hysteresis_min_core_area_px=1,
    )

    assert out.core_pixel_count == 0
    assert out.binary.sum() == 0


def test_vectorize_chip_mask_uses_physical_area_tiers():
    mask = np.full((20, 20), 100, dtype=np.uint8)
    transform = Affine(0.5, 0.0, 0.0, 0.0, -0.5, 0.0)

    out = vectorize_chip_mask(
        mask,
        (0, 0),
        threshold=0.3,
        window_transform=transform,
        source_crs="EPSG:32735",
        threshold_area_tiers=[(90, 0.5)],
        threshold_area_scale=affine_pixel_area(transform),
    )

    assert out.effective_threshold == 0.5
    assert out.geoms == []


def test_pixel_or_hysteresis_breaks_weak_bridge_between_detections():
    crop_a = np.full((10, 10), 220, dtype=np.uint8)
    crop_b = np.full((10, 10), 220, dtype=np.uint8)
    bridge = np.full((10, 10), 90, dtype=np.uint8)
    dets = [
        {"mask_crop_uint8": crop_a, "source_offset": (0, 0), "score": 0.9, "label": 1},
        {"mask_crop_uint8": bridge, "source_offset": (10, 0), "score": 0.8, "label": 1},
        {"mask_crop_uint8": crop_b, "source_offset": (20, 0), "score": 0.85, "label": 1},
    ]

    no_hysteresis = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20,
        raster_width=40,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
    )
    with_hysteresis = paint_and_vectorize_pixel_or(
        dets,
        raster_height=20,
        raster_width=40,
        source_transform=_identity_transform(),
        source_crs="EPSG:32735",
        mask_threshold=0.3,
        hysteresis_high_threshold=0.7,
        hysteresis_min_core_area_px=1,
    )

    assert len(no_hysteresis) == 1
    assert len(with_hysteresis) == 2


def test_parse_threshold_area_tiers_accepts_dicts_and_sorts_desc():
    out = parse_threshold_area_tiers([
        {"min_area_m2": 100, "threshold": 0.45},
        {"min_area_m2": 200, "threshold": 0.55},
    ])

    assert out == ((200.0, 0.55), (100.0, 0.45))
