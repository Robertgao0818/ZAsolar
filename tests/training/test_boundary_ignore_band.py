"""CPU tests for the C-3(b) area-adaptive boundary ignore band.

Covers:
- area tier selection (small/medium/large by pixel area) + S-class force-large.
- per-pixel weight map: band gets the source boundary_w (R/S=0 ignore),
  foreground core + background stay 1.0 (R-class "band-ignore-core-supervised").
- band WIDTH grows with the tier (small < medium < large ring thickness).
- H (trusted) sources get an all-ones map (no band carved, full edge supervision).
- legacy-equivalence regression: the FIXED-band reference (a verbatim copy of
  train.py's _boundary_pixel_weights, the off-path code) reproduces the adaptive
  band when the adaptive width matches the fixed iters — proving the new lever
  is opt-in and the off-path is unchanged.
- threshold spec parsing + validation.

All CPU (numpy + cv2). No train.py import (CUDA-asserting).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.boundary_trust import boundary_w_map
from core.training.boundary_ignore_band import (
    BandConfig,
    adaptive_boundary_pixel_weights,
    parse_band_thresholds,
)


# ── Legacy reference: verbatim copy of train.py _boundary_pixel_weights ──
# This mirrors the OFF-path (fixed band_iters) code so the test can assert
# that the new module reduces to legacy behavior at a matching width.
_FIXED_BOUNDARY_W = boundary_w_map()


def _legacy_fixed_band(mask_np, label_source, band_iters=2):
    bw = _FIXED_BOUNDARY_W.get(label_source, 1.0)
    if bw == 1.0:
        return np.ones_like(mask_np, dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dil = cv2.dilate(mask_np, kernel, iterations=band_iters)
    ero = cv2.erode(mask_np, kernel, iterations=band_iters)
    band = (dil.astype(np.int8) ^ ero.astype(np.int8)) > 0
    out = np.ones_like(mask_np, dtype=np.float32)
    out[band] = bw
    return out


def _square_mask(side, total=None):
    total = total or (side + 20)
    m = np.zeros((total, total), dtype=np.uint8)
    off = (total - side) // 2
    m[off:off + side, off:off + side] = 1
    return m


# ── Tier selection ───────────────────────────────────────────────────────
def test_tier_selection_by_area():
    cfg = BandConfig()  # small<400, medium<2500, else large
    assert cfg.tier_iters(100.0, "reviewed_prediction") == 1
    assert cfg.tier_iters(1000.0, "reviewed_prediction") == 2
    assert cfg.tier_iters(5000.0, "reviewed_prediction") == 3


def test_s_class_forced_to_large_band_regardless_of_area():
    cfg = BandConfig()
    # tiny S-class instance still gets the widest (large) band
    assert cfg.tier_iters(10.0, "sam_refined_review") == cfg.large_iters
    assert cfg.tier_iters(10.0, "sam_added_true_fn") == cfg.large_iters


def test_tier_boundaries_are_half_open():
    cfg = BandConfig(small_max_area_px=400.0, medium_max_area_px=2500.0)
    assert cfg.tier_iters(399.9, "reviewed_prediction") == 1
    assert cfg.tier_iters(400.0, "reviewed_prediction") == 2   # == small_max → medium
    assert cfg.tier_iters(2499.9, "reviewed_prediction") == 2
    assert cfg.tier_iters(2500.0, "reviewed_prediction") == 3  # == medium_max → large


# ── R-class band-ignore-core-supervised semantics ────────────────────────
def test_r_class_band_zeroed_core_and_bg_supervised():
    # large R instance → 3px band, core + bg stay 1.0, band == 0.0
    m = _square_mask(100, total=140)  # area 10000 → large
    w = adaptive_boundary_pixel_weights(m, "reviewed_prediction")
    assert w.shape == m.shape
    # background far from edge is 1.0
    assert w[0, 0] == pytest.approx(1.0)
    # deep interior (core) is 1.0 — model still supervised on "panel here"
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=5).astype(bool)
    assert (w[eroded] == 1.0).all()
    # there IS an ignored band (some zeros)
    assert (w == 0.0).any()
    # every zero pixel lies in the edge ring (not in deep core, not far bg)
    band = (cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=3).astype(np.int8)
            ^ cv2.erode(m, np.ones((3, 3), np.uint8), iterations=3).astype(np.int8)) > 0
    assert np.array_equal(w == 0.0, band)


def test_h_class_gets_all_ones_no_band():
    m = _square_mask(40)
    for src in ("human_manual", "human_manual_sam_assisted", "sam_added_browser"):
        w = adaptive_boundary_pixel_weights(m, src)
        assert (w == 1.0).all(), f"{src} must keep full edge supervision"


def test_unknown_source_defaults_to_full_weight():
    m = _square_mask(40)
    w = adaptive_boundary_pixel_weights(m, "some_unknown_source")
    assert (w == 1.0).all()


def test_none_source_defaults_to_full_weight():
    m = _square_mask(40)
    w = adaptive_boundary_pixel_weights(m, None)
    assert (w == 1.0).all()


# ── Band width grows with tier ───────────────────────────────────────────
def test_band_width_increases_with_target_size():
    # Three R instances spanning the three tiers; count ignored (band) pixels.
    small = _square_mask(15, total=60)    # area 225  → small (1px)
    medium = _square_mask(45, total=90)   # area 2025 → medium (2px)
    large = _square_mask(80, total=140)   # area 6400 → large (3px)
    n_small = int((adaptive_boundary_pixel_weights(small, "reviewed_prediction") == 0.0).sum())
    n_medium = int((adaptive_boundary_pixel_weights(medium, "reviewed_prediction") == 0.0).sum())
    n_large = int((adaptive_boundary_pixel_weights(large, "reviewed_prediction") == 0.0).sum())
    # wider band → more ignored pixels (also perimeter grows, both push same way)
    assert n_small < n_medium < n_large


# ── Legacy-equivalence regression (off-path unchanged) ───────────────────
def test_adaptive_reduces_to_fixed_band_at_matching_width():
    """A medium R instance (adaptive width 2) must produce the SAME map as the
    legacy fixed band_iters=2 path — i.e. the off-path code is unchanged and the
    new lever only changes *which* width is chosen per instance."""
    m = _square_mask(45, total=90)  # area 2025 → medium → 2 iters
    adaptive = adaptive_boundary_pixel_weights(m, "reviewed_prediction")
    legacy = _legacy_fixed_band(m, "reviewed_prediction", band_iters=2)
    assert np.array_equal(adaptive, legacy)


def test_small_instance_band_narrower_than_legacy_fixed_2():
    """A small R instance gets 1px (adaptive) vs 2px (legacy fixed) → strictly
    fewer ignored pixels. This is the whole point of the lever: don't erase tiny
    sub-arrays with an over-wide fixed band."""
    m = _square_mask(15, total=60)  # area 225 → small → 1 iter
    adaptive = adaptive_boundary_pixel_weights(m, "reviewed_prediction")
    legacy = _legacy_fixed_band(m, "reviewed_prediction", band_iters=2)
    assert int((adaptive == 0.0).sum()) < int((legacy == 0.0).sum())


def test_h_class_identical_between_adaptive_and_legacy():
    m = _square_mask(40)
    assert np.array_equal(
        adaptive_boundary_pixel_weights(m, "human_manual"),
        _legacy_fixed_band(m, "human_manual"),
    )


# ── Threshold spec parsing ───────────────────────────────────────────────
def test_parse_thresholds_default_is_none_spec():
    cfg = parse_band_thresholds(None)
    assert cfg.small_max_area_px == BandConfig().small_max_area_px
    assert cfg.medium_max_area_px == BandConfig().medium_max_area_px


def test_parse_thresholds_custom():
    cfg = parse_band_thresholds("500,3000")
    assert cfg.small_max_area_px == 500.0
    assert cfg.medium_max_area_px == 3000.0


def test_parse_thresholds_rejects_bad_order():
    with pytest.raises(ValueError):
        parse_band_thresholds("3000,500")


def test_parse_thresholds_rejects_wrong_arity():
    with pytest.raises(ValueError):
        parse_band_thresholds("500")
    with pytest.raises(ValueError):
        parse_band_thresholds("100,200,300")
