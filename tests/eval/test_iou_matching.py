"""Unit tests for the extracted IoU matcher (``core.eval_matching``).

These pin the F1 main-judge logic that was lifted verbatim out of
``detect_and_evaluate.py`` (candidate #1 of the 2026-06-12 architecture review).
Pure synthetic shapely fixtures only — no real GeoTIFFs, no GPU.

Coverage:
  * basic TP / FP / FN counting,
  * IoU threshold boundary (>= is inclusive),
  * the ``installation`` profile's pred-side many-to-one merge in BOTH modes
    (``merge_preds=True`` unions overlapping preds; ``merge_preds=False`` keeps
    strict 1:1),
  * empty inputs,
  * the re-export shim: ``from detect_and_evaluate import iou_matching`` is the
    same object as ``core.eval_matching.iou_matching`` (no duplicate impl).
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.eval_matching import compute_iou, iou_matching  # noqa: E402


def sq(x0, y0, s):
    """Axis-aligned square with lower-left corner (x0, y0) and side s."""
    return box(x0, y0, x0 + s, y0 + s)


def rect(x0, y0, w, h):
    """Axis-aligned rectangle with lower-left corner (x0, y0), width w, height h."""
    return box(x0, y0, x0 + w, y0 + h)


def gdf(*geoms):
    return gpd.GeoDataFrame(geometry=list(geoms))


# ─────────────────────────────────────────────────────────────────────────
# compute_iou
# ─────────────────────────────────────────────────────────────────────────
def test_compute_iou_identical_is_one():
    assert compute_iou(sq(0, 0, 10), sq(0, 0, 10)) == pytest.approx(1.0)


def test_compute_iou_disjoint_is_zero():
    assert compute_iou(sq(0, 0, 10), sq(100, 100, 10)) == 0.0


def test_compute_iou_half_overlap():
    # sq A area 100, sq B area 100, intersection 50 -> IoU = 50 / 150.
    a = sq(0, 0, 10)
    b = sq(5, 0, 10)
    assert compute_iou(a, b) == pytest.approx(50.0 / 150.0)


def test_compute_iou_empty_geometry_is_zero():
    from shapely.geometry import Polygon
    assert compute_iou(Polygon(), sq(0, 0, 10)) == 0.0


# ─────────────────────────────────────────────────────────────────────────
# basic TP / FP / FN
# ─────────────────────────────────────────────────────────────────────────
def test_basic_tp_fp_fn_strict():
    gt = gdf(sq(0, 0, 10), sq(100, 0, 10))     # GT1 has no matching pred
    pred = gdf(sq(1, 1, 9), sq(500, 500, 10))  # pred0 hits GT0, pred1 is a far FP
    r = iou_matching(gt, pred, iou_threshold=0.3, merge_preds=False)
    assert r["tp"] == 1
    assert r["fn"] == 1            # GT1 unmatched
    assert r["fp"] == 1            # pred1 unmatched
    assert r["matched_gt_indices"] == {0}
    assert r["matched_pred_indices"] == {0}
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(0.5)
    assert len(r["iou_scores"]) == 1


def test_perfect_match_all_modes():
    gt = gdf(sq(0, 0, 10))
    pred = gdf(sq(0, 0, 10))
    for merge in (True, False):
        r = iou_matching(gt, pred, iou_threshold=0.3, merge_preds=merge)
        assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
        assert r["f1"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────
# IoU threshold boundary (>= is inclusive)
# ─────────────────────────────────────────────────────────────────────────
def test_threshold_boundary_inclusive():
    a = sq(0, 0, 10)
    b = sq(1, 1, 9)
    iou_exact = compute_iou(a, b)
    gt = gdf(a)
    pred = gdf(b)
    for merge in (True, False):
        # exactly at threshold -> match (>=)
        at = iou_matching(gt, pred, iou_threshold=iou_exact, merge_preds=merge)
        assert at["tp"] == 1, f"merge={merge}: at-threshold should match"
        # just above threshold -> no match
        above = iou_matching(gt, pred, iou_threshold=iou_exact + 1e-9, merge_preds=merge)
        assert above["tp"] == 0, f"merge={merge}: above-threshold should not match"


# ─────────────────────────────────────────────────────────────────────────
# installation-profile pred-side many-to-one merge
# ─────────────────────────────────────────────────────────────────────────
def test_merge_mode_unions_fragments_into_one_tp():
    # One GT square (10x10) tiled exactly by two non-overlapping pred halves
    # (left 5x10, right 5x10). Each half alone is IoU 0.5; their union == GT.
    gt = gdf(sq(0, 0, 10))
    pred = gdf(rect(0, 0, 5, 10), rect(5, 0, 5, 10))  # two halves (left / right)
    r = iou_matching(gt, pred, iou_threshold=0.6, merge_preds=True)
    # Union of the two halves == GT -> IoU 1.0, both preds consumed as the TP.
    assert r["tp"] == 1
    assert r["fp"] == 0
    assert r["fn"] == 0
    assert r["matched_pred_indices"] == {0, 1}
    assert r["iou_scores"][0] == pytest.approx(1.0)


def test_strict_mode_does_not_union_fragments():
    # Same geometry as above, but strict 1:1: each half is IoU 0.5 with the GT.
    gt = gdf(sq(0, 0, 10))
    pred = gdf(rect(0, 0, 5, 10), rect(5, 0, 5, 10))
    # threshold 0.6 -> neither half qualifies alone (each IoU 0.5) -> no match.
    r_high = iou_matching(gt, pred, iou_threshold=0.6, merge_preds=False)
    assert r_high["tp"] == 0
    assert r_high["fn"] == 1
    assert r_high["fp"] == 2
    # threshold 0.5 -> one half matches (greedy), the other half is a spare FP.
    r_low = iou_matching(gt, pred, iou_threshold=0.5, merge_preds=False)
    assert r_low["tp"] == 1
    assert r_low["fp"] == 1   # the second half left over
    assert r_low["fn"] == 0


def test_merge_mode_greedy_does_not_double_assign_preds():
    # Two adjacent GT squares; a single wide pred straddles both. In merge mode
    # the wide pred can only be consumed by ONE gt (greedy IoU-descending), the
    # other gt becomes a FN.
    gt = gdf(sq(0, 0, 10), sq(10, 0, 10))
    pred = gdf(sq(0, 0, 20))  # one pred over both GTs (each IoU 100/400 = 0.25)
    r = iou_matching(gt, pred, iou_threshold=0.2, merge_preds=True)
    assert r["tp"] == 1
    assert r["fn"] == 1
    assert r["fp"] == 0       # the single pred was consumed by the one TP
    assert r["matched_pred_indices"] == {0}


# ─────────────────────────────────────────────────────────────────────────
# match details
# ─────────────────────────────────────────────────────────────────────────
def test_return_match_details_merge():
    gt = gdf(sq(0, 0, 10))
    pred = gdf(rect(0, 0, 5, 10), rect(5, 0, 5, 10))
    r = iou_matching(gt, pred, iou_threshold=0.6, merge_preds=True,
                     return_match_details=True)
    assert "match_details" in r
    assert len(r["match_details"]) == 1
    d = r["match_details"][0]
    assert d["gt_idx"] == 0
    assert d["pred_indices"] == {0, 1}
    assert d["iou"] == pytest.approx(1.0)
    assert d["gt_area"] == pytest.approx(100.0)
    assert d["pred_area"] == pytest.approx(100.0)
    assert d["intersection_area"] == pytest.approx(100.0)


def test_return_match_details_strict():
    gt = gdf(sq(0, 0, 10))
    pred = gdf(sq(0, 0, 10))
    r = iou_matching(gt, pred, iou_threshold=0.3, merge_preds=False,
                     return_match_details=True)
    assert len(r["match_details"]) == 1
    d = r["match_details"][0]
    assert d["pred_indices"] == {0}
    assert d["iou"] == pytest.approx(1.0)


def test_no_match_details_key_when_not_requested():
    gt = gdf(sq(0, 0, 10))
    pred = gdf(sq(0, 0, 10))
    r = iou_matching(gt, pred, return_match_details=False)
    assert "match_details" not in r


# ─────────────────────────────────────────────────────────────────────────
# empty inputs
# ─────────────────────────────────────────────────────────────────────────
def test_empty_pred_all_fn():
    gt = gdf(sq(0, 0, 10), sq(100, 0, 10))
    empty = gpd.GeoDataFrame(geometry=[])
    for merge in (True, False):
        r = iou_matching(gt, empty, merge_preds=merge)
        assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 2)
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0
        assert r["f1"] == 0.0


def test_empty_gt_all_fp():
    pred = gdf(sq(0, 0, 10), sq(100, 0, 10))
    empty = gpd.GeoDataFrame(geometry=[])
    for merge in (True, False):
        r = iou_matching(empty, pred, merge_preds=merge)
        assert (r["tp"], r["fp"], r["fn"]) == (0, 2, 0)
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0
        assert r["f1"] == 0.0


def test_empty_both():
    empty = gpd.GeoDataFrame(geometry=[])
    for merge in (True, False):
        r = iou_matching(empty, empty, merge_preds=merge)
        assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 0)
        assert r["f1"] == 0.0


# ─────────────────────────────────────────────────────────────────────────
# default arguments
# ─────────────────────────────────────────────────────────────────────────
def test_defaults_are_merge_true_thr_0_3():
    # Two opposite-corner 5x5 fragments of a 10x10 GT: each alone is IoU 0.25
    # (< 0.3 -> would fail strict), but their union is IoU 0.5 (>= 0.3). With
    # the defaults (merge_preds=True, iou_threshold=0.3) this is one TP; under
    # strict it would be zero. Pins the default arguments.
    gt = gdf(sq(0, 0, 10))
    pred = gdf(sq(0, 0, 5), sq(5, 5, 5))  # bottom-left + top-right corners
    r_default = iou_matching(gt, pred)
    assert r_default["tp"] == 1
    assert r_default["matched_pred_indices"] == {0, 1}
    # Sanity: same geometry under strict yields no match (defaults differ).
    r_strict = iou_matching(gt, pred, merge_preds=False)
    assert r_strict["tp"] == 0


# ─────────────────────────────────────────────────────────────────────────
# re-export shim: no duplicate implementation
# ─────────────────────────────────────────────────────────────────────────
def test_detect_and_evaluate_reexports_same_objects():
    import detect_and_evaluate as dae
    import core.eval_matching as em
    assert dae.iou_matching is em.iou_matching
    assert dae.compute_iou is em.compute_iou
