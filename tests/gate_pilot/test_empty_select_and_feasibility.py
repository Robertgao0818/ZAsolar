"""Pure checks for the locked empty-grid draw and the step-0 stop rule."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
from shapely.geometry import box

from scripts.gate_pilot.empty_select import select_empty_grids
from scripts.gate_pilot.feasibility import g2_minimum_pass
from scripts.gate_pilot.intersect import chips_hit_mask, iou_component_count
from scripts.gate_pilot.step1_imagery_parity import parity_decision, parse_detect_rate


def test_draw_is_stable_and_comes_from_the_upper_half():
    records = [
        ("CPT0001", 1),
        ("CPT0002", 4),
        ("CPT0003", 4),
        ("CPT0004", 9),
        ("CPT0005", 0),
        ("CPT0006", 2),
    ]
    first = select_empty_grids(records, seed=20260923)
    second = select_empty_grids(records, seed=20260923)
    assert first == second
    assert first.e_train in first.draw_population
    assert first.e_test in first.draw_population
    assert first.e_train != first.e_test
    assert not first.used_fallback
    # Sorted by count then id: 1, 2, 4/CPT0002, 4/CPT0003, 9. Upper half
    # starts at index 5//2 = 2, so the two count-4 grids and the count-9 grid.
    assert first.upper_half == ("CPT0002", "CPT0003", "CPT0004")
    assert first.cutoff_pre_count == 4
    assert "CPT0005" not in first.upper_half


def test_two_candidate_pool_falls_back_to_both():
    drawn = select_empty_grids([("CPT0002", 3), ("CPT0001", 8)], seed=20260923)
    assert drawn.used_fallback
    assert drawn.draw_population == ("CPT0001", "CPT0002")
    assert {drawn.e_train, drawn.e_test} == {"CPT0001", "CPT0002"}


def test_written_stop_rule_uses_the_continuous_product():
    # 0.99 * (420 / 2058) = 0.202137... > 0.20
    blocked = g2_minimum_pass(2058, 420)
    assert blocked["stop"]
    assert blocked["continuous_min_pass"] > 0.20

    # 0.99 * (400 / 2058) = 0.192419... <= 0.20
    open_case = g2_minimum_pass(2058, 400)
    assert not open_case["stop"]
    assert open_case["integer_k_min"] == 396
    assert open_case["integer_min_pass"] == 396 / 2058


def test_shared_edge_has_no_area_and_does_not_count():
    chips = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )
    touching = gpd.GeoDataFrame(geometry=[box(1, 0, 2, 1)], crs="EPSG:4326")
    inside = gpd.GeoDataFrame(geometry=[box(0.2, 0.2, 0.4, 0.4)], crs="EPSG:4326")
    assert chips_hit_mask(chips, touching).tolist() == [False, False]
    assert chips_hit_mask(chips, inside).tolist() == [True, False]


def test_hit_test_reprojects_from_the_metric_crs():
    chips = gpd.GeoDataFrame(
        geometry=[box(18.37, -34.20, 18.372, -34.198)],
        crs="EPSG:4326",
    )
    inside = gpd.GeoDataFrame(
        geometry=[box(18.3705, -34.1995, 18.3710, -34.1990)],
        crs="EPSG:4326",
    ).to_crs(32734)
    outside = gpd.GeoDataFrame(
        geometry=[box(19.0, -33.0, 19.001, -32.999)],
        crs="EPSG:4326",
    ).to_crs(32734)
    assert chips_hit_mask(chips, inside).tolist() == [True]
    assert chips_hit_mask(chips, outside).tolist() == [False]


def test_iou_components_link_only_the_overlapping_pair():
    frame = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 2), box(1, 1, 3, 3), box(10, 10, 12, 12)],
        crs="EPSG:32734",
    )
    assert iou_component_count(frame, 0.10) == 2
    assert iou_component_count(frame, 0.50) == 3


def test_parity_rule_is_inclusive_at_the_published_cut():
    exact = parity_decision(100, 105, 0.95, 0.95)
    assert exact["pass"]
    assert exact["count_rel_diff"] == 0.05
    short = parity_decision(200, 180, 0.96, 0.94)
    assert not short["pass"]


def test_detect_log_rate_uses_the_last_line():
    text = (
        "[detect_direct] 10 chips → 1 detections in 2.0s (5.00 chips/s, peak VRAM 1.00 GB)\n"
        "[detect_direct] 2058 chips → 253 detections in 80.0s (25.72 chips/s, peak VRAM 4.00 GB)\n"
    )
    parsed = parse_detect_rate(text)
    assert parsed is not None
    assert parsed["n_chips"] == 2058
    assert parsed["chips_per_s"] == 25.72


def test_integer_k_min_is_ceil_of_99_percent():
    assert g2_minimum_pass(100, 1)["integer_k_min"] == 1
    assert g2_minimum_pass(100, 100)["integer_k_min"] == 99
    assert g2_minimum_pass(200, 101)["integer_k_min"] == 100
    assert g2_minimum_pass(10, 0)["integer_k_min"] == 0
    assert not g2_minimum_pass(10, 0)["stop"]
