"""Locked folds and rule-A tau for the 2026-09-28 learning curve."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import shapely

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot.learning_curve_select import SELECTION_PATH  # noqa: E402
from scripts.gate_pilot.learning_curve_train import (  # noqa: E402
    binding_tau_for_polygon,
    active_folds,
    deferred_grids,
    folds_for_arm,
    folds_from_selection,
    interim_reading,
    losses_for_polygon,
    render_interim_section,
    render_result_section,
    s4_reading,
    select_tau_from_bindings,
)

EXPECTED = {
    0: {"CPT2932", "CPT3695", "CPT2373", "CPT4235", "CPT2216", "CPT2729", "CPT2330", "CPT3865"},
    1: {"CPT3468", "CPT3734", "CPT2182", "CPT2776", "CPT2332", "CPT2406", "CPT3802"},
    2: {"CPT3635", "CPT2174", "CPT2718", "CPT2997", "CPT2040", "CPT3021", "CPT2207"},
    3: {"CPT3246", "CPT1506", "CPT2390", "CPT2616"},
    4: {"CPT3638", "CPT2496", "CPT2167", "CPT3194"},
}
TEN_BY_FOLD = {
    0: {"CPT2932", "CPT2373", "CPT2216", "CPT2330"},
    1: {"CPT3468", "CPT2182", "CPT2332", "CPT3802"},
    2: {"CPT3635", "CPT2718"},
}


def _selection() -> dict:
    if not SELECTION_PATH.exists():
        pytest.skip("learning-curve selection.json is not on this machine")
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def test_fold_table_matches_the_handoff():
    folds = folds_from_selection(_selection())
    got = {fold: set() for fold in range(5)}
    for grid_id, fold in folds.items():
        got[fold].add(grid_id)
    assert got == EXPECTED
    assert set().union(*EXPECTED.values()).isdisjoint({"CPT1238", "CPT2172", "CPT3247", "CPT2035"})


def test_ten_grid_arm_uses_only_folds_0_to_2():
    grouped = folds_for_arm(_selection(), "yolo10")
    assert set(grouped) == {0, 1, 2}
    assert {fold: set(grids) for fold, grids in grouped.items()} == TEN_BY_FOLD


def test_large_install_tau_is_the_score_that_completes_coverage():
    polygon = shapely.box(0, 0, 10, 4)
    left = shapely.box(0, 0, 6, 4)
    right = shapely.box(4, 0, 10, 4)
    scores = [0.9, 0.3]
    geoms = [left, right]
    tau = binding_tau_for_polygon(scores, geoms, polygon, 40.0)
    assert tau == 0.3
    whole, partial = losses_for_polygon(scores, geoms, polygon, 40.0, tau)
    assert (whole, partial) == (0, 0.0)
    higher = math.nextafter(tau, math.inf)
    whole_hi, partial_hi = losses_for_polygon(scores, geoms, polygon, 40.0, higher)
    assert whole_hi > 0 or partial_hi > 0


def test_small_install_only_needs_its_highest_chip():
    polygon = shapely.box(0, 0, 2, 2)
    tau = binding_tau_for_polygon(
        [0.2, 0.8],
        [polygon, shapely.box(50, 50, 51, 51)],
        polygon,
        4.0,
    )
    assert tau == 0.8


def test_uncovered_geometry_uses_the_all_chips_pass_shortcut():
    polygon = shapely.box(0, 0, 10, 4)
    chips = [shapely.box(0, 0, 9, 4), shapely.box(0, 0, 8, 4)]
    tau = binding_tau_for_polygon([0.9, 0.4], chips, polygon, 40.0)
    assert tau == 0.4
    assert losses_for_polygon([0.9, 0.4], chips, polygon, 40.0, tau) == (0, 0.0)
    higher = math.nextafter(0.4, math.inf)
    _whole, partial = losses_for_polygon([0.9, 0.4], chips, polygon, 40.0, higher)
    assert partial > 0


def test_zero_score_needed_for_coverage_fails_the_arm():
    polygon = shapely.box(0, 0, 10, 4)
    binding = binding_tau_for_polygon(
        [0.9, 0.0],
        [shapely.box(0, 0, 6, 4), shapely.box(4, 0, 10, 4)],
        polygon,
        40.0,
    )
    chosen = select_tau_from_bindings([binding])
    assert chosen["tau"] == 0.0
    assert chosen["loss_met"] is False


def test_global_tau_is_the_minimum_binding_and_empty_is_one():
    chosen = select_tau_from_bindings([0.8, 0.25, 0.9])
    assert chosen == {"tau": 0.25, "loss_met": True, "n_polygons": 3, "binding_min": 0.25}
    empty = select_tau_from_bindings([])
    assert empty["tau"] == 1.0 and empty["loss_met"] is True


def test_s4_reads_both_zeros_as_ten_grids_already_enough():
    assert "10 格已经够" in s4_reading(True, True)
    assert "加数据有用" in s4_reading(False, True)
    assert "没有把丢失条件救回来" in s4_reading(False, False)
    assert "约束更严" in s4_reading(True, False)


def test_result_section_leaves_the_speed_verdict_open():
    grid = {
        "stratum": "dense",
        "pass_rate": 0.1,
        "whole_loss": 0,
        "partial_area_ge_30_m2": 0.0,
    }
    arm = {
        "tau": 0.2,
        "oof_loss_met": True,
        "oof_pass_rate": 0.2,
        "whole_loss": 0,
        "partial_area_ge_30_m2": 0.0,
        "chips_per_s": 100.0,
        "local_relative_time_vs_20_24_cps": 0.2,
        "grids": {grid_id: dict(grid) for grid_id in (
            "CPT3247", "CPT2119", "CPT2714", "CPT2308", "CPT2957",
            "CPT4369", "CPT2035", "CPT3577", "CPT3758",
        )},
    }
    text = render_result_section({
        "arms": {"yolo10": arm, "yolo30": arm, "vit30": arm},
        "s4": {"yolo10_loss_met": True, "yolo30_loss_met": True},
    })
    assert "S1" in text
    assert "不判" in text
    assert "16.7" in text


def test_interim_comparison_drops_fold_4_and_says_so():
    folds = {fold: tuple(sorted(grids)) for fold, grids in EXPECTED.items()}
    kept = active_folds(folds)
    assert set(kept) == {0, 1, 2, 3}
    assert set(deferred_grids(folds)) == EXPECTED[4]
    assert "26" in interim_reading(True, True)
    assert "DINOv2" in interim_reading(False, True)
    assert "YOLO" in interim_reading(True, False)
    assert "都没有" in interim_reading(False, False)
    text = render_interim_section({
        "deferred_grids": ["CPT3638", "CPT2496", "CPT2167", "CPT3194"],
        "reading": interim_reading(True, False),
        "yolo10": {"tau": 0.2, "loss_met": True},
        "yolo30_folds_0_to_3": {
            "tau": 0.1, "loss_met": True, "whole_loss_at_tau": 0,
            "partial_area_ge_30_m2_at_tau": 0.0, "oof_pass_rate": 0.5,
        },
        "vit30_folds_0_to_3": {
            "tau": 0.0, "loss_met": False, "whole_loss_at_tau": 3,
            "partial_area_ge_30_m2_at_tau": 12.5, "oof_pass_rate": 0.2,
        },
    })
    assert "CPT3638" in text
    assert "26" in text
    assert "S1" in text
    assert "留出的 9 个格子没有评分" in text
    assert "第五折未跑" in text


def test_holdout_predict_refuses_without_a_frozen_tau(tmp_path, monkeypatch):
    from scripts.gate_pilot import learning_curve_train as train

    monkeypatch.setattr(train, "TRAIN_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="frozen_tau"):
        train._holdout_guard("yolo10", train.HOLDOUT)
    (tmp_path / "frozen_tau_yolo10.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="CPT1238"):
        train._holdout_guard("yolo10", ["CPT1238"])
