"""The 2026-09-28 learning-curve draw is deterministic and stays in its strata."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot.learning_curve_select import (  # noqa: E402
    EMPTY_PRE_COUNTS,
    INVENTORY_CSV,
    SEEN_HOLDOUT,
    draw_learning_curve,
    load_and_draw,
    stratum_name,
)

TRAIN_30 = (
    "CPT2932", "CPT3468", "CPT3635", "CPT3246", "CPT3638", "CPT3695", "CPT3734", "CPT2174",
    "CPT2373", "CPT2182", "CPT2718", "CPT1506", "CPT2496", "CPT4235", "CPT2776", "CPT2997",
    "CPT2216", "CPT2332", "CPT2040", "CPT2390", "CPT2167", "CPT2729", "CPT2406", "CPT3021",
    "CPT2330", "CPT3802", "CPT2207", "CPT2616", "CPT3194", "CPT3865",
)
TRAIN_10 = (
    "CPT2932", "CPT3468", "CPT3635",
    "CPT2373", "CPT2182", "CPT2718",
    "CPT2216", "CPT2332",
    "CPT2330", "CPT3802",
)
HOLDOUT = (
    "CPT3247", "CPT2119", "CPT2714",
    "CPT2308", "CPT2957", "CPT4369",
    "CPT2035", "CPT3577", "CPT3758",
)


def test_fresh_draw_matches_the_locked_ids():
    if not INVENTORY_CSV.exists() or not EMPTY_PRE_COUNTS.exists():
        pytest.skip("citywide inventory or empty-grid counts are not on this machine")
    first = load_and_draw()
    second = load_and_draw()
    assert first == second
    assert first.train_30 == TRAIN_30
    assert first.train_10 == TRAIN_10
    assert first.holdout == HOLDOUT
    assert set(TRAIN_10) < set(TRAIN_30)
    assert set(TRAIN_30).isdisjoint(HOLDOUT)
    assert set(SEEN_HOLDOUT).isdisjoint(set(TRAIN_30) | set(HOLDOUT))
    assert first.population["occupied"] == 1301
    assert first.population["gap"] == 551


def test_locked_ids_sit_in_the_stratum_they_were_drawn_from():
    if not INVENTORY_CSV.exists() or not EMPTY_PRE_COUNTS.exists():
        pytest.skip("citywide inventory or empty-grid counts are not on this machine")
    from scripts.gate_pilot.learning_curve_select import load_inputs

    delivered, pre_counts = load_inputs()
    expected = {
        "CPT2932": "dense", "CPT3468": "dense", "CPT3635": "dense",
        "CPT3246": "dense", "CPT3638": "dense", "CPT3695": "dense",
        "CPT3734": "dense", "CPT2174": "dense",
        "CPT2373": "mid", "CPT2182": "mid", "CPT2718": "mid",
        "CPT3247": "dense", "CPT2119": "dense", "CPT2714": "dense",
        "CPT2308": "mid", "CPT2957": "mid", "CPT4369": "mid",
        "CPT2216": "low", "CPT2332": "low",
        "CPT2330": "empty", "CPT3802": "empty",
        "CPT2035": "empty", "CPT3577": "empty", "CPT3758": "empty",
    }
    for grid_id, name in expected.items():
        count = delivered.get(grid_id)
        pre = pre_counts.get(grid_id)
        assert stratum_name(count, pre) == name


def test_sampler_rejects_a_forced_grid_that_leaves_its_stratum():
    delivered = {"CPT2932": 10, "CPT2373": 35, "CPT2182": 35, "CPT1238": 183}
    empty = {"CPT2330": 10, "CPT2172": 7}
    with pytest.raises(ValueError, match="CPT2932"):
        draw_learning_curve(delivered, empty)
