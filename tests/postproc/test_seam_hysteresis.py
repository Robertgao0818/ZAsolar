"""Constrained seam hysteresis. Seeds are the original accepts only."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from core.postproc import (
    DEFAULT_CONF_TIERED,
    apply_postproc_filters,
    load_postproc_config,
    rescue_seam_truncated,
)

ROOT = Path(__file__).resolve().parents[2]
ON = {
    "hysteresis_rescue": True,
    "hysteresis_s_min": 0.5,
    "hysteresis_min_area_m2": 200,
    "hysteresis_touch_m": 0.5,
    "hysteresis_edge_px": 2,
}


def _gdf(rows):
    copied = [dict(row) for row in rows]
    geoms = [box(*row.pop("bbox")) for row in copied]
    return gpd.GeoDataFrame(copied, geometry=geoms, crs="EPSG:32735")


def _meta(box_chip, *, window=(0, 0, 400, 400), valid=(0, 0, 400, 400), tif=(1000, 1000)):
    return {
        "_hyst_box": box_chip,
        "_hyst_window": window,
        "_hyst_valid": valid,
        "_hyst_tif_shape": tif,
    }


def _seed():
    return {
        "bbox": (0, 0, 10, 10),
        "chip_index": 0,
        "source_tif": "a.tif",
        "score": 0.9,
        "confidence": 0.9,
        "area_m2": 400,
        **_meta((100, 100, 140, 140)),
    }


def _candidate(**overrides):
    row = {
        "bbox": (10.2, 0, 25, 10),
        "chip_index": 1,
        "source_tif": "a.tif",
        "score": 0.5,
        "confidence": 0.5,
        "area_m2": 200,
        **_meta((0, 40, 30, 80)),
    }
    row.update(overrides)
    return row


def _ids(frame):
    return [
        (int(row.chip_index), bool(row.hysteresis_rescued))
        for row in frame.itertuples(index=False)
    ]


def test_rescue_happy_path():
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate()])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False), (1, True)]
    assert not any(str(column).startswith("_hyst_") for column in out.columns)


def test_same_chip_seed_is_rejected():
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate(chip_index=0, source_tif="a.tif")])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False)]


def test_different_tif_same_chip_index_is_allowed():
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate(chip_index=0, source_tif="b.tif")])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False), (0, True)]


def test_not_touching_edge_is_rejected():
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate(**_meta((100, 100, 140, 140)))])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False)]


def test_tif_boundary_counts_when_the_valid_window_edge_does_not():
    # Valid data is inset by 10px. The box sits on the tif edge, 5px
    # outside that inset, so only the tif-boundary rule can rescue it.
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate(**_meta(
        (0, 200, 5, 220),
        valid=(10, 10, 380, 380),
        tif=(400, 400),
    ))])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False), (1, True)]


def test_score_below_s_min_is_rejected():
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate(score=0.49, confidence=0.49)])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False)]


def test_area_below_200_is_rejected():
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate(area_m2=199)])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False)]


def test_chaining_is_prevented():
    accepted = _gdf([_seed()])
    rejected = _gdf([
        _candidate(chip_index=1, bbox=(10.2, 0, 20, 10)),
        _candidate(chip_index=2, bbox=(20.2, 0, 30, 10)),
    ])
    out = rescue_seam_truncated(accepted, rejected, ON)
    assert _ids(out) == [(0, False), (1, True)]


def test_disabled_config_is_a_noop():
    accepted = _gdf([_seed()])
    rejected = _gdf([_candidate()])
    assert rescue_seam_truncated(accepted, rejected, {}) is accepted
    assert rescue_seam_truncated(accepted, rejected, {"hysteresis_rescue": False}) is accepted
    assert "hysteresis_rescued" not in accepted.columns


def test_score_rejects_are_only_the_tiered_confidence_drop():
    """Area and elongation drops are not offered to the rescue."""
    assert DEFAULT_CONF_TIERED == [(200.0, 0.70), (100.0, 0.65), (0.0, 0.85)]
    gdf = _gdf([
        {"bbox": (0, 0, 1, 1), "area_m2": 250, "elongation": 2, "confidence": 0.9,
         "score": 0.9, "mean_r": 100, "mean_g": 100, "mean_b": 100},
        {"bbox": (2, 0, 3, 1), "area_m2": 250, "elongation": 2, "confidence": 0.6,
         "score": 0.6, "mean_r": 100, "mean_g": 100, "mean_b": 100},
        {"bbox": (4, 0, 5, 1), "area_m2": 3, "elongation": 2, "confidence": 0.99,
         "score": 0.99, "mean_r": 100, "mean_g": 100, "mean_b": 100},
        {"bbox": (6, 0, 7, 1), "area_m2": 50, "elongation": 20, "confidence": 0.99,
         "score": 0.99, "mean_r": 100, "mean_g": 100, "mean_b": 100},
    ])
    kept, _stats = apply_postproc_filters(gdf, {})
    kept_again, _stats2, rejected = apply_postproc_filters(
        gdf, {}, return_score_rejects=True,
    )
    assert list(kept.index) == list(kept_again.index)
    assert len(kept) == 1
    assert float(kept.iloc[0]["confidence"]) == 0.9
    assert len(rejected) == 1
    assert float(rejected.iloc[0]["confidence"]) == 0.6


def test_hysteresis_config_extends_v4_canonical_without_editing_it():
    canonical = json.loads((ROOT / "configs/postproc/v4_canonical.json").read_text())
    seam_path = ROOT / "configs/postproc/v4_canonical_seam_hysteresis.json"
    hyst = json.loads(seam_path.read_text())
    assert "hysteresis_rescue" not in canonical
    for key, value in canonical.items():
        assert hyst[key] == value
    assert hyst["hysteresis_rescue"] is True
    assert hyst["hysteresis_s_min"] == 0.5
    assert hyst["hysteresis_min_area_m2"] == 200
    assert hyst["hysteresis_touch_m"] == 0.5
    assert hyst["hysteresis_edge_px"] == 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_postproc_config(seam_path)
    assert not any("unknown postproc" in str(item.message) for item in caught)
    assert loaded["hysteresis_rescue"] is True
    assert loaded["pre_vector_score_threshold"] == 0.3

    # The pre-existing mask-pixel experiment keeps its own filename.
    mask = json.loads((ROOT / "configs/postproc/v4_canonical_hysteresis.json").read_text())
    assert mask["mask_threshold"] == 0.18
    assert mask["mask_hysteresis_high_threshold"] == 0.50
    assert mask["mask_hysteresis_min_core_area_px"] == 200
    assert mask["merge_mode"] == "pixel-or"
    assert "hysteresis_rescue" not in mask
