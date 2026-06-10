"""Regression tests for the 2026-06-10 legacy post_conf/conf_tiered fix (F1-gap Tier A2 / C11).

Pre-fix behavior on the legacy path (detect_and_evaluate.py geoai path A):
as soon as predictions had an ``area_m2`` column, the hardcoded module-level
``CONF_TIERED`` was applied unconditionally; ``--postproc-config`` could not
inject ``conf_tiered`` at all (unknown key, silently ignored) and the config's
``post_conf_threshold`` was dead. These tests pin:

1. the legacy config parser now accepts ``conf_tiered``;
2. an injected ``conf_tiered`` actually takes effect (fails pre-fix);
3. default behavior (no config) is per-polygon identical to the old inline
   hardcoded logic;
4. the re-pinned ``configs/postproc/batch003_best_f1.json`` reproduces the
   pre-fix hardcoded behavior per polygon (re-pin correctness);
5. the documented legacy-vs-direct tier-iteration divergence (fall-through vs
   first-match-wins) stays exactly where documented: area>=200 m² with
   conf in [0.65, 0.70).
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect_and_evaluate import (  # noqa: E402
    CONF_TIERED,
    apply_conf_filter,
    load_postproc_config,
)
from core.postproc import apply_postproc_filters  # noqa: E402

REPINNED_CONFIG = ROOT / "configs" / "postproc" / "batch003_best_f1.json"


def _gdf(rows):
    geoms = [box(0, 0, 1, 1) for _ in rows]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:32734")


def _legacy_inline_conf_filter(pred_gdf, tiers):
    """Verbatim replica of the pre-2026-06-10 inline filter (oracle).

    Source: detect_and_evaluate.py:796-804 @ commit 0b6147e.
    """
    keep_mask = pd.Series(False, index=pred_gdf.index)
    for min_area, thresh in tiers:
        tier_mask = (pred_gdf["area_m2"] >= min_area) & ~keep_mask
        keep_mask |= tier_mask & (pred_gdf["confidence"] >= thresh)
    return pred_gdf[keep_mask].copy()


def _synthetic_frame():
    """Boundary-heavy synthetic predictions covering every tier edge."""
    rng = np.random.default_rng(20260610)
    rows = []
    # Dense boundary cases around tier edges (areas 99/100/101, 199/200/201;
    # confs at 0.649/0.65/0.651, 0.699/0.70/0.701, 0.849/0.85/0.851).
    for area in [1, 5, 50, 99, 100, 101, 150, 199, 200, 201, 500, 2000]:
        for conf in [0.10, 0.649, 0.65, 0.66, 0.699, 0.70, 0.71,
                     0.849, 0.85, 0.851, 0.92, 0.99]:
            rows.append({"area_m2": float(area), "confidence": float(conf)})
    # Random fuzz.
    for _ in range(500):
        rows.append({
            "area_m2": float(rng.uniform(0, 1000)),
            "confidence": float(rng.uniform(0, 1)),
        })
    return _gdf(rows)


def test_load_postproc_config_accepts_conf_tiered(tmp_path):
    cfg = tmp_path / "pp.json"
    cfg.write_text(
        '{"conf_tiered": [[200, 0.7], [100, 0.65], [0, 0.85]],'
        ' "post_conf_threshold": 0.92}',
        encoding="utf-8",
    )
    params = load_postproc_config(cfg)
    assert params["conf_tiered"] == [(200.0, 0.7), (100.0, 0.65), (0.0, 0.85)]
    assert params["post_conf_threshold"] == 0.92


def test_injected_conf_tiered_takes_effect():
    """Pre-fix this fails: no injection point existed; CONF_TIERED always won."""
    gdf = _gdf([
        {"area_m2": 50.0, "confidence": 0.60},   # default tiers: drop (needs 0.85)
        {"area_m2": 50.0, "confidence": 0.90},   # default tiers: keep
    ])
    strict, _ = apply_conf_filter(gdf, conf_tiered=[(0.0, 0.95)])
    assert len(strict) == 0
    loose, _ = apply_conf_filter(gdf, conf_tiered=[(0.0, 0.5)])
    assert len(loose) == 2
    default, _ = apply_conf_filter(gdf)
    assert len(default) == 1


def test_default_behavior_identical_to_old_inline_logic():
    gdf = _synthetic_frame()
    expected = _legacy_inline_conf_filter(gdf, CONF_TIERED)
    actual, _ = apply_conf_filter(gdf)
    assert list(actual.index) == list(expected.index)


def test_repinned_batch003_reproduces_hardcoded_behavior_per_polygon():
    """Acceptance check: the re-pinned preset == pre-fix hardcoded behavior."""
    params = load_postproc_config(REPINNED_CONFIG)
    assert "conf_tiered" in params, "batch003_best_f1.json must be re-pinned"
    gdf = _synthetic_frame()
    expected = _legacy_inline_conf_filter(gdf, CONF_TIERED)
    actual, _ = apply_conf_filter(gdf, conf_tiered=params["conf_tiered"])
    assert list(actual.index) == list(expected.index)


def test_fallthrough_divergence_zone_pinned():
    """Legacy fall-through keeps area>=200 & conf in [0.65,0.70); direct drops it.

    This divergence predates the fix (legacy hardcoded loop used ~keep_mask).
    Pinned here so any future unification is an explicit decision, not drift.
    """
    row = {"area_m2": 250.0, "confidence": 0.66,
           "elongation": 1.0, "mean_r": 100, "mean_g": 100, "mean_b": 100}
    gdf = _gdf([row])
    legacy_kept, _ = apply_conf_filter(gdf)
    assert len(legacy_kept) == 1, "legacy fall-through must keep it"
    direct_kept, _ = apply_postproc_filters(gdf, {})
    assert len(direct_kept) == 0, "direct first-match-wins must drop it"


def test_no_area_column_falls_back_to_post_conf():
    gdf = _gdf([{"confidence": 0.90}, {"confidence": 0.93}])
    out, desc = apply_conf_filter(gdf, conf_tiered=[(0.0, 0.5)],
                                  post_conf_threshold=0.92)
    assert len(out) == 1
    assert "0.92" in desc
