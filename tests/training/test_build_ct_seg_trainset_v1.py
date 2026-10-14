"""Unit tests for the CT segmentation training-set builder (pure helpers)."""
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from scripts.training.build_ct_seg_trainset_v1 import (
    BG, HN, IGNORE, PV, band_px, compose_mask, dedupe_polygons, split_grids,
)


def _sq(n, y0, y1, x0, x1):
    a = np.zeros((n, n), bool)
    a[y0:y1, x0:x1] = True
    return a


def test_band_px_rounding():
    assert band_px(0.3, 0.074) == 4
    assert band_px(0.15, 0.074) == 2
    assert band_px(0.0, 0.074) == 0
    assert band_px(0.01, 0.074) == 1


def test_compose_mask_values_and_priority():
    n = 60
    pv = _sq(n, 10, 30, 10, 30)
    hn = _sq(n, 20, 50, 20, 50)          # overlaps PV
    valid = np.ones((n, n), bool)
    valid[:, 55:] = False                # outside grid
    m = compose_mask(pv, np.zeros_like(pv), hn, valid, r_b=2, r_a=0)
    assert set(np.unique(m)) <= {BG, PV, HN, IGNORE}
    assert m[20, 20] == PV                # PV interior beats HN
    assert m[45, 45] == HN                # HN-only region
    assert m[10, 20] == IGNORE            # PV edge band
    assert m[8, 20] == IGNORE and m[6, 20] == BG
    assert (m[:, 55:] == IGNORE).all()    # outside imagery/grid
    assert m[49, 49] == HN                # no band around HN


def test_class_a_band_narrower():
    n = 40
    pv = _sq(n, 10, 30, 10, 30)
    z = np.zeros_like(pv)
    mb = compose_mask(pv, z, z, np.ones_like(pv), 4, 2)
    ma = compose_mask(z, pv, z, np.ones_like(pv), 4, 2)
    assert (mb == IGNORE).sum() > (ma == IGNORE).sum() > 0


def test_split_by_grid_stratified_and_deterministic():
    gb = {f"G{i:04d}": ("b1" if i < 20 else "b2") for i in range(26)}
    s1, s2 = split_grids(gb, 0.15, seed=0), split_grids(gb, 0.15, seed=0)
    assert s1 == s2
    assert set(s1.values()) == {"train", "val"}
    for b in ("b1", "b2"):
        assert any(s1[g] == "val" for g, bb in gb.items() if bb == b)


def test_dedupe_drops_near_duplicates_only():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(0.02, 0, 1.02, 1), box(5, 5, 6, 6)])
    assert len(dedupe_polygons(g)) == 2
