import numpy as np, geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_origin
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xtile_merge import TileIn, merge_tiles, per_tile_masks, polygons, assign_owner, global_origin

def two_tiles(seed_left=True):
    a = np.zeros((10, 10), np.float32); b = np.zeros((10, 10), np.float32)
    a[3:6, 6:10] = 0.6; b[3:6, 0:4] = 0.6
    (a if seed_left else b)[4, 7 if seed_left else 2] = 0.95
    return [TileIn('L', 0, 0, lambda: a), TileIn('R', 10, 0, lambda: b)]

def test_straddle_one_object_both_sides():
    m, ids, objs = merge_tiles(two_tiles(), 'h0.9_0.5')
    assert len(objs) == 1 and m['L'].sum() == 12 and m['R'].sum() == 12
    assert set(np.unique(ids['L'])) - {0} == set(np.unique(ids['R'])) - {0}
    assert list(objs.values())[0]['n_tiles'] == 2 and list(objs.values())[0]['n_px'] == 24

def test_per_tile_truncates():
    m = per_tile_masks(two_tiles(seed_left=False), 'h0.9_0.5')
    assert m['L'].sum() == 0 and m['R'].sum() == 12

def test_diagonal_8conn():
    a = np.zeros((4, 4), np.float32); b = np.zeros((4, 4), np.float32); a[3, 3] = 0.95; b[0, 0] = 0.6
    _, _, objs = merge_tiles([TileIn('A', 0, 0, lambda: a), TileIn('B', 4, 4, lambda: b)], 'h0.9_0.5')
    assert len(objs) == 1

def test_no_merge_with_gap():
    a = np.zeros((4, 4), np.float32); b = np.zeros((4, 4), np.float32); a[1, 3] = 0.95; b[1, 1] = 0.95
    _, _, objs = merge_tiles([TileIn('A', 0, 0, lambda: a), TileIn('B', 4, 0, lambda: b)], 't0.5')
    assert len(objs) == 2

def test_polygon_and_single_owner():
    tiles = two_tiles(); _, ids, _ = merge_tiles(tiles, 'h0.9_0.5')
    tr = {'L': from_origin(0, 10, 1, 1), 'R': from_origin(10, 10, 1, 1)}
    assert global_origin(tr['R'], 1.0) == (10, -10)
    g = polygons(ids, tr, 'EPSG:32734'); assert len(g) == 1 and abs(g.area.iloc[0] - 24) < 1e-6
    cells = gpd.GeoDataFrame(dict(gridcell_id=['C1', 'C2']), geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs='EPSG:32734')
    o = assign_owner(g, cells); assert o.owner_cell.notna().all() and len(o) == 1

def test_threshold_cfg_matches_plain():
    p = np.random.default_rng(0).random((20, 20)).astype(np.float32)
    m, _, _ = merge_tiles([TileIn('A', 0, 0, lambda: p)], 't0.5'); assert (m['A'] == (p >= 0.5)).all()

# --- r2 tests for Codex r1 findings ---
import pytest
from xtile_merge import merge_plan, plan_tile_ids, check_no_overlap

def test_overlap_detected_full_extent():
    a = np.zeros((5, 5), np.float32); a[2, 2] = 0.95
    with pytest.raises(ValueError):
        merge_tiles([TileIn('A', 0, 0, lambda: a), TileIn('B', 0, 0, lambda: a)], 't0.5')
    z = np.zeros((5, 5), np.float32)
    with pytest.raises(ValueError):
        merge_tiles([TileIn('A', 0, 0, lambda: z), TileIn('B', 3, 3, lambda: z)], 't0.5')
    check_no_overlap([(0, 0, 5, 5, 'A'), (5, 0, 10, 5, 'B'), (0, 5, 5, 10, 'C')])

def test_diagonal_polygon_valid():
    a = np.array([[0.95, 0], [0, 0.95]], np.float32)
    _, ids, _ = merge_tiles([TileIn('A', 0, 0, lambda: a)], 't0.5')
    g = polygons(ids, {'A': from_origin(0, 2, 1, 1)}, 'EPSG:32734')
    assert len(g) == 1 and g.is_valid.all() and abs(g.area.iloc[0] - 2) < 1e-9

def test_n_tiles_dedup():
    a = np.zeros((6, 6), np.float32); b = np.zeros((6, 6), np.float32)
    a[0, 5] = 0.95; a[4, 5] = 0.95; b[0:5, 0] = 0.6
    _, _, objs = merge_tiles([TileIn('A', 0, 0, lambda: a), TileIn('B', 6, 0, lambda: b)], 'h0.9_0.5')
    assert len(objs) == 1 and list(objs.values())[0]['n_tiles'] == 2

def test_plan_matches_merge_tiles():
    rng = np.random.default_rng(1)
    arrs = {f'T{i}{j}': rng.random((16, 16)).astype(np.float32) ** 3 for i in range(3) for j in range(3)}
    tiles = [TileIn(k, int(k[1]) * 16, int(k[2]) * 16, (lambda v: lambda: v)(v)) for k, v in arrs.items()]
    m, ids, _ = merge_tiles(tiles, 'h0.9_0.5'); plan = merge_plan(tiles, 'h0.9_0.5')
    for t in tiles:
        assert (plan_tile_ids(plan, t) == ids[t.key]).all()
    big = np.block([[arrs[f'T{i}{j}'] for i in range(3)] for j in range(3)])
    from scipy import ndimage
    lab, nl = ndimage.label(big >= 0.5, np.ones((3, 3), bool))
    keep = np.r_[False, ndimage.maximum(big, lab, np.arange(1, nl + 1)) >= 0.9][lab]
    whole = np.block([[m[f'T{i}{j}'] for i in range(3)] for j in range(3)])
    assert (whole == keep).all()

def test_owner_tie_lowest_and_nearest():
    cells = gpd.GeoDataFrame(dict(gridcell_id=['C2', 'C1']), geometry=[box(10, 0, 20, 10), box(0, 0, 10, 10)], crs='EPSG:32734')
    objs = gpd.GeoDataFrame(dict(oid=[1, 2]), geometry=[box(9.5, 4, 10.5, 6), box(25, 4, 26, 6)], crs='EPSG:32734')
    o = assign_owner(objs, cells)
    assert o.set_index('oid').owner_cell.to_dict() == {1: 'C1', 2: 'C2'}
