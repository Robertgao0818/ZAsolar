import sys
from pathlib import Path
import numpy as np, pytest

pytest.importorskip('geopandas'); pytest.importorskip('rasterio')
D = Path(__file__).resolve().parents[2] / 'scripts/production_lock_20261020/citywide_infer'
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent / 'ct_xtile_merge'))
import merge_city_v3 as V  # noqa: E402
import ct_city_infer_v2 as C  # noqa: E402
from xtile_merge import TileIn, merge_tiles  # noqa: E402

PX = 20; GSD = 5.0   # 100 m tiles of 20 px


class Loader:
    """picklable dict loader: tile -> float prob [PX,PX]"""
    def __init__(self, arrs): self.arrs = arrs
    def __call__(self, out, tile): return self.arrs[tile]


def name(i, j): return f'E{i * 100}_N{j * 100}.tif'


def run(arrs, tmp_path, sub='m'):
    G, s = V.run(sorted(arrs), None, tmp_path / sub, workers=0, loader=Loader(arrs), gsd=GSD)
    return G, s


def reference(arrs):
    """xtile_merge.merge_tiles over all tiles at once (the unmodified A3 rule) -> sorted list of (n_px, max_p)."""
    tiles = []
    for t, a in arrs.items():
        x, y = C.tile_xy(t)
        tiles.append(TileIn(t, x // 5, -(y + 100) // 5, (lambda a=a: a)))
    _, _, objs = merge_tiles(tiles, 'h0.9_0.5')
    return sorted((int(o['n_px']), round(float(o['max_p']), 5)) for o in objs.values())


def long_row(n_tiles, seed_tile):
    """one object crossing n_tiles tiles west->east along a row (Codex r3 long-object case: x=0..2000 m)."""
    arrs = {}
    for i in range(n_tiles):
        a = np.zeros((PX, PX), np.float32); a[9:11, :] = 0.6
        if i == seed_tile: a[10, 5] = 0.95
        arrs[name(i, 50)] = a
    return arrs


def test_long_object_across_blocks_counted_once(tmp_path):
    arrs = long_row(20, seed_tile=19)              # 2 km, crosses the x=1000 block edge, seed only at the far end
    G, s = run(arrs, tmp_path)
    assert len(G) == 1 and s['n_cross_block'] == 1 and int(G.n_tiles[0]) == 20 and int(G.n_blocks[0]) == 2
    assert int(G.n_px[0]) == 20 * 2 * PX and G.area_m2[0] == pytest.approx(20 * 2 * PX * GSD * GSD)
    assert sorted((int(n), round(float(m), 5)) for n, m in zip(G.n_px, G.max_p)) == reference(arrs)


def test_unseeded_long_object_dropped(tmp_path):
    arrs = long_row(15, seed_tile=-1)
    G, _ = run(arrs, tmp_path); assert len(G) == 0


def test_diagonal_corner_and_vertical_joins(tmp_path):
    a = np.zeros((PX, PX), np.float32); b = np.zeros((PX, PX), np.float32); c = np.zeros((PX, PX), np.float32)
    a[-1, -1] = 0.6; a[-3:, -3:] = 0.95          # bottom-right corner of tile (9,50)
    b[0, 0] = 0.6                                  # top-left of SE tile (10,49) -> joined only through the corner
    c[0, 5:8] = 0.6; a[-1, 5:8] = 0.6              # vertical join with S tile (9,49) through column 5..7 ... separate object
    a[-2, 5:8] = 0.92
    arrs = {name(9, 50): a, name(10, 49): b, name(9, 49): c}
    G, s = run(arrs, tmp_path)
    assert sorted((int(n), round(float(m), 5)) for n, m in zip(G.n_px, G.max_p)) == reference(arrs)
    assert s['n_cross_block'] == 2 and len(G) == 2 and G.oid.is_unique
    assert G.area_m2.sum() == pytest.approx(G.n_px.sum() * GSD * GSD)


def test_random_city_matches_global_reference_and_conserves_area(tmp_path):
    rng = np.random.default_rng(7); arrs = {}
    for i in range(8, 13):
        for j in range(48, 52):
            a = (rng.random((PX, PX)) ** 6).astype(np.float32)   # sparse blobs, some > 0.9
            arrs[name(i, j)] = a
    G, s = run(arrs, tmp_path)
    assert sorted((int(n), round(float(m), 5)) for n, m in zip(G.n_px, G.max_p)) == reference(arrs)
    assert G.oid.is_unique and G.area_m2.sum() == pytest.approx(G.n_px.sum() * GSD * GSD)
    assert all(G.geometry.is_valid)
    assert s['n_objects'] == len(G) and s['n_cross_block'] >= 1


def test_inside_block_objects_identical_to_v2_per_block(tmp_path):
    rng = np.random.default_rng(3); arrs = {}
    for i in range(1, 9):                    # block E0 only, away from its edges
        for j in range(21, 29):
            arrs[name(i, j)] = (rng.random((PX, PX)) ** 6).astype(np.float32)
    out = tmp_path / 'inf'
    for t, a in arrs.items():               # write as uint8 COGs both models (avg2 = same value)
        x, y = C.tile_xy(t); u = np.rint(a * 255).astype(np.uint8)
        from rasterio.transform import from_origin
        for m in C.MODELS:
            C.save_cog(C.out_path(out, m, 'pv', t), u, 'EPSG:32734', from_origin(x, y + 100, GSD, GSD).to_gdal())
    G, _ = V.run(sorted(arrs), out, tmp_path / 'v3', workers=0, gsd=GSD)
    by_xy = {C.tile_xy(t): t for t in arrs}
    old = C.merge_block(out, (0, 2000), by_xy, gsd=GSD)
    key = lambda g: sorted((int(n), round(float(a), 3)) for n, a in zip(g.n_px, g.geometry.area))
    assert key(G) == key(old)
    assert G.geometry.union_all().symmetric_difference(old.geometry.union_all()).area == pytest.approx(0, abs=1e-6)


def test_refuses_nonempty_dest(tmp_path):
    (tmp_path / 'm').mkdir(); (tmp_path / 'm' / 'x').write_text('1')
    with pytest.raises(FileExistsError): run({name(0, 0): np.zeros((PX, PX), np.float32)}, tmp_path)
