"""JHB tile names (grid-prefixed) and CRS taken from the inference outputs, for ct_city_infer_v2 + merge_city_v3."""
import sys
from pathlib import Path
import numpy as np, pytest

pytest.importorskip('geopandas'); pytest.importorskip('rasterio')
D = Path(__file__).resolve().parents[2] / 'scripts/production_lock_20261020/citywide_infer'
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent / 'ct_xtile_merge'))
import merge_city_v3 as V  # noqa: E402
import ct_city_infer_v2 as C  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

PX = 20; GSD = 5.0


def test_tile_xy_ct_unchanged_and_jhb_prefixed():
    assert C.tile_xy('E251300_N6230400.tif') == (251300, 6230400)
    assert C.tile_xy('G0772_E601162_N7099458.tif') == (601162, 7099458)
    assert C.tile_xy('G0772/G0772_E601162_N7099458.tif') == (601162, 7099458)
    assert C.tile_xy('JNB0001/JNB0001_E612000_N7114000.tif') == (612000, 7114000)
    assert C.block_of('JNB0001_E612300_N7114900.tif') == (612000, 7114000)
    for bad in ['g0772_E1_N2.tif', 'G0772_E1_N2_x.tif', 'E1_N2_G0772.tif', '_E1_N2.tif', 'G_E1_N2.tif']:
        with pytest.raises(ValueError): C.tile_xy(bad)


def test_unique_xy_refuses_same_corner_under_two_prefixes():
    with pytest.raises(ValueError): C.unique_xy(['G0001_E100_N200.tif', 'G0002_E100_N200.tif'])
    assert C.unique_xy(['E100_N200.tif', 'E200_N200.tif']) == {(100, 200): 'E100_N200.tif', (200, 200): 'E200_N200.tif'}


def _write(out, arrs, epsg):
    for t, a in arrs.items():
        x, y = C.tile_xy(t); u = np.rint(a * 255).astype(np.uint8)
        for m in C.MODELS:
            C.save_cog(C.out_path(out, m, 'pv', t), u, f'EPSG:{epsg}', from_origin(x, y + 100, GSD, GSD).to_gdal())


def _arrs(prefix):
    a = np.zeros((PX, PX), np.float32); a[5:15, 15:] = 0.95; b = np.zeros((PX, PX), np.float32); b[5:15, :6] = 0.95
    return {f'{prefix}E600000_N7100000.tif': a, f'{prefix}E600100_N7100000.tif': b}


def test_merge_v3_jhb_names_take_crs_from_outputs(tmp_path):
    arrs = _arrs('JNB0001/JNB0001_'); _write(tmp_path / 'inf', arrs, 32735)
    G, s = V.run(sorted(arrs), tmp_path / 'inf', tmp_path / 'm', workers=0, gsd=GSD, return_gdf=True)
    assert G.crs.to_epsg() == 32735 and len(G) == 1 and int(G.n_tiles.iloc[0]) == 2
    assert G.block.iloc[0] == 'E600000_N7100000'
    G2, _ = V.run(sorted(arrs), tmp_path / 'inf', tmp_path / 'm2', workers=0, gsd=GSD, return_gdf=True, crs='EPSG:32735')
    assert G2.crs.to_epsg() == 32735
    with pytest.raises(ValueError): V.run(sorted(arrs), tmp_path / 'inf', tmp_path / 'm3', workers=0, gsd=GSD, crs='EPSG:32734')


def test_merge_v3_ct_crs_is_32734(tmp_path):
    arrs = _arrs(''); _write(tmp_path / 'inf', arrs, 32734)
    assert C.tiles_crs(tmp_path / 'inf', sorted(arrs)) == 'EPSG:32734'
    G, _ = V.run(sorted(arrs), tmp_path / 'inf', tmp_path / 'm', workers=0, gsd=GSD, return_gdf=True)
    assert G.crs.to_epsg() == 32734


def test_tiles_crs_refuses_mixed(tmp_path):
    arrs = _arrs(''); t = sorted(arrs)
    _write(tmp_path / 'inf', {t[0]: arrs[t[0]]}, 32734); _write(tmp_path / 'inf', {t[1]: arrs[t[1]]}, 32735)
    with pytest.raises(ValueError, match='mixed'): C.tiles_crs(tmp_path / 'inf', t)


def test_legacy_merge_block_path_uses_output_crs(tmp_path):
    arrs = _arrs('G0772_'); _write(tmp_path / 'inf', arrs, 32735)
    by_xy = C.unique_xy(sorted(arrs)); crs = C.tiles_crs(tmp_path / 'inf', sorted(arrs))
    P = C.merge_block(tmp_path / 'inf', (600000, 7100000), by_xy, gsd=GSD, crs=crs)
    assert P.crs.to_epsg() == 32735 and len(P) == 1


def test_tiles_crs_checks_every_tile_not_a_sample(tmp_path):
    a = np.zeros((PX, PX), np.float32); names = [f'E{600000 + 100 * i}_N7100000.tif' for i in range(401)]
    for i, t in enumerate(names): _write(tmp_path / 'inf', {t: a}, 32734 if i == 1 else 32735)
    with pytest.raises(ValueError, match='mixed'): C.tiles_crs(tmp_path / 'inf', names, 'EPSG:32735')


def test_blocks_of_tile_offset_and_aligned():
    assert C.blocks_of_tile('E251300_N6230400.tif') == {(251000, 6230000)}
    assert C.blocks_of_tile('E251900_N6230900.tif') == {(251000, 6230000)}
    assert C.blocks_of_tile('G0772_E601962_N7099458.tif') == {(601000, 7099000), (602000, 7099000)}
    assert C.blocks_of_tile('G0772_E601962_N7099958.tif') == {(601000, 7099000), (602000, 7099000), (601000, 7100000), (602000, 7100000)}


def test_legacy_merge_keeps_object_in_block_without_tile_corner(tmp_path):
    """Codex r1: tile at E601962 spans into block E602000; an object there must not be lost."""
    a = np.zeros((PX, PX), np.float32); a[8:12, 14:18] = 0.95       # x 601962+70..90 = 602032..602052 -> block 602000
    t = 'G0772_E601962_N7099458.tif'; _write(tmp_path / 'inf', {t: a}, 32735)
    r = C.run_merge([t], tmp_path / 'inf', workers=1, gsd=GSD)
    assert r['objects'] == 1 and r['ok'] >= 1


def test_v3_n_blocks_from_object_extent_on_offset_tile(tmp_path):
    """Codex r2: object E601992-602012 on tile G0772_E601962 spans blocks 601000 and 602000."""
    a = np.zeros((PX, PX), np.float32); a[8:12, 6:10] = 0.95          # px cols 6..9 -> x 601992..602012
    t = 'G0772_E601962_N7099458.tif'; _write(tmp_path / 'inf', {t: a}, 32735)
    G, s = V.run([t], tmp_path / 'inf', tmp_path / 'm', workers=0, gsd=GSD, return_gdf=True)
    assert len(G) == 1 and int(G.n_blocks.iloc[0]) == 2 and s['n_cross_block'] == 1


def test_n_blocks_aligned_matches_tile_blocks():
    from shapely.geometry import box
    assert V._n_blocks(box(251010, 6230010, 251090, 6230090)) == 1
    assert V._n_blocks(box(251900, 6230010, 252000, 6230090)) == 1      # touches block edge only
    assert V._n_blocks(box(251900, 6230010, 252100, 6230090)) == 2
    assert V._n_blocks(box(251900, 6230900, 252100, 6231100)) == 4
