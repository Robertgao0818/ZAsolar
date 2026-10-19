import sys
from pathlib import Path
import numpy as np, pytest

rasterio = pytest.importorskip('rasterio')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/production_lock_20261020/citywide_infer'))
import ct_city_infer_v2 as M  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

VAL = {'convnext_t_s0': (0.2, 0.6), 'mit_b2_s2': (0.4, 0.1)}


def fake_prob(name, x):
    pv, swh = VAL[name]
    p = np.zeros((3,) + x.shape[1:], np.float32); p[1], p[2] = pv, swh; p[0] = 1 - pv - swh
    return p


def mk_tiles(d, names, px=8):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        x, y = M.tile_xy(n)
        with rasterio.open(d / n, 'w', driver='GTiff', width=px, height=px, count=3, dtype='uint8', crs='EPSG:32734',
                           transform=from_origin(x, y + 100, 100 / px, 100 / px)) as f:
            f.write(np.zeros((3, px, px), np.uint8))


def test_cog_opts_zstd3_pred2():
    assert M.COG_OPTS['COMPRESS'] == 'ZSTD' and M.COG_OPTS['LEVEL'] == 3 and M.COG_OPTS['PREDICTOR'] == 2


def test_tile_list_and_names(tmp_path):
    c = tmp_path / 't.csv'; c.write_text('tile,reason,score\nE251300_N6230400.tif,keep,0.8\nE251300_N6230100.tif,qa,0.1\n')
    assert M.read_tile_list(c) == ['E251300_N6230400.tif', 'E251300_N6230100.tif']
    assert M.tile_xy('E251300_N6230400.tif') == (251300, 6230400) and M.block_of('E251300_N6230400.tif') == (251000, 6230000)
    c.write_text('tile\nA.tif\nA.tif\n')
    with pytest.raises(ValueError): M.read_tile_list(c)


def test_infer_writes_pv_swh_lossless_and_resumes(tmp_path):
    names = ['E1000_N2000.tif', 'E1100_N2000.tif', 'E1200_N2000.tif']; mk_tiles(tmp_path / 'tiles', names)
    out = tmp_path / 'out'
    r = M.run_infer(names, tmp_path / 'tiles', out, fake_prob, writers=2, max_pending=3)
    assert (r['done'], r['skipped'], r['absent'], r['failed']) == (3, 0, 0, 0)
    exp = {('convnext_t_s0', 'pv'): 51, ('convnext_t_s0', 'swh'): 153, ('mit_b2_s2', 'pv'): 102, ('mit_b2_s2', 'swh'): 26}
    for (m, c), v in exp.items():
        with rasterio.open(M.out_path(out, m, c, names[0])) as d:
            assert d.dtypes[0] == 'uint8' and (d.read(1) == v).all()
            assert d.profile['compress'].lower() == 'zstd' and d.transform == from_origin(1000, 2100, 12.5, 12.5)
    assert not list(out.rglob('*.tmp*'))
    keep = M.out_path(out, 'convnext_t_s0', 'pv', names[0]).stat().st_mtime_ns
    assert M.run_infer(names, tmp_path / 'tiles', out, fake_prob, writers=1)['skipped'] == 3
    M.out_path(out, 'mit_b2_s2', 'swh', names[1]).rename(tmp_path / 'x.tif')
    r = M.run_infer(names, tmp_path / 'tiles', out, fake_prob, writers=1)
    assert r['done'] == 1 and r['skipped'] == 2 and M.out_path(out, 'mit_b2_s2', 'swh', names[1]).exists()
    assert M.out_path(out, 'convnext_t_s0', 'pv', names[0]).stat().st_mtime_ns == keep


def test_infer_skips_absent_tiles_then_picks_them_up(tmp_path):
    names = ['E1000_N2000.tif', 'E1100_N2000.tif']; mk_tiles(tmp_path / 'tiles', names[:1])
    r = M.run_infer(names, tmp_path / 'tiles', tmp_path / 'out', fake_prob, writers=1)
    assert (r['done'], r['absent']) == (1, 1)
    mk_tiles(tmp_path / 'tiles', names[1:])
    r = M.run_infer(names, tmp_path / 'tiles', tmp_path / 'out', fake_prob, writers=1)
    assert (r['done'], r['skipped'], r['absent']) == (1, 1, 0)


def test_infer_stops_after_two_consecutive_failures(tmp_path):
    names = ['E1000_N2000.tif', 'E1100_N2000.tif', 'E1200_N2000.tif']
    d = tmp_path / 'tiles'; d.mkdir()
    for n in names: (d / n).write_bytes(b'not a tif')
    with pytest.raises(RuntimeError, match='2 consecutive'):
        M.run_infer(names, d, tmp_path / 'out', fake_prob, writers=1)


def test_publish_never_overwrites(tmp_path):
    p = tmp_path / 'a.tif'; p.write_bytes(b'old')
    assert M.save_cog(p, np.zeros((4, 4), np.uint8), 'EPSG:32734', from_origin(0, 4, 1, 1).to_gdal()) == 'exists'
    assert p.read_bytes() == b'old' and not list(tmp_path.glob('*.tmp*'))


def _write_pv(out, name, arr):
    x, y = M.tile_xy(name); px = arr.shape[0]
    for m in M.MODELS:
        for c in M.CLASS_IDX:
            M.save_cog(M.out_path(out, m, c, name), arr if c == 'pv' else np.zeros_like(arr), 'EPSG:32734',
                       from_origin(x, y + 100, 100 / px, 100 / px).to_gdal())


def test_merge_cross_block_object_written_once(tmp_path):
    out = tmp_path / 'out'; px = 20
    a = np.zeros((px, px), np.uint8); b = np.zeros((px, px), np.uint8)
    a[8:12, 14:20] = 153; b[8:12, 0:6] = 153; a[9, 16] = 240        # seed (>=0.9) only on the left tile
    _write_pv(out, 'E900_N2000.tif', a); _write_pv(out, 'E1000_N2000.tif', b)
    by_xy = {M.tile_xy(t): t for t in ['E900_N2000.tif', 'E1000_N2000.tif']}
    g0 = M.merge_block(out, (0, 2000), by_xy, gsd=5.0); g1 = M.merge_block(out, (1000, 2000), by_xy, gsd=5.0)
    assert len(g0) + len(g1) == 1
    g = g0 if len(g0) else g1
    assert int(g.n_tiles.iloc[0]) == 2 and int(g.n_px.iloc[0]) == 48 and not bool(g.halo_cut.iloc[0])
    assert abs(g.geometry.iloc[0].area - 48 * 25) < 1e-6
    assert M.merge_block(out, (5000, 2000), by_xy, gsd=5.0) is None


def test_merge_avg2_is_float_mean_like_bench(tmp_path):
    out = tmp_path / 'out'; n = 'E0_N0.tif'
    for m, v in zip(M.MODELS, (229, 230)):
        M.save_cog(M.out_path(out, m, 'pv', n), np.full((4, 4), v, np.uint8), 'EPSG:32734', from_origin(0, 100, 25, 25).to_gdal())
    assert np.allclose(M.load_avg2(out, n), 229.5 / 255)


def test_run_merge_writes_gpkg_once_and_resumes(tmp_path):
    gpd = pytest.importorskip('geopandas')
    out = tmp_path / 'out'; px = 20
    a = np.zeros((px, px), np.uint8); b = np.zeros((px, px), np.uint8)
    a[8:12, 14:20] = 153; b[8:12, 0:6] = 153; a[9, 16] = 240
    names = ['E900_N2000.tif', 'E1000_N2000.tif']
    _write_pv(out, names[0], a); _write_pv(out, names[1], b)
    r = M.run_merge(names, out, workers=1, gsd=5.0)
    assert r['ok'] == 2 and r['objects'] == 1 and r['halo_cut'] == 0
    g = pd_concat([gpd.read_file(f, layer='objects') for f in sorted((out / 'merge').glob('*.gpkg'))])
    assert len(g) == 1 and not list((out / 'merge').glob('*.tmp*'))
    assert M.run_merge(names, out, workers=1, gsd=5.0)['skip'] == 2
    with pytest.raises(RuntimeError, match='lack PV'):
        M.run_merge(names + ['E1100_N2000.tif'], out, workers=1, gsd=5.0)


def pd_concat(x):
    import pandas as pd
    return pd.concat(x, ignore_index=True)


def test_follow_ready_file_and_done_manifest(tmp_path, monkeypatch):
    names = ['E1000_N2000.tif', 'E1100_N2000.tif']; d = tmp_path / 'tiles'; mk_tiles(d, names)
    ready = tmp_path / 'READY.txt'; ready.write_text(names[0] + '\n'); out = tmp_path / 'out'
    calls = []
    def fake_sleep(s):
        calls.append(s); ready.write_text('\n'.join(names) + '\n')   # second tile becomes ready
    monkeypatch.setattr(M.time, 'sleep', fake_sleep)
    r = M.follow_infer(names, d, out, fake_prob, ready_file=ready, poll_s=1, writers=1)
    assert r['done_total'] == 2 and len(calls) == 1
    assert (out / 'done_manifest.txt').read_text().split() == names
    assert M.update_done_manifest(names, out) == 2 and (out / 'done_manifest.txt').read_text().split() == names
