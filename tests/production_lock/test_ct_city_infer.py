import importlib.util
from pathlib import Path
import numpy as np, pytest

P = Path(__file__).resolve().parents[2] / 'scripts/production_lock_20261020/citywide_infer/ct_city_infer_v1.py'
spec = importlib.util.spec_from_file_location('ct_city_infer_v1', P); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def test_class_indices_pv_swh():
    assert M.CLASS_IDX == {'pv': 1, 'swh': 2}


def test_out_paths_pv_and_swh_per_model():
    p = M.out_paths('/o', '/x/tiles/T1.tif', ['convnext_t_s0', 'mit_b2_s2'], save_avg=False)
    assert set(p) == {(m, c) for m in ['convnext_t_s0', 'mit_b2_s2'] for c in ['pv', 'swh']}
    assert p[('mit_b2_s2', 'swh')] == Path('/o/mit_b2_s2/swh/T1.tif')
    assert len(M.out_paths('/o', 'T1.tif', ['a', 'b'], save_avg=True)) == 6


def test_uint8_convention():
    assert M.to_u8(np.array([0.0, 0.5, 1.0, 1.2, -0.1])).tolist() == [0, 128, 255, 255, 0]
    assert M.avg_u8(np.array([0, 255, 3], np.uint8), np.array([0, 0, 4], np.uint8)).tolist() == [0, 128, 4]


def test_save_cog_refuses_overwrite(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    class S: crs = 'EPSG:32734'; transform = from_origin(0, 0, 0.05, 0.05)
    p = tmp_path / 'm' / 'pv' / 't.tif'; a = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256).astype(np.uint8)
    M.save_cog(p, a, S)
    with rasterio.open(p) as d: assert d.dtypes[0] == 'uint8' and (d.read(1) == a).all()
    with pytest.raises(FileExistsError): M.save_cog(p, a, S)
    assert not p.with_suffix('.tif.tmp').exists()


def _tile(tmp_path, name='T1.tif', h=8, w=8):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    p = tmp_path / 'tiles' / name; p.parent.mkdir(exist_ok=True)
    with rasterio.open(p, 'w', driver='GTiff', width=w, height=h, count=3, dtype='uint8', crs='EPSG:32734',
                       transform=from_origin(0, 0, 0.05, 0.05)) as d:
        d.write(np.zeros((3, h, w), np.uint8))
    return p


def _fake(val):
    def fn(m, x):
        pr = np.zeros((3,) + x.shape[1:], np.float32)
        pr[1] = val[m]['pv']; pr[2] = val[m]['swh']; pr[0] = 1 - pr[1] - pr[2]
        return pr
    return fn


VAL = {'c': {'pv': 0.2, 'swh': 0.6}, 'b': {'pv': 0.4, 'swh': 0.1}}
MODELS = {'convnext_t_s0': 'c', 'mit_b2_s2': 'b'}


def test_process_tile_writes_pv_and_swh_per_model_and_avg(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    t = _tile(tmp_path); out = tmp_path / 'out'
    assert M.process_tile(str(t), MODELS, out, True, _fake(VAL)) == 6
    rd = lambda n, c: rasterio.open(out / n / c / 'T1.tif').read(1)
    assert (rd('convnext_t_s0', 'pv') == 51).all() and (rd('convnext_t_s0', 'swh') == 153).all()
    assert (rd('mit_b2_s2', 'pv') == 102).all() and (rd('mit_b2_s2', 'swh') == 26).all()
    assert (rd('avg2', 'pv') == 76).all() and (rd('avg2', 'swh') == 90).all()   # rint(76.5)=76, rint(89.5)=90
    assert rasterio.open(out / 'mit_b2_s2' / 'swh' / 'T1.tif').dtypes[0] == 'uint8'
    assert not list(out.rglob('*.tmp'))


def test_process_tile_skips_complete_and_fills_missing_without_overwrite(tmp_path):
    t = _tile(tmp_path); out = tmp_path / 'out'
    M.process_tile(str(t), MODELS, out, False, _fake(VAL))
    keep = (out / 'convnext_t_s0' / 'pv' / 'T1.tif').stat().st_mtime_ns
    assert M.process_tile(str(t), MODELS, out, False, _fake(VAL)) == 0
    (out / 'mit_b2_s2' / 'swh' / 'T1.tif').rename(tmp_path / 'moved.tif')
    assert M.process_tile(str(t), MODELS, out, False, _fake(VAL)) == 1
    assert (out / 'convnext_t_s0' / 'pv' / 'T1.tif').stat().st_mtime_ns == keep


def test_save_cog_publish_refuses_existing_even_if_created_late(tmp_path, monkeypatch):
    pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    class S: crs = 'EPSG:32734'; transform = from_origin(0, 0, 0.05, 0.05)
    p = tmp_path / 'x' / 't.tif'; p.parent.mkdir(); real_link = M.os.link
    def racing_link(src, dst):
        Path(dst).write_bytes(b'other writer'); return real_link(src, dst)
    monkeypatch.setattr(M.os, 'link', racing_link)
    with pytest.raises(FileExistsError):
        M.save_cog(p, np.zeros((4, 4), np.uint8), S)
    assert p.read_bytes() == b'other writer' and not list(p.parent.glob('*.tmp'))
