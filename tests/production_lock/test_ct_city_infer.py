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
