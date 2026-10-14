"""Unit tests for the resolution-agnostic CT seg trainset v2 helpers."""
import pandas as pd
import pytest
from affine import Affine
from rasterio.crs import CRS

from scripts.training.build_ct_seg_trainset_v1 import band_px
from scripts.training.build_ct_seg_trainset_v2 import norm_grid, pixel_size_m, read_tile_manifest


def test_norm_grid():
    assert norm_grid("CPT1631") == "G1631"
    assert norm_grid("cpt0042") == "G0042"
    assert norm_grid("G1631") == "G1631"


def test_manifest_flexible_columns(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"grid_id": ["CPT1631"], "tile_path": ["a/CPT1631_0_0.tif"]}).to_csv(p, index=False)
    m = read_tile_manifest(p, root=tmp_path)
    assert m.grid.tolist() == ["G1631"] and m.image.iloc[0] == str(tmp_path / "a/CPT1631_0_0.tif")


def test_manifest_infers_grid_from_filename(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"path": ["/x/CPT1700_3_2_geo.tif"]}).to_csv(p, index=False)
    assert read_tile_manifest(p).grid.tolist() == ["G1700"]


def test_manifest_requires_path_column(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"grid": ["G1"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        read_tile_manifest(p)


def test_pixel_size_projected_and_geographic():
    assert pixel_size_m(Affine(0.05, 0, 0, 0, -0.05, 0), CRS.from_epsg(32734), -34) == (0.05, 0.05)
    w, h = pixel_size_m(Affine(8e-7, 0, 0, 0, -8e-7, 0), CRS.from_epsg(4326), -34.0)
    assert w == pytest.approx(0.0738, abs=1e-3) and h == pytest.approx(0.0885, abs=1e-3)


def test_metric_bands_scale_with_resolution():
    assert band_px(0.3, 0.05) == 6 and band_px(0.15, 0.05) == 3
    assert band_px(0.3, 0.074) == 4


def test_tiles_from_native_dir_parses_vrts(tmp_path):
    from scripts.training.build_ct_seg_trainset_v2 import tiles_from_native_dir

    (tmp_path / "vrt").mkdir()
    (tmp_path / "vrt" / "CPT1349.vrt").write_text(
        '<VRTDataset><VRTRasterBand band="1">'
        '<SimpleSource><SourceFilename relativeToVRT="1">../tiles/E1_N2.tif</SourceFilename></SimpleSource>'
        '<SimpleSource><SourceFilename relativeToVRT="0">/abs/E3_N4.tif</SourceFilename></SimpleSource>'
        '</VRTRasterBand><VRTRasterBand band="2">'
        '<SimpleSource><SourceFilename relativeToVRT="1">../tiles/E1_N2.tif</SourceFilename></SimpleSource>'
        '</VRTRasterBand></VRTDataset>')
    t = tiles_from_native_dir(tmp_path)
    assert t.grid.unique().tolist() == ["G1349"]
    assert sorted(t.image) == ["/abs/E3_N4.tif", str(tmp_path / "tiles" / "E1_N2.tif")]
