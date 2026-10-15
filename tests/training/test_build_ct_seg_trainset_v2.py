"""Unit tests for the resolution-agnostic CT seg trainset v2 helpers."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from scripts.training.build_ct_seg_trainset_v1 import HN, IGNORE, band_px, compose_mask
from scripts.training.build_ct_seg_trainset_v2 import (
    assemble_collectors, collector_mask_value, merge_class_a, norm_grid, pixel_size_m,
    read_tile_manifest, route_sam_fn_polygons,
)


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
        '<SimpleSource><SourceFilename relativeToVRT="0">tiles/E5_N6.tif</SourceFilename></SimpleSource>'
        '</VRTRasterBand><VRTRasterBand band="2">'
        '<SimpleSource><SourceFilename relativeToVRT="1">../tiles/E1_N2.tif</SourceFilename></SimpleSource>'
        '</VRTRasterBand></VRTDataset>')
    t = tiles_from_native_dir(tmp_path)
    assert t.grid.unique().tolist() == ["G1349"]
    # relativeToVRT="1" is relative to vrt/; relativeToVRT="0" non-absolute
    # paths are relative to native_dir (parent of vrt/), not left as cwd-relative.
    assert sorted(t.image) == [
        "/abs/E3_N4.tif",
        str(tmp_path / "tiles" / "E1_N2.tif"),
        str(tmp_path / "tiles" / "E5_N6.tif"),
    ]


def test_collector_mask_value_confirmed_is_hn_likely_is_ignore():
    # Human-confirmed collectors train as hard negatives. Likely and anything
    # unaudited are ignore, so the loss does not treat them as background.
    assert collector_mask_value("confirmed") == HN
    assert collector_mask_value("CONFIRMED") == HN
    assert collector_mask_value("likely") == IGNORE
    assert collector_mask_value("Likely") == IGNORE
    assert collector_mask_value("unaudited") == IGNORE
    assert collector_mask_value(None) == IGNORE
    assert collector_mask_value("") == IGNORE
    # Pool heater is confirmed even if the swh row said likely.
    assert collector_mask_value("likely", is_pool_heater=True) == HN
    assert collector_mask_value(None, is_pool_heater=True) == HN


def test_confirmed_collector_joins_hn_likely_painted_ignore():
    n = 20
    z = np.zeros((n, n), bool)
    confirmed = np.zeros((n, n), bool)
    confirmed[2:6, 2:6] = True
    likely = np.zeros((n, n), bool)
    likely[10:14, 10:14] = True
    valid = np.ones((n, n), bool)
    mask = compose_mask(z, z, z | confirmed, valid, r_b=3, r_a=1)
    mask[likely] = IGNORE
    assert (mask[2:6, 2:6] == HN).all()
    assert mask[1, 3] == 0  # no ignore band around the collector hard negative
    assert (mask[10:14, 10:14] == IGNORE).all()


def test_route_sam_fn_polygons_to_class_a():
    g = gpd.GeoDataFrame({
        "source": ["sam_fn_review", "sam_fn_marker", "SAM_FN_extra", "sam2", None, "classB"],
        "geometry": [box(i, 0, i + 1, 1) for i in range(6)],
    })
    class_b, class_a = route_sam_fn_polygons(g)
    assert set(class_a["source"].str.lower()) == {"sam_fn_review", "sam_fn_marker", "sam_fn_extra"}
    assert set(class_b["source"].dropna()) == {"sam2", "classB"}
    assert class_b["source"].isna().sum() == 1
    no_src = g.drop(columns="source")
    class_b2, class_a2 = route_sam_fn_polygons(no_src)
    assert len(class_b2) == 6 and len(class_a2) == 0


def test_merge_class_a_keeps_sam_fn_replaces_explicit():
    existing = gpd.GeoDataFrame({
        "source": ["sam_fn_review", "older_explicit"],
        "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
    })
    explicit = gpd.GeoDataFrame({"source": ["human"], "geometry": [box(3, 0, 4, 1)]})
    out = merge_class_a(existing, explicit)
    assert set(out["source"]) == {"sam_fn_review", "human"}


def test_assemble_collectors_confirmed_likely_and_pool_heater():
    swh = gpd.GeoDataFrame({
        "grid": ["G1", "G1", "G1"],
        "origin": ["classB_label", "fp_review_delete", "classB_label"],
        "ref": ["a.gpkg#0", "b.gpkg#1", "c.gpkg#2"],
        "confidence": ["confirmed", "likely", None],
        "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
    }, crs="EPSG:4326")
    pool = [
        {"grid": "G1", "ref": "b.gpkg#1", "geometry": box(2, 0, 3, 1)},
        {"grid": "G2", "ref": "d.gpkg#4", "geometry": box(6, 0, 7, 1)},
    ]
    out = assemble_collectors(swh, pool)
    by_ref = out.set_index("ref")
    assert by_ref.loc["a.gpkg#0", "mask_value"] == HN
    assert by_ref.loc["a.gpkg#0", "confidence"] == "confirmed"
    assert by_ref.loc["a.gpkg#0", "kind"] == "collector"
    # Already in the swh table as likely, but it is a pool heater → confirmed, no duplicate.
    assert (out.ref == "b.gpkg#1").sum() == 1
    assert by_ref.loc["b.gpkg#1", "mask_value"] == HN
    assert by_ref.loc["b.gpkg#1", "confidence"] == "confirmed"
    assert by_ref.loc["b.gpkg#1", "confidence_src"] == "likely"
    assert by_ref.loc["b.gpkg#1", "subtype"] == "pool_heater"
    assert by_ref.loc["c.gpkg#2", "mask_value"] == IGNORE
    assert by_ref.loc["c.gpkg#2", "confidence"] == "unaudited"
    assert by_ref.loc["d.gpkg#4", "mask_value"] == HN
    assert by_ref.loc["d.gpkg#4", "confidence"] == "confirmed"
    assert by_ref.loc["d.gpkg#4", "kind"] == "collector"
    assert by_ref.loc["d.gpkg#4", "subtype"] == "pool_heater"
    assert by_ref.loc["d.gpkg#4", "origin"] == "hn_pool_heater"
    assert (out.kind == "collector").all()
