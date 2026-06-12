"""Unit tests for the negative-pool ingest / backfill / leakage tooling.

Covers the F1-gap C-1 deliverables (CPU-only, no GPU, no network):

- agreement filter rules (gemini_says_nonpv, cls non-PV index, veto on
  actually_pv_mislabeled)
- the BFN0126 / DBN0044 hard block (irreversible-pollution guard)
- the chip_id -> cascade chip_id backfill join mapping + training_eligible gate
- the eval-leakage helper (mined-grid derivation + filter_eval_grids)
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name: str, rel_path: str):
    """Import a script module by path (the negative_pool scripts are not a
    package; load them directly so the tests don't depend on packaging)."""
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ingest = _load_module(
    "np_ingest_fp_audit",
    "scripts/training/negative_pool/ingest_fp_audit.py",
)
backfill = _load_module(
    "np_backfill_geometry",
    "scripts/training/negative_pool/backfill_geometry.py",
)
from core import negative_pool_leakage as leakage  # noqa: E402


# ── agreement filter: gemini_says_nonpv ─────────────────────────────────────

def test_gemini_says_nonpv_accepts_confident_drop():
    rec = {
        "production_action": "drop",
        "label": "not_pv",
        "pv_present": False,
        "requires_human_review": False,
    }
    assert ingest.gemini_says_nonpv(rec) is True


def test_gemini_says_nonpv_rejects_keep():
    rec = {
        "production_action": "keep",
        "label": "pv",
        "pv_present": True,
        "requires_human_review": False,
    }
    assert ingest.gemini_says_nonpv(rec) is False


def test_gemini_says_nonpv_rejects_when_human_review_required():
    rec = {
        "production_action": "drop",
        "label": "not_pv",
        "pv_present": False,
        "requires_human_review": True,
    }
    assert ingest.gemini_says_nonpv(rec) is False


def test_gemini_says_nonpv_rejects_label_pv_even_if_action_drop():
    # internally inconsistent record must not be admitted
    rec = {
        "production_action": "drop",
        "label": "pv",
        "pv_present": True,
        "requires_human_review": False,
    }
    assert ingest.gemini_says_nonpv(rec) is False


# ── agreement filter: cls non-PV index ──────────────────────────────────────

def test_cls_index_records_nonpv_and_vetoes_pv(tmp_path):
    csv_path = tmp_path / "subtype.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chip_filename", "subtype"])
        w.writeheader()
        w.writerow({"chip_filename": "jhbcbd_v3c_G0772_p0001__skylight_roof_window.png",
                    "subtype": "skylight_roof_window"})
        w.writerow({"chip_filename": "jhbcbd_v3c_G0772_p0002__actually_pv_mislabeled.png",
                    "subtype": "actually_pv_mislabeled"})
    idx = ingest.load_cls_nonpv_index(csv_path)
    assert idx[("G0772", "v3c", "1")] == "skylight_roof_window"
    # actually_pv_mislabeled is a veto sentinel (cls says it IS pv)
    assert idx[("G0772", "v3c", "2")] == "__pv__"


# ── BFN / DBN hard block ────────────────────────────────────────────────────

def test_blocked_grids_constant():
    assert "BFN0126" in ingest.BLOCKED_GRIDS
    assert "DBN0044" in ingest.BLOCKED_GRIDS


@pytest.mark.parametrize("grid", ["BFN0126", "DBN0044"])
def test_assert_not_blocked_raises_for_blocked(grid):
    with pytest.raises(ValueError, match="BLOCKED_GRIDS"):
        ingest.assert_not_blocked(grid)


def test_assert_not_blocked_allows_normal_grid():
    # should not raise
    ingest.assert_not_blocked("GQB0202")


def test_gemini_fpcut_skips_blocked_grid(tmp_path, monkeypatch):
    """A Gemini drop on a blocked grid must never be admitted."""
    import json

    verdict = tmp_path / "verdict.jsonl"
    recs = [
        # blocked grid, confident non-PV drop -> must be blocked
        {"grid_id": "DBN0044", "region_key": "durban", "pred_id": 0,
         "production_action": "drop", "label": "not_pv", "pv_present": False,
         "requires_human_review": False, "lookalike_type": "skylight",
         "human_label": "skylight"},
        # normal grid, same verdict, has human label -> admitted
        {"grid_id": "GQB0202", "region_key": "gqeberha", "pred_id": 1,
         "production_action": "drop", "label": "not_pv", "pv_present": False,
         "requires_human_review": False, "lookalike_type": "skylight",
         "human_label": "skylight"},
    ]
    with verdict.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("")  # empty -> append path

    monkeypatch.setattr(ingest, "MANIFEST_CSV", manifest)

    captured = {}

    def fake_append(rows, *, dry_run):
        captured["rows"] = rows

    monkeypatch.setattr(ingest, "append_rows", fake_append)

    args = _Args(
        verdict_jsonl=str(verdict),
        cls_subtype_csv=None,
        imagery_layer="vexcel",
        source_run="test",
        dry_run=True,
    )
    rc = ingest.ingest_gemini_fpcut(args)
    assert rc == 0
    admitted = {r["grid_id"] for r in captured["rows"]}
    assert "DBN0044" not in admitted
    assert "GQB0202" in admitted


def test_gemini_fpcut_requires_agreement(tmp_path, monkeypatch):
    """With no cls csv and no human record, nothing is admitted."""
    import json

    verdict = tmp_path / "verdict.jsonl"
    rec = {"grid_id": "GQB0202", "region_key": "gqeberha", "pred_id": 1,
           "production_action": "drop", "label": "not_pv", "pv_present": False,
           "requires_human_review": False, "lookalike_type": "skylight"}
    verdict.write_text(json.dumps(rec) + "\n")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("")
    monkeypatch.setattr(ingest, "MANIFEST_CSV", manifest)
    captured = {}
    monkeypatch.setattr(ingest, "append_rows",
                        lambda rows, *, dry_run: captured.update(rows=rows))

    args = _Args(verdict_jsonl=str(verdict), cls_subtype_csv=None,
                 imagery_layer="vexcel", source_run="test", dry_run=True)
    assert ingest.ingest_gemini_fpcut(args) == 0
    assert captured["rows"] == []


# ── backfill join + training_eligible gate ──────────────────────────────────

def test_pool_chip_to_cascade_chip_mapping():
    assert (backfill.pool_chip_to_cascade_chip("johannesburg_G0772_v3c_p0000")
            == "v3c_G0772_p0000")
    assert (backfill.pool_chip_to_cascade_chip("johannesburg_G0816_v4_2_p0123")
            == "v4_2_G0816_p0123")


def test_pool_chip_to_cascade_chip_rejects_bad_id():
    assert backfill.pool_chip_to_cascade_chip("not-a-chip-id") is None


def test_geid_layer_is_provenance_only():
    assert "geid_2024_02" in backfill.PROVENANCE_ONLY_LAYERS


def test_hn_ops_training_eligible_gate():
    from pipeline.hn_ops import _is_training_eligible
    assert _is_training_eligible({"training_eligible": "true"}) is True
    assert _is_training_eligible({"training_eligible": "false"}) is False
    # legacy rows without the column default to eligible (additive gate)
    assert _is_training_eligible({}) is True
    assert _is_training_eligible({"training_eligible": ""}) is True


# ── cropper CRS reproject (4326 geometry vs non-4326 tile) ───────────────────

def _write_tile(path: Path, *, epsg: int, lon: float, lat: float,
                half_m: float = 60.0):
    """Write a small single-band GeoTIFF centred on (lon, lat) in ``epsg``.

    The tile spans roughly ``2*half_m`` metres so the reprojected centroid
    lands comfortably inside its bounds.  Returns the centre (lon, lat) so the
    caller can build a matching bbox_geo_wkt in EPSG:4326.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform

    if epsg == 4326:
        # ~half_m metres expressed in degrees (rough; good enough for a tiny tile)
        deg = half_m / 111_320.0
        west, north = lon - deg, lat + deg
        res = (2 * deg) / 256.0
        crs = "EPSG:4326"
    else:
        xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
        cx, cy = xs[0], ys[0]
        west, north = cx - half_m, cy + half_m
        res = (2 * half_m) / 256.0
        crs = f"EPSG:{epsg}"

    transform = from_origin(west, north, res, res)
    data = np.full((1, 256, 256), 100, dtype="uint8")
    profile = dict(driver="GTiff", height=256, width=256, count=1,
                   dtype="uint8", crs=crs, transform=transform)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_pool_manifest(path: Path, *, region: str, grid_id: str,
                         imagery_layer: str, lon: float, lat: float):
    from shapely.geometry import box
    # tiny EPSG:4326 bbox around the point
    d = 0.0001
    wkt = box(lon - d, lat - d, lon + d, lat + d).wkt
    cols = ["chip_id", "archetype", "archetype_confidence", "region",
            "imagery_layer", "grid_id", "bbox_geo_wkt", "training_eligible"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({
            "chip_id": f"{region}_{grid_id}_v3c_p0001",
            "archetype": "skylight_roof_window",
            "archetype_confidence": "A2",
            "region": region,
            "imagery_layer": imagery_layer,
            "grid_id": grid_id,
            "bbox_geo_wkt": wkt,
            "training_eligible": "true",
        })


@pytest.mark.parametrize("epsg", [4326, 3857])
def test_cropper_resolves_tile_for_non4326_layer(tmp_path, epsg):
    """The cropper must reproject the EPSG:4326 bbox into the tile's native CRS.

    Regression for the C-1 blocker: a vexcel_2024 / aerial_legacy tile is in
    EPSG:3857 (metre-scale bounds ~3.1e6); comparing the raw lon/lat (~28)
    against those bounds silently resolves no tile and crops 0 chips.  This
    asserts a chip IS produced for both a 4326 layer (no-op transform) and a
    3857 layer (real reproject).
    """
    from pipeline.hn_ops import extract_negative_pool_hn

    lon, lat = 28.05, -26.20  # JHB-ish
    grid_id = "G0772"
    grid_dir = tmp_path / "tiles" / grid_id
    grid_dir.mkdir(parents=True)
    _write_tile(grid_dir / f"{grid_id}_0_0_geo.tif",
                epsg=epsg, lon=lon, lat=lat)

    manifest = tmp_path / "manifest.csv"
    _write_pool_manifest(manifest, region="johannesburg", grid_id=grid_id,
                         imagery_layer="vexcel_2024", lon=lon, lat=lat)

    out_dir = tmp_path / "out"
    result = extract_negative_pool_hn(
        archetypes=["skylight_roof_window"],
        output_dir=out_dir,
        chip_size=128,
        regions=["johannesburg"],
        tiles_root=tmp_path / "tiles",
        manifest_csv=manifest,
    )
    assert result.n_chips == 1, (
        f"EPSG:{epsg} layer should yield 1 chip; the unreprojected code path "
        f"silently drops non-4326 tiles"
    )
    assert result.images[0]["positive"] is False


def test_geom_xy_in_crs_reproject_and_noop():
    from shapely.geometry import Point
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform
    from pipeline.hn_ops import _geom_xy_in_crs

    pt = Point(28.05, -26.20)
    # 4326 -> 4326 is a no-op (returns lon/lat unchanged)
    x, y = _geom_xy_in_crs(pt, CRS.from_epsg(4326))
    assert (x, y) == (28.05, -26.20)
    # 4326 -> 3857 matches rasterio's transform and is metre-scale
    x3857, y3857 = _geom_xy_in_crs(pt, CRS.from_epsg(3857))
    xs, ys = warp_transform("EPSG:4326", "EPSG:3857", [28.05], [-26.20])
    assert abs(x3857 - xs[0]) < 1e-6 and abs(y3857 - ys[0]) < 1e-6
    assert x3857 > 1e6  # confirms it is NOT lon/lat anymore


# ── eval-leakage helper ─────────────────────────────────────────────────────

def _write_manifest(path: Path, rows: list[dict]):
    cols = ["chip_id", "archetype", "region", "imagery_layer", "grid_id",
            "training_eligible"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def test_mined_grid_keys_uses_explicit_region(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"chip_id": "a", "region": "gqeberha", "grid_id": "GQB0202"},
        {"chip_id": "b", "region": "durban", "grid_id": "DBN0402"},
        # same grid_id text, different region — must stay distinct
        {"chip_id": "c", "region": "johannesburg", "grid_id": "GQB0202"},
    ])
    keys = leakage.mined_grid_keys(manifest)
    assert ("gqeberha", "GQB0202") in keys
    assert ("johannesburg", "GQB0202") in keys
    assert len(keys) == 3


def test_mined_grids_for_region(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"chip_id": "a", "region": "gqeberha", "grid_id": "GQB0202"},
        {"chip_id": "b", "region": "gqeberha", "grid_id": "GQB0203"},
        {"chip_id": "c", "region": "durban", "grid_id": "DBN0402"},
    ])
    assert leakage.mined_grids_for_region("gqeberha", manifest) == {
        "GQB0202", "GQB0203"}
    assert leakage.mined_grids_for_region("durban", manifest) == {"DBN0402"}


def test_filter_eval_grids_excludes_mined(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"chip_id": "a", "region": "gqeberha", "grid_id": "GQB0202"},
    ])
    kept, excluded = leakage.filter_eval_grids(
        "gqeberha", ["GQB0202", "GQB0999", "GQB0327"], manifest)
    assert kept == ["GQB0999", "GQB0327"]
    assert excluded == ["GQB0202"]


def test_is_mined(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"chip_id": "a", "region": "gqeberha", "grid_id": "GQB0202"},
    ])
    assert leakage.is_mined("gqeberha", "GQB0202", manifest) is True
    assert leakage.is_mined("gqeberha", "GQB0999", manifest) is False
    # region must match — grid_id alone is not enough (rule 06-multi-city)
    assert leakage.is_mined("durban", "GQB0202", manifest) is False


def test_mined_keys_ignores_provenance_only_rows(tmp_path):
    """training_eligible=false rows do not contaminate eval (no model trains
    on them) — but they ARE counted in the full provenance footprint."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        # eligible: blank flag defaults eligible -> mined
        {"chip_id": "a", "region": "gqeberha", "grid_id": "GQB0202"},
        # provenance-only: training_eligible=false -> NOT mined for eval
        {"chip_id": "b", "region": "pretoria", "grid_id": "PTA0738",
         "training_eligible": "false"},
        # explicit true -> mined
        {"chip_id": "c", "region": "durban", "grid_id": "DBN0402",
         "training_eligible": "true"},
    ])
    keys = leakage.mined_grid_keys(manifest)
    assert ("gqeberha", "GQB0202") in keys
    assert ("durban", "DBN0402") in keys
    assert ("pretoria", "PTA0738") not in keys  # provenance-only stays clean
    # full provenance footprint includes the gated row
    full = leakage.mined_grid_keys(manifest, include_provenance_only=True)
    assert ("pretoria", "PTA0738") in full
    # filter keeps a provenance-only grid on the eval surface by default
    kept, excluded = leakage.filter_eval_grids(
        "pretoria", ["PTA0738", "PTA0999"], manifest)
    assert kept == ["PTA0738", "PTA0999"]
    assert excluded == []


# ── helper ──────────────────────────────────────────────────────────────────

class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)
