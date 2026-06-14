"""Unit tests for ``core.chip_extraction`` (HN / COCO tile-crop primitives).

CPU-only, no GPU, no network. Builds synthetic GeoTIFFs with rasterio to
exercise the shared crop / write / tile-resolve logic that used to be copied
verbatim across the HN exporters, and fixates the mosaic-layout regression:
the legacy chunked-only ``find_tile`` glob silently returned nothing for
``mosaic`` imagery layers (vexcel_2024 / aerial_legacy), dropping whole HN
batches (rule 06-multi-city).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.chip_extraction import (  # noqa: E402
    crop_chip,
    point_xy_in_crs,
    resolve_tile_for_point,
    write_chip_geotiff,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _write_geotiff(
    path: Path,
    *,
    width: int,
    height: int,
    fill: int | np.ndarray = 100,
    crs="EPSG:4326",
    west: float = 28.0,
    north: float = -26.0,
    pixel: float = 1e-4,
    bands: int = 3,
) -> Path:
    """Write a tiny RGB GeoTIFF. ``fill`` is a scalar or a (bands,h,w) array."""
    transform = from_origin(west, north, pixel, pixel)
    if isinstance(fill, np.ndarray):
        data = fill
    else:
        data = np.full((bands, height, width), fill, dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": bands,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)
    return path


# ── point_xy_in_crs ──────────────────────────────────────────────────────────
def test_point_xy_in_crs_noop_for_4326():
    # 4326 -> 4326 (and None) is a no-op: returns lon/lat unchanged so the
    # refactor is byte-identical to the legacy direct src.index(lon, lat).
    assert point_xy_in_crs(28.05, -26.20, CRS.from_epsg(4326)) == (28.05, -26.20)
    assert point_xy_in_crs(28.05, -26.20, None) == (28.05, -26.20)


def test_point_xy_in_crs_reprojects_to_3857():
    from rasterio.warp import transform as warp_transform

    x, y = point_xy_in_crs(28.05, -26.20, CRS.from_epsg(3857))
    xs, ys = warp_transform("EPSG:4326", "EPSG:3857", [28.05], [-26.20])
    assert abs(x - xs[0]) < 1e-6 and abs(y - ys[0]) < 1e-6
    assert x > 1e6  # metre-scale, no longer lon/lat


# ── crop_chip ────────────────────────────────────────────────────────────────
def test_crop_chip_centered_full_window(tmp_path):
    tif = _write_geotiff(tmp_path / "t.tif", width=200, height=200, fill=100)
    chip_size = 50
    with rasterio.open(tif) as src:
        # centre of the raster
        lon, lat = 28.0 + 100 * 1e-4, -26.0 - 100 * 1e-4
        out = crop_chip(src, lon, lat, chip_size)
    assert out is not None
    data, window, x0, y0, w, h = out
    assert data.shape == (3, chip_size, chip_size)
    assert (w, h) == (chip_size, chip_size)
    assert window.col_off == x0 and window.row_off == y0
    # centred: 100 - 25 = 75
    assert x0 == 75 and y0 == 75


def test_crop_chip_clamps_and_pads_at_edge(tmp_path):
    # Raster smaller than chip_size in one dim forces clamp + zero-pad.
    tif = _write_geotiff(tmp_path / "t.tif", width=60, height=200, fill=100)
    chip_size = 100
    with rasterio.open(tif) as src:
        lon, lat = 28.0 + 30 * 1e-4, -26.0 - 100 * 1e-4
        out = crop_chip(src, lon, lat, chip_size)
    assert out is not None
    data, window, x0, y0, w, h = out
    # padded array is always chip_size square
    assert data.shape == (3, chip_size, chip_size)
    # width clamped to raster width (60 < 100); read window is unpadded
    assert w == 60 and h == 100
    assert x0 == 0  # clamped to min(0, width-chip)=0
    # the padded region (cols >= 60) is zero
    assert np.all(data[:, :, 60:] == 0)
    assert np.all(data[:, :, :60] == 100)


def test_crop_chip_skips_tiny_edge_chip(tmp_path):
    # Raster narrower than 50% of chip_size -> tiny edge chip -> None.
    tif = _write_geotiff(tmp_path / "t.tif", width=40, height=200, fill=100)
    chip_size = 100  # 40 < 0.5 * 100
    with rasterio.open(tif) as src:
        lon, lat = 28.0 + 20 * 1e-4, -26.0 - 100 * 1e-4
        out = crop_chip(src, lon, lat, chip_size)
    assert out is None


def test_crop_chip_skips_blank(tmp_path):
    # All-white (>=245) chip is skipped when skip_blank (default).
    tif = _write_geotiff(tmp_path / "blank.tif", width=200, height=200, fill=255)
    chip_size = 50
    with rasterio.open(tif) as src:
        lon, lat = 28.0 + 100 * 1e-4, -26.0 - 100 * 1e-4
        out = crop_chip(src, lon, lat, chip_size)
    assert out is None
    # ...but kept when skip_blank=False
    with rasterio.open(tif) as src:
        out2 = crop_chip(src, lon, lat, chip_size, skip_blank=False)
    assert out2 is not None


def test_crop_chip_reprojects_for_non_4326_tile(tmp_path):
    # A 3857 tile: passing a 4326 lon/lat must reproject before src.index so
    # the chip lands on the right pixel (mosaic-layer CRS branch).
    from rasterio.warp import transform as warp_transform

    lon, lat = 28.05, -26.20
    xs, ys = warp_transform("EPSG:4326", "EPSG:3857", [lon], [lat])
    cx, cy = xs[0], ys[0]
    # Build a 3857 raster centred on the point with 1 m pixels.
    width = height = 200
    west = cx - 100.0
    north = cy + 100.0
    transform = from_origin(west, north, 1.0, 1.0)
    data = np.full((3, height, width), 100, dtype=np.uint8)
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 3,
        "dtype": "uint8", "crs": "EPSG:3857", "transform": transform,
    }
    tif = tmp_path / "mosaic_3857.tif"
    with rasterio.open(str(tif), "w", **profile) as dst:
        dst.write(data)

    chip_size = 50
    with rasterio.open(tif) as src:
        out = crop_chip(src, lon, lat, chip_size)
    assert out is not None
    _data, _window, x0, y0, _w, _h = out
    # point is at raster centre (pixel ~100,100) -> chip top-left ~75,75
    assert x0 == 75 and y0 == 75


# ── write_chip_geotiff ───────────────────────────────────────────────────────
def test_write_chip_geotiff_lzw_and_shape(tmp_path):
    arr = (np.arange(3 * 200 * 200, dtype=np.uint8) % 200).reshape(3, 200, 200)
    tif = _write_geotiff(tmp_path / "src.tif", width=200, height=200, fill=arr)
    chip_size = 50
    with rasterio.open(tif) as src:
        out = crop_chip(src, 28.0 + 100 * 1e-4, -26.0 - 100 * 1e-4, chip_size)
        assert out is not None
        data, window, *_ = out
        chip_path = tmp_path / "chip.tif"
        write_chip_geotiff(src, data, window, chip_path, chip_size)

    assert chip_path.exists()
    with rasterio.open(chip_path) as dst:
        assert dst.width == chip_size and dst.height == chip_size
        assert (dst.compression.name.lower() == "lzw")
        # source photometric/JPEG keys are stripped, not carried through
        assert dst.profile.get("photometric") is None
        read_back = dst.read()
    assert read_back.shape == (3, chip_size, chip_size)
    # round-trips the cropped pixels
    assert np.array_equal(read_back, data)


# ── resolve_tile_for_point: chunked layout ───────────────────────────────────
def test_resolve_tile_for_point_chunked(tmp_path):
    grid = "G9999"
    grid_dir = tmp_path / grid
    grid_dir.mkdir()
    # two chunks side by side
    _write_geotiff(grid_dir / f"{grid}_0_0_geo.tif", width=100, height=100,
                   west=28.0, north=-26.0, pixel=1e-3)
    _write_geotiff(grid_dir / f"{grid}_1_0_geo.tif", width=100, height=100,
                   west=28.1, north=-26.0, pixel=1e-3)
    # point inside the SECOND chunk (lon between 28.1 and 28.2)
    lon, lat = 28.15, -26.05
    hit = resolve_tile_for_point(lon, lat, grid, tiles_root=tmp_path)
    assert hit is not None
    assert hit.name == f"{grid}_1_0_geo.tif"

    # point outside both chunks -> None
    miss = resolve_tile_for_point(29.0, -26.05, grid, tiles_root=tmp_path)
    assert miss is None


# ── resolve_tile_for_point: mosaic layout (the regression) ───────────────────
def _legacy_chunked_only_find_tile(lon, lat, grid_id, tiles_root):
    """Replica of the pre-refactor chunked-only find_tile (the buggy form).

    It globs ``{grid}_{col}_{row}_geo.tif`` only, so for a mosaic layer (a
    single ``{grid}_mosaic.tif``) it matches nothing and returns None — the
    silent HN-dropping defect this module fixes.
    """
    grid_dir = Path(tiles_root) / grid_id
    if not grid_dir.exists():
        return None
    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            left, bottom, right, top = src.bounds
            if left <= lon <= right and bottom <= lat <= top:
                return tif
    return None


def test_resolve_tile_for_point_mosaic_regression(tmp_path):
    grid = "JNB0042"
    # mosaic layout: a single {grid}_mosaic.tif directly under tiles_root
    mosaic = _write_geotiff(
        tmp_path / f"{grid}_mosaic.tif",
        width=200, height=200, west=28.0, north=-26.0, pixel=1e-3,
    )
    lon, lat = 28.05, -26.05  # inside the mosaic bounds

    # New resolver: resolves the single mosaic file.
    hit = resolve_tile_for_point(lon, lat, grid, tiles_root=mosaic)
    assert hit is not None and hit == mosaic

    # Legacy chunked-only logic: silently returns None for the same mosaic
    # (this is the defect being fixed).
    legacy = _legacy_chunked_only_find_tile(lon, lat, grid, tmp_path)
    assert legacy is None


def test_resolve_tile_for_point_mosaic_via_dir(tmp_path):
    # tiles_root pointing at a DIR that holds {grid}_mosaic.tif also resolves.
    grid = "JNB0042"
    _write_geotiff(
        tmp_path / f"{grid}_mosaic.tif",
        width=200, height=200, west=28.0, north=-26.0, pixel=1e-3,
    )
    hit = resolve_tile_for_point(28.05, -26.05, grid, tiles_root=tmp_path)
    assert hit is not None and hit.name == f"{grid}_mosaic.tif"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
