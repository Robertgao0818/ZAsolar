"""Tests for SlidingWindowDataset using synthetic GeoTIFFs in tmpdir.

No real grids needed; portable on any machine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from core.inference.tile_dataset import (
    ChipMeta,
    SlidingWindowDataset,
    list_collate,
)


def _write_synthetic_tif(path: Path, width: int, height: int, *, value: int = 100) -> None:
    """Write a 3-band uint8 GeoTIFF filled with `value`. EPSG:4326."""
    arr = np.full((3, height, width), value, dtype=np.uint8)
    transform = from_origin(18.0, -34.0, 0.0001, 0.0001)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width,
        count=3, dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(arr)


# ─────────────────────────────────────────────────────────────────────────
# Window coverage math
# ─────────────────────────────────────────────────────────────────────────
def test_chunked_chip_count_no_overlap(tmp_path):
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=400, height=400)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.0)
    assert len(ds) == 1


def test_chunked_chip_count_with_overlap(tmp_path):
    """800×800, chip 400, overlap 0.25 → stride 300.

    Cols: [0, 300, 400 (anchor)] = 3
    Rows: [0, 300, 400 (anchor)] = 3
    Total = 9 chips.
    """
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=800, height=800)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.25)
    assert len(ds) == 9


def test_undersized_raster_yields_one_padded_chip(tmp_path):
    """100×100 raster, chip 400 → exactly one (0, 0) origin; chip is padded."""
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=100, height=100)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.25)
    assert len(ds) == 1
    chip_tensor, meta = ds[0]
    assert chip_tensor.shape == (3, 400, 400)
    assert meta.valid_shape == (100, 100)
    assert meta.valid_window == (0, 0, 100, 100)
    assert meta.chip_shape == (400, 400)


def test_chip_dtype_and_range(tmp_path):
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=400, height=400, value=255)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.0)
    chip_tensor, _ = ds[0]
    assert chip_tensor.dtype.is_floating_point
    # uint8(255) → /255 → 1.0
    assert chip_tensor.min() == 1.0
    assert chip_tensor.max() == 1.0


def test_edge_padding_outside_valid_is_zero(tmp_path):
    """Pad region of a small raster must be 0.0."""
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=100, height=100, value=200)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.0)
    chip_tensor, meta = ds[0]
    # Inside valid window: ≈200/255
    inside = chip_tensor[:, :100, :100].numpy()
    assert (inside > 0.5).all()
    # Outside valid window: 0
    outside = chip_tensor[:, 100:, 100:].numpy()
    assert (outside == 0.0).all()


def test_multiple_chunked_tifs(tmp_path):
    """Two 400×400 TIFs → 2 chips total at chip_size 400, overlap 0."""
    p1 = tmp_path / "G1234_0_0_geo.tif"
    p2 = tmp_path / "G1234_0_1_geo.tif"
    _write_synthetic_tif(p1, width=400, height=400)
    _write_synthetic_tif(p2, width=400, height=400)
    ds = SlidingWindowDataset([p1, p2], chip_size=400, overlap=0.0)
    assert len(ds) == 2
    _, meta0 = ds[0]
    _, meta1 = ds[1]
    assert meta0.source_tile_id == "G1234_0_0_geo"
    assert meta1.source_tile_id == "G1234_0_1_geo"


def test_max_chips_caps(tmp_path):
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=2000, height=2000)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.25, max_chips=3)
    assert len(ds) == 3


def test_metadata_round_trip(tmp_path):
    p = tmp_path / "G1234_0_0_geo.tif"
    _write_synthetic_tif(p, width=400, height=400)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.0)
    _, meta = ds[0]
    assert isinstance(meta, ChipMeta)
    assert meta.source_crs == "EPSG:4326"
    assert meta.source_tif == str(p)
    assert meta.window == (0, 0, 400, 400)
    assert meta.chip_shape == (400, 400)
    assert meta.valid_shape == (400, 400)


def test_tif_meta_attribute(tmp_path):
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=500, height=500)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.0)
    info = ds.tif_meta
    assert len(info) == 1
    assert info[0]["width"] == 500
    assert info[0]["height"] == 500
    assert info[0]["crs"] == "EPSG:4326"


# ─────────────────────────────────────────────────────────────────────────
# Validation errors
# ─────────────────────────────────────────────────────────────────────────
def test_empty_tif_paths_raises():
    with pytest.raises(ValueError, match="empty"):
        SlidingWindowDataset([])


def test_invalid_overlap_raises(tmp_path):
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=400, height=400)
    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowDataset([p], overlap=1.0)


def test_chip_size_zero_raises(tmp_path):
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=400, height=400)
    with pytest.raises(ValueError, match="chip_size"):
        SlidingWindowDataset([p], chip_size=0)


# ─────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────
def test_list_collate_returns_lists(tmp_path):
    p = tmp_path / "tile.tif"
    _write_synthetic_tif(p, width=800, height=400)
    ds = SlidingWindowDataset([p], chip_size=400, overlap=0.0)
    batch = [ds[i] for i in range(len(ds))]
    tensors, metas = list_collate(batch)
    assert isinstance(tensors, list)
    assert isinstance(metas, list)
    assert all(t.shape == (3, 400, 400) for t in tensors)
    assert all(isinstance(m, ChipMeta) for m in metas)
