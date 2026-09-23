"""The chip grid has to be the production SlidingWindowDataset grid."""

from __future__ import annotations

import sys
from pathlib import Path

import rasterio
from rasterio.transform import from_bounds

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate_pilot.chip_windows import (
    EXPECTED_ORIGINS_2000,
    _write_zero_geotiff,
    chip_windows_from_tifs,
)


def test_2000px_tile_uses_anchored_origins(tmp_path):
    path = tmp_path / "CPT0000_0_0_geo.tif"
    transform = from_bounds(18.0, -34.0, 18.0016, -33.9984, 2000, 2000)
    _write_zero_geotiff(path, transform)
    copied = tmp_path / "CPT0000_1_0_geo.tif"
    copied.write_bytes(path.read_bytes())
    shifted = from_bounds(18.0016, -34.0, 18.0032, -33.9984, 2000, 2000)
    with rasterio.open(copied, "r+") as dst:
        dst.transform = shifted
    with rasterio.open(copied) as src:
        assert abs(src.bounds.left - 18.0016) < 1e-8

    frame = chip_windows_from_tifs([path, copied], "CPT0000")
    assert len(frame) == 98
    first = frame[frame["source_tile"] == path.name]
    origins = set(zip(first["col_off"], first["row_off"]))
    assert origins == {(col, row) for row in EXPECTED_ORIGINS_2000 for col in EXPECTED_ORIGINS_2000}
    assert frame["chip_index"].tolist() == list(range(98))
    assert frame["valid_w"].eq(400).all()
    # Second tile is one tile-width to the east, so its chip 0 starts there.
    second_origin = frame[frame["source_tile"] == copied.name].iloc[0]
    assert abs(second_origin["minx"] - 18.0016) < 1e-6
