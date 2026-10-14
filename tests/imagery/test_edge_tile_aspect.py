"""Full-edge-tile mode: last col/row tiles are full unclipped squares."""
import os
from types import SimpleNamespace

import pytest

from scripts.imagery.download_tiles import full_tile_bounds, tile_request_bounds


def _spec():
    # 7 cols x 6 rows; grid extent ends mid-tile on the right and bottom.
    d = 0.0016
    return SimpleNamespace(pixel_size=2000, tile_size_deg=d, n_cols=7, n_rows=6,
                           xmin=0.0, xmax=6.94 * d, ymax=0.0, ymin=-5.772 * d)


def test_interior_tile_identical_in_both_modes():
    s = _spec()
    assert tile_request_bounds(s, 2, 2) == tile_request_bounds(s, 2, 2, full_edge_tiles=True)


@pytest.mark.parametrize("col,row", [(6, 0), (0, 5), (6, 5)])
def test_edge_tile_is_full_square_anchored_at_origin(col, row):
    s = _spec()
    x0, y0, x1, y1 = tile_request_bounds(s, col, row, full_edge_tiles=True)
    assert x1 - x0 == pytest.approx(s.tile_size_deg)
    assert y1 - y0 == pytest.approx(s.tile_size_deg)
    assert x0 == pytest.approx(s.xmin + col * s.tile_size_deg)
    assert y1 == pytest.approx(s.ymax - row * s.tile_size_deg)
    # extends past the grid boundary on the clipped side
    if col == s.n_cols - 1:
        assert x1 > s.xmax
    if row == s.n_rows - 1:
        assert y0 < s.ymin


def test_legacy_mode_still_clips():
    s = _spec()
    x0, y0, x1, y1 = tile_request_bounds(s, 6, 5)
    assert x1 == pytest.approx(s.xmax) and y0 == pytest.approx(s.ymin)


@pytest.mark.skipif(not os.environ.get("ZASOLAR_NETWORK_TESTS"),
                    reason="set ZASOLAR_NETWORK_TESTS=1 to hit the City WMS")
def test_live_full_edge_tile_square_pixels(tmp_path):
    import rasterio
    from core.grid_utils import get_grid_spec
    from scripts.imagery.download_tiles import download_tile

    spec = get_grid_spec("CPT1692", region="ct")
    out = tmp_path / "edge.tif"
    download_tile(spec, spec.n_cols - 1, spec.n_rows - 1, out, full_edge_tiles=True)
    with rasterio.open(out) as s:
        assert (s.width, s.height) == (spec.pixel_size, spec.pixel_size)
        assert abs(abs(s.transform.a) - abs(s.transform.e)) / abs(s.transform.a) < 1e-3
        assert s.bounds.right > spec.xmax and s.bounds.bottom < spec.ymin
