"""Unit tests for grid-level spatial_nms."""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from core.postproc import spatial_nms


def _gdf(geoms):
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:32734")


def test_no_overlap_keeps_all():
    out = spatial_nms(_gdf([box(0, 0, 1, 1), box(10, 10, 11, 11)]))
    assert len(out) == 2


def test_full_overlap_drops_smaller():
    """A 10×10 fully contains a 5×5 → IoU = 25/100 = 0.25; not over 0.5 default.
    Use a tighter threshold to verify the smaller one is dropped."""
    geoms = [box(0, 0, 10, 10), box(0, 0, 5, 5)]  # IoU = 0.25
    # default threshold 0.5: both kept
    out = spatial_nms(_gdf(geoms))
    assert len(out) == 2
    # threshold 0.2: smaller dropped (IoU=0.25 > 0.2)
    out = spatial_nms(_gdf(geoms), iou_threshold=0.2)
    assert len(out) == 1
    # The kept polygon should be the larger (10×10 = area 100)
    assert out.iloc[0].geometry.area == 100.0


def test_high_overlap_keeps_larger():
    """Two near-identical polygons → keep larger, drop smaller."""
    geoms = [box(0, 0, 10, 10), box(0, 0, 9.9, 9.9)]
    out = spatial_nms(_gdf(geoms), iou_threshold=0.5)
    assert len(out) == 1
    assert out.iloc[0].geometry.area == 100.0


def test_single_polygon_returned_unchanged():
    out = spatial_nms(_gdf([box(0, 0, 1, 1)]))
    assert len(out) == 1


def test_empty_gdf_returned_unchanged():
    out = spatial_nms(_gdf([]))
    assert len(out) == 0
