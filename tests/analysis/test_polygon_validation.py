"""Unit tests for core.polygon_validation — the canonical polygon geometry
validity + area-cap filtering pipeline extracted from
scripts/analysis/area_aggregate_eval.py (2026-06-19, ADR-0001 deepening track).

CPU-only, no GPU and no real GeoPackage data: every fixture is a small
in-memory GeoDataFrame, written to a temp .gpkg only for the path-level
``read_polygons`` cases. Covers every divergence the migration must preserve:
invalid geom, non-finite bounds (incl. the exact 1e18 boundary), the
zero-area keep/drop split (§4), the 20000 m² cap at/over boundary, empty gpkg,
missing-layer fallback, and CRS already-metric vs needs-reproject.
"""
from __future__ import annotations

import math
import unittest

import geopandas as gpd
from shapely.geometry import Point, Polygon, box

from core.polygon_validation import (
    MAX_PLAUSIBLE_POLY_M2,
    _MAX_PLAUSIBLE_POLY_M2,
    _read_polys_geom,
    _sum_area_m2,
    clean_metric_gdf,
    geometry_finite,
    read_polygons,
)

METRIC = "EPSG:32734"  # Cape Town UTM 34S


def _gdf(geoms, crs=METRIC):
    return gpd.GeoDataFrame(geometry=list(geoms), crs=crs)


# --------------------------------------------------------------------------
# geometry_finite predicate
# --------------------------------------------------------------------------
class TestGeometryFinite(unittest.TestCase):
    def test_normal_polygon_is_finite(self):
        self.assertTrue(geometry_finite(box(0, 0, 1, 1)))

    def test_inf_coordinate_rejected(self):
        self.assertFalse(geometry_finite(Polygon([(0, 0), (0, 1), (float("inf"), 1)])))

    def test_nan_bounds_rejected(self):
        # empty polygon → bounds are all NaN
        self.assertFalse(geometry_finite(Polygon()))

    def test_at_1e18_boundary_kept(self):
        # canonical rule rejects abs(coord) > 1e18, so exactly 1e18 is KEPT
        self.assertTrue(geometry_finite(box(0, 0, 1e18, 1)))

    def test_over_1e18_rejected(self):
        self.assertFalse(geometry_finite(box(0, 0, 1.1e18, 1)))

    def test_bounds_raises_rejected(self):
        class NoBounds:
            @property
            def bounds(self):
                raise RuntimeError("no bounds")

        self.assertFalse(geometry_finite(NoBounds()))


# --------------------------------------------------------------------------
# clean_metric_gdf — the gdf-level primitive
# --------------------------------------------------------------------------
class TestCleanMetricGdf(unittest.TestCase):
    def test_empty_gdf(self):
        g, n = clean_metric_gdf(_gdf([]), metric_crs=METRIC, drop_zero_area=True)
        self.assertTrue(g.empty)
        self.assertEqual(n, 0)

    def test_drops_invalid_geometry(self):
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])  # self-intersecting → invalid
        g, n = clean_metric_gdf(
            _gdf([box(0, 0, 1, 1), bowtie]), metric_crs=METRIC, drop_zero_area=False
        )
        self.assertEqual(len(g), 1)
        self.assertEqual(n, 0)  # invalid drops are NOT counted in n_dropped

    def test_drops_nonfinite_geometry(self):
        infp = Polygon([(0, 0), (0, 1), (float("inf"), 1)])
        g, n = clean_metric_gdf(
            _gdf([box(0, 0, 1, 1), infp]), metric_crs=METRIC, drop_zero_area=False
        )
        self.assertEqual(len(g), 1)
        self.assertEqual(n, 0)

    def test_area_cap_over_dropped_and_counted(self):
        big = box(0, 0, 200, 200)  # 40000 m² > 20000
        g, n = clean_metric_gdf(
            _gdf([box(0, 0, 10, 10), big]), metric_crs=METRIC, drop_zero_area=False
        )
        self.assertEqual(len(g), 1)
        self.assertEqual(n, 1)  # cap exceedance IS counted

    def test_area_cap_at_boundary_kept(self):
        # exactly 20000 m² (100 x 200) → kept (cap is `<= MAX`)
        at_cap = box(0, 0, 100, 200)
        self.assertEqual(at_cap.area, MAX_PLAUSIBLE_POLY_M2)
        g, n = clean_metric_gdf(_gdf([at_cap]), metric_crs=METRIC, drop_zero_area=False)
        self.assertEqual(len(g), 1)
        self.assertEqual(n, 0)

    def test_zero_area_kept_when_flag_false(self):
        # Point is valid + finite + area==0
        g, n = clean_metric_gdf(
            _gdf([box(0, 0, 10, 10), Point(5, 5)]),
            metric_crs=METRIC, drop_zero_area=False,
        )
        self.assertEqual(len(g), 2)
        self.assertEqual(n, 0)

    def test_zero_area_dropped_when_flag_true(self):
        g, n = clean_metric_gdf(
            _gdf([box(0, 0, 10, 10), Point(5, 5)]),
            metric_crs=METRIC, drop_zero_area=True,
        )
        self.assertEqual(len(g), 1)
        self.assertEqual(n, 0)  # zero-area drops are NOT counted in n_dropped

    def test_no_reproject_when_already_metric(self):
        g, _ = clean_metric_gdf(
            _gdf([box(0, 0, 10, 10)], crs=METRIC),
            metric_crs=METRIC, drop_zero_area=False,
        )
        self.assertAlmostEqual(float(g.geometry.area.iloc[0]), 100.0)

    def test_reproject_when_needs_metric(self):
        # a ~0.001° box near Cape Town in 4326 reprojects to a real m² area
        poly = box(18.42, -33.92, 18.421, -33.919)
        g, _ = clean_metric_gdf(
            _gdf([poly], crs="EPSG:4326"),
            metric_crs=METRIC, drop_zero_area=False,
        )
        area = float(g.geometry.area.iloc[0])
        self.assertGreater(area, 1.0)        # not degrees-squared (~1e-6)
        self.assertLess(area, MAX_PLAUSIBLE_POLY_M2)


# --------------------------------------------------------------------------
# read_polygons — path-level loader + the two backward-compat wrappers
# --------------------------------------------------------------------------
class TestReadPolygons(unittest.TestCase):
    def _write(self, geoms, crs=METRIC, layer=None):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "polys.gpkg")
        kw = {"layer": layer} if layer else {}
        _gdf(geoms, crs=crs).to_file(path, driver="GPKG", **kw)
        return path

    def test_empty_gpkg_4tuple(self):
        path = self._write([])
        self.assertEqual(read_polygons(path, metric_crs=METRIC, drop_zero_area=False),
                         (0, 0.0, 0.0, 0))

    def test_empty_gpkg_5tuple(self):
        path = self._write([])
        self.assertEqual(
            read_polygons(path, metric_crs=METRIC, drop_zero_area=True, with_union=True),
            (0, 0.0, 0.0, 0, None),
        )

    def test_4tuple_sum_keeps_zero_area(self):
        path = self._write([box(0, 0, 10, 10), Point(5, 5)])
        n, total, mx, ndrop = read_polygons(
            path, metric_crs=METRIC, drop_zero_area=False)
        self.assertEqual(n, 2)            # Point kept
        self.assertAlmostEqual(total, 100.0)
        self.assertAlmostEqual(mx, 100.0)
        self.assertEqual(ndrop, 0)

    def test_5tuple_union_drops_zero_area(self):
        path = self._write([box(0, 0, 10, 10), box(5, 5, 15, 15), Point(50, 50)])
        n, total, mx, ndrop, union = read_polygons(
            path, metric_crs=METRIC, drop_zero_area=True, with_union=True)
        self.assertEqual(n, 2)            # Point dropped
        self.assertAlmostEqual(total, 200.0)   # naive sum (overlap not removed)
        self.assertEqual(ndrop, 0)
        # union removes the 25 m² overlap → 175 m²
        self.assertAlmostEqual(union.area, 175.0)

    def test_missing_layer_falls_back_to_first(self):
        path = self._write([box(0, 0, 10, 10)], layer="solar")
        # ask for a layer that doesn't exist → falls back to the (only) layer
        n, total, _, _ = read_polygons(
            path, metric_crs=METRIC, drop_zero_area=False, layer="does_not_exist")
        self.assertEqual(n, 1)
        self.assertAlmostEqual(total, 100.0)

    def test_named_layer_selected(self):
        path = self._write([box(0, 0, 10, 10)], layer="solar")
        n, _, _, _ = read_polygons(
            path, metric_crs=METRIC, drop_zero_area=False, layer="solar")
        self.assertEqual(n, 1)


# --------------------------------------------------------------------------
# backward-compat aliases preserve the original semantics exactly
# --------------------------------------------------------------------------
class TestBackwardCompatWrappers(unittest.TestCase):
    def _write(self, geoms, crs=METRIC):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "polys.gpkg")
        _gdf(geoms, crs=crs).to_file(path, driver="GPKG")
        return path

    def test_sum_area_m2_keeps_zero_area_4tuple(self):
        path = self._write([box(0, 0, 10, 10), Point(5, 5)])
        result = _sum_area_m2(path, METRIC, None)
        self.assertEqual(len(result), 4)
        n, total, mx, ndrop = result
        self.assertEqual(n, 2)            # _sum_area_m2 KEEPS zero-area
        self.assertAlmostEqual(total, 100.0)

    def test_read_polys_geom_drops_zero_area_5tuple(self):
        path = self._write([box(0, 0, 10, 10), Point(5, 5)])
        result = _read_polys_geom(path, METRIC, None)
        self.assertEqual(len(result), 5)
        n, total, mx, ndrop, union = result
        self.assertEqual(n, 1)            # _read_polys_geom DROPS zero-area
        self.assertAlmostEqual(total, 100.0)
        self.assertAlmostEqual(union.area, 100.0)

    def test_constant_alias_identity(self):
        self.assertEqual(_MAX_PLAUSIBLE_POLY_M2, 20_000.0)
        self.assertIs(_MAX_PLAUSIBLE_POLY_M2, MAX_PLAUSIBLE_POLY_M2)

    def test_geometry_finite_alias_identity(self):
        from core.polygon_validation import _geometry_finite
        self.assertIs(_geometry_finite, geometry_finite)


if __name__ == "__main__":
    unittest.main()
