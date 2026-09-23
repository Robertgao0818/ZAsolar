"""Chip/polygon intersection and IoU helpers.

Chip membership uses intersection area > 0 in the tile CRS. A shared edge
has area 0 and does not count. There is no square-metre floor.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import shapely


def _geoms_in_crs(frame: gpd.GeoDataFrame, crs) -> np.ndarray:
    """Row-aligned geometries in ``crs``. Empty inputs stay empty so ids do not shift."""
    if frame.empty:
        return np.empty(0, dtype=object)
    if frame.crs is None:
        raise ValueError("frame has no CRS")
    projected = frame if frame.crs.equals(crs) else frame.to_crs(crs)
    geoms = []
    for geom in projected.geometry.values:
        if geom is None or geom.is_empty:
            geoms.append(shapely.GeometryCollection())
        else:
            geoms.append(geom)
    return shapely.make_valid(np.asarray(geoms, dtype=object))


def chips_hit_mask(chips: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> np.ndarray:
    """True for each chip whose valid footprint intersects a polygon with area > 0.

    Polygons are projected into the chip CRS before the test. The caller
    passes chips in the tile CRS (EPSG:4326 for this pilot).
    """
    hit = np.zeros(len(chips), dtype=bool)
    if polygons is None or polygons.empty:
        return hit
    if polygons.crs is None:
        raise ValueError("polygons have no CRS")
    if chips.crs is None:
        raise ValueError("chips have no CRS")
    poly_geoms = _geoms_in_crs(polygons, chips.crs)
    if len(poly_geoms) == 0:
        return hit
    chip_geoms = np.asarray(chips.geometry.values)
    tree = shapely.STRtree(poly_geoms)
    pairs = tree.query(chip_geoms, predicate="intersects")
    if pairs.size == 0:
        return hit
    chip_index = pairs[0]
    poly_index = pairs[1]
    areas = shapely.area(shapely.intersection(chip_geoms[chip_index], poly_geoms[poly_index]))
    hit[np.unique(chip_index[areas > 0])] = True
    return hit


def count_with_partner(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, iou_min: float) -> int:
    """How many left geometries have some right geometry at IoU >= iou_min.

    IoU is computed in ``left``'s CRS. Both frames must already be in a
    projected CRS if the caller wants metre-based IoU; this function does
    not pick a CRS of its own beyond matching right onto left.
    """
    if left.empty or right.empty:
        return 0
    if left.crs is None:
        raise ValueError("left has no CRS")
    left_geoms = _geoms_in_crs(left, left.crs)
    right_geoms = _geoms_in_crs(right, left.crs)
    if len(left_geoms) == 0 or len(right_geoms) == 0:
        return 0
    tree = shapely.STRtree(right_geoms)
    pairs = tree.query(left_geoms, predicate="intersects")
    if pairs.size == 0:
        return 0
    left_index = pairs[0]
    right_index = pairs[1]
    inter = shapely.area(shapely.intersection(left_geoms[left_index], right_geoms[right_index]))
    union = (
        shapely.area(left_geoms[left_index])
        + shapely.area(right_geoms[right_index])
        - inter
    )
    ok = (union > 0) & (inter / np.where(union > 0, union, 1.0) >= iou_min)
    return int(np.unique(left_index[ok]).size)


def iou_component_count(frame: gpd.GeoDataFrame, iou_min: float = 0.10) -> int:
    """Connected components, linking two polygons when their IoU is >= iou_min."""
    geoms = _geoms_in_crs(frame, frame.crs) if not frame.empty else np.empty(0, dtype=object)
    n = len(geoms)
    if n == 0:
        return 0
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    tree = shapely.STRtree(geoms)
    pairs = tree.query(geoms, predicate="intersects")
    if pairs.size:
        left = pairs[0]
        right = pairs[1]
        keep = left < right
        left = left[keep]
        right = right[keep]
        if len(left):
            inter = shapely.area(shapely.intersection(geoms[left], geoms[right]))
            union_area = shapely.area(geoms[left]) + shapely.area(geoms[right]) - inter
            linked = (union_area > 0) & (inter / np.where(union_area > 0, union_area, 1.0) >= iou_min)
            for i, j in zip(left[linked].tolist(), right[linked].tolist()):
                union(int(i), int(j))
    return len({find(i) for i in range(n)})


def best_partner_grid(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    right_grid_ids: np.ndarray,
    iou_min: float,
) -> list[str | None]:
    """For each left geometry, the right-hand grid id of its best IoU partner.

    None when no partner reaches ``iou_min``. ``right_grid_ids`` aligns with
    ``right``'s rows.
    """
    n = len(left)
    result: list[str | None] = [None] * n
    if n == 0 or right.empty:
        return result
    if left.crs is None:
        raise ValueError("left has no CRS")
    if len(right_grid_ids) != len(right):
        raise ValueError("right_grid_ids must align with right rows")
    left_geoms = _geoms_in_crs(left, left.crs)
    right_geoms = _geoms_in_crs(right, left.crs)
    if len(left_geoms) != n or len(right_geoms) != len(right):
        raise RuntimeError("geometry cleaning changed the row count")
    tree = shapely.STRtree(right_geoms)
    pairs = tree.query(left_geoms, predicate="intersects")
    if pairs.size == 0:
        return result
    left_index = pairs[0]
    right_index = pairs[1]
    inter = shapely.area(shapely.intersection(left_geoms[left_index], right_geoms[right_index]))
    union = (
        shapely.area(left_geoms[left_index])
        + shapely.area(right_geoms[right_index])
        - inter
    )
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    order = np.argsort(-iou)
    seen = np.zeros(n, dtype=bool)
    for pos in order:
        i = int(left_index[pos])
        if seen[i]:
            continue
        if iou[pos] < iou_min:
            break
        seen[i] = True
        result[i] = str(right_grid_ids[int(right_index[pos])])
    return result
