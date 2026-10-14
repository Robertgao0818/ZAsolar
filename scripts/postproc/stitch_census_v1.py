#!/usr/bin/python3
"""Stitch v1 for the Durban census polygons.

The 2026-10-08 delivery glued detections with IoU > 0.1 (intersection area
divided by union area) and a plain union. That rule misses fragments of one
installation that were cut on a 400 px detector-window edge: many of those
pieces only touch, or overlap in a sliver whose IoU is at most 0.1, so they
stay separate and the same roof is counted twice.

This script replaces that rule with a gap graph. Two raw polygons are linked
when they overlap or the gap between them is at most g metres. Links are
transitive: a connected component (scipy.sparse.csgraph) becomes one block.
Block geometry is the union of its members. A block with more than one member
then gets a morphological closing of fixed radius ``close_m`` (buffer +r with
mitre joins, then buffer -r with mitre joins). The radius does not grow with
g. A gap larger than about 2 * close_m can remain a MultiPolygon; ``n_parts``
records that.

v1 is this gap rule. The delivery layer is reproduced only by ``--also-legacy``
and is not the default output.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely import MultiPolygon, STRtree, area, buffer, get_num_geometries, get_type_id, is_empty, is_valid, make_valid
from shapely import intersection as shp_intersection
from shapely import touches as shp_touches
from shapely import union_all as shp_union_all
from shapely.geometry import box

DEFAULT_INPUT = Path(
    "/home/gao/zasolar_data/gate_pilot_2026-09/durban_census_delivery_20261008/durban_census_inventory.gpkg"
)
DEFAULT_OUT = Path("/home/gao/zasolar_data/durban_stitch_sam3_20261009/A_stitch")
GATE_PILOT = Path("/home/gao/zasolar_data/gate_pilot_2026-09")
CRS = "EPSG:32736"
READ_COLUMNS = [
    "gridcell_id",
    "source_tile",
    "score",
    "area_m2",
    "n_components_dropped",
]
# Delivered IoU>0.1 layer, from delivery_manifest.json. The 0.1% check is
# relative to these two numbers.
LEGACY_N_BLOCKS = 36654
LEGACY_AREA_M2 = 1927374.3858453454
LEGACY_TOL = 0.001
IOU_THRESH = 0.1
GEOM_COLLECTION = 7


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gap_tag(gap: float) -> str:
    return f"{gap}"


def parse_gaps(text: str) -> list[float]:
    gaps: list[float] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value < 0:
            raise SystemExit(f"gap must be >= 0, got {token}")
        gaps.append(value)
    if not gaps:
        raise SystemExit("no gaps given")
    return gaps


def as_geom_array(geoms) -> np.ndarray:
    """Object array of geometries. np.asarray would unpack coordinate sequences."""
    src = list(geoms)
    out = np.empty(len(src), dtype=object)
    for i, geom in enumerate(src):
        out[i] = geom
    return out


def upper_pairs(idx: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Undirected i<j pairs from an STRtree query of shape (2, n_pairs)."""
    if idx.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    if idx.ndim != 2 or idx.shape[0] != 2:
        raise RuntimeError(f"expected STRtree query shape (2, n), got {idx.shape}")
    left = idx[0]
    right = idx[1]
    keep = left < right
    left = np.asarray(left[keep], dtype=np.int64)
    right = np.asarray(right[keep], dtype=np.int64)
    if len(left) == 0:
        return left, right
    key = left * np.int64(n) + right
    key = np.unique(key)
    width = np.int64(n)
    return key // width, key % width


def pairwise_intersection_area(geoms: np.ndarray, left: np.ndarray, right: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Vectorized shapely intersection area, in chunks so geometries can be freed."""
    n_pairs = len(left)
    out = np.empty(n_pairs, dtype=np.float64)
    if n_pairs == 0:
        return out
    for start in range(0, n_pairs, chunk):
        stop = min(start + chunk, n_pairs)
        sl = slice(start, stop)
        out[sl] = area(shp_intersection(geoms[left[sl]], geoms[right[sl]]))
        if n_pairs > chunk:
            log(f"  intersection area {stop}/{n_pairs}")
    return out


def labels_from_edges(n: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if len(left) == 0:
        return np.arange(n, dtype=np.int32)
    data = np.ones(len(left), dtype=np.uint8)
    graph = csr_matrix((data, (left, right)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False, return_labels=True)
    log(f"  components={n_comp} edges={len(left)}")
    return np.asarray(labels, dtype=np.int32)


def edges_for_gap(geoms: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Gap graph. g=0 keeps positive-area overlaps and touches. g>0 keeps dwithin."""
    n = len(geoms)
    tree = STRtree(geoms)
    if gap == 0.0:
        left, right = upper_pairs(tree.query(geoms, predicate="intersects"), n)
        if len(left) == 0:
            log("  g=0 intersecting pairs=0")
            return left, right
        inter = pairwise_intersection_area(geoms, left, right)
        keep = inter > 0
        zero = np.flatnonzero(~keep)
        n_touch = 0
        if len(zero):
            touch = np.asarray(shp_touches(geoms[left[zero]], geoms[right[zero]]), dtype=bool)
            keep[zero] = touch
            n_touch = int(touch.sum())
        log(
            f"  g=0 pairs={len(left)} area>0={int((inter > 0).sum())} "
            f"touch_only={n_touch} dropped={int((~keep).sum())}"
        )
        return left[keep], right[keep]
    left, right = upper_pairs(tree.query(geoms, predicate="dwithin", distance=float(gap)), n)
    log(f"  g={gap_tag(gap)} dwithin pairs={len(left)}")
    return left, right


def edges_for_legacy(geoms: np.ndarray, areas_m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Delivery rule: bbox candidates, then intersection area > 0 and IoU > 0.1.

    No make_valid and no closing, matching build_delivery.merge_overlapping.
    The comparison is strict (> 0.1), same as the delivery script.
    """
    n = len(geoms)
    tree = STRtree(geoms)
    left, right = upper_pairs(tree.query(geoms), n)
    log(f"  legacy bbox pairs={len(left)}")
    if len(left) == 0:
        return left, right
    inter = pairwise_intersection_area(geoms, left, right)
    union = areas_m2[left] + areas_m2[right] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(inter, union, out=np.zeros(len(inter), dtype=np.float64), where=union > 0)
    keep = (inter > 0) & (union > 0) & (ratio > IOU_THRESH)
    log(f"  legacy IoU>0.1 edges={int(keep.sum())}")
    return left[keep], right[keep]


def polygonal_only(geom):
    """Drop non-polygon parts of a GeometryCollection produced by make_valid or buffer."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return geom
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    polys = []
    for part in parts:
        kind = part.geom_type
        if kind == "Polygon":
            polys.append(part)
        elif kind == "MultiPolygon":
            polys.extend(list(part.geoms))
        elif kind == "GeometryCollection":
            extracted = polygonal_only(part)
            if extracted is None or extracted.is_empty:
                continue
            if extracted.geom_type == "Polygon":
                polys.append(extracted)
            elif extracted.geom_type == "MultiPolygon":
                polys.extend(list(extracted.geoms))
    if not polys:
        return geom
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def _copy_geoms(geoms: np.ndarray) -> np.ndarray:
    out = np.empty(len(geoms), dtype=object)
    for i in range(len(geoms)):
        out[i] = geoms[i]
    return out


def repair_geometries(geoms: np.ndarray) -> np.ndarray:
    out = _copy_geoms(geoms)
    invalid = ~np.asarray(is_valid(out), dtype=bool)
    if invalid.any():
        fixed = make_valid(out[invalid])
        for slot, src in enumerate(np.flatnonzero(invalid).tolist()):
            out[src] = fixed[slot]
    collections = np.asarray(get_type_id(out), dtype=np.int32) == GEOM_COLLECTION
    for src in np.flatnonzero(collections).tolist():
        out[src] = polygonal_only(out[src])
    return out


def group_slices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort by component label. Returns order, label ids, starts, counts."""
    order = np.argsort(labels, kind="mergesort")
    ordered = labels[order]
    if len(ordered) == 0:
        empty = np.empty(0, dtype=np.int64)
        return order, empty, empty, empty
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ordered)])
    uniq = ordered[starts]
    return order, uniq, starts.astype(np.int64), counts.astype(np.int64)


def build_block_geometries(
    geoms: np.ndarray,
    order: np.ndarray,
    uniq: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    close_m: float,
    repair: bool,
) -> np.ndarray:
    n_blocks = len(uniq)
    out = np.empty(n_blocks, dtype=object)
    single = counts == 1
    for pos in np.flatnonzero(single).tolist():
        out[pos] = geoms[int(order[int(starts[pos])])]
    multi = np.flatnonzero(~single)
    for done, pos in enumerate(multi.tolist(), start=1):
        start = int(starts[pos])
        count = int(counts[pos])
        out[pos] = shp_union_all(geoms[order[start:start + count]])
        if done % 5000 == 0:
            log(f"  unioned {done}/{len(multi)} multi-piece blocks")
    if close_m > 0 and len(multi):
        # Fixed radius for every gap. Do not use g/2 + close_m: a wide gap is
        # allowed to stay a MultiPolygon, and n_parts records the pieces.
        current = np.empty(len(multi), dtype=object)
        for i, pos in enumerate(multi.tolist()):
            current[i] = out[pos]
        closed = buffer(current, float(close_m), join_style="mitre", mitre_limit=5.0)
        closed = buffer(closed, -float(close_m), join_style="mitre", mitre_limit=5.0)
        empty = np.asarray(is_empty(closed), dtype=bool)
        for i in np.flatnonzero(empty).tolist():
            closed[i] = current[i]
        for i, pos in enumerate(multi.tolist()):
            out[pos] = closed[i]
    if repair:
        fallback = _copy_geoms(out)
        out = repair_geometries(out)
        empty = np.asarray(is_empty(out), dtype=bool)
        for i in np.flatnonzero(empty).tolist():
            out[i] = fallback[i]
    return out


def unique_join(values: np.ndarray, order: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = values[order]
    joined = np.empty(len(starts), dtype=object)
    nunique = np.empty(len(starts), dtype=np.int32)
    for i, (start, count) in enumerate(zip(starts.tolist(), counts.tolist())):
        if count == 1:
            joined[i] = ordered[int(start)]
            nunique[i] = 1
            continue
        items = sorted(set(ordered[int(start):int(start) + int(count)].tolist()))
        joined[i] = ",".join(items)
        nunique[i] = len(items)
    return joined, nunique


def _contained_count_from_pairs(
    n: int,
    left: np.ndarray,
    right: np.ndarray,
    inter: np.ndarray,
    areas_m2: np.ndarray,
) -> int:
    """Blocks whose intersection with one other block is more than half their area.

    True is written through unique indices. Fancy-index ``|=`` drops a True when
    the same block index appears again with a False later in the pair list.
    """
    contained = np.zeros(n, dtype=bool)
    left_hit = (areas_m2[left] > 0) & (inter > 0.5 * areas_m2[left])
    right_hit = (areas_m2[right] > 0) & (inter > 0.5 * areas_m2[right])
    if left_hit.any():
        contained[np.unique(left[left_hit])] = True
    if right_hit.any():
        contained[np.unique(right[right_hit])] = True
    return int(contained.sum())


def overlap_stats(geoms: np.ndarray, areas_m2: np.ndarray) -> tuple[float, int]:
    """Sum of pairwise intersection area (i<j) and blocks >50% covered by another."""
    n = len(geoms)
    if n == 0:
        return 0.0, 0
    left, right = upper_pairs(STRtree(geoms).query(geoms, predicate="intersects"), n)
    if len(left) == 0:
        return 0.0, 0
    inter = pairwise_intersection_area(geoms, left, right)
    contained = _contained_count_from_pairs(n, left, right, inter, areas_m2)
    return float(inter.sum()), contained


def piece_quantiles(n_pieces: np.ndarray) -> dict:
    if len(n_pieces) == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0, "mean": 0.0}
    vals = np.quantile(n_pieces.astype(np.float64), [0.5, 0.9, 0.99])
    return {
        "p50": float(vals[0]),
        "p90": float(vals[1]),
        "p99": float(vals[2]),
        "max": int(n_pieces.max()),
        "mean": float(n_pieces.mean()),
    }


def size_hist(areas_m2: np.ndarray) -> dict:
    return {
        "lt20": int((areas_m2 < 20).sum()),
        "b20_100": int(((areas_m2 >= 20) & (areas_m2 < 100)).sum()),
        "b100_1000": int(((areas_m2 >= 100) & (areas_m2 < 1000)).sum()),
        "ge1000": int((areas_m2 >= 1000).sum()),
    }


def stitch(
    geoms: np.ndarray,
    areas_m2: np.ndarray,
    scores: np.ndarray,
    cells: np.ndarray,
    tiles: np.ndarray,
    *,
    rule: str,
    gap: float | None,
    close_m: float,
    repair: bool,
) -> tuple[gpd.GeoDataFrame, dict, np.ndarray]:
    t0 = time.perf_counter()
    log(f"stitch {rule}")
    if gap is None:
        left, right = edges_for_legacy(geoms, areas_m2)
    else:
        left, right = edges_for_gap(geoms, gap)
    labels = labels_from_edges(len(geoms), left, right)
    order, uniq, starts, counts = group_slices(labels)
    if int(counts.sum()) != len(geoms):
        raise RuntimeError(f"{rule}: component sizes do not partition the input")
    block_geoms = build_block_geometries(geoms, order, uniq, starts, counts, close_m=close_m, repair=repair)
    n_pieces = counts.astype(np.int32)
    gridcell_ids, n_cells = unique_join(cells, order, starts, counts)
    _, n_tiles = unique_join(tiles, order, starts, counts)
    scores_ord = scores[order]
    areas_ord = areas_m2[order]
    score_max = np.maximum.reduceat(scores_ord, starts)
    raw_area_sum = np.add.reduceat(areas_ord, starts)
    weighted = np.add.reduceat(scores_ord * areas_ord, starts)
    score_mean = weighted / np.maximum(raw_area_sum, 1e-12)
    final_area = np.asarray(area(block_geoms), dtype=np.float64)
    bounds = np.asarray([geom.bounds for geom in block_geoms], dtype=np.float64)
    bbox_w = bounds[:, 2] - bounds[:, 0]
    bbox_h = bounds[:, 3] - bounds[:, 1]
    max_side = np.maximum(bbox_w, bbox_h)
    n_parts = np.asarray(get_num_geometries(block_geoms), dtype=np.int32)
    if gap is None:
        block_ids = np.array([f"legacy_{int(lab):06d}" for lab in uniq.tolist()])
        row_ids = np.array([f"legacy_{int(lab):06d}" for lab in labels.tolist()])
    else:
        prefix = f"g{gap_tag(gap)}_"
        block_ids = np.char.add(prefix, np.char.zfill(uniq.astype(np.int64).astype(str), 6))
        row_ids = np.char.add(prefix, np.char.zfill(labels.astype(np.int64).astype(str), 6))
    log(f"  overlap scan on {len(block_geoms)} blocks")
    residual, n_contained = overlap_stats(block_geoms, final_area)
    wide = max_side > 75
    multi_part = n_parts > 1
    stats = {
        "n_blocks": int(len(block_geoms)),
        "total_area_m2": float(final_area.sum()),
        "n_ge100": int((final_area >= 100).sum()),
        "max_area_m2": float(final_area.max()) if len(final_area) else 0.0,
        "residual_overlap_m2": residual,
        "n_blocks_multi_cell": int((n_cells > 1).sum()),
        "n_blocks_multi_tile": int((n_tiles > 1).sum()),
        "n_blocks_max_side_gt75": int(wide.sum()),
        "area_max_side_gt75_m2": float(final_area[wide].sum()) if wide.any() else 0.0,
        "hist": size_hist(final_area),
        "pieces_quantiles": piece_quantiles(n_pieces),
        "n_edges": int(len(left)),
        "n_blocks_multi_piece": int((n_pieces > 1).sum()),
        "n_blocks_n_parts_gt1": int(multi_part.sum()),
        "area_n_parts_gt1_m2": float(final_area[multi_part].sum()) if multi_part.any() else 0.0,
        "n_blocks_gt50pct_contained": n_contained,
        "close_m": float(close_m if gap is not None else 0.0),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    frame = pd.DataFrame(
        {
            "block_id": block_ids,
            "n_pieces": n_pieces,
            "gridcell_ids": gridcell_ids,
            "n_cells": n_cells,
            "n_tiles": n_tiles,
            "score_max": score_max,
            "score_mean_area_weighted": score_mean,
            "area_m2": final_area,
            "raw_area_sum_m2": raw_area_sum,
            "bbox_w_m": bbox_w,
            "bbox_h_m": bbox_h,
            "max_side_m": max_side,
            "n_parts": n_parts,
        }
    )
    out = gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries(block_geoms, crs=CRS), crs=CRS)
    log(
        f"  {rule}: blocks={stats['n_blocks']} area_m2={stats['total_area_m2']:.1f} "
        f"ge100={stats['n_ge100']} residual_m2={stats['residual_overlap_m2']:.1f} "
        f"elapsed_s={stats['elapsed_s']}"
    )
    return out, stats, row_ids


def union_area_by_key(geoms: np.ndarray, keys: np.ndarray) -> float:
    """Union within each key, then union those parts. Equals a full union_all."""
    order = np.argsort(keys, kind="mergesort")
    ordered = keys[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(ordered)])
    parts = np.empty(len(starts), dtype=object)
    for i, (start, count) in enumerate(zip(starts.tolist(), counts.tolist())):
        member_idx = order[int(start):int(start) + int(count)]
        if count == 1:
            parts[i] = geoms[int(member_idx[0])]
            continue
        members = geoms[member_idx]
        try:
            parts[i] = shp_union_all(members)
        except Exception:
            parts[i] = shp_union_all(make_valid(members))
            log(f"  group union needed make_valid ({i})")
        if i and i % 200 == 0:
            log(f"  group unions {i}/{len(starts)}")
    log(f"  union of {len(parts)} group unions")
    return float(area(shp_union_all(parts)))


def render_markdown(raw: dict, rules: list[tuple[str, dict]], close_m: float) -> str:
    header = (
        "| rule | n_blocks | area_m2 | ge100 | max_area | residual_overlap_m2 "
        "| blocks>75m side (n, area) | multi-cell blocks |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Durban census stitch v1",
        "",
        (
            f"Raw polygons: {raw['n']}. "
            f"Sum of polygon areas: {raw['total_area_m2']:.1f} m2. "
            f"Area of the union of all raw polygons: {raw['union_all_area_m2']:.1f} m2. "
            f"Largest raw polygon: {raw['max_area_m2']:.1f} m2. "
            f"Share of rows with n_components_dropped>0: {raw['share_n_components_dropped_gt0']:.4f}."
        ),
        "",
        (
            f"Gap blocks are connected components of overlap-or-gap<=g. "
            f"Closing radius is fixed at {close_m} m and is not increased for larger gaps, "
            f"so a block can stay a MultiPolygon."
        ),
        "",
        header,
        sep,
    ]
    for name, stats in rules:
        wide = f"{stats['n_blocks_max_side_gt75']}, {stats['area_max_side_gt75_m2']:.1f}"
        lines.append(
            "| {rule} | {n} | {area:.1f} | {ge100} | {max_area:.1f} | {residual:.1f} | {wide} | {multi} |".format(
                rule=name,
                n=stats["n_blocks"],
                area=stats["total_area_m2"],
                ge100=stats["n_ge100"],
                max_area=stats["max_area_m2"],
                residual=stats["residual_overlap_m2"],
                wide=wide,
                multi=stats["n_blocks_multi_cell"],
            )
        )
    lines.append("")
    lines.append("Pieces merged per block (p50, p90, p99, max):")
    lines.append("")
    for name, stats in rules:
        q = stats["pieces_quantiles"]
        lines.append(f"- {name}: {q['p50']:.1f}, {q['p90']:.1f}, {q['p99']:.1f}, {q['max']}")
    legacy = dict(rules).get("legacy_iou_0.1")
    if legacy is not None:
        lines.append("")
        lines.append(
            f"Legacy blocks >50% contained in another legacy block: "
            f"{legacy['n_blocks_gt50pct_contained']} (expected about 2378). "
            f"Legacy pairwise overlap: {legacy['residual_overlap_m2']:.1f} m2 "
            f"(expected about 41562)."
        )
    lines.append("")
    return "\n".join(lines)


def assert_legacy(stats: dict) -> None:
    n_rel = abs(stats["n_blocks"] - LEGACY_N_BLOCKS) / LEGACY_N_BLOCKS
    a_rel = abs(stats["total_area_m2"] - LEGACY_AREA_M2) / LEGACY_AREA_M2
    log(
        f"legacy check: n_blocks={stats['n_blocks']} expected {LEGACY_N_BLOCKS} "
        f"rel={n_rel:.6f}; area_m2={stats['total_area_m2']:.3f} expected {LEGACY_AREA_M2:.3f} "
        f"rel={a_rel:.6f}"
    )
    if n_rel > LEGACY_TOL or a_rel > LEGACY_TOL:
        raise SystemExit(
            f"legacy IoU>0.1 did not reproduce the delivery "
            f"({stats['n_blocks']} blocks, {stats['total_area_m2']:.1f} m2; "
            f"expected {LEGACY_N_BLOCKS} and {LEGACY_AREA_M2:.1f}, tolerance {LEGACY_TOL:.1%})"
        )


def write_gpkg(gdf: gpd.GeoDataFrame, path: Path) -> None:
    if path.exists():
        path.unlink()
    gdf.to_file(path, layer="blocks", driver="GPKG", index=False)
    info = pyogrio.read_info(path, layer="blocks")
    if int(info["features"]) != len(gdf):
        raise RuntimeError(f"{path} wrote {info['features']} features, expected {len(gdf)}")


def synthetic_check() -> None:
    """Small rectangles covering touch, IoU, containment, and closing."""
    squares = as_geom_array(
        [
            box(0, 0, 10, 10),       # touches the next square along x=10
            box(10, 0, 20, 10),      # overlaps the third square
            box(18, 0, 28, 10),
            box(100, 0, 110, 10),    # 0.2 m gap to the next square
            box(110.2, 0, 120.2, 10),
            box(200, 0, 210, 10),    # 0.4 m gap to the next square
            box(210.4, 0, 220.4, 10),
            box(300, 0, 310, 10),    # contains the 2x2 square; IoU = 0.04
            box(304, 4, 306, 6),
        ]
    )
    areas_m2 = np.asarray(area(squares), dtype=np.float64)
    scores = np.array([1, 3, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    cells = np.array(["A", "B", "B", "C", "C", "D", "D", "E", "E"])
    tiles = np.array(["t0", "t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"])

    g0, stats0, _ = stitch(
        squares, areas_m2, scores, cells, tiles, rule="synthetic_g0", gap=0.0, close_m=0.15, repair=True
    )
    # Touch A-B plus overlap B-C is one block. Each gapped pair stays apart.
    # The contained pair overlaps, so it is one block. 1 + 2 + 2 + 1 = 6.
    if stats0["n_blocks"] != 6:
        raise SystemExit(f"synthetic g=0 expected 6 blocks, got {stats0['n_blocks']}")
    touch = g0.loc[g0["n_pieces"] == 3].iloc[0]
    if abs(float(touch.area_m2) - 280) > 1e-4:
        raise SystemExit(f"synthetic touched union area {touch.area_m2}")
    if int(touch.n_cells) != 2 or touch.gridcell_ids != "A,B":
        raise SystemExit(f"synthetic gridcell join {touch.gridcell_ids}")
    if abs(float(touch.score_max) - 3) > 1e-9:
        raise SystemExit("synthetic score_max")
    # Area-weighted mean of scores 1, 3, 1 on areas 100, 100, 100.
    if abs(float(touch.score_mean_area_weighted) - (1 + 3 + 1) / 3) > 1e-9:
        raise SystemExit("synthetic weighted score")

    g25, stats25, _ = stitch(
        squares, areas_m2, scores, cells, tiles, rule="synthetic_g0.25", gap=0.25, close_m=0.15, repair=True
    )
    # 0.2 m is inside the gap, and 2*0.15 m closing seals it. 0.4 m is not a g=0.25 edge.
    bridged = g25.loc[np.isclose(g25.area_m2, 202, atol=1e-3)]
    if len(bridged) != 1 or int(bridged.iloc[0].n_parts) != 1:
        raise SystemExit(
            f"synthetic 0.2 m gap did not close: parts={g25.n_parts.tolist()} areas={g25.area_m2.tolist()}"
        )
    if stats25["n_blocks"] != 5:
        raise SystemExit(f"synthetic g=0.25 expected 5 blocks, got {stats25['n_blocks']}")

    g50, _, _ = stitch(
        squares, areas_m2, scores, cells, tiles, rule="synthetic_g0.5", gap=0.5, close_m=0.15, repair=True
    )
    # 0.4 m is an edge at g=0.5, but it is wider than 2*close_m, so the block stays split.
    two_piece = g50.loc[g50["n_pieces"] == 2]
    wide_gap = two_piece.loc[np.isclose(two_piece["area_m2"].to_numpy(), 200, atol=1e-3)]
    if len(wide_gap) != 1 or int(wide_gap.iloc[0].n_parts) != 2:
        raise SystemExit(
            f"synthetic 0.4 m gap should stay multi-part: parts={g50.n_parts.tolist()} areas={g50.area_m2.tolist()}"
        )
    if len(g50) != 4:
        raise SystemExit(f"synthetic g=0.5 expected 4 blocks, got {len(g50)}")

    legacy, stats_l, _ = stitch(
        squares, areas_m2, scores, cells, tiles, rule="synthetic_legacy", gap=None, close_m=0.0, repair=False
    )
    # Only the 20 m2 overlap (IoU ~0.111) merges. The contained 4 m2 square has IoU 0.04.
    if stats_l["n_blocks"] != 8:
        raise SystemExit(f"synthetic legacy expected 8 blocks, got {stats_l['n_blocks']}")
    if stats_l["n_blocks_gt50pct_contained"] != 1:
        raise SystemExit(f"synthetic contained count {stats_l['n_blocks_gt50pct_contained']}")
    if abs(stats_l["residual_overlap_m2"] - 4) > 1e-4:
        raise SystemExit(f"synthetic legacy overlap {stats_l['residual_overlap_m2']}")
    # One polygon is >50% covered by the first neighbour and only slightly by the
    # second. Both pairs put it on the same side of i<j, which used to drop it.
    contained = _contained_count_from_pairs(
        3,
        np.array([0, 1]),
        np.array([2, 2]),
        np.array([4.0, 0.8]),
        np.array([100.0, 100.0, 5.2]),
    )
    if contained != 1:
        raise SystemExit(f"duplicate-index containment count {contained}")
    log("synthetic check ok")


def refuse_output_dir(out_dir: Path, input_path: Path) -> None:
    resolved = out_dir.resolve()
    input_dir = input_path.resolve().parent
    gate = GATE_PILOT.resolve()
    if resolved == input_dir or input_dir in resolved.parents:
        raise SystemExit(f"refusing to write into the input directory {input_dir}")
    if resolved == gate or gate in resolved.parents:
        raise SystemExit(f"refusing to write under read-only {gate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch Durban census fragments by gap, v1.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gaps", default="0,0.25,0.5,1.0")
    parser.add_argument("--close-m", type=float, default=0.15)
    parser.add_argument("--write-gpkg-for", default="0.25")
    parser.add_argument("--also-legacy", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.close_m < 0:
        raise SystemExit("--close-m must be >= 0")
    gaps = parse_gaps(args.gaps)
    write_gaps = parse_gaps(args.write_gpkg_for)
    refuse_output_dir(args.out_dir, args.input)
    if not args.skip_synthetic:
        synthetic_check()

    log(f"reading {args.input}")
    t0 = time.perf_counter()
    gdf = pyogrio.read_dataframe(args.input, columns=READ_COLUMNS)
    raw_geoms = as_geom_array(gdf.geometry)
    log(f"  read {len(raw_geoms)} polygons in {time.perf_counter() - t0:.1f}s")
    geom_area = np.asarray(area(raw_geoms), dtype=np.float64)
    valid_mask = np.asarray(is_valid(raw_geoms), dtype=bool)
    n_invalid = int((~valid_mask).sum())
    log(f"  invalid polygons={n_invalid}")
    if n_invalid:
        valid_geoms = _copy_geoms(raw_geoms)
        fixed = make_valid(raw_geoms[~valid_mask])
        for slot, src in enumerate(np.flatnonzero(~valid_mask).tolist()):
            valid_geoms[src] = fixed[slot]
        valid_geoms = repair_geometries(valid_geoms)
    else:
        valid_geoms = raw_geoms
    valid_area = np.asarray(area(valid_geoms), dtype=np.float64)
    scores = pd.to_numeric(gdf["score"], errors="coerce").to_numpy(dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0)
    cells = gdf["gridcell_id"].fillna("").astype(str).to_numpy()
    tiles = gdf["source_tile"].fillna("").astype(str).to_numpy()
    dropped = pd.to_numeric(gdf["n_components_dropped"], errors="coerce").to_numpy()
    dropped = np.nan_to_num(dropped, nan=0.0)

    rules: list[tuple[str, dict]] = []
    written: list[str] = []
    legacy_stats = None
    if args.also_legacy:
        # Original geometries, no make_valid and no closing, so the partition
        # can be checked against the delivered 36,654-block layer.
        _, legacy_stats, _ = stitch(
            raw_geoms,
            geom_area,
            scores,
            cells,
            tiles,
            rule="legacy_iou_0.1",
            gap=None,
            close_m=0.0,
            repair=False,
        )
        rules.append(("legacy_iou_0.1", legacy_stats))

    log("raw union_all")
    t1 = time.perf_counter()
    union_area = union_area_by_key(raw_geoms, cells)
    log(f"  raw union area={union_area:.1f} m2 in {time.perf_counter() - t1:.1f}s")
    raw = {
        "n": int(len(raw_geoms)),
        "total_area_m2": float(geom_area.sum()),
        "area_column_sum_m2": float(pd.to_numeric(gdf["area_m2"], errors="coerce").sum()),
        "union_all_area_m2": union_area,
        "max_area_m2": float(geom_area.max()),
        "share_n_components_dropped_gt0": float((dropped > 0).sum() / len(dropped)),
        "n_invalid": n_invalid,
    }
    log(
        f"raw n={raw['n']} total_area_m2={raw['total_area_m2']:.1f} "
        f"union_all_m2={raw['union_all_area_m2']:.1f} max_area_m2={raw['max_area_m2']:.1f} "
        f"share_dropped_gt0={raw['share_n_components_dropped_gt0']:.4f}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for gap in gaps:
        tag = gap_tag(gap)
        rule = f"gap_{tag}"
        blocks, stats, row_ids = stitch(
            valid_geoms,
            valid_area,
            scores,
            cells,
            tiles,
            rule=rule,
            gap=gap,
            close_m=float(args.close_m),
            repair=True,
        )
        rules.append((rule, stats))
        if any(abs(gap - wanted) <= 1e-9 for wanted in write_gaps):
            gpkg_path = args.out_dir / f"durban_census_stitched_gap{tag}_v1.gpkg"
            parquet_path = args.out_dir / f"piece_to_block_gap{tag}.parquet"
            if set(pd.unique(row_ids)) != set(blocks["block_id"]):
                raise RuntimeError(f"{rule}: piece map is not the block id set")
            log(f"writing {gpkg_path.name}")
            write_gpkg(blocks, gpkg_path)
            mapping = pd.DataFrame({"row_index": np.arange(len(row_ids), dtype=np.int64), "block_id": row_ids})
            mapping.to_parquet(parquet_path, index=False)
            written.append(str(gpkg_path))
            written.append(str(parquet_path))
        del blocks, row_ids

    payload = {
        "params": {
            "input": str(args.input),
            "out_dir": str(args.out_dir),
            "gaps": gaps,
            "close_m": float(args.close_m),
            "write_gpkg_for": write_gaps,
            "also_legacy": bool(args.also_legacy),
            "rule": (
                "Edge if polygons overlap or the gap is <= g (g=0: intersection area > 0 or touch). "
                "Connected components are unioned. Multi-piece blocks are closed with fixed close_m. "
                "Delivery baseline is IoU > 0.1 with union only."
            ),
        },
        "raw": raw,
    }
    for name, stats in rules:
        payload[name] = stats
    stats_json = args.out_dir / "stats.json"
    stats_md = args.out_dir / "stats.md"
    stats_json.write_text(json.dumps(payload, indent=2) + "\n")
    markdown = render_markdown(raw, rules, close_m=float(args.close_m))
    stats_md.write_text(markdown)
    print("\n" + markdown, flush=True)
    for path in written:
        log(f"wrote {path}")
    log(f"wrote {stats_json}")
    log(f"wrote {stats_md}")
    if legacy_stats is not None:
        assert_legacy(legacy_stats)


if __name__ == "__main__":
    main()
