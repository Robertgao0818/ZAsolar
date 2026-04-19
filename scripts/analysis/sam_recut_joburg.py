#!/usr/bin/env python3
"""
SAM 2.1 Large 重切：针对 Joburg CBD batch1 review 数据，对三类问题做有针对性的重切。

三类输入：
  1. EDIT_INNER  : edit 状态 pred 内的 FN markers → 该 pred 形状错了，包含非面板区域。
                   用 FN markers 做正点提示，让 SAM 重新生成正确的多边形。
                   忽略原 pred 多边形（因为它本身是错的）。
  2. NEAR_BOUNDARY: correct/edit 周围 15m 内的 FN markers，但不在任何 pred 内 → 边界缺失/碎片。
                   用 (邻近 pred 的 bbox + FN points) 让 SAM 扩展。
  3. HARD_MISS   : >15m 远离任何 pred 的 FN markers → 模型完全漏检。
                   用单点提示让 SAM 从零分割。

NOT touched:
  - correct 内的 FN markers (review 噪声，模型已经检测对了)

输出：
  results/analysis/sam_recut_joburg/<run_id>/
    ├── recut_polygons.gpkg     最终重切结果，可加载到 QGIS
    ├── recut_results.csv       每条记录的 metadata
    └── summary.json            按类别统计

用法：
    python scripts/analysis/sam_recut_joburg.py
    python scripts/analysis/sam_recut_joburg.py --grids G0890 G0816  # 单 grid
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape, box, Point

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from core.grid_utils import get_grid_spec, get_tile_bounds, normalize_grid_id

# DEPRECATED (2026-04-19): hardcoded paths retained via symlinks. New code
# should use core.region_registry.get_model_run_path("johannesburg", "v4_aerial_2023")
# and get_imagery_layer_path("johannesburg", "aerial_2023").
JHB_RESULTS = BASE_DIR / "results_joburg"                 # → results/johannesburg/v4_aerial_2023
JHB_TILES = Path("/mnt/d/ZAsolar/tiles_joburg")           # → tiles/johannesburg/aerial_2023
METRIC_CRS = "EPSG:32735"  # Joburg UTM

CBD_GRIDS = [
    "G0772","G0773","G0774","G0775","G0776","G0814","G0815","G0816","G0817","G0818",
    "G0853","G0854","G0855","G0856","G0857","G0888","G0889","G0890","G0891","G0892",
    "G0922","G0923","G0924","G0925","G0926",
]

SAM2_CHECKPOINT = Path(
    "/mnt/c/Users/gaosh/AppData/Roaming/QGIS/QGIS3/profiles/default/"
    "python/plugins/GeoOSAM/sam2/checkpoints/sam2.1_hiera_large.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l"

OUTPUT_BASE = BASE_DIR / "results" / "analysis" / "sam_recut_joburg"


def _load_sam_model():
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 2.1 Large → {device}...")
    sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("  SAM 2.1 ready")
    return predictor, device


def _classify_fn_markers(grid_id: str):
    """Classify FN markers into EDIT_INNER, NEAR_BOUNDARY, HARD_MISS, IGNORE.

    Returns list of dicts: {marker_geom, klass, anchor_pred (or None), tile_key (or None)}
    """
    rev_path = JHB_RESULTS / grid_id / "review" / f"{grid_id}_reviewed.gpkg"
    fn_path = JHB_RESULTS / grid_id / "review" / f"{grid_id}_fn_markers.gpkg"
    fn_csv = JHB_RESULTS / grid_id / "review" / "fn_markers.csv"

    if not rev_path.exists() or not fn_path.exists():
        return []

    rev = gpd.read_file(rev_path).to_crs(METRIC_CRS)
    fn = gpd.read_file(fn_path).to_crs(METRIC_CRS)

    # Read tile_key per FN marker (csv keeps the original tile_key the user clicked on)
    fn_tile_keys: list[str] = []
    if fn_csv.exists():
        with open(fn_csv) as f:
            for row in csv.DictReader(f):
                fn_tile_keys.append(row.get("tile_key", ""))
    while len(fn_tile_keys) < len(fn):
        fn_tile_keys.append("")

    classified = []
    for i, m in fn.iterrows():
        if not m.geometry:
            continue
        # Inside any kept (correct/edit) prediction?
        kept = rev[rev["review_status"].isin(["correct", "edit", "accept"])]
        contains_mask = kept.contains(m.geometry)
        if contains_mask.any():
            inside_pred = kept[contains_mask].iloc[0]
            status = inside_pred.get("review_status", "")
            if status == "edit":
                klass = "EDIT_INNER"
                anchor = inside_pred.geometry
            else:
                klass = "IGNORE"  # inside a correct pred = noise
                anchor = None
        else:
            # outside all preds — distance to nearest kept pred
            if len(kept) > 0:
                dist = kept.distance(m.geometry).min()
                if dist <= 15:
                    klass = "NEAR_BOUNDARY"
                    distances = kept.distance(m.geometry)
                    nearest_pos = int(np.argmin(distances.values))
                    anchor = kept.iloc[nearest_pos].geometry
                else:
                    klass = "HARD_MISS"
                    anchor = None
            else:
                klass = "HARD_MISS"
                anchor = None

        classified.append({
            "marker_geom": m.geometry,
            "klass": klass,
            "anchor_pred": anchor,
            "tile_key": fn_tile_keys[i] if i < len(fn_tile_keys) else "",
            "marker_idx": i,
        })

    return classified


def _resolve_tile_path(grid_id: str, tile_key: str) -> Path | None:
    if not tile_key:
        return None
    p = JHB_TILES / grid_id / f"{tile_key}_geo.tif"
    if p.exists():
        return p
    return None


def _find_tile_for_point(grid_id: str, point_geom):
    """Given a marker point in EPSG:32735, find the tile that contains it (geographic test)."""
    from pyproj import Transformer
    t = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(point_geom.x, point_geom.y)
    spec = get_grid_spec(grid_id, region="jhb")
    for c in range(spec.n_cols):
        for r in range(spec.n_rows):
            txmin, tymin, txmax, tymax = get_tile_bounds(spec, c, r)
            if txmin <= lon <= txmax and tymin <= lat <= tymax:
                tk = f"{grid_id}_{c}_{r}"
                p = JHB_TILES / grid_id / f"{tk}_geo.tif"
                if p.exists():
                    return tk, (lon, lat)
    return None, None


def _geo_to_px(tile_path: Path, lon: float, lat: float):
    import rasterio
    with rasterio.open(tile_path) as src:
        col, row = ~src.transform * (lon, lat)
    return float(col), float(row)


def _sam_segment(predictor, tile_path: Path, points_px: list[tuple[float, float]],
                 box_px: tuple | None = None):
    """Run SAM 2.1 with point prompts and optional bbox prompt.
    Returns (polygon_in_4326, score) or (None, 0)."""
    import rasterio
    from rasterio.features import shapes as rio_shapes

    with rasterio.open(tile_path) as src:
        img = src.read()
        transform = src.transform

    img_rgb = np.moveaxis(img[:3], 0, -1)
    predictor.set_image(img_rgb)

    point_coords = np.array(points_px) if points_px else None
    point_labels = np.array([1] * len(points_px)) if points_px else None
    box_arr = np.array(box_px) if box_px else None

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_arr,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8)
    score = float(scores[best_idx])

    candidates = []
    for geom, val in rio_shapes(mask, transform=transform):
        if val == 1:
            poly = shape(geom)
            if poly.is_valid and poly.area > 0:
                candidates.append(poly)
    if candidates:
        return max(candidates, key=lambda p: p.area), score
    return None, 0.0


def _polygon_bbox_px(polygon_4326, tile_path):
    import rasterio
    from rasterio.transform import rowcol
    with rasterio.open(tile_path) as src:
        transform = src.transform
        H, W = src.height, src.width
    minx, miny, maxx, maxy = polygon_4326.bounds
    c1, r1 = ~transform * (minx, maxy)
    c2, r2 = ~transform * (maxx, miny)
    x1, x2 = sorted([c1, c2])
    y1, y2 = sorted([r1, r2])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    return (x1, y1, x2, y2)


def run_recut(grids: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor, device = _load_sam_model()

    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    to_4326_for_anchor = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)

    rows = []
    polygons = []  # (geometry_metric, klass, grid_id, sam_score, area_m2, source)
    counts = {"EDIT_INNER": 0, "NEAR_BOUNDARY": 0, "HARD_MISS": 0, "IGNORE": 0}
    succ = {"EDIT_INNER": 0, "NEAR_BOUNDARY": 0, "HARD_MISS": 0}

    # Group markers by tile to feed multi-point prompts within same anchor
    for grid_id in grids:
        classified = _classify_fn_markers(grid_id)
        if not classified:
            continue
        for c in classified:
            counts[c["klass"]] += 1

        # Group EDIT_INNER and NEAR_BOUNDARY by their anchor pred (so multi-point prompts use FN cluster)
        # Group HARD_MISS by tile + spatial proximity (15m)
        edit_groups: dict[int, list] = {}
        near_groups: dict[int, list] = {}
        hard_items: list = []

        for c in classified:
            if c["klass"] == "EDIT_INNER":
                key = id(c["anchor_pred"])
                edit_groups.setdefault(key, []).append(c)
            elif c["klass"] == "NEAR_BOUNDARY":
                key = id(c["anchor_pred"])
                near_groups.setdefault(key, []).append(c)
            elif c["klass"] == "HARD_MISS":
                hard_items.append(c)

        print(f"\n[{grid_id}] EDIT_INNER groups={len(edit_groups)} "
              f"NEAR groups={len(near_groups)} HARD={len(hard_items)} "
              f"(IGNORE={sum(1 for x in classified if x['klass']=='IGNORE')})")

        # --- EDIT_INNER: only points, ignore the wrong polygon shape
        for group in edit_groups.values():
            anchor = group[0]["anchor_pred"]
            # Pick a tile that contains the anchor centroid
            centroid = anchor.centroid
            tile_key, lonlat = _find_tile_for_point(grid_id, centroid)
            if tile_key is None:
                continue
            tile_path = _resolve_tile_path(grid_id, tile_key)
            if tile_path is None:
                continue
            # Convert all FN points to pixel coords
            pts_px = []
            for c in group:
                lon, lat = to_4326_for_anchor.transform(c["marker_geom"].x, c["marker_geom"].y)
                pts_px.append(_geo_to_px(tile_path, lon, lat))
            poly, score = _sam_segment(predictor, tile_path, pts_px, box_px=None)
            if poly is None:
                continue
            poly_m = shp_transform(to_metric.transform, poly)
            area = poly_m.area
            polygons.append((poly_m, "EDIT_INNER", grid_id, score, area))
            succ["EDIT_INNER"] += 1
            rows.append({"grid": grid_id, "klass": "EDIT_INNER", "tile_key": tile_key,
                         "n_points": len(pts_px), "score": score, "area_m2": area})
            print(f"  EDIT_INNER {tile_key} pts={len(pts_px)} score={score:.3f} area={area:.0f}m²")

        # --- NEAR_BOUNDARY: anchor pred bbox + FN points
        for group in near_groups.values():
            anchor = group[0]["anchor_pred"]
            # Tile = the tile containing the FN cluster centroid
            xs = [c["marker_geom"].x for c in group]
            ys = [c["marker_geom"].y for c in group]
            cluster_pt = Point(sum(xs)/len(xs), sum(ys)/len(ys))
            tile_key, _ = _find_tile_for_point(grid_id, cluster_pt)
            if tile_key is None:
                # Fall back to anchor centroid tile
                tile_key, _ = _find_tile_for_point(grid_id, anchor.centroid)
            if tile_key is None:
                continue
            tile_path = _resolve_tile_path(grid_id, tile_key)
            if tile_path is None:
                continue
            # Convert anchor (metric) to 4326 for bbox computation
            anchor_4326 = shp_transform(to_4326_for_anchor.transform, anchor)
            bbox_px = _polygon_bbox_px(anchor_4326, tile_path)
            pts_px = []
            for c in group:
                lon, lat = to_4326_for_anchor.transform(c["marker_geom"].x, c["marker_geom"].y)
                pts_px.append(_geo_to_px(tile_path, lon, lat))
            poly, score = _sam_segment(predictor, tile_path, pts_px, box_px=bbox_px)
            if poly is None:
                continue
            poly_m = shp_transform(to_metric.transform, poly)
            area = poly_m.area
            polygons.append((poly_m, "NEAR_BOUNDARY", grid_id, score, area))
            succ["NEAR_BOUNDARY"] += 1
            rows.append({"grid": grid_id, "klass": "NEAR_BOUNDARY", "tile_key": tile_key,
                         "n_points": len(pts_px), "score": score, "area_m2": area})
            print(f"  NEAR_BOUNDARY {tile_key} pts={len(pts_px)} score={score:.3f} area={area:.0f}m²")

        # --- HARD_MISS: single point each
        for c in hard_items:
            tile_key, _ = _find_tile_for_point(grid_id, c["marker_geom"])
            if tile_key is None:
                continue
            tile_path = _resolve_tile_path(grid_id, tile_key)
            if tile_path is None:
                continue
            lon, lat = to_4326_for_anchor.transform(c["marker_geom"].x, c["marker_geom"].y)
            pt_px = _geo_to_px(tile_path, lon, lat)
            poly, score = _sam_segment(predictor, tile_path, [pt_px], box_px=None)
            if poly is None:
                continue
            poly_m = shp_transform(to_metric.transform, poly)
            area = poly_m.area
            polygons.append((poly_m, "HARD_MISS", grid_id, score, area))
            succ["HARD_MISS"] += 1
            rows.append({"grid": grid_id, "klass": "HARD_MISS", "tile_key": tile_key,
                         "n_points": 1, "score": score, "area_m2": area})
            print(f"  HARD_MISS {tile_key} score={score:.3f} area={area:.0f}m²")

    # Save outputs
    if polygons:
        gdf = gpd.GeoDataFrame(
            {"klass": [p[1] for p in polygons],
             "grid_id": [p[2] for p in polygons],
             "sam_score": [p[3] for p in polygons],
             "area_m2": [p[4] for p in polygons]},
            geometry=[p[0] for p in polygons],
            crs=METRIC_CRS,
        )
        gpkg = output_dir / "recut_polygons.gpkg"
        gdf.to_file(gpkg, driver="GPKG")
        print(f"\nSaved {len(gdf)} polygons → {gpkg}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "recut_results.csv", index=False)

    summary = {
        "total_classified": sum(counts.values()),
        "by_class": counts,
        "successful_recut": succ,
        "ignored_review_noise": counts["IGNORE"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    grids = [normalize_grid_id(g) for g in (args.grids or CBD_GRIDS)]
    run_id = args.output or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    print(f"Output → {output_dir}")
    print(f"Grids: {grids}")
    run_recut(grids, output_dir)


if __name__ == "__main__":
    main()
