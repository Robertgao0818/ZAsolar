#!/usr/bin/env python3
"""
小目标 FP 专项分析：对 <20m² 的 reviewed predictions 做分类、HN 安全筛选、chip 导出。

主分析集固定为一个 batch，对照集（如有）单独输出到 reference/ 子目录。
FN marker 真源为 review/fn_markers.csv（像素坐标），需通过 tile transform 转 geo。

用法：
    # 仅 batch004 主分析（跳过 chip）
    python scripts/analysis/analyze_small_fp.py --primary-batch batch004 --skip-chips

    # 完整运行（含 chip 导出 + batch003 对照）
    python scripts/analysis/analyze_small_fp.py \
        --primary-batch batch004 --reference-batch batch003

    # 指定 grid 子集
    python scripts/analysis/analyze_small_fp.py \
        --primary-batch batch004 --grid-ids G1919 G2029
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point

BASE_DIR = Path(__file__).resolve().parents[2]
TILES_ROOT = Path("/mnt/d/ZAsolar/tiles")

RESULTS_ROOTS = {
    "batch003": BASE_DIR / "results",
    "batch004": Path("/mnt/d/ZAsolar/results"),
}

BATCH_GRIDS = {
    "batch003": [
        "G1682", "G1683", "G1685", "G1686", "G1687", "G1688",
        "G1689", "G1690", "G1691", "G1692", "G1693", "G1743",
        "G1744", "G1747", "G1749", "G1750", "G1798", "G1800",
        "G1806", "G1807",
    ],
    "batch004": [
        "G1855", "G1856", "G1862", "G1863", "G1864",
        "G1909", "G1910", "G1911", "G1917", "G1918",
        "G1919", "G1920", "G1921", "G1966", "G1970",
        "G1971", "G1972", "G1973", "G1974", "G1975",
        "G1976", "G1977", "G1979", "G1981",
        "G2025", "G2026", "G2027", "G2028", "G2029",
        "G2030", "G2031", "G2032", "G2033", "G2034",
        "G2035", "G2037", "G2038",
    ],
}

METRIC_CRS = "EPSG:32734"
SEG_ERROR_BUFFER_M = 5.0
TILE_EDGE_BUFFER_PX = 50

CONFIDENCE_BUCKETS = [
    (0.95, 1.01, ">=0.95"),
    (0.90, 0.95, "0.90-0.95"),
    (0.80, 0.90, "0.80-0.90"),
    (0.00, 0.80, "<0.80"),
]

TAXONOMY_LABELS = [
    "solar_thermal_water_heater",
    "pergola_carport_shadow",
    "skylight_roof_window",
    "roof_shadow_dark_fixture",
    "blue_tarp_roof_cover",
    "vehicle_or_road_marking",
    "fragment_near_true_panel",
    "other_unknown",
]


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Load & Merge
# ═══════════════════════════════════════════════════════════════════════


def load_fn_markers_csv(
    grid_id: str, results_root: Path, tiles_root: Path,
) -> gpd.GeoDataFrame:
    """Load FN markers from review/fn_markers.csv, convert pixel→geo (EPSG:32734)."""
    fn_csv = results_root / grid_id / "review" / "fn_markers.csv"
    if not fn_csv.exists():
        return gpd.GeoDataFrame(columns=["geometry", "tile_key"], crs=METRIC_CRS)

    markers = []
    with open(fn_csv) as f:
        for row in csv.DictReader(f):
            markers.append({
                "tile_key": row["tile_key"],
                "px": float(row.get("px", row.get("x", 0))),
                "py": float(row.get("py", row.get("y", 0))),
            })

    if not markers:
        return gpd.GeoDataFrame(columns=["geometry", "tile_key"], crs=METRIC_CRS)

    # Convert pixel coords to geo coords via tile transforms
    transformer = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    tile_transforms = {}  # cache per tile
    points = []
    tile_keys = []

    for m in markers:
        tk = m["tile_key"]
        if tk not in tile_transforms:
            tile_path = tiles_root / grid_id / f"{tk}_geo.tif"
            if not tile_path.exists():
                print(f"  [WARN] tile not found for FN marker: {tile_path}")
                continue
            with rasterio.open(str(tile_path)) as src:
                tile_transforms[tk] = src.transform

        if tk not in tile_transforms:
            continue

        tf = tile_transforms[tk]
        # pixel (col, row) → geo (lon, lat) in EPSG:4326
        lon, lat = tf * (m["px"], m["py"])
        # → EPSG:32734
        x, y = transformer.transform(lon, lat)
        points.append(Point(x, y))
        tile_keys.append(tk)

    if not points:
        return gpd.GeoDataFrame(columns=["geometry", "tile_key"], crs=METRIC_CRS)

    return gpd.GeoDataFrame(
        {"tile_key": tile_keys}, geometry=points, crs=METRIC_CRS,
    )


def detect_seg_errors(
    preds_gdf: gpd.GeoDataFrame,
    fn_gdf: gpd.GeoDataFrame,
    batch: str,
) -> set:
    """Identify segmentation errors: predictions that overlap with FN markers.

    batch003: delete + FN within polygon → seg_error
    batch004: edit + FN within 5m buffer → seg_error
    """
    seg_error_ids = set()
    if len(fn_gdf) == 0:
        return seg_error_ids

    # batch003 rule: delete polygon contains FN marker
    delete_mask = preds_gdf["status"] == "delete"
    delete_preds = preds_gdf[delete_mask].copy()
    if len(delete_preds) > 0 and len(fn_gdf) > 0:
        joined = gpd.sjoin(fn_gdf, delete_preds, predicate="within", how="inner")
        if "pred_id" in joined.columns:
            seg_error_ids.update(joined["pred_id"].unique())

    # batch004 rule: edit polygon + FN within 5m buffer
    if batch == "batch004":
        edit_mask = preds_gdf["status"] == "edit"
        edit_preds = preds_gdf[edit_mask].copy()
        if len(edit_preds) > 0 and len(fn_gdf) > 0:
            buffered = edit_preds.copy()
            buffered["geometry"] = buffered.geometry.buffer(SEG_ERROR_BUFFER_M)
            joined = gpd.sjoin(fn_gdf, buffered, predicate="within", how="inner")
            if "pred_id" in joined.columns:
                seg_error_ids.update(joined["pred_id"].unique())

    return seg_error_ids


def load_grid_data(
    grid_id: str,
    results_root: Path,
    tiles_root: Path,
    batch: str,
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame]:
    """Load predictions + review decisions + FN markers for one grid."""
    pred_path = results_root / grid_id / "predictions_metric.gpkg"
    csv_path = results_root / grid_id / "review" / "detection_review_decisions.csv"

    if not pred_path.exists():
        return None, gpd.GeoDataFrame()

    preds = gpd.read_file(str(pred_path))
    preds["pred_id"] = preds.index

    # Merge review decisions
    if csv_path.exists():
        review = pd.read_csv(str(csv_path))
        preds = preds.merge(
            review[["pred_id", "status"]],
            on="pred_id", how="left",
        )
    else:
        preds["status"] = ""

    # Fill missing status as unreviewed
    preds["status"] = preds["status"].fillna("").replace({"": "unreviewed"})

    # Load FN markers
    fn_gdf = load_fn_markers_csv(grid_id, results_root, tiles_root)

    # Detect seg errors
    seg_error_ids = detect_seg_errors(preds, fn_gdf, batch)
    preds["is_seg_error"] = preds["pred_id"].isin(seg_error_ids)

    # Add metadata
    preds["grid_id"] = grid_id
    preds["batch_id"] = batch

    return preds, fn_gdf


def load_all_grids(
    batch: str,
    tiles_root: Path,
    grid_ids: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, dict, dict]:
    """Load all grids for a batch.

    Returns:
        all_preds: all predictions with status + is_seg_error columns
        correct_by_grid: {grid_id: GeoDataFrame of correct predictions}
        fn_by_grid: {grid_id: GeoDataFrame of FN markers}
    """
    results_root = RESULTS_ROOTS[batch]
    grids = grid_ids or BATCH_GRIDS[batch]

    all_dfs = []
    correct_by_grid = {}
    fn_by_grid = {}

    for grid_id in grids:
        preds, fn_gdf = load_grid_data(grid_id, results_root, tiles_root, batch)
        if preds is None or len(preds) == 0:
            print(f"  [SKIP] {grid_id}: no predictions")
            continue

        n_rev = (preds["status"] != "unreviewed").sum()
        n_del = (preds["status"] == "delete").sum()
        n_seg = preds["is_seg_error"].sum()
        print(f"  [{grid_id}] {len(preds)} preds, {n_rev} reviewed, "
              f"{n_del} delete, {n_seg} seg_error")

        all_dfs.append(preds)
        correct_by_grid[grid_id] = preds[preds["status"] == "correct"].copy()
        fn_by_grid[grid_id] = fn_gdf

    if not all_dfs:
        print(f"  [ERROR] No data loaded for {batch}")
        sys.exit(1)

    return pd.concat(all_dfs, ignore_index=True), correct_by_grid, fn_by_grid


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Compute Features
# ═══════════════════════════════════════════════════════════════════════


def compute_size_band(area: float) -> str:
    return "<5" if area < 5 else "5-20"


def compute_confidence_bucket(conf: float) -> str:
    for lo, hi, label in CONFIDENCE_BUCKETS:
        if lo <= conf < hi:
            return label
    return "<0.80"


def compute_geometry_bucket(elongation: float, solidity: float) -> str:
    if elongation > 6 or solidity < 0.7:
        return "sliver_fragment"
    if elongation > 2.5:
        return "elongated"
    return "compact"


def compute_tile_edge_flags(
    gdf: gpd.GeoDataFrame, tiles_root: Path,
) -> pd.Series:
    """Check if prediction centroid is near or outside tile edge.

    Returns two Series:
      tile_edge_flag: True if centroid within TILE_EDGE_BUFFER_PX of edge or outside
      tile_oob_flag:  True if centroid is completely outside tile bounds
    """
    flags = pd.Series(False, index=gdf.index)
    oob = pd.Series(False, index=gdf.index)
    transformer = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    tile_info_cache = {}  # (grid_id, source_tile) → (transform, width, height)

    for idx, row in gdf.iterrows():
        grid_id = row["grid_id"]
        source_tile = row.get("source_tile", "")
        if not source_tile:
            continue

        cache_key = (grid_id, source_tile)
        if cache_key not in tile_info_cache:
            tile_path = tiles_root / grid_id / f"{source_tile}.tif"
            if not tile_path.exists():
                continue
            with rasterio.open(str(tile_path)) as src:
                tile_info_cache[cache_key] = (src.transform, src.width, src.height)

        if cache_key not in tile_info_cache:
            continue

        tf, w, h = tile_info_cache[cache_key]
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        # EPSG:32734 → EPSG:4326
        lon, lat = transformer.transform(cx, cy)
        # geo → pixel
        col, rw = ~tf * (lon, lat)
        col, rw = int(col), int(rw)

        buf = TILE_EDGE_BUFFER_PX
        # Outside tile entirely or near edge
        if col < buf or col > w - buf or rw < buf or rw > h - buf:
            flags.at[idx] = True
        # Completely out of bounds
        if col < 0 or col > w or rw < 0 or rw > h:
            oob.at[idx] = True

    return flags, oob


def compute_nearest_correct_dist(
    gdf: gpd.GeoDataFrame, correct_by_grid: dict,
) -> pd.Series:
    """Compute distance to nearest correct prediction centroid (same grid, meters)."""
    dists = pd.Series(np.nan, index=gdf.index, dtype=float)

    for grid_id, group in gdf.groupby("grid_id"):
        correct = correct_by_grid.get(grid_id)
        if correct is None or len(correct) == 0:
            continue

        correct_centroids = np.array([
            [g.centroid.x, g.centroid.y] for g in correct.geometry
        ])
        tree = cKDTree(correct_centroids)

        for idx, row in group.iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            dist, _ = tree.query([cx, cy])
            dists.at[idx] = dist

    return dists


def compute_fn_proximity(
    gdf: gpd.GeoDataFrame, fn_by_grid: dict,
) -> tuple[pd.Series, pd.Series]:
    """Compute fn_within_3m and tile_has_fn."""
    fn_within_3m = pd.Series(False, index=gdf.index)
    tile_has_fn = pd.Series(False, index=gdf.index)

    for grid_id, group in gdf.groupby("grid_id"):
        fn_gdf = fn_by_grid.get(grid_id)
        if fn_gdf is None or len(fn_gdf) == 0:
            continue

        # tile_has_fn: check if source_tile matches any FN marker tile_key
        fn_tiles = set()
        for _, fn_row in fn_gdf.iterrows():
            tk = fn_row.get("tile_key", "")
            if tk:
                fn_tiles.add(tk + "_geo")  # source_tile includes _geo suffix

        for idx, row in group.iterrows():
            st = row.get("source_tile", "")
            # source_tile is like "G1919_1_0_geo", tile_key is "G1919_1_0"
            if st in fn_tiles or st.replace("_geo", "") in {t.replace("_geo", "") for t in fn_tiles}:
                tile_has_fn.at[idx] = True

            # fn_within_3m: buffer prediction by 3m, check intersection
            buffered = row.geometry.buffer(3.0)
            for _, fn_row in fn_gdf.iterrows():
                if buffered.contains(fn_row.geometry):
                    fn_within_3m.at[idx] = True
                    break

    return fn_within_3m, tile_has_fn


def compute_tile_context(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute tile_has_edit and tile_large_array_flag from the full prediction set."""
    tile_has_edit = pd.Series(False, index=gdf.index)
    tile_large_array_flag = pd.Series(False, index=gdf.index)

    for (grid_id, source_tile), group in gdf.groupby(["grid_id", "source_tile"]):
        has_edit = (group["status"] == "edit").any()
        has_large = (group["area_m2"] >= 100).any()

        for idx in group.index:
            tile_has_edit.at[idx] = has_edit
            tile_large_array_flag.at[idx] = has_large

    return tile_has_edit, tile_large_array_flag


def compute_all_features(
    gdf: gpd.GeoDataFrame,
    correct_by_grid: dict,
    fn_by_grid: dict,
    tiles_root: Path,
) -> gpd.GeoDataFrame:
    """Compute all Phase 2 features on the small-prediction subset.

    Note: tile_has_edit and tile_large_array_flag must be pre-computed on the
    full reviewed set (not just small) and set on gdf before calling this function,
    so that tile_large_array_flag reflects >=100m² preds outside the small subset.
    """
    print("\n=== Phase 2: Computing features ===")

    gdf = gdf.copy()

    # Simple column-level features
    gdf["size_band"] = gdf["area_m2"].apply(compute_size_band)
    gdf["confidence_bucket"] = gdf["confidence"].apply(compute_confidence_bucket)
    gdf["geometry_bucket"] = gdf.apply(
        lambda r: compute_geometry_bucket(
            r.get("elongation", 1.0), r.get("solidity", 1.0),
        ), axis=1,
    )

    print("  Computing tile edge flags...")
    gdf["tile_edge_flag"], gdf["tile_oob_flag"] = compute_tile_edge_flags(gdf, tiles_root)

    print("  Computing FN proximity...")
    gdf["fn_within_3m"], gdf["tile_has_fn"] = compute_fn_proximity(gdf, fn_by_grid)

    # tile_has_edit and tile_large_array_flag are expected to be pre-set
    if "tile_has_edit" not in gdf.columns or "tile_large_array_flag" not in gdf.columns:
        print("  [WARN] tile context not pre-computed, computing on subset only")
        gdf["tile_has_edit"], gdf["tile_large_array_flag"] = compute_tile_context(gdf)

    print("  Computing nearest correct distance...")
    gdf["nearest_correct_dist_m"] = compute_nearest_correct_dist(gdf, correct_by_grid)

    return gdf


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Classify
# ═══════════════════════════════════════════════════════════════════════


def classify_fp_bucket(row: pd.Series) -> str:
    if row["status"] == "correct":
        return "not_fp"
    if row["is_seg_error"]:
        return "segmentation_error"
    if row["status"] == "edit":
        return "boundary_editable"
    # Out-of-bounds: centroid outside tile, always a postproc artifact
    if row["status"] == "delete" and row.get("tile_oob_flag", False):
        return "uncertain_postproc"
    if row["status"] == "delete" and row.get("tile_edge_flag") and row.get("confidence", 1) < 0.85:
        return "uncertain_postproc"
    return "true_fp"


def classify_hn_safety(row: pd.Series) -> str:
    if row["status"] != "delete":
        return "not_applicable"
    if row["fp_bucket"] == "segmentation_error":
        return "exclude_seg_error"
    if row.get("tile_oob_flag", False):
        return "exclude_postproc_artifact"
    if row.get("nearest_correct_dist_m", 999) <= 6:
        return "exclude_near_true_panel"
    if row.get("tile_large_array_flag", False):
        return "exclude_large_array_context"
    if row.get("fn_within_3m", False):
        return "exclude_near_true_panel"
    if row.get("tile_has_edit", False):
        return "exclude_near_true_panel"
    return "safe_true_fp"


def classify_all(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply fp_bucket and hn_safety classification."""
    print("\n=== Phase 3: Classifying ===")
    gdf = gdf.copy()
    gdf["fp_bucket"] = gdf.apply(classify_fp_bucket, axis=1)
    gdf["hn_safety"] = gdf.apply(classify_hn_safety, axis=1)
    return gdf


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Export Chips
# ═══════════════════════════════════════════════════════════════════════


def draw_polygon_outline(draw, geom, transform, color="red", width=2):
    """Draw polygon outline on image, handling Polygon and MultiPolygon."""
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        coords = []
        for x, y in poly.exterior.coords:
            px, py = ~transform * (x, y)
            coords.append((int(px), int(py)))
        if len(coords) >= 3:
            draw.polygon(coords, outline=color, width=width)


def export_chips(
    gdf: gpd.GeoDataFrame,
    all_preds: gpd.GeoDataFrame,
    output_dir: Path,
    tiles_root: Path,
    chip_size: int = 400,
    max_chips: int = 500,
) -> int:
    """Export chip images for safe_true_fp predictions."""
    print("\n=== Phase 4: Exporting chips ===")

    safe = gdf[gdf["hn_safety"] == "safe_true_fp"]
    if len(safe) == 0:
        print("  No safe_true_fp to export")
        return 0

    # Sample if needed
    if len(safe) > max_chips:
        safe = safe.sample(max_chips, random_state=42)
        print(f"  Sampled {max_chips} of {len(gdf[gdf['hn_safety'] == 'safe_true_fp'])} safe_true_fp")

    chips_dir = output_dir / "chips"
    transformer = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    half = chip_size // 2
    count = 0

    for _, row in safe.iterrows():
        grid_id = row["grid_id"]
        source_tile = row.get("source_tile", "")
        if not source_tile:
            continue

        tile_path = tiles_root / grid_id / f"{source_tile}.tif"
        if not tile_path.exists():
            continue

        # Create subdirectory by fp_bucket
        bucket_dir = chips_dir / "safe_true_fp"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir = chips_dir / "safe_true_fp_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(str(tile_path)) as src:
                tile_crs = src.crs

                # Reproject prediction geometry to tile CRS
                row_geom = gpd.GeoSeries(
                    [row.geometry], crs=METRIC_CRS,
                ).to_crs(tile_crs).iloc[0]

                cx, cy = row_geom.centroid.x, row_geom.centroid.y
                px, py = ~src.transform * (cx, cy)
                px, py = int(px), int(py)

                row_start = max(0, py - half)
                col_start = max(0, px - half)
                row_stop = min(src.height, py + half)
                col_stop = min(src.width, px + half)

                window = rasterio.windows.Window.from_slices(
                    (row_start, row_stop), (col_start, col_stop),
                )
                data = src.read(window=window)

                if data.shape[1] < 20 or data.shape[2] < 20:
                    continue

                # Pad to full chip_size if clipped at tile edge
                actual_h, actual_w = data.shape[1], data.shape[2]
                if actual_h < chip_size or actual_w < chip_size:
                    padded = np.zeros((data.shape[0], chip_size, chip_size),
                                     dtype=data.dtype)
                    # Place data at the correct offset within the padded canvas
                    pad_y = (py - half) - row_start  # 0 if not clipped at top
                    pad_x = (px - half) - col_start  # 0 if not clipped at left
                    # If clipped at top/left, offset is 0; if clipped at bottom/right, also 0
                    # We need to figure out where the data sits in the full chip
                    y_off = half - (py - row_start)
                    x_off = half - (px - col_start)
                    padded[:, y_off:y_off + actual_h, x_off:x_off + actual_w] = data
                    data = padded
                    # Update window transform to reflect the padded canvas
                    from rasterio.transform import Affine
                    gt = src.transform
                    win_origin_x = gt.c + (col_start - x_off) * gt.a
                    win_origin_y = gt.f + (row_start - y_off) * gt.e
                    window_transform_override = Affine(gt.a, gt.b, win_origin_x,
                                                       gt.d, gt.e, win_origin_y)
                else:
                    window_transform_override = None

                # Pure RGB chip
                img_rgb = Image.fromarray(data[:3].transpose(1, 2, 0))
                fname = (
                    f"{grid_id}_pred{row['pred_id']}_{source_tile}.png"
                )
                img_rgb.save(str(bucket_dir / fname))

                # Overlay version
                img_overlay = img_rgb.copy()
                draw = ImageDraw.Draw(img_overlay)
                if window_transform_override is not None:
                    win_transform = window_transform_override
                else:
                    win_transform = rasterio.windows.transform(window, src.transform)

                # Red outline for target prediction
                draw_polygon_outline(draw, row_geom, win_transform, "red", 2)

                # Green outlines for correct predictions on same tile
                correct_on_tile = all_preds[
                    (all_preds["grid_id"] == grid_id)
                    & (all_preds["source_tile"] == source_tile)
                    & (all_preds["status"] == "correct")
                ]
                for _, cr in correct_on_tile.iterrows():
                    cr_geom = gpd.GeoSeries(
                        [cr.geometry], crs=METRIC_CRS,
                    ).to_crs(tile_crs).iloc[0]
                    draw_polygon_outline(draw, cr_geom, win_transform, "lime", 1)

                # Text annotation
                conf = row.get("confidence", 0)
                area = row.get("area_m2", 0)
                draw.text(
                    (4, 4),
                    f"safe_true_fp | conf={conf:.2f} | {area:.1f}m²",
                    fill="yellow",
                )

                img_overlay.save(str(overlay_dir / fname))
                count += 1

        except Exception as e:
            print(f"  [WARN] chip export failed for {grid_id} pred {row['pred_id']}: {e}")

    print(f"  Exported {count} chips")
    return count


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Aggregate & Report
# ═══════════════════════════════════════════════════════════════════════


def run_sanity_checks(gdf: gpd.GeoDataFrame, max_area: float):
    """Validate classification consistency."""
    print("\n=== Sanity Checks ===")
    errors = []

    # All records within size threshold
    over = gdf[gdf["area_m2"] >= max_area]
    if len(over) > 0:
        errors.append(f"  FAIL: {len(over)} records with area >= {max_area}")

    # No unclassified
    if gdf["fp_bucket"].isna().any():
        errors.append(f"  FAIL: {gdf['fp_bucket'].isna().sum()} unclassified fp_bucket")
    if gdf["hn_safety"].isna().any():
        errors.append(f"  FAIL: {gdf['hn_safety'].isna().sum()} unclassified hn_safety")

    # seg_error not in safe_true_fp
    bad = gdf[(gdf["is_seg_error"]) & (gdf["hn_safety"] == "safe_true_fp")]
    if len(bad) > 0:
        errors.append(f"  FAIL: {len(bad)} seg_error in safe_true_fp")

    # tile_has_edit delete not in safe_true_fp
    bad = gdf[
        (gdf["status"] == "delete") & gdf["tile_has_edit"]
        & (gdf["hn_safety"] == "safe_true_fp")
    ]
    if len(bad) > 0:
        errors.append(f"  FAIL: {len(bad)} tile_has_edit delete in safe_true_fp")

    # nearest_correct_dist_m <= 6 should be exclude_near_true_panel
    bad = gdf[
        (gdf["status"] == "delete")
        & (gdf["nearest_correct_dist_m"] <= 6)
        & (gdf["hn_safety"] == "safe_true_fp")
    ]
    if len(bad) > 0:
        errors.append(f"  FAIL: {len(bad)} near-panel delete in safe_true_fp")

    # tile_large_array_flag should not be in safe_true_fp
    bad = gdf[
        (gdf["status"] == "delete") & gdf["tile_large_array_flag"]
        & (gdf["hn_safety"] == "safe_true_fp")
    ]
    if len(bad) > 0:
        errors.append(f"  FAIL: {len(bad)} large_array delete in safe_true_fp")

    # hn_shortlist only safe_true_fp and only delete
    safe = gdf[gdf["hn_safety"] == "safe_true_fp"]
    if len(safe) > 0:
        non_del = safe[safe["status"] != "delete"]
        if len(non_del) > 0:
            errors.append(f"  FAIL: {len(non_del)} non-delete in safe_true_fp")

    # Status consistency
    correct_not_fp = gdf[(gdf["status"] == "correct") & (gdf["fp_bucket"] != "not_fp")]
    if len(correct_not_fp) > 0:
        errors.append(f"  FAIL: {len(correct_not_fp)} correct not marked as not_fp")

    if errors:
        for e in errors:
            print(e)
        print(f"\n  {len(errors)} sanity check(s) FAILED")
    else:
        print("  All sanity checks PASSED")


def generate_report(
    gdf: gpd.GeoDataFrame,
    unreviewed_df: pd.DataFrame,
    output_dir: Path,
    batch: str,
    max_area: float,
    is_reference: bool = False,
):
    """Generate all output CSVs and recommendations."""
    out = output_dir / "reference" if is_reference else output_dir
    out.mkdir(parents=True, exist_ok=True)

    label = "reference" if is_reference else "primary"
    print(f"\n=== Phase 5: Generating {label} report ({batch}) ===")

    # Drop geometry for CSV export
    audit_cols = [
        "batch_id", "grid_id", "pred_id", "source_tile", "status",
        "area_m2", "size_band", "confidence", "confidence_bucket",
        "elongation", "solidity", "geometry_bucket",
        "tile_edge_flag", "tile_oob_flag", "fn_within_3m", "tile_has_fn", "tile_has_edit",
        "tile_large_array_flag", "nearest_correct_dist_m",
        "is_seg_error", "fp_bucket", "hn_safety",
    ]
    available_cols = [c for c in audit_cols if c in gdf.columns]

    # 1. Main audit table
    gdf[available_cols].to_csv(out / "small_fp_audit.csv", index=False)

    # 2. By grid
    grid_agg = gdf.groupby("grid_id").agg(
        n_small_reviewed=("pred_id", "count"),
        n_correct=("status", lambda s: (s == "correct").sum()),
        n_delete=("status", lambda s: (s == "delete").sum()),
        n_edit=("status", lambda s: (s == "edit").sum()),
        n_seg_error=("is_seg_error", "sum"),
        n_safe_hn=("hn_safety", lambda s: (s == "safe_true_fp").sum()),
    ).reset_index()
    grid_agg["precision_small"] = (
        grid_agg["n_correct"]
        / (grid_agg["n_correct"] + grid_agg["n_delete"] + grid_agg["n_edit"])
    ).round(4)
    grid_agg.to_csv(out / "small_fp_by_grid.csv", index=False)

    # 3. By confidence bucket
    delete_and_correct = gdf[gdf["status"].isin(["correct", "delete"])]
    if len(delete_and_correct) > 0:
        conf_agg = delete_and_correct.groupby("confidence_bucket").agg(
            count=("pred_id", "count"),
            n_delete=("status", lambda s: (s == "delete").sum()),
            n_correct=("status", lambda s: (s == "correct").sum()),
            mean_area=("area_m2", "mean"),
        ).reset_index()
        conf_agg["fp_rate"] = (
            conf_agg["n_delete"] / conf_agg["count"]
        ).round(4)
        conf_agg.to_csv(out / "small_fp_by_confidence.csv", index=False)

    # 4. HN shortlist (primary only)
    if not is_reference:
        shortlist = gdf[gdf["hn_safety"] == "safe_true_fp"]
        hn_cols = [
            "grid_id", "pred_id", "source_tile", "status",
            "confidence", "area_m2", "size_band", "fp_bucket", "hn_safety",
            "tile_large_array_flag", "nearest_correct_dist_m",
        ]
        hn_available = [c for c in hn_cols if c in shortlist.columns]
        shortlist[hn_available].to_csv(out / "hn_small_fp_shortlist.csv", index=False)

    # 5. Unreviewed reconciliation (primary only)
    if not is_reference and len(unreviewed_df) > 0:
        unreviewed_df.to_csv(out / "unreviewed_reconciliation.csv", index=False)

    # 6. Taxonomy template (primary only)
    if not is_reference:
        safe = gdf[gdf["hn_safety"] == "safe_true_fp"].copy()
        if len(safe) > 0:
            tmpl = safe[["pred_id", "grid_id", "source_tile", "fp_bucket",
                         "hn_safety", "confidence", "area_m2"]].copy()
            tmpl["chip_path"] = tmpl.apply(
                lambda r: f"chips/safe_true_fp/{r['grid_id']}_pred{r['pred_id']}_{r['source_tile']}.png",
                axis=1,
            )
            tmpl["human_label"] = ""
            tmpl["notes"] = ""
            tmpl.to_csv(out / "small_fp_taxonomy_template.csv", index=False)

    # 7. Recommendations (primary only)
    if not is_reference:
        _write_recommendations(gdf, out, max_area)


def _write_recommendations(gdf: gpd.GeoDataFrame, output_dir: Path, max_area: float):
    """Auto-generate recommendations.md answering the four Codex questions."""
    delete = gdf[gdf["status"] == "delete"]
    correct = gdf[gdf["status"] == "correct"]
    safe = gdf[gdf["hn_safety"] == "safe_true_fp"]

    # Q1: <5 vs 5-20 pain point
    small5 = delete[delete["area_m2"] < 5]
    mid5_20 = delete[(delete["area_m2"] >= 5) & (delete["area_m2"] < 20)]
    c_small5 = correct[correct["area_m2"] < 5]
    c_mid5_20 = correct[(correct["area_m2"] >= 5) & (correct["area_m2"] < 20)]

    fp_rate_lt5 = len(small5) / (len(small5) + len(c_small5)) if (len(small5) + len(c_small5)) > 0 else 0
    fp_rate_5_20 = len(mid5_20) / (len(mid5_20) + len(c_mid5_20)) if (len(mid5_20) + len(c_mid5_20)) > 0 else 0

    # Q2: concentration in few grids
    if len(delete) > 0:
        top_grids = delete.groupby("grid_id").size().sort_values(ascending=False).head(5)
        top_pct = top_grids.sum() / len(delete)
    else:
        top_grids = pd.Series(dtype=int)
        top_pct = 0

    # Q2b: high-confidence FP concentration
    hi_conf = delete[delete["confidence"] >= 0.90]

    # Q3: HN categories
    safe_by_bucket = safe.groupby("geometry_bucket").size() if len(safe) > 0 else pd.Series(dtype=int)

    # Q4: postproc/fragment
    uncertain = gdf[gdf["fp_bucket"] == "uncertain_postproc"]
    seg_err = gdf[gdf["fp_bucket"] == "segmentation_error"]

    lines = [
        f"# Small FP Analysis Recommendations",
        f"",
        f"Analysis scope: reviewed predictions with area < {max_area}m²",
        f"",
        f"## Q1: <5m² vs 5-20m² — which is the main precision pain point?",
        f"",
        f"- <5m²: {len(small5)} delete / {len(small5) + len(c_small5)} reviewed = "
        f"**FP rate {fp_rate_lt5:.1%}**",
        f"- 5-20m²: {len(mid5_20)} delete / {len(mid5_20) + len(c_mid5_20)} reviewed = "
        f"**FP rate {fp_rate_5_20:.1%}**",
        f"",
        f"{'<5m² is the larger pain point.' if fp_rate_lt5 > fp_rate_5_20 else '5-20m² is the larger pain point.' if fp_rate_5_20 > fp_rate_lt5 else 'Both bands have similar FP rates.'}",
        f"",
        f"## Q2: Are high-confidence small FPs concentrated in a few grids?",
        f"",
        f"Top 5 grids by small delete count ({top_pct:.0%} of all small deletes):",
        f"",
    ]
    for gid, cnt in top_grids.items():
        lines.append(f"- {gid}: {cnt}")
    lines += [
        f"",
        f"High-confidence (conf>=0.90) small deletes: {len(hi_conf)}",
    ]
    if len(hi_conf) > 0:
        hi_top = hi_conf.groupby("grid_id").size().sort_values(ascending=False).head(5)
        for gid, cnt in hi_top.items():
            lines.append(f"  - {gid}: {cnt}")
    lines += [
        f"",
        f"## Q3: Which categories are best suited for targeted hard negatives?",
        f"",
        f"Safe HN candidates: {len(safe)}",
        f"",
        f"By geometry bucket:",
    ]
    for bucket, cnt in safe_by_bucket.items():
        lines.append(f"  - {bucket}: {cnt}")
    lines += [
        f"",
        f"**Recommendation**: Compact, high-confidence safe_true_fp are the best HN "
        f"candidates — they represent confident model mistakes that are not boundary "
        f"issues or segmentation fragments.",
        f"",
        f"## Q4: Which small deletes are postproc/fragment problems (not suitable for HN)?",
        f"",
        f"- uncertain_postproc (tile edge + low conf): {len(uncertain)}",
        f"- segmentation_error (FN inside delete/edit): {len(seg_err)}",
        f"",
        f"These should NOT be used as hard negatives. uncertain_postproc may benefit "
        f"from tile overlap or edge masking. segmentation_error items are actually "
        f"correct detections with boundary issues.",
        f"",
    ]

    (output_dir / "recommendations.md").write_text("\n".join(lines))


def print_summary(gdf: gpd.GeoDataFrame, batch: str, max_area: float):
    """Print console summary."""
    reviewed = gdf[gdf["status"].isin(["correct", "delete", "edit"])]
    n_correct = (reviewed["status"] == "correct").sum()
    n_delete = (reviewed["status"] == "delete").sum()
    n_edit = (reviewed["status"] == "edit").sum()

    print(f"\n{'=' * 60}")
    print(f"Small FP Analysis Summary ({batch}, area < {max_area}m²)")
    print(f"{'=' * 60}")
    print(f"Total reviewed: {len(reviewed)} "
          f"(correct={n_correct}, delete={n_delete}, edit={n_edit})")
    print(f"Grids: {gdf['grid_id'].nunique()}")

    print(f"\nFP Bucket Distribution (delete + edit only):")
    fp_rows = gdf[gdf["status"].isin(["delete", "edit"])]
    for bucket, group in fp_rows.groupby("fp_bucket"):
        pct = len(group) / len(fp_rows) * 100 if len(fp_rows) > 0 else 0
        mean_conf = group["confidence"].mean()
        print(f"  {bucket:25s} {len(group):5d} ({pct:5.1f}%)  mean_conf={mean_conf:.2f}")

    print(f"\nHN Safety (delete only):")
    del_rows = gdf[gdf["status"] == "delete"]
    for safety, group in del_rows.groupby("hn_safety"):
        pct = len(group) / len(del_rows) * 100 if len(del_rows) > 0 else 0
        print(f"  {safety:30s} {len(group):5d} ({pct:5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Small-target FP analysis (<20m² reviewed predictions)",
    )
    parser.add_argument("--primary-batch", required=True, choices=list(BATCH_GRIDS.keys()))
    parser.add_argument("--reference-batch", choices=list(BATCH_GRIDS.keys()), default=None)
    parser.add_argument("--grid-ids", nargs="+", default=None,
                        help="Override grid list for primary batch")
    parser.add_argument("--max-area", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tiles-root", type=Path, default=TILES_ROOT)
    parser.add_argument("--skip-chips", action="store_true")
    parser.add_argument("--max-chips", type=int, default=500)
    parser.add_argument("--chip-size", type=int, default=400)

    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = BASE_DIR / "results" / "analysis" / "small_fp" / run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # ── Phase 1: Load primary batch ──
    print(f"\n=== Phase 1: Loading {args.primary_batch} ===")
    all_preds, correct_by_grid, fn_by_grid = load_all_grids(
        args.primary_batch, args.tiles_root, args.grid_ids,
    )

    # Split reviewed vs unreviewed
    unreviewed = all_preds[all_preds["status"] == "unreviewed"]
    reviewed = all_preds[all_preds["status"] != "unreviewed"]

    print(f"\n  Total: {len(all_preds)} predictions, "
          f"{len(reviewed)} reviewed, {len(unreviewed)} unreviewed")

    # Filter to small predictions
    small = reviewed[reviewed["area_m2"] < args.max_area].copy()
    print(f"  Small (<{args.max_area}m²): {len(small)} reviewed")

    # Unreviewed small for reconciliation
    unreviewed_small = unreviewed[unreviewed["area_m2"] < args.max_area]
    unreviewed_export = unreviewed_small[
        ["grid_id", "pred_id", "source_tile", "area_m2", "confidence"]
    ].copy() if len(unreviewed_small) > 0 else pd.DataFrame()

    # ── Phase 2: Compute features ──
    # Note: we compute tile context on ALL reviewed preds (not just small)
    # so tile_large_array_flag reflects >=100m² preds too
    tile_has_edit_full, tile_large_array_full = compute_tile_context(reviewed)
    # Map back to small subset
    small["tile_has_edit"] = small.index.map(tile_has_edit_full)
    small["tile_large_array_flag"] = small.index.map(tile_large_array_full)

    small = compute_all_features(small, correct_by_grid, fn_by_grid, args.tiles_root)

    # ── Phase 3: Classify ──
    small = classify_all(small)

    # ── Sanity checks ──
    run_sanity_checks(small, args.max_area)

    # ── Phase 4: Export chips ──
    if not args.skip_chips:
        export_chips(
            small, all_preds, args.output_dir, args.tiles_root,
            args.chip_size, args.max_chips,
        )

    # ── Phase 5: Report ──
    generate_report(small, unreviewed_export, args.output_dir,
                    args.primary_batch, args.max_area, is_reference=False)
    print_summary(small, args.primary_batch, args.max_area)

    # ── Reference batch ──
    if args.reference_batch:
        print(f"\n{'=' * 60}")
        print(f"=== Reference batch: {args.reference_batch} ===")
        print(f"{'=' * 60}")

        ref_preds, ref_correct, ref_fn = load_all_grids(
            args.reference_batch, args.tiles_root,
        )
        ref_reviewed = ref_preds[ref_preds["status"] != "unreviewed"]
        ref_small = ref_reviewed[ref_reviewed["area_m2"] < args.max_area].copy()

        if len(ref_small) > 0:
            ref_tile_edit, ref_tile_large = compute_tile_context(ref_reviewed)
            ref_small["tile_has_edit"] = ref_small.index.map(ref_tile_edit)
            ref_small["tile_large_array_flag"] = ref_small.index.map(ref_tile_large)

            ref_small = compute_all_features(
                ref_small, ref_correct, ref_fn, args.tiles_root,
            )
            ref_small = classify_all(ref_small)

            ref_unreviewed = ref_preds[ref_preds["status"] == "unreviewed"]
            ref_unrev_small = ref_unreviewed[ref_unreviewed["area_m2"] < args.max_area]
            ref_unrev_export = ref_unrev_small[
                ["grid_id", "pred_id", "source_tile", "area_m2", "confidence"]
            ].copy() if len(ref_unrev_small) > 0 else pd.DataFrame()

            generate_report(ref_small, ref_unrev_export, args.output_dir,
                            args.reference_batch, args.max_area, is_reference=True)
            print_summary(ref_small, args.reference_batch, args.max_area)
        else:
            print(f"  No small reviewed predictions in {args.reference_batch}")

    print(f"\n✓ Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
