#!/usr/bin/env python3
"""
SAM 重切实验：对 FN markers 和现有 mask 区域用 SAM point-prompt 重新分割，
对比 orthogonalize 矢量化 vs SAM 直接矢量化的 polygon 质量差异。

不修改任何现有数据，结果输出到 results/analysis/sam_recut/<run_id>/

两种模式：
  1. FN markers：对 review 中标记的 FN 位置用 SAM 重切，看能否恢复漏检
  2. TP predictions：对已有 correct predictions 的中心点用 SAM 重切，
     对比 SAM polygon vs orthogonalize polygon 的 IoU，量化矢量化精度损失

用法：
    # 对 batch 004 全部 grids 做 FN 重切
    python scripts/analysis/sam_recut_experiment.py --batches batch004

    # 对指定 grids 做 FN + TP 对比
    python scripts/analysis/sam_recut_experiment.py --grid-ids G1919 G2029 --mode both

    # 只做 TP 对比（量化 orthogonalize 损失）
    python scripts/analysis/sam_recut_experiment.py --grid-ids G1919 --mode tp-compare
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
from shapely.geometry import shape

BASE_DIR = Path(__file__).resolve().parents[2]
TILES_ROOT = Path("/mnt/d/ZAsolar/tiles")

# Results root per batch
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

OUTPUT_BASE = BASE_DIR / "results" / "analysis" / "sam_recut"


def _get_results_root(grid_id: str) -> Path:
    """Resolve results root for a grid (local or D drive)."""
    for batch, grids in BATCH_GRIDS.items():
        if grid_id in grids:
            return RESULTS_ROOTS[batch]
    # Fallback: check both
    local = BASE_DIR / "results" / grid_id
    if local.exists():
        return BASE_DIR / "results"
    return Path("/mnt/d/ZAsolar/results")


def _load_fn_markers(grid_id: str, results_root: Path) -> list[dict]:
    """Load FN markers from review CSV."""
    fn_csv = results_root / grid_id / "review" / "fn_markers.csv"
    if not fn_csv.exists():
        return []
    markers = []
    with open(fn_csv) as f:
        for row in csv.DictReader(f):
            markers.append({
                "tile_key": row["tile_key"],
                "px": float(row.get("px", row.get("x", 0))),
                "py": float(row.get("py", row.get("y", 0))),
            })
    return markers


def _load_correct_predictions(grid_id: str, results_root: Path) -> gpd.GeoDataFrame | None:
    """Load reviewed-correct predictions for TP comparison."""
    decisions_csv = results_root / grid_id / "review" / "detection_review_decisions.csv"
    preds_path = results_root / grid_id / "predictions_metric.gpkg"
    if not decisions_csv.exists() or not preds_path.exists():
        return None

    preds = gpd.read_file(str(preds_path))
    decisions = pd.read_csv(decisions_csv)
    if len(decisions) == 0:
        return None

    correct_ids = set(decisions[decisions["status"] == "correct"]["pred_id"].astype(int))
    correct = preds[preds.index.isin(correct_ids)].copy()
    return correct if len(correct) > 0 else None


SAM2_CHECKPOINT = Path(
    "/mnt/c/Users/gaosh/AppData/Roaming/QGIS/QGIS3/profiles/default/"
    "python/plugins/GeoOSAM/sam2/checkpoints/sam2.1_hiera_large.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l"


def _load_sam_model():
    """Lazy-load SAM 2.1 Large model. Returns (predictor, device)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 2.1 Large → {device}...")

    sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("  SAM 2.1 ready")
    return predictor, device


def _sam_segment_points(predictor, _device, tile_path: Path,
                        points: list[tuple[float, float]]):
    """Run SAM 2.1 with one or more foreground point prompts.
    Returns (polygon, score) or (None, 0)."""
    import rasterio
    from rasterio.features import shapes as rio_shapes

    with rasterio.open(tile_path) as src:
        img_array = src.read()
        transform = src.transform

    img_rgb = np.moveaxis(img_array[:3], 0, -1)  # (H, W, 3)

    predictor.set_image(img_rgb)
    point_coords = np.array(points)
    point_labels = np.array([1] * len(points))  # all foreground

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    # masks shape: (3, H, W), scores shape: (3,)
    # Pick mask with highest predicted IoU
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
        best = max(candidates, key=lambda p: p.area)
        return best, score
    return None, 0.0


def _sam_segment_point(predictor, device, tile_path: Path, px: float, py: float):
    """Single-point convenience wrapper."""
    return _sam_segment_points(predictor, device, tile_path, [(px, py)])


def _compute_polygon_centroid_px(polygon, transform, src_crs="EPSG:32734"):
    """Get pixel coordinates of polygon centroid. Handles CRS conversion if needed."""
    from rasterio.transform import rowcol
    from pyproj import Transformer
    cx, cy = polygon.centroid.x, polygon.centroid.y
    # Tile transforms are in EPSG:4326; convert if polygon is in metric CRS
    if src_crs and src_crs != "EPSG:4326":
        t = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        cx, cy = t.transform(cx, cy)
    row, col = rowcol(transform, cx, cy)
    return float(col), float(row)


def _group_markers_by_edit(markers: list[dict], results_root: Path, grid_id: str,
                           proximity_px: float = 100) -> list[list[dict]]:
    """Group FN markers that belong to the same edit polygon or are close together
    on the same tile. Returns list of marker groups (each group → one multi-point prompt)."""
    from collections import defaultdict

    # Load edit predictions if available
    decisions_csv = results_root / grid_id / "review" / "detection_review_decisions.csv"
    preds_path = results_root / grid_id / "predictions_metric.gpkg"
    edit_polys = []  # list of (tile_key, polygon_in_px_approx, [marker_indices])

    if decisions_csv.exists() and preds_path.exists():
        try:
            preds = gpd.read_file(str(preds_path))
            decisions = pd.read_csv(decisions_csv)
            edit_ids = set(decisions[decisions["status"] == "edit"]["pred_id"].astype(int))
            edit_preds = preds[preds.index.isin(edit_ids)]
            for idx, row in edit_preds.iterrows():
                tile_key = row.get("source_tile", "").replace("_geo", "")
                if tile_key:
                    edit_polys.append((tile_key, row.geometry))
        except Exception:
            pass

    # Group by tile first
    by_tile = defaultdict(list)
    for i, m in enumerate(markers):
        by_tile[m["tile_key"]].append((i, m))

    groups = []
    used = set()

    for tile_key, tile_markers in by_tile.items():
        # Try to match markers to edit polygons on same tile
        for edit_tile, edit_geom in edit_polys:
            if edit_tile != tile_key:
                continue
            group = []
            for i, m in tile_markers:
                if i in used:
                    continue
                # Check pixel proximity — rough: if markers are near each other
                # they likely belong to the same object
                px, py = m["px"], m["py"]
                # Can't easily check if marker falls in edit poly (CRS mismatch),
                # so use pixel proximity between markers
                if not group:
                    group.append(m)
                    used.add(i)
                else:
                    # Check if close to any marker already in group
                    for gm in group:
                        dist = ((px - gm["px"])**2 + (py - gm["py"])**2)**0.5
                        if dist < proximity_px:
                            group.append(m)
                            used.add(i)
                            break
            if len(group) > 1:
                groups.append(group)

        # Also cluster remaining ungrouped markers by proximity
        remaining = [(i, m) for i, m in tile_markers if i not in used]
        while remaining:
            seed_i, seed_m = remaining.pop(0)
            if seed_i in used:
                continue
            used.add(seed_i)
            cluster = [seed_m]
            changed = True
            while changed:
                changed = False
                still_remaining = []
                for ri, rm in remaining:
                    if ri in used:
                        continue
                    for cm in cluster:
                        dist = ((rm["px"] - cm["px"])**2 + (rm["py"] - cm["py"])**2)**0.5
                        if dist < proximity_px:
                            cluster.append(rm)
                            used.add(ri)
                            changed = True
                            break
                    else:
                        still_remaining.append((ri, rm))
                remaining = still_remaining
            groups.append(cluster)

    # Add any truly isolated markers as single-element groups
    for i, m in enumerate(markers):
        if i not in used:
            groups.append([m])

    return groups


def run_fn_recut(grids: list[str], output_dir: Path, predictor, device):
    """Recut FN markers with SAM, output to isolated directory.
    Groups nearby markers on the same tile into multi-point prompts."""
    all_rows = []

    for grid_id in grids:
        results_root = _get_results_root(grid_id)
        markers = _load_fn_markers(grid_id, results_root)
        if not markers:
            print(f"[{grid_id}] No FN markers, skipping")
            continue

        # Group markers for multi-point prompts
        marker_groups = _group_markers_by_edit(markers, results_root, grid_id)
        n_multi = sum(1 for g in marker_groups if len(g) > 1)
        print(f"[{grid_id}] {len(markers)} FN markers → {len(marker_groups)} prompts "
              f"({n_multi} multi-point)")
        tiles_dir = TILES_ROOT / grid_id

        for group in marker_groups:
            tile_key = group[0]["tile_key"]
            points = [(m["px"], m["py"]) for m in group]

            geo_path = tiles_dir / f"{tile_key}_geo.tif"
            if not geo_path.exists():
                geo_path = tiles_dir / f"{tile_key}.tif"
            if not geo_path.exists():
                print(f"  [{tile_key}] tile not found, skipping")
                for m in group:
                    all_rows.append({
                        "grid_id": grid_id, "tile_key": tile_key,
                        "marker_px": m["px"], "marker_py": m["py"],
                        "status": "tile_missing", "prompt_type": "multi" if len(group) > 1 else "single",
                        "group_size": len(group),
                    })
                continue

            poly, score = _sam_segment_points(predictor, device, geo_path, points)

            prompt_type = "multi" if len(group) > 1 else "single"
            pts_str = " + ".join(f"({p[0]:.0f},{p[1]:.0f})" for p in points)

            if poly is None:
                print(f"  [{tile_key}] {pts_str} → no mask ({prompt_type})")
                for m in group:
                    all_rows.append({
                        "grid_id": grid_id, "tile_key": tile_key,
                        "marker_px": m["px"], "marker_py": m["py"],
                        "status": "no_mask", "sam_score": 0,
                        "prompt_type": prompt_type, "group_size": len(group),
                    })
                continue

            # Compute area in metric CRS
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32734", always_xy=True)
            from shapely.ops import transform as shp_transform
            poly_m = shp_transform(transformer.transform, poly)
            area_m2 = poly_m.area

            # Check if there's an existing mask at this location
            mask_path = results_root / grid_id / "masks" / f"{tile_key}_geo_mask.tif"
            has_existing_mask = False
            if mask_path.exists():
                import rasterio
                with rasterio.open(mask_path) as src:
                    from rasterio.transform import rowcol
                    r, c = rowcol(src.transform, poly.centroid.x, poly.centroid.y)
                    if 0 <= r < src.height and 0 <= c < src.width:
                        band1 = src.read(1)
                        has_existing_mask = bool(band1[int(r), int(c)] > 0)

            # Check if there's an existing vector at this location (compare in metric CRS)
            vector_path = results_root / grid_id / "vectors" / f"{tile_key}_geo_vectors.geojson"
            has_existing_vector = False
            existing_vector_iou = 0.0
            if vector_path.exists():
                try:
                    vgdf = gpd.read_file(str(vector_path))
                    for _, vrow in vgdf.iterrows():
                        vec_m = shp_transform(transformer.transform, vrow.geometry)
                        if poly_m.intersects(vec_m):
                            inter = poly_m.intersection(vec_m).area
                            union = poly_m.union(vec_m).area
                            iou = inter / union if union > 0 else 0
                            if iou > existing_vector_iou:
                                existing_vector_iou = iou
                                has_existing_vector = True
                except Exception:
                    pass

            print(f"  [{tile_key}] {pts_str} → score={score:.3f} "
                  f"area={area_m2:.1f}m² mask={has_existing_mask} "
                  f"vec={has_existing_vector}(iou={existing_vector_iou:.2f}) [{prompt_type}]")

            # One row per group (not per marker) for multi-point prompts
            all_rows.append({
                "grid_id": grid_id, "tile_key": tile_key,
                "marker_px": points[0][0], "marker_py": points[0][1],
                "all_points": ";".join(f"{p[0]:.1f},{p[1]:.1f}" for p in points),
                "status": "segmented",
                "prompt_type": prompt_type,
                "group_size": len(group),
                "sam_score": round(score, 4),
                "sam_area_m2": round(area_m2, 2),
                "has_existing_mask": has_existing_mask,
                "has_existing_vector": has_existing_vector,
                "existing_vector_iou": round(existing_vector_iou, 4),
                "sam_wkt": poly.wkt,
            })

    # Save results
    fn_csv = output_dir / "fn_recut_results.csv"
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(fn_csv, index=False)
        print(f"\nFN recut: {len(df)} markers → {fn_csv}")

        # Summary
        segmented = df[df["status"] == "segmented"]
        print(f"  Segmented: {len(segmented)}/{len(df)}")
        if len(segmented) > 0:
            print(f"  Has existing mask: {segmented['has_existing_mask'].sum()}")
            print(f"  Has existing vector: {segmented['has_existing_vector'].sum()}")
            no_mask = segmented[~segmented["has_existing_mask"]]
            has_mask_no_vec = segmented[segmented["has_existing_mask"] & ~segmented["has_existing_vector"]]
            has_vec = segmented[segmented["has_existing_vector"]]
            print(f"  → NO_MASK (model missed): {len(no_mask)}")
            print(f"  → VECTORIZATION_LOSS (mask ok, no vector): {len(has_mask_no_vec)}")
            print(f"  → HAS_VECTOR (iou issue or filter): {len(has_vec)}")
    return all_rows


def run_tp_compare(grids: list[str], output_dir: Path, predictor, device,
                   sample_per_grid: int = 30):
    """Compare SAM re-segmentation vs orthogonalize for existing correct predictions."""
    import rasterio

    all_rows = []

    for grid_id in grids:
        results_root = _get_results_root(grid_id)
        correct = _load_correct_predictions(grid_id, results_root)
        if correct is None or len(correct) == 0:
            print(f"[{grid_id}] No correct predictions, skipping")
            continue

        # Sample uniformly
        if len(correct) > sample_per_grid:
            correct = correct.sample(n=sample_per_grid, random_state=42)

        print(f"[{grid_id}] Comparing {len(correct)} correct predictions")
        tiles_dir = TILES_ROOT / grid_id

        for idx, row in correct.iterrows():
            tile_key = row.get("source_tile", "")
            if not tile_key:
                continue
            # Strip _geo suffix if present for tile lookup
            tile_base = tile_key.replace("_geo", "")

            geo_path = tiles_dir / f"{tile_base}_geo.tif"
            if not geo_path.exists():
                geo_path = tiles_dir / f"{tile_base}.tif"
            if not geo_path.exists():
                continue

            # Get centroid in pixel coords
            with rasterio.open(geo_path) as src:
                transform = src.transform
            cx, cy = _compute_polygon_centroid_px(row.geometry, transform)

            # SAM re-segment at centroid
            poly_sam, score = _sam_segment_point(predictor, device, geo_path, cx, cy)
            if poly_sam is None:
                all_rows.append({
                    "grid_id": grid_id, "pred_id": int(idx),
                    "tile_key": tile_key, "status": "sam_no_mask",
                    "ortho_area_m2": row.get("area_m2", 0),
                })
                continue

            # Convert SAM polygon (EPSG:4326) to metric CRS (EPSG:32734) for comparison
            from pyproj import Transformer
            from shapely.ops import transform as shp_transform
            transformer_to_m = Transformer.from_crs("EPSG:4326", "EPSG:32734", always_xy=True)
            sam_m = shp_transform(transformer_to_m.transform, poly_sam)
            # predictions are already in EPSG:32734
            ortho_m = row.geometry

            # IoU in metric CRS
            try:
                intersection = ortho_m.intersection(sam_m).area
                union = ortho_m.union(sam_m).area
                iou = intersection / union if union > 0 else 0
            except Exception:
                iou = 0

            all_rows.append({
                "grid_id": grid_id, "pred_id": int(idx),
                "tile_key": tile_key, "status": "compared",
                "sam_score": round(score, 4),
                "ortho_area_m2": round(ortho_m.area, 2),
                "sam_area_m2": round(sam_m.area, 2),
                "area_ratio": round(sam_m.area / ortho_m.area, 3) if ortho_m.area > 0 else 0,
                "iou_sam_vs_ortho": round(iou, 4),
                "confidence": row.get("confidence", 0),
                "elongation": row.get("elongation", 0),
            })

        print(f"  [{grid_id}] done")

    tp_csv = output_dir / "tp_compare_results.csv"
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(tp_csv, index=False)
        compared = df[df["status"] == "compared"]
        print(f"\nTP compare: {len(compared)} predictions → {tp_csv}")
        if len(compared) > 0:
            print(f"  Mean IoU (SAM vs ortho): {compared['iou_sam_vs_ortho'].mean():.3f}")
            print(f"  Median IoU: {compared['iou_sam_vs_ortho'].median():.3f}")
            print(f"  IoU < 0.5: {(compared['iou_sam_vs_ortho'] < 0.5).sum()} "
                  f"({(compared['iou_sam_vs_ortho'] < 0.5).mean()*100:.1f}%)")
            print(f"  Mean area ratio (SAM/ortho): {compared['area_ratio'].mean():.3f}")
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="SAM re-cut experiment")
    parser.add_argument("--grid-ids", nargs="+", help="Specific grid IDs")
    parser.add_argument("--batches", nargs="+", choices=["batch003", "batch004"],
                        help="Process all grids in batch(es)")
    parser.add_argument("--mode", choices=["fn", "tp-compare", "both"], default="fn",
                        help="fn=FN markers only, tp-compare=TP polygon comparison, both=all")
    parser.add_argument("--sample-per-grid", type=int, default=30,
                        help="Max TP predictions to sample per grid for comparison")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID for output directory")
    args = parser.parse_args()

    # Resolve grids
    grids = []
    if args.grid_ids:
        grids = args.grid_ids
    elif args.batches:
        for b in args.batches:
            grids.extend(BATCH_GRIDS.get(b, []))
    else:
        parser.error("Specify --grid-ids or --batches")

    # Output directory
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "grids": grids,
        "mode": args.mode,
        "sample_per_grid": args.sample_per_grid,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"SAM Recut Experiment: {len(grids)} grids, mode={args.mode}")
    print(f"Output: {output_dir}\n")

    # Load SAM 2.1 once
    predictor, device = _load_sam_model()

    if args.mode in ("fn", "both"):
        print("=" * 60)
        print("Stage 1: FN Marker Recut")
        print("=" * 60)
        run_fn_recut(grids, output_dir, predictor, device)

    if args.mode in ("tp-compare", "both"):
        print("\n" + "=" * 60)
        print("Stage 2: TP Polygon Comparison (SAM vs Orthogonalize)")
        print("=" * 60)
        run_tp_compare(grids, output_dir, predictor, device,
                       sample_per_grid=args.sample_per_grid)

    print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()
