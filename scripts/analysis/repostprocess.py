"""
Re-run post-processing on existing per-tile vectors + masks without re-running inference.

Reads vectors/*.geojson + masks/*_mask.tif, recomputes confidence from mask band 2,
applies tiered confidence thresholds, and overwrites predictions_metric.gpkg.

Usage:
  python scripts/analysis/repostprocess.py --grids G2030 G1864
  python scripts/analysis/repostprocess.py --all          # all grids on D drive
  python scripts/analysis/repostprocess.py --grids G2030 --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from detect_and_evaluate import (
    CONF_TIERED,
    ELONGATION_TIERED,
    MAX_ELONGATION,
    MIN_OBJECT_AREA,
    SHADOW_RGB_THRESH,
    spatial_nms,
)
from core.grid_utils import TILES_ROOT

RESULTS_DIRS = [
    Path("/mnt/d/ZAsolar/results"),
    Path(__file__).resolve().parent.parent.parent / "results",
]
METRIC_CRS = "EPSG:32734"


def find_results_dir(grid_id: str) -> Path | None:
    for d in RESULTS_DIRS:
        p = d / grid_id
        if (p / "vectors").exists() and (p / "masks").exists():
            return p
    return None


def repostprocess_grid(grid_id: str, dry_run: bool = False) -> dict | None:
    gdir = find_results_dir(grid_id)
    if gdir is None:
        print(f"  [SKIP] {grid_id}: no vectors/masks found")
        return None

    vectors_dir = gdir / "vectors"
    masks_dir = gdir / "masks"
    tiles_dir = TILES_ROOT / grid_id

    all_gdfs = []
    for vf in sorted(vectors_dir.glob("*_vectors.geojson")):
        tile_name = vf.stem.replace("_vectors", "")
        mask_path = masks_dir / f"{tile_name}_mask.tif"
        tif_path = tiles_dir / f"{tile_name}.tif"

        gdf = gpd.read_file(str(vf))
        if len(gdf) == 0:
            continue

        # Confidence from mask band 2
        if mask_path.exists():
            try:
                conf_stats = rasterstats.zonal_stats(
                    gdf, str(mask_path), band=2, stats=["mean"], nodata=0
                )
                gdf["confidence"] = [
                    (s["mean"] / 255.0) if s["mean"] is not None else 0.0
                    for s in conf_stats
                ]
            except Exception as e:
                print(f"    [WARN] {tile_name} confidence failed: {e}")
                gdf["confidence"] = 0.5
        else:
            gdf["confidence"] = 0.5

        # NOTE: RGB shadow/overexposure filter is SKIPPED here because
        # per-tile vectors were already filtered during inference.
        # Applying it again would double-filter and drop valid detections.

        if len(gdf) > 0:
            try:
                import geoai
                gdf = geoai.add_geometric_properties(gdf)
            except Exception:
                # Fallback: compute basic metrics manually
                gdf_proj = gdf.to_crs(METRIC_CRS)
                gdf["area_m2"] = gdf_proj.geometry.area
            gdf["source_tile"] = tile_name
            all_gdfs.append(gdf)

    if not all_gdfs:
        print(f"  [SKIP] {grid_id}: no detections in vectors")
        return None

    pred_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    if pred_gdf.crs is None:
        pred_gdf = pred_gdf.set_crs(METRIC_CRS)
    pred_gdf = pred_gdf.to_crs(METRIC_CRS)

    raw_count = len(pred_gdf)

    # NMS
    pred_gdf = spatial_nms(pred_gdf, iou_threshold=0.5)

    # Area filter
    if "area_m2" in pred_gdf.columns:
        pred_gdf = pred_gdf[pred_gdf["area_m2"] >= MIN_OBJECT_AREA].copy()

    # Elongation filter (tiered by area)
    if "elongation" in pred_gdf.columns and "area_m2" in pred_gdf.columns:
        elong_keep = pd.Series(False, index=pred_gdf.index)
        for min_area, max_elong in ELONGATION_TIERED:
            tier_mask = (pred_gdf["area_m2"] >= min_area) & ~elong_keep
            elong_keep |= tier_mask & (pred_gdf["elongation"] <= max_elong)
        pred_gdf = pred_gdf[elong_keep].copy()
    elif "elongation" in pred_gdf.columns:
        pred_gdf = pred_gdf[pred_gdf["elongation"] <= MAX_ELONGATION].copy()

    # Tiered confidence filter
    if "area_m2" in pred_gdf.columns:
        keep_mask = pd.Series(False, index=pred_gdf.index)
        for min_area, thresh in CONF_TIERED:
            tier_mask = (pred_gdf["area_m2"] >= min_area) & ~keep_mask
            keep_mask |= tier_mask & (pred_gdf["confidence"] >= thresh)
        pred_gdf = pred_gdf[keep_mask].copy()

    # Stable index: keep original predictions in their original order,
    # append newly recovered detections at the end.
    old_path = gdir / "predictions_metric.gpkg"
    n_recovered = 0
    if old_path.exists():
        old_gdf = gpd.read_file(str(old_path))
        if old_gdf.crs and old_gdf.crs.to_epsg() != pred_gdf.crs.to_epsg():
            old_gdf = old_gdf.to_crs(pred_gdf.crs)

        # Match new predictions to old ones by spatial overlap (IoU > 0.5)
        matched_new = set()
        ordered = []
        for oi in range(len(old_gdf)):
            old_geom = old_gdf.iloc[oi].geometry
            best_j, best_iou = -1, 0.5
            for nj in range(len(pred_gdf)):
                if nj in matched_new:
                    continue
                new_geom = pred_gdf.iloc[nj].geometry
                if not old_geom.intersects(new_geom):
                    continue
                inter = old_geom.intersection(new_geom).area
                union = old_geom.area + new_geom.area - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_j, best_iou = nj, iou
            if best_j >= 0:
                matched_new.add(best_j)
                ordered.append(best_j)

        # Append unmatched new predictions (recovered) at the end
        for nj in range(len(pred_gdf)):
            if nj not in matched_new:
                ordered.append(nj)
                n_recovered += 1

        pred_gdf = pred_gdf.iloc[ordered].reset_index(drop=True)
    else:
        n_recovered = len(pred_gdf[pred_gdf["confidence"] < 0.85]) if len(pred_gdf) > 0 else 0

    result = {
        "grid": grid_id,
        "raw": raw_count,
        "post_filter": len(pred_gdf),
        "recovered": n_recovered,
    }

    if not dry_run:
        out_path = gdir / "predictions_metric.gpkg"
        pred_gdf.to_file(str(out_path), driver="GPKG")
        export_gdf = pred_gdf.to_crs("EPSG:4326")
        export_gdf.to_file(str(gdir / "predictions.geojson"), driver="GeoJSON")
        result["output"] = str(out_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Re-run post-processing with tiered confidence")
    parser.add_argument("--grids", nargs="+", help="Grid IDs to reprocess")
    parser.add_argument("--all", action="store_true", help="Reprocess all grids on D drive")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write files")
    args = parser.parse_args()

    if args.all:
        d = Path("/mnt/d/ZAsolar/results")
        grids = sorted(g.name for g in d.iterdir() if g.is_dir() and g.name.startswith("G") and (g / "vectors").exists())
    elif args.grids:
        grids = args.grids
    else:
        print("Provide --grids or --all")
        sys.exit(1)

    tier_desc = ", ".join(f"≥{a}m²→conf≥{t}" for a, t in CONF_TIERED)
    print(f"Tiered confidence: {tier_desc}")
    print(f"Area ≥ {MIN_OBJECT_AREA}m², elongation ≤ {MAX_ELONGATION}")
    print(f"{'Dry run' if args.dry_run else 'Writing results'}")
    print(f"Grids: {len(grids)}\n")

    results = []
    for gid in grids:
        print(f"Processing {gid}...")
        r = repostprocess_grid(gid, dry_run=args.dry_run)
        if r:
            results.append(r)
            print(f"  raw={r['raw']} → final={r['post_filter']} (recovered={r['recovered']})")

    if results:
        total_recovered = sum(r["recovered"] for r in results)
        total_final = sum(r["post_filter"] for r in results)
        print(f"\nTotal: {total_final} predictions, {total_recovered} recovered by tiered thresholds")


if __name__ == "__main__":
    main()
