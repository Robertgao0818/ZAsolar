#!/usr/bin/env python3
"""Post-processing ablation experiment.

Re-runs vectorization + filtering on existing mask outputs with different
parameters, saving results in parallel directories for comparison.
Does NOT overwrite existing predictions.

Usage:
    python scripts/analysis/postprocess_ablation.py --config A
    python scripts/analysis/postprocess_ablation.py --config B
    python scripts/analysis/postprocess_ablation.py --config all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geoai

# ---------------------------------------------------------------------------
# Experiment configs: each is a named variant of post-processing parameters
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline": {
        "description": "Current production parameters (control)",
        "epsilon": 0.2,
        "min_object_area": 5,
        "elongation_tiered": [(100, 15.0), (0, 8.0)],
        "conf_tiered": [(200, 0.70), (100, 0.65), (0, 0.85)],
        "shadow_rgb_thresh": 60,
    },
    "A": {
        "description": "Lower epsilon (less orthogonalize distortion)",
        "epsilon": 0.05,
        "min_object_area": 5,
        "elongation_tiered": [(100, 15.0), (0, 8.0)],
        "conf_tiered": [(200, 0.70), (100, 0.65), (0, 0.85)],
        "shadow_rgb_thresh": 60,
    },
    "B": {
        "description": "Lower epsilon + relaxed elongation",
        "epsilon": 0.05,
        "min_object_area": 5,
        "elongation_tiered": [(100, 20.0), (0, 12.0)],
        "conf_tiered": [(200, 0.70), (100, 0.65), (0, 0.85)],
        "shadow_rgb_thresh": 60,
    },
    "C": {
        "description": "Lower epsilon + relaxed elongation + lower confidence",
        "epsilon": 0.05,
        "min_object_area": 5,
        "elongation_tiered": [(100, 20.0), (0, 12.0)],
        "conf_tiered": [(200, 0.60), (100, 0.55), (0, 0.75)],
        "shadow_rgb_thresh": 60,
    },
    "D": {
        "description": "rasterio.features.shapes vectorization (no orthogonalize)",
        "vectorizer": "rasterio",
        "simplify_tolerance": 0.5,  # meters, applied after vectorization
        "min_object_area": 5,
        "elongation_tiered": [(100, 15.0), (0, 8.0)],
        "conf_tiered": [(200, 0.70), (100, 0.65), (0, 0.85)],
        "shadow_rgb_thresh": 60,
    },
}

# Grids to test on (subset for speed; mix of batch 003 + 004)
SAMPLE_GRIDS = [
    "G1688",  # batch003, high FP rate, diverse
    "G1690",  # batch003, many FN
    "G1749",  # batch003
    "G1855",  # batch004, many FN
    "G1920",  # batch004, many FN
    "G1971",  # batch004, most FN
    "G1976",  # batch004
    "G2027",  # batch004
]

METRIC_CRS = "EPSG:32734"
INPUT_CRS = "EPSG:4326"


def find_results_base(grid_id: str) -> Path | None:
    """Find the results directory for a grid."""
    for base in [Path("results"), Path("/mnt/d/ZAsolar/results")]:
        p = base / grid_id
        if p.exists():
            return p
    return None


def find_mask_files(grid_dir: Path) -> list[Path]:
    """Find all detection mask TIF files for a grid."""
    masks_dir = grid_dir / "masks"
    if not masks_dir.exists():
        return []
    return sorted(masks_dir.glob("*_mask.tif"))


def find_tile_file(grid_dir: Path, mask_path: Path) -> Path | None:
    """Find the corresponding tile TIF for a mask file."""
    # mask name: GridID_col_row_geo_mask.tif -> tile: GridID_col_row_geo.tif
    tile_name = mask_path.stem.replace("_mask", "") + ".tif"
    tiles_root = Path("/mnt/d/ZAsolar/tiles")
    grid_id = grid_dir.name
    tile_path = tiles_root / grid_id / tile_name
    if tile_path.exists():
        return tile_path
    return None


def vectorize_orthogonalize(mask_path: Path, epsilon: float) -> gpd.GeoDataFrame | None:
    """Vectorize mask using geoai.orthogonalize."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        gdf = geoai.orthogonalize(
            input_path=str(mask_path),
            output_path=tmp_path,
            epsilon=epsilon,
        )
        return gdf
    except Exception as exc:
        # Empty geometry or single-feature edge case
        msg = str(exc).lower()
        if ("empty" in msg or "no valid" in msg or "geometry column" in msg
                or "without a geometry" in msg):
            return None
        raise
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def vectorize_rasterio(mask_path: Path, simplify_tolerance: float = 0.5) -> gpd.GeoDataFrame | None:
    """Vectorize mask using rasterio.features.shapes (preserves exact mask shape)."""
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(mask_path) as src:
        mask_band = src.read(1)
        transform = src.transform
        crs = src.crs

    # Binary mask: presence band > 0
    binary = (mask_band > 0).astype("uint8")

    polygons = []
    for geom, value in shapes(binary, mask=binary > 0, transform=transform):
        if value > 0:
            poly = shape(geom)
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)
            if not poly.is_empty and poly.area > 0:
                polygons.append(poly)

    if not polygons:
        return None

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    # Backfill confidence from band 2
    try:
        import rasterstats
        conf_stats = rasterstats.zonal_stats(gdf, str(mask_path), band=2,
                                              stats=["mean"], nodata=0)
        gdf["confidence"] = [
            (s["mean"] / 255.0) if s["mean"] is not None else 0.0
            for s in conf_stats
        ]
    except Exception:
        gdf["confidence"] = 0.5

    return gdf


def apply_filters(gdf: gpd.GeoDataFrame, config: dict,
                  tile_path: Path | None = None) -> gpd.GeoDataFrame:
    """Apply post-processing filters matching detect_and_evaluate.py logic."""
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame()

    # Ensure metric CRS
    if gdf.crs and str(gdf.crs) != METRIC_CRS:
        gdf = gdf.to_crs(METRIC_CRS)
    elif not gdf.crs:
        gdf = gdf.set_crs(METRIC_CRS)

    # Color filter
    if tile_path and tile_path.exists():
        try:
            import rasterstats
            stats_r = rasterstats.zonal_stats(gdf, str(tile_path), band=1, stats=['mean'], nodata=0)
            stats_g = rasterstats.zonal_stats(gdf, str(tile_path), band=2, stats=['mean'], nodata=0)
            stats_b = rasterstats.zonal_stats(gdf, str(tile_path), band=3, stats=['mean'], nodata=0)
            gdf["mean_r"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_r]
            gdf["mean_g"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_g]
            gdf["mean_b"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_b]

            thresh = config["shadow_rgb_thresh"]
            is_shadow = ((gdf["mean_r"] < thresh) & (gdf["mean_g"] < thresh) & (gdf["mean_b"] < thresh))
            is_bright = (gdf["mean_r"] > 250) & (gdf["mean_g"] > 250) & (gdf["mean_b"] > 250)
            gdf = gdf[~(is_shadow | is_bright)].copy()
        except Exception:
            pass

    if len(gdf) == 0:
        return gdf

    # Add geometric properties
    gdf = geoai.add_geometric_properties(gdf)

    # Area filter
    min_area = config["min_object_area"]
    if "area_m2" in gdf.columns:
        gdf = gdf[gdf["area_m2"] >= min_area].copy()

    # Elongation filter (tiered)
    if "elongation" in gdf.columns and "area_m2" in gdf.columns:
        keep = pd.Series(False, index=gdf.index)
        for min_a, max_e in config["elongation_tiered"]:
            tier = (gdf["area_m2"] >= min_a) & ~keep
            keep |= tier & (gdf["elongation"] <= max_e)
        gdf = gdf[keep].copy()

    # Confidence filter (tiered)
    if "confidence" in gdf.columns and "area_m2" in gdf.columns:
        keep = pd.Series(False, index=gdf.index)
        for min_a, thresh in config["conf_tiered"]:
            tier = (gdf["area_m2"] >= min_a) & ~keep
            keep |= tier & (gdf["confidence"] >= thresh)
        gdf = gdf[keep].copy()

    return gdf


def spatial_nms(gdf: gpd.GeoDataFrame, iou_threshold: float = 0.5) -> gpd.GeoDataFrame:
    """Simple spatial NMS to remove overlapping detections."""
    if len(gdf) <= 1:
        return gdf

    from shapely.strtree import STRtree

    geoms = gdf.geometry.values
    confs = gdf["confidence"].values if "confidence" in gdf.columns else [1.0] * len(gdf)
    order = sorted(range(len(gdf)), key=lambda i: -confs[i])

    tree = STRtree(geoms)
    keep = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        candidates = tree.query(geoms[i])
        for j in candidates:
            if j == i or j in suppressed:
                continue
            intersection = geoms[i].intersection(geoms[j]).area
            union = geoms[i].area + geoms[j].area - intersection
            if union > 0 and intersection / union > iou_threshold:
                suppressed.add(j)

    return gdf.iloc[keep].copy()


def run_config(config_name: str, config: dict, grids: list[str],
               output_base: Path) -> pd.DataFrame:
    """Run a single config across all grids, return summary DataFrame."""
    out_dir = output_base / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    use_rasterio = config.get("vectorizer") == "rasterio"

    for grid_id in grids:
        grid_dir = find_results_base(grid_id)
        if grid_dir is None:
            print(f"  [{grid_id}] results dir not found, skip")
            continue

        mask_files = find_mask_files(grid_dir)
        if not mask_files:
            print(f"  [{grid_id}] no mask files found, skip")
            continue

        all_gdfs = []
        for mask_path in mask_files:
            tile_path = find_tile_file(grid_dir, mask_path)
            tile_name = mask_path.stem.replace("_masks", "")

            # Vectorize
            if use_rasterio:
                gdf = vectorize_rasterio(mask_path, config.get("simplify_tolerance", 0.5))
            else:
                gdf = vectorize_orthogonalize(mask_path, config["epsilon"])

            if gdf is None or len(gdf) == 0:
                continue

            # Backfill confidence for orthogonalize path
            if not use_rasterio and "confidence" not in gdf.columns:
                try:
                    import rasterstats
                    conf_stats = rasterstats.zonal_stats(gdf, str(mask_path), band=2,
                                                         stats=["mean"], nodata=0)
                    gdf["confidence"] = [
                        (s["mean"] / 255.0) if s["mean"] is not None else 0.0
                        for s in conf_stats
                    ]
                except Exception:
                    gdf["confidence"] = 0.5

            # Apply filters
            gdf = apply_filters(gdf, config, tile_path)
            if len(gdf) > 0:
                gdf["source_tile"] = tile_name
                all_gdfs.append(gdf)

        if not all_gdfs:
            print(f"  [{grid_id}] no predictions after filtering")
            results.append({
                "grid_id": grid_id, "config": config_name,
                "n_preds": 0, "n_preds_baseline": 0,
            })
            continue

        pred_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        if pred_gdf.crs is None:
            pred_gdf = pred_gdf.set_crs(METRIC_CRS)
        pred_gdf = pred_gdf.to_crs(METRIC_CRS)

        # Spatial NMS
        pred_gdf = spatial_nms(pred_gdf)

        # Save
        grid_out = out_dir / grid_id
        grid_out.mkdir(parents=True, exist_ok=True)
        pred_gdf.to_file(str(grid_out / "predictions_metric.gpkg"), driver="GPKG")

        # Load baseline for comparison
        baseline_gpkg = grid_dir / "predictions_metric.gpkg"
        n_baseline = 0
        if baseline_gpkg.exists():
            n_baseline = len(gpd.read_file(baseline_gpkg))

        print(f"  [{grid_id}] {config_name}: {len(pred_gdf)} preds (baseline: {n_baseline})")

        results.append({
            "grid_id": grid_id,
            "config": config_name,
            "n_preds": len(pred_gdf),
            "n_preds_baseline": n_baseline,
            "mean_area": pred_gdf["area_m2"].mean() if "area_m2" in pred_gdf.columns else 0,
            "mean_conf": pred_gdf["confidence"].mean() if "confidence" in pred_gdf.columns else 0,
        })

    return pd.DataFrame(results)


def evaluate_against_gt(config_name: str, grids: list[str],
                        output_base: Path) -> pd.DataFrame:
    """Compare ablation predictions against reviewed GT (correct+edit decisions)."""
    from shapely.strtree import STRtree

    eval_rows = []
    pred_dir = output_base / config_name

    for grid_id in grids:
        # Load ablation predictions
        pred_path = pred_dir / grid_id / "predictions_metric.gpkg"
        if not pred_path.exists():
            continue
        preds = gpd.read_file(pred_path)
        if preds.crs and preds.crs.to_epsg() != 32734:
            preds = preds.to_crs(epsg=32734)

        # Load GT (reviewed correct+edit predictions from baseline)
        grid_dir = find_results_base(grid_id)
        if grid_dir is None:
            continue
        dec_csv = grid_dir / "review" / "detection_review_decisions.csv"
        gt_gpkg = grid_dir / "predictions_metric.gpkg"
        if not dec_csv.exists() or not gt_gpkg.exists():
            continue

        decisions = pd.read_csv(dec_csv)
        gt_all = gpd.read_file(gt_gpkg)
        accepted = set(decisions[decisions['status'].isin(['correct', 'edit'])]['pred_id'])
        gt = gt_all[gt_all.index.isin(accepted)].copy()
        if gt.crs and gt.crs.to_epsg() != 32734:
            gt = gt.to_crs(epsg=32734)

        if len(gt) == 0:
            continue

        # Match predictions to GT at IoU 0.3
        iou_thresh = 0.3
        gt_tree = STRtree(gt.geometry.values)

        tp = 0
        for pred_geom in preds.geometry:
            candidates = gt_tree.query(pred_geom)
            for ci in candidates:
                gt_geom = gt.geometry.values[ci]
                inter = pred_geom.intersection(gt_geom).area
                union = pred_geom.area + gt_geom.area - inter
                if union > 0 and inter / union >= iou_thresh:
                    tp += 1
                    break

        fp = len(preds) - tp
        # Count FN: GT not matched by any prediction
        pred_tree = STRtree(preds.geometry.values)
        fn = 0
        for gt_geom in gt.geometry:
            matched = False
            candidates = pred_tree.query(gt_geom)
            for ci in candidates:
                pred_geom = preds.geometry.values[ci]
                inter = gt_geom.intersection(pred_geom).area
                union = gt_geom.area + pred_geom.area - inter
                if union > 0 and inter / union >= iou_thresh:
                    matched = True
                    break
            if not matched:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        eval_rows.append({
            "grid_id": grid_id,
            "config": config_name,
            "n_gt": len(gt),
            "n_preds": len(preds),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    return pd.DataFrame(eval_rows)


def main():
    parser = argparse.ArgumentParser(description="Post-processing ablation experiment")
    parser.add_argument("--config", default="all",
                        help="Config name (A/B/C/D/baseline/all)")
    parser.add_argument("--grids", nargs="*", default=None,
                        help="Grid IDs to process (default: SAMPLE_GRIDS)")
    parser.add_argument("--output-dir", default="results/analysis/postprocess_ablation",
                        help="Output directory")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip vectorization, only run evaluation")
    args = parser.parse_args()

    grids = args.grids or SAMPLE_GRIDS
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    if args.config == "all":
        config_names = list(CONFIGS.keys())
    else:
        config_names = [args.config]

    all_summaries = []
    all_evals = []

    for cname in config_names:
        if cname not in CONFIGS:
            print(f"Unknown config: {cname}")
            continue

        config = CONFIGS[cname]
        print(f"\n{'='*60}")
        print(f"Config: {cname} — {config['description']}")
        print(f"{'='*60}")

        if not args.eval_only:
            t0 = time.time()
            summary = run_config(cname, config, grids, output_base)
            elapsed = time.time() - t0
            print(f"\n  Completed in {elapsed:.1f}s")
            all_summaries.append(summary)

        # Evaluate
        eval_df = evaluate_against_gt(cname, grids, output_base)
        all_evals.append(eval_df)

    # Save combined results
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(output_base / "summary.csv", index=False)

    if all_evals:
        eval_combined = pd.concat(all_evals, ignore_index=True)
        eval_combined.to_csv(output_base / "evaluation.csv", index=False)

        if len(eval_combined) == 0:
            print("\nNo evaluation results (no grids with GT found)")
            return

        # Print comparison table
        print(f"\n{'='*80}")
        print("EVALUATION COMPARISON (IoU@0.3)")
        print(f"{'='*80}")

        pivot = eval_combined.groupby("config").agg({
            "tp": "sum", "fp": "sum", "fn": "sum",
            "n_preds": "sum", "n_gt": "sum",
        })
        pivot["precision"] = pivot["tp"] / (pivot["tp"] + pivot["fp"])
        pivot["recall"] = pivot["tp"] / (pivot["tp"] + pivot["fn"])
        pivot["f1"] = 2 * pivot["precision"] * pivot["recall"] / (pivot["precision"] + pivot["recall"])

        for cname in config_names:
            if cname in pivot.index:
                r = pivot.loc[cname]
                desc = CONFIGS[cname]["description"]
                print(f"\n  {cname}: {desc}")
                print(f"    Preds={int(r['n_preds']):4d}  GT={int(r['n_gt']):4d}  "
                      f"TP={int(r['tp']):4d}  FP={int(r['fp']):4d}  FN={int(r['fn']):4d}")
                print(f"    P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")

    # Save config metadata
    with open(output_base / "configs.json", "w") as f:
        json.dump(CONFIGS, f, indent=2)

    print(f"\nResults saved to {output_base}/")


if __name__ == "__main__":
    main()
