"""Post-hoc spatial NMS on 382-grid predictions + spatial-clip eval vs clean_gt.

Two steps:
  1. Apply greedy NMS at low IoU (default 0.1) to every grid's
     predictions_metric.gpkg, writing predictions_metric_nms01.gpkg
     beside it. Also applies an optional polygon-confidence floor.
  2. Spatially clip the union of all post-NMS predictions to the
     CBD-25 GT footprint, compare against clean_gt total area, and
     emit a Tier-1-style summary against per-grid clean_gt where
     spatial overlap exists.

Coordinate-based, not grid-id-based — handles the JNB-vs-G grid id
mismatch by intersecting polygons in metric CRS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.strtree import STRtree

REPO = Path(__file__).resolve().parents[2]
METRIC_CRS = "EPSG:32735"


def greedy_nms(gdf: gpd.GeoDataFrame, iou_thresh: float, score_col: str) -> gpd.GeoDataFrame:
    """Greedy spatial NMS: sort by score desc, drop later polygons whose
    IoU with any already-kept polygon exceeds the threshold."""
    if len(gdf) == 0:
        return gdf
    g = gdf.sort_values(score_col, ascending=False).reset_index(drop=True)
    geoms = list(g.geometry.values)
    keep = [False] * len(geoms)
    kept_geoms: list = []
    tree: STRtree | None = None

    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        drop = False
        if kept_geoms:
            tree = STRtree(kept_geoms)
            cand_idx = tree.query(geom)
            for j in cand_idx:
                other = kept_geoms[j]
                inter = geom.intersection(other).area
                if inter == 0:
                    continue
                union = geom.area + other.area - inter
                if union > 0 and inter / union > iou_thresh:
                    drop = True
                    break
        if not drop:
            keep[i] = True
            kept_geoms.append(geom)

    return g.iloc[[i for i, k in enumerate(keep) if k]].reset_index(drop=True)


def process_grid(grid_dir: Path, out_name: str, iou_thresh: float,
                 conf_min: float, score_col: str) -> dict:
    src = grid_dir / "predictions_metric.gpkg"
    if not src.exists():
        return {"grid": grid_dir.name, "n_in": 0, "n_after_conf": 0, "n_after_nms": 0,
                "area_in_m2": 0.0, "area_after_conf_m2": 0.0, "area_after_nms_m2": 0.0}

    gdf = gpd.read_file(src)
    n_in = len(gdf)
    area_in = float(gdf.geometry.area.sum())

    use_col = score_col if score_col in gdf.columns else "score"
    if conf_min > 0 and use_col in gdf.columns:
        gdf = gdf[gdf[use_col] >= conf_min].copy()
    n_after_conf = len(gdf)
    area_after_conf = float(gdf.geometry.area.sum())

    gdf_nms = greedy_nms(gdf, iou_thresh=iou_thresh, score_col=use_col)
    n_after_nms = len(gdf_nms)
    area_after_nms = float(gdf_nms.geometry.area.sum())

    out_path = grid_dir / out_name
    if out_path.exists():
        out_path.unlink()
    gdf_nms.to_file(out_path, driver="GPKG")
    return {
        "grid": grid_dir.name,
        "n_in": n_in,
        "n_after_conf": n_after_conf,
        "n_after_nms": n_after_nms,
        "area_in_m2": area_in,
        "area_after_conf_m2": area_after_conf,
        "area_after_nms_m2": area_after_nms,
    }


def cbd25_footprint(task_grid_path: Path, cbd25_ids: list[str]) -> gpd.GeoDataFrame:
    task = gpd.read_file(task_grid_path)
    cbd = task[task["gridcell_id"].isin(cbd25_ids)].copy()
    return cbd.to_crs(METRIC_CRS)


def spatial_eval(results_root: Path, pred_name: str,
                 gt_root: Path, cbd25_fp: gpd.GeoDataFrame) -> dict:
    """Aggregate area comparison over the CBD25 footprint and per-GT-grid
    spatial intersection."""
    cbd_union = unary_union(cbd25_fp.geometry.values)
    pred_polys = []
    grid_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    for d in grid_dirs:
        f = d / pred_name
        if not f.exists():
            continue
        g = gpd.read_file(f).to_crs(METRIC_CRS)
        if len(g) == 0:
            continue
        clipped = g.clip(cbd_union)
        clipped = clipped[~clipped.geometry.is_empty]
        if len(clipped) > 0:
            pred_polys.append(clipped)
    if pred_polys:
        pred_in_cbd = pd.concat(pred_polys, ignore_index=True)
        pred_in_cbd = gpd.GeoDataFrame(pred_in_cbd, geometry="geometry", crs=METRIC_CRS)
    else:
        pred_in_cbd = gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)
    pred_area = float(pred_in_cbd.geometry.area.sum())

    gt_per_grid = {}
    gt_total = 0.0
    for sub in sorted(gt_root.iterdir()):
        gf = sub / f"{sub.name}_clean_gt.gpkg"
        if not gf.exists():
            continue
        gg = gpd.read_file(gf).to_crs(METRIC_CRS)
        a = float(gg.geometry.area.sum())
        gt_per_grid[sub.name] = a
        gt_total += a

    per_grid_rows = []
    cbd_grid_lookup = {row["gridcell_id"]: row.geometry for _, row in cbd25_fp.iterrows()}
    for gid, gt_a in gt_per_grid.items():
        gpoly = cbd_grid_lookup.get(gid)
        if gpoly is None:
            per_grid_rows.append({"grid": gid, "gt_m2": gt_a, "pred_m2": 0.0, "missing_grid_geom": True})
            continue
        pred_in_g = pred_in_cbd.clip(gpoly)
        pred_in_g = pred_in_g[~pred_in_g.geometry.is_empty]
        per_grid_rows.append({
            "grid": gid,
            "gt_m2": gt_a,
            "pred_m2": float(pred_in_g.geometry.area.sum()),
        })

    return {
        "pred_total_m2_in_cbd25": pred_area,
        "gt_total_m2": gt_total,
        "bulk": (pred_area / gt_total) if gt_total > 0 else None,
        "per_grid": per_grid_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path,
                   default=REPO / "results/johannesburg/unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox")
    p.add_argument("--task-grid", type=Path, default=REPO / "data/jhb_task_grid_unified.gpkg")
    p.add_argument("--gt-root", type=Path, default=REPO / "data/annotations_channel2_clean")
    p.add_argument("--iou", type=float, default=0.1)
    p.add_argument("--conf-min", type=float, default=0.925)
    p.add_argument("--score-col", default="confidence")
    p.add_argument("--out-name", default="predictions_metric_nms01_c0925.gpkg")
    p.add_argument("--summary-dir", type=Path,
                   default=REPO / "results/analysis/full382_nms01_2026-05-15")
    args = p.parse_args()

    args.summary_dir.mkdir(parents=True, exist_ok=True)

    cbd25_ids = ['G0772','G0773','G0774','G0775','G0776','G0814','G0815','G0816','G0817','G0818',
                 'G0853','G0854','G0855','G0856','G0857','G0888','G0889','G0890','G0891','G0892',
                 'G0922','G0923','G0924','G0925','G0926']

    grid_dirs = sorted([d for d in args.results_root.iterdir() if d.is_dir()])
    print(f"[nms] processing {len(grid_dirs)} grids — IoU={args.iou} conf_min={args.conf_min}")
    nms_rows = []
    for i, d in enumerate(grid_dirs):
        row = process_grid(d, args.out_name, args.iou, args.conf_min, args.score_col)
        nms_rows.append(row)
        if (i + 1) % 50 == 0 or i == len(grid_dirs) - 1:
            print(f"  [{i+1}/{len(grid_dirs)}] {d.name} "
                  f"in={row['n_in']} c≥{args.conf_min}→{row['n_after_conf']} nms→{row['n_after_nms']}")
    df = pd.DataFrame(nms_rows)
    df.to_csv(args.summary_dir / "nms_per_grid.csv", index=False)

    totals = {
        "n_polys_in": int(df["n_in"].sum()),
        "n_polys_after_conf": int(df["n_after_conf"].sum()),
        "n_polys_after_nms": int(df["n_after_nms"].sum()),
        "area_m2_in": float(df["area_in_m2"].sum()),
        "area_m2_after_conf": float(df["area_after_conf_m2"].sum()),
        "area_m2_after_nms": float(df["area_after_nms_m2"].sum()),
        "drop_by_conf_pct": (1 - df["n_after_conf"].sum() / df["n_in"].sum()) * 100,
        "drop_by_nms_pct": (1 - df["n_after_nms"].sum() / max(df["n_after_conf"].sum(), 1)) * 100,
    }
    print(f"\n[totals] polys {totals['n_polys_in']} -> {totals['n_polys_after_conf']} (c≥{args.conf_min}) "
          f"-> {totals['n_polys_after_nms']} (NMS@{args.iou})")
    print(f"[totals] area  {totals['area_m2_in']:.0f} -> "
          f"{totals['area_m2_after_conf']:.0f} -> {totals['area_m2_after_nms']:.0f} m²")
    print(f"[totals] dropped {totals['drop_by_conf_pct']:.1f}% by conf, "
          f"{totals['drop_by_nms_pct']:.1f}% by NMS")

    print(f"\n[eval] spatial clip → CBD25 footprint vs clean_gt")
    cbd25_fp = cbd25_footprint(args.task_grid, cbd25_ids)
    summary = spatial_eval(args.results_root, args.out_name, args.gt_root, cbd25_fp)
    summary["nms_totals"] = totals
    summary["params"] = {"iou": args.iou, "conf_min": args.conf_min,
                         "score_col": args.score_col, "out_name": args.out_name}
    with open(args.summary_dir / "spatial_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    pg = pd.DataFrame(summary["per_grid"])
    pg["pred_gt_ratio"] = pg["pred_m2"] / pg["gt_m2"]
    pg.to_csv(args.summary_dir / "spatial_eval_per_grid.csv", index=False)

    print(f"\n[summary] pred_in_CBD25 = {summary['pred_total_m2_in_cbd25']:.0f} m²")
    print(f"[summary] gt_total       = {summary['gt_total_m2']:.0f} m²")
    print(f"[summary] bulk           = {summary['bulk']:.4f}")
    print(f"[summary] CBD25 grids w/ both pred+gt: {len(pg)}")
    print(f"\n[summary] outputs at {args.summary_dir}")


if __name__ == "__main__":
    main()
