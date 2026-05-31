"""Post-hoc UNION-merge (not NMS drop) at low IoU on 382-grid predictions +
spatial-clip eval vs clean_gt.

Difference from posthoc_nms_and_spatial_eval.py:
  - When two polygons have IoU > threshold, take their geometric UNION
    instead of dropping the lower-score one. Preserves all unique area.
  - Uses union-find on the polygon adjacency graph so chains of
    overlapping polygons get fused into one merged geometry.
  - Merged polygon keeps the attributes (incl. score) of the
    highest-scoring member of its cluster; area_m2 is recomputed.

This is the natural stitching/de-fragmentation operation — it fixes
chunk-boundary fragmentation without dropping coverage.
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


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def merge_overlapping(gdf: gpd.GeoDataFrame, iou_thresh: float,
                      score_col: str) -> gpd.GeoDataFrame:
    """Polygon-level UNION-merge under IoU gating. Polygons with IoU >
    threshold are clustered (transitively) and each cluster becomes one
    unary_union polygon. Attributes come from the cluster's max-score
    member; area_m2 is recomputed."""
    if len(gdf) == 0:
        return gdf
    geoms = list(gdf.geometry.values)
    n = len(geoms)
    uf = UnionFind(n)
    tree = STRtree(geoms)

    for i, gi in enumerate(geoms):
        if gi is None or gi.is_empty:
            continue
        cand = tree.query(gi)
        ai = gi.area
        for j in cand:
            if j <= i:
                continue
            gj = geoms[j]
            if gj is None or gj.is_empty:
                continue
            inter = gi.intersection(gj).area
            if inter == 0:
                continue
            union = ai + gj.area - inter
            if union > 0 and inter / union > iou_thresh:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        clusters.setdefault(r, []).append(i)

    rows = []
    df = gdf.reset_index(drop=True)
    for root, members in clusters.items():
        if len(members) == 1:
            rows.append(df.iloc[members[0]].to_dict())
            continue
        merged_geom = unary_union([geoms[m] for m in members])
        scores = [df.iloc[m].get(score_col, 0) or 0 for m in members]
        best = members[int(np.argmax(scores))]
        rec = df.iloc[best].to_dict()
        rec["geometry"] = merged_geom
        rec["area_m2"] = float(merged_geom.area) if hasattr(merged_geom, "area") else None
        rec["n_merged"] = len(members)
        rows.append(rec)

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
    return out


def process_grid(grid_dir: Path, out_name: str, iou_thresh: float,
                 conf_min: float, score_col: str) -> dict:
    src = grid_dir / "predictions_metric.gpkg"
    if not src.exists():
        return {"grid": grid_dir.name, "n_in": 0, "n_after_conf": 0, "n_after_merge": 0,
                "area_in_m2": 0.0, "area_after_conf_m2": 0.0, "area_after_merge_m2": 0.0}
    gdf = gpd.read_file(src)
    n_in = len(gdf)
    area_in = float(gdf.geometry.area.sum())

    use_col = score_col if score_col in gdf.columns else "score"
    if conf_min > 0 and use_col in gdf.columns:
        gdf = gdf[gdf[use_col] >= conf_min].copy()
    n_after_conf = len(gdf)
    area_after_conf = float(gdf.geometry.area.sum())

    merged = merge_overlapping(gdf, iou_thresh=iou_thresh, score_col=use_col)
    n_after_merge = len(merged)
    area_after_merge = float(merged.geometry.area.sum())

    out_path = grid_dir / out_name
    if out_path.exists():
        out_path.unlink()
    if len(merged) > 0:
        # Drop columns we don't need to write
        merged.to_file(out_path, driver="GPKG")
    return {
        "grid": grid_dir.name,
        "n_in": n_in,
        "n_after_conf": n_after_conf,
        "n_after_merge": n_after_merge,
        "area_in_m2": area_in,
        "area_after_conf_m2": area_after_conf,
        "area_after_merge_m2": area_after_merge,
    }


def cbd25_footprint(task_grid_path: Path, cbd25_ids: list[str]) -> gpd.GeoDataFrame:
    task = gpd.read_file(task_grid_path)
    cbd = task[task["gridcell_id"].isin(cbd25_ids)].copy()
    return cbd.to_crs(METRIC_CRS)


def spatial_eval(results_root: Path, pred_name: str, gt_root: Path,
                 cbd25_fp: gpd.GeoDataFrame) -> dict:
    cbd_union = unary_union(cbd25_fp.geometry.values)
    pred_polys = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
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

    cbd_grid_lookup = {row["gridcell_id"]: row.geometry for _, row in cbd25_fp.iterrows()}
    per_grid_rows = []
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
    p.add_argument("--out-name", default="predictions_metric_merge01_c0925.gpkg")
    p.add_argument("--summary-dir", type=Path,
                   default=REPO / "results/analysis/full382_merge01_2026-05-15")
    args = p.parse_args()

    args.summary_dir.mkdir(parents=True, exist_ok=True)

    cbd25_ids = ['G0772','G0773','G0774','G0775','G0776','G0814','G0815','G0816','G0817','G0818',
                 'G0853','G0854','G0855','G0856','G0857','G0888','G0889','G0890','G0891','G0892',
                 'G0922','G0923','G0924','G0925','G0926']

    grid_dirs = sorted([d for d in args.results_root.iterdir() if d.is_dir()])
    print(f"[merge] processing {len(grid_dirs)} grids — IoU>{args.iou} ⇒ union, conf_min={args.conf_min}")
    rows = []
    for i, d in enumerate(grid_dirs):
        row = process_grid(d, args.out_name, args.iou, args.conf_min, args.score_col)
        rows.append(row)
        if (i + 1) % 50 == 0 or i == len(grid_dirs) - 1:
            print(f"  [{i+1}/{len(grid_dirs)}] {d.name} "
                  f"in={row['n_in']} c≥{args.conf_min}→{row['n_after_conf']} merge→{row['n_after_merge']}")
    df = pd.DataFrame(rows)
    df.to_csv(args.summary_dir / "merge_per_grid.csv", index=False)

    totals = {
        "n_polys_in": int(df["n_in"].sum()),
        "n_polys_after_conf": int(df["n_after_conf"].sum()),
        "n_polys_after_merge": int(df["n_after_merge"].sum()),
        "area_m2_in": float(df["area_in_m2"].sum()),
        "area_m2_after_conf": float(df["area_after_conf_m2"].sum()),
        "area_m2_after_merge": float(df["area_after_merge_m2"].sum()),
        "drop_by_conf_pct": (1 - df["n_after_conf"].sum() / max(df["n_in"].sum(), 1)) * 100,
        "collapse_by_merge_pct": (1 - df["n_after_merge"].sum() / max(df["n_after_conf"].sum(), 1)) * 100,
    }
    print(f"\n[totals] polys {totals['n_polys_in']} -> {totals['n_polys_after_conf']} (c≥{args.conf_min}) "
          f"-> {totals['n_polys_after_merge']} (merge@{args.iou})")
    print(f"[totals] area  {totals['area_m2_in']:.0f} -> "
          f"{totals['area_m2_after_conf']:.0f} -> {totals['area_m2_after_merge']:.0f} m²")
    print(f"[totals] dropped {totals['drop_by_conf_pct']:.1f}% polys by conf, "
          f"{totals['collapse_by_merge_pct']:.1f}% polys collapsed by merge")
    print(f"[totals] area lost in merge (only via dedupe of overlap): "
          f"{(1 - totals['area_m2_after_merge']/max(totals['area_m2_after_conf'],1))*100:.1f}%")

    print(f"\n[eval] spatial clip → CBD25 footprint vs clean_gt")
    cbd25_fp = cbd25_footprint(args.task_grid, cbd25_ids)
    summary = spatial_eval(args.results_root, args.out_name, args.gt_root, cbd25_fp)
    summary["merge_totals"] = totals
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
    print(f"\n[summary] outputs at {args.summary_dir}")


if __name__ == "__main__":
    main()
