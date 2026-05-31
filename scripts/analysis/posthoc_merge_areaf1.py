"""Apply post-hoc MERGE (polygon union under IoU gating) + polygon-conf
filter on a results dir, then compute apples-to-apples Area F1 vs
clean_gt for each grid and aggregate.

Reuses the merge_overlapping logic from posthoc_merge_and_spatial_eval.py
but adds pixel-level set-theoretic Area F1 (intersection / pred area /
gt area) per the Tier-1 metric system.

Two modes:
  - --no-merge      : skip merging, evaluate baseline at given conf
  - default         : merge at --iou, evaluate after merging

Use this on the 25-grid CBD native predictions for an apples-to-apples
comparison with the production audit's agg_area_F1 = 0.839 (NMS=0.5,
per-detection, c=0.925).
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
    if len(gdf) == 0:
        return gdf
    geoms = list(gdf.geometry.values)
    n = len(geoms)
    uf = UnionFind(n)
    tree = STRtree(geoms)
    for i, gi in enumerate(geoms):
        if gi is None or gi.is_empty:
            continue
        ai = gi.area
        for j in tree.query(gi):
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
        clusters.setdefault(uf.find(i), []).append(i)
    rows = []
    df = gdf.reset_index(drop=True)
    for root, members in clusters.items():
        if len(members) == 1:
            rows.append(df.iloc[members[0]].to_dict())
            continue
        merged = unary_union([geoms[m] for m in members])
        scores = [df.iloc[m].get(score_col, 0) or 0 for m in members]
        best = members[int(np.argmax(scores))]
        rec = df.iloc[best].to_dict()
        rec["geometry"] = merged
        rec["area_m2"] = float(merged.area) if hasattr(merged, "area") else None
        rec["n_merged"] = len(members)
        rows.append(rec)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def area_f1(pred_gdf: gpd.GeoDataFrame, gt_gdf: gpd.GeoDataFrame) -> dict:
    """Pixel-level (set-theoretic on polygons in metric CRS) area
    precision / recall / F1."""
    if len(pred_gdf) == 0 and len(gt_gdf) == 0:
        return {"pred_area": 0.0, "gt_area": 0.0, "inter_area": 0.0,
                "agg_R": None, "agg_P": None, "agg_F1": None}
    pred_union = unary_union(pred_gdf.geometry.values) if len(pred_gdf) else None
    gt_union = unary_union(gt_gdf.geometry.values) if len(gt_gdf) else None
    pred_area = pred_union.area if pred_union is not None and not pred_union.is_empty else 0.0
    gt_area = gt_union.area if gt_union is not None and not gt_union.is_empty else 0.0
    inter_area = (pred_union.intersection(gt_union).area
                  if (pred_union and gt_union and not pred_union.is_empty and not gt_union.is_empty) else 0.0)
    R = inter_area / gt_area if gt_area > 0 else None
    P = inter_area / pred_area if pred_area > 0 else None
    F1 = (2 * R * P / (R + P)) if (R and P and (R + P) > 0) else None
    return {"pred_area": pred_area, "gt_area": gt_area, "inter_area": inter_area,
            "agg_R": R, "agg_P": P, "agg_F1": F1}


def load_pred_for_grid(grid_dir: Path, conf_min: float, score_col: str,
                       apply_merge: bool, iou_thresh: float) -> gpd.GeoDataFrame:
    f = grid_dir / "predictions_metric.gpkg"
    if not f.exists():
        return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)
    gdf = gpd.read_file(f).to_crs(METRIC_CRS)
    use_col = score_col if score_col in gdf.columns else "score"
    if conf_min > 0 and use_col in gdf.columns:
        gdf = gdf[gdf[use_col] >= conf_min].copy()
    if apply_merge and len(gdf) > 0:
        gdf = merge_overlapping(gdf, iou_thresh=iou_thresh, score_col=use_col)
    return gdf


def load_gt_for_grid(gt_root: Path, grid_id: str) -> gpd.GeoDataFrame:
    f = gt_root / grid_id / f"{grid_id}_clean_gt.gpkg"
    if not f.exists():
        return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)
    return gpd.read_file(f).to_crs(METRIC_CRS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, default=REPO / "data/annotations_channel2_clean")
    p.add_argument("--iou", type=float, default=0.1)
    p.add_argument("--conf-min", type=float, default=0.925)
    p.add_argument("--score-col", default="confidence")
    p.add_argument("--no-merge", action="store_true",
                   help="evaluate baseline (no merge), only apply conf filter")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--label", default="(unlabeled)")
    args = p.parse_args()

    gt_grids = [d.name for d in sorted(args.gt_root.iterdir())
                if d.is_dir() and (d / f"{d.name}_clean_gt.gpkg").exists()]

    rows = []
    agg_pred_area = 0.0
    agg_gt_area = 0.0
    agg_inter_area = 0.0
    for gid in gt_grids:
        pred = load_pred_for_grid(args.results_root / gid,
                                  args.conf_min, args.score_col,
                                  not args.no_merge, args.iou)
        gt = load_gt_for_grid(args.gt_root, gid)
        m = area_f1(pred, gt)
        m["grid"] = gid
        m["n_pred"] = len(pred)
        m["n_gt"] = len(gt)
        m["bulk"] = (m["pred_area"] / m["gt_area"]) if m["gt_area"] > 0 else None
        rows.append(m)
        agg_pred_area += m["pred_area"]
        agg_gt_area += m["gt_area"]
        agg_inter_area += m["inter_area"]

    df = pd.DataFrame(rows)
    df = df[["grid","n_pred","n_gt","pred_area","gt_area","inter_area",
             "agg_R","agg_P","agg_F1","bulk"]]
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    aR = agg_inter_area / agg_gt_area if agg_gt_area > 0 else None
    aP = agg_inter_area / agg_pred_area if agg_pred_area > 0 else None
    aF1 = (2*aR*aP/(aR+aP)) if (aR and aP and (aR+aP)>0) else None
    bulk = agg_pred_area / agg_gt_area if agg_gt_area > 0 else None
    pg_F1 = df["agg_F1"].dropna().mean()
    sigma_Bw = df["bulk"].dropna().std()
    rmse = np.sqrt(((df["pred_area"] - df["gt_area"])**2).mean())

    print(f"\n=== {args.label} ===")
    print(f"  conf_min={args.conf_min}  merge={not args.no_merge}  iou={args.iou}")
    print(f"  n_grids: {len(df)}")
    print(f"  pred_area_total: {agg_pred_area:.0f} m²")
    print(f"  gt_area_total:   {agg_gt_area:.0f} m²")
    print(f"  inter_total:     {agg_inter_area:.0f} m²")
    print(f"  agg_R:           {aR:.4f}")
    print(f"  agg_P:           {aP:.4f}")
    print(f"  **agg_area_F1:   {aF1:.4f}**")
    print(f"  per-grid F1 avg: {pg_F1:.4f}")
    print(f"  bulk:            {bulk:.4f}")
    print(f"  sigma_Bw:        {sigma_Bw:.4f}")
    print(f"  RMSE:            {rmse:.0f}")


if __name__ == "__main__":
    main()
