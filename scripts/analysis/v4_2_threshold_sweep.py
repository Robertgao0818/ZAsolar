"""V4.2 threshold sweep on val 10 grids.

Same IoU-based installation-profile matching as v4_vs_v4_2_val10_compare.py,
but sweeps post_conf from 0.60 to 0.99 to find V4.2's optimal operating point.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

PROJECT = Path(__file__).resolve().parents[2]
IOU_THRESH = 0.5
METRIC_CRS = "EPSG:32735"

VAL_CBD = ["G0776", "G0814", "G0854", "G0856", "G0891"]
VAL_SANDTON = ["G1110", "G1111", "G1179", "G1250", "G1251"]

V4_2_RESULTS = "results/johannesburg/v4_2_jhb_ft_aerial_2023"
V4_RESULTS = "results/johannesburg/v4_aerial_2023"


def load_gt(grid: str) -> gpd.GeoDataFrame:
    f = PROJECT / f"data/annotations/Joburg/{grid}_V4_{'260407' if grid < 'G1000' else '260421'}.gpkg"
    gdf = gpd.read_file(f).to_crs(METRIC_CRS)
    return gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)


def load_raw_preds(results_dir: str, grid: str) -> gpd.GeoDataFrame:
    f = PROJECT / results_dir / grid / "predictions_metric.gpkg"
    gdf = gpd.read_file(f)
    if gdf.crs is None or str(gdf.crs) != METRIC_CRS:
        gdf = gdf.to_crs(METRIC_CRS)
    return gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)


def installation_match(preds: gpd.GeoDataFrame, gt: gpd.GeoDataFrame):
    if len(preds) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(preds), 0
    preds = preds.reset_index(drop=True)
    pred_used = [False] * len(preds)
    tp_preds = 0
    gt_matched = 0
    pred_sindex = preds.sindex
    for _, gt_row in gt.iterrows():
        gt_geom = gt_row.geometry
        matched_idx = []
        matched_geoms = []
        for ci in pred_sindex.intersection(gt_geom.bounds):
            if pred_used[ci]:
                continue
            pg = preds.geometry.iloc[ci]
            if pg.intersects(gt_geom):
                matched_idx.append(ci)
                matched_geoms.append(pg)
        if not matched_geoms:
            continue
        union = unary_union(matched_geoms)
        inter = union.intersection(gt_geom).area
        uarea = union.union(gt_geom).area
        iou = inter / uarea if uarea > 0 else 0
        if iou >= IOU_THRESH:
            gt_matched += 1
            tp_preds += len(matched_idx)
            for ci in matched_idx:
                pred_used[ci] = True
    fp = sum(1 for u in pred_used if not u)
    return tp_preds, fp, len(gt) - gt_matched


def sweep_model(results_dir: str, label: str, thresholds: np.ndarray):
    rows = []
    gt_by_grid = {g: load_gt(g) for g in VAL_CBD + VAL_SANDTON}
    raw_by_grid = {g: load_raw_preds(results_dir, g) for g in VAL_CBD + VAL_SANDTON}
    for t in thresholds:
        for grp_name, grp in [("CBD", VAL_CBD), ("Sandton", VAL_SANDTON), ("ALL", VAL_CBD + VAL_SANDTON)]:
            tp_tot = fp_tot = fn_tot = 0
            for g in grp:
                preds_t = raw_by_grid[g][raw_by_grid[g]["confidence"] >= t]
                tp, fp, fn = installation_match(preds_t, gt_by_grid[g])
                tp_tot += tp; fp_tot += fp; fn_tot += fn
            p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0
            r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            rows.append({
                "model": label, "threshold": round(float(t), 2), "group": grp_name,
                "tp": tp_tot, "fp": fp_tot, "fn": fn_tot,
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            })
    return pd.DataFrame(rows)


def main():
    thresholds = np.arange(0.50, 0.99, 0.02)
    v4_df = sweep_model(V4_RESULTS, "V4", thresholds)
    v4_2_df = sweep_model(V4_2_RESULTS, "V4.2", thresholds)
    df = pd.concat([v4_df, v4_2_df], ignore_index=True)
    out = PROJECT / "results/analysis/v4_vs_v4_2_val10_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[write] {out}")

    print()
    for grp in ["CBD", "Sandton", "ALL"]:
        print(f"=== {grp} ===")
        for model in ["V4", "V4.2"]:
            sub = df[(df["group"] == grp) & (df["model"] == model)]
            best = sub.loc[sub["f1"].idxmax()]
            print(f"  [{model:5s}] best F1={best['f1']:.4f} at thresh={best['threshold']} "
                  f"(P={best['precision']:.3f}, R={best['recall']:.3f}, "
                  f"TP={int(best['tp'])}, FP={int(best['fp'])}, FN={int(best['fn'])})")
        print()

    print()
    print("=== Aligned comparison at key thresholds (ALL) ===")
    print(f"{'Thresh':>6s}  {'Model':6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}")
    for t in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        for model in ["V4", "V4.2"]:
            sub = df[(df["group"] == "ALL") & (df["model"] == model) & (df["threshold"] == round(t, 2))]
            if len(sub):
                r = sub.iloc[0]
                print(f"{t:>6.2f}  {model:6s}  {int(r['tp']):4d}  {int(r['fp']):4d}  {int(r['fn']):4d}  "
                      f"{r['precision']:.3f}  {r['recall']:.3f}  {r['f1']:.3f}")
        print()


if __name__ == "__main__":
    main()
