"""Apples-to-apples V4 vs V4.2 comparison on val 10 grids.

Uses IoU-based presence matching (greedy max-IoU, threshold 0.5) against the
installation-level GT (reviewed annotation source files). Matches the
detect_and_evaluate 'installation' profile semantics: multiple predictions
hitting the same GT are merged before IoU (pred-side many-to-one).

Output: one row per grid per model with TP/FP/FN/P/R/F1 + group aggregates.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

PROJECT = Path(__file__).resolve().parents[2]
POST_CONF = 0.85
IOU_THRESH = 0.5
METRIC_CRS = "EPSG:32735"

VAL_CBD = ["G0776", "G0814", "G0854", "G0856", "G0891"]
VAL_SANDTON = ["G1110", "G1111", "G1179", "G1250", "G1251"]
VAL_ALL = VAL_CBD + VAL_SANDTON

MODELS = {
    "V4":   "results/johannesburg/v4_aerial_2023",
    "V4.2": "results/johannesburg/v4_2_jhb_ft_aerial_2023",
}


def load_gt(grid: str) -> gpd.GeoDataFrame:
    if grid < "G1000":
        f = PROJECT / f"data/annotations/Joburg/{grid}_V4_260407.gpkg"
    else:
        f = PROJECT / f"data/annotations/Joburg/{grid}_V4_260421.gpkg"
    gdf = gpd.read_file(f).to_crs(METRIC_CRS)
    return gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)


def load_preds(model: str, grid: str) -> gpd.GeoDataFrame:
    f = PROJECT / MODELS[model] / grid / "predictions_metric.gpkg"
    gdf = gpd.read_file(f)
    if gdf.crs is None or str(gdf.crs) != METRIC_CRS:
        gdf = gdf.to_crs(METRIC_CRS)
    gdf = gdf[gdf["confidence"] >= POST_CONF].reset_index(drop=True)
    return gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)


def installation_match(preds: gpd.GeoDataFrame, gt: gpd.GeoDataFrame) -> tuple[int, int, int]:
    """Installation-profile matching.

    For each GT installation:
      - Collect all pred polygons that spatially intersect it (any overlap).
      - Union those preds → compute IoU(union, GT).
      - If IoU >= threshold → GT matched (count preds as TP).
      - Preds that intersect a GT are 'assigned' and cannot be reused.
    After matching, unassigned preds = FP; unmatched GTs = FN.
    """
    if len(preds) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(preds), 0

    pred_used = [False] * len(preds)
    tp_preds = 0
    gt_matched = 0

    # Build spatial index
    pred_sindex = preds.sindex

    for gi, gt_row in gt.iterrows():
        gt_geom = gt_row.geometry
        candidates = list(pred_sindex.intersection(gt_geom.bounds))
        matched_idx = []
        matched_geoms = []
        for ci in candidates:
            if pred_used[ci]:
                continue
            pred_geom = preds.geometry.iloc[ci]
            if pred_geom.intersects(gt_geom):
                matched_idx.append(ci)
                matched_geoms.append(pred_geom)
        if not matched_geoms:
            continue
        union = unary_union(matched_geoms)
        inter_area = union.intersection(gt_geom).area
        union_area = union.union(gt_geom).area
        iou = inter_area / union_area if union_area > 0 else 0
        if iou >= IOU_THRESH:
            gt_matched += 1
            tp_preds += len(matched_idx)
            for ci in matched_idx:
                pred_used[ci] = True

    fp = sum(1 for u in pred_used if not u)
    fn = len(gt) - gt_matched
    return tp_preds, fp, fn


def main() -> None:
    rows = []
    for grid in VAL_ALL:
        gt = load_gt(grid)
        for model_name in MODELS:
            preds = load_preds(model_name, grid)
            tp, fp, fn = installation_match(preds, gt)
            rows.append({
                "grid": grid,
                "group": "CBD" if grid < "G1000" else "Sandton",
                "model": model_name,
                "n_gt": len(gt),
                "n_pred": len(preds),
                "tp": tp, "fp": fp, "fn": fn,
            })
    df = pd.DataFrame(rows)
    df["precision"] = df["tp"] / (df["tp"] + df["fp"]).clip(lower=1)
    df["recall"]    = df["tp"] / (df["tp"] + df["fn"]).clip(lower=1)
    df["f1"]        = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"]).clip(lower=1e-9)
    df.loc[(df["tp"] + df["fp"]) == 0, "precision"] = 0
    df.loc[(df["tp"] + df["fn"]) == 0, "recall"] = 0

    print("=== Per-grid ===")
    print(df.to_string(index=False))

    print()
    print("=== Group aggregate (installation-profile, IoU>=0.5, post_conf=0.85) ===")
    print(f"{'Group':10s}  {'Model':6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}")
    print("-" * 70)
    for grp_name, grp_grids in [("CBD", VAL_CBD), ("Sandton", VAL_SANDTON), ("ALL", VAL_ALL)]:
        for model_name in MODELS:
            sub = df[(df["grid"].isin(grp_grids)) & (df["model"] == model_name)]
            tp = int(sub["tp"].sum()); fp = int(sub["fp"].sum()); fn = int(sub["fn"].sum())
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print(f"{grp_name:10s}  {model_name:6s}  {tp:4d}  {fp:4d}  {fn:4d}  "
                  f"{p:.3f}  {r:.3f}  {f1:.3f}")
        print()

    out = PROJECT / "results/analysis/v4_vs_v4_2_val10.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
