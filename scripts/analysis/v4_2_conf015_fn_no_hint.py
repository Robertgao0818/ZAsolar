"""V4.2 @ conf=0.15 on 25 Sandton grids: how many GT have no prediction hint?

Complement of v4_2_conf015_no_hint.py:
  - "no hint GT" = GT polygon with NO intersecting prediction at all → pure miss (RPN never proposed)
  - "hint-only GT" = GT with intersecting pred(s) but installation-profile IoU < 0.5 → localization/fragmentation issue
  - "matched GT" = IoU >= 0.5 (TP)

Outputs:
  - results/analysis/v4_2_conf015_sandton_fn_no_hint.csv
  - results/analysis/v4_2_conf015_sandton_fn_no_hint/<grid>_{no_hint_gt,hint_only_gt}.gpkg
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

PROJECT = Path(__file__).resolve().parents[2]
METRIC_CRS = "EPSG:32735"
IOU_THRESH = 0.5
RESULTS_DIR = PROJECT / "results/johannesburg/v4_2_jhb_ft_aerial_2023_conf015"
OUT_DIR = PROJECT / "results/analysis/v4_2_conf015_sandton_fn_no_hint"
OUT_CSV = PROJECT / "results/analysis/v4_2_conf015_sandton_fn_no_hint.csv"

SANDTON = [
    "G1110", "G1111", "G1112", "G1113", "G1114",
    "G1144", "G1145", "G1146", "G1147", "G1148",
    "G1179", "G1180", "G1181", "G1182", "G1183",
    "G1214", "G1215", "G1216", "G1217", "G1218",
    "G1250", "G1251", "G1252", "G1253", "G1254",
]


def load_gt(grid: str) -> gpd.GeoDataFrame:
    f = PROJECT / f"data/annotations/Joburg/{grid}_V4_260421.gpkg"
    gdf = gpd.read_file(f).to_crs(METRIC_CRS)
    return gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)


def load_preds(grid: str) -> gpd.GeoDataFrame:
    f = RESULTS_DIR / grid / "predictions_metric.gpkg"
    if not f.exists():
        return gpd.GeoDataFrame({"geometry": [], "confidence": []}, crs=METRIC_CRS)
    gdf = gpd.read_file(f)
    if gdf.crs is None or str(gdf.crs) != METRIC_CRS:
        gdf = gdf.to_crs(METRIC_CRS)
    return gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)


def analyze(grid: str) -> dict:
    gt = load_gt(grid)
    preds = load_preds(grid)

    if len(gt) == 0:
        return {"grid": grid, "n_gt": 0, "n_pred": len(preds),
                "n_matched": 0, "n_fn_total": 0, "n_fn_no_hint": 0, "n_fn_hint_only": 0,
                "pct_fn_no_hint": 0.0}

    if len(preds) == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        gt.to_file(OUT_DIR / f"{grid}_no_hint_gt.gpkg", driver="GPKG")
        return {"grid": grid, "n_gt": len(gt), "n_pred": 0,
                "n_matched": 0, "n_fn_total": len(gt),
                "n_fn_no_hint": len(gt), "n_fn_hint_only": 0,
                "pct_fn_no_hint": 100.0}

    pred_used = [False] * len(preds)
    pred_sindex = preds.sindex
    gt_status = []  # "matched" | "hint_only" | "no_hint"

    for _, gt_row in gt.iterrows():
        gt_geom = gt_row.geometry
        candidates = list(pred_sindex.intersection(gt_geom.bounds))
        matched_idx = []
        matched_geoms = []
        any_intersect = False
        for ci in candidates:
            pg = preds.geometry.iloc[ci]
            if pg.intersects(gt_geom):
                any_intersect = True
                if not pred_used[ci]:
                    matched_idx.append(ci)
                    matched_geoms.append(pg)
        if not any_intersect:
            gt_status.append("no_hint")
            continue
        if not matched_geoms:
            gt_status.append("hint_only")
            continue
        union = unary_union(matched_geoms)
        inter_area = union.intersection(gt_geom).area
        union_area = union.union(gt_geom).area
        iou = inter_area / union_area if union_area > 0 else 0
        if iou >= IOU_THRESH:
            gt_status.append("matched")
            for ci in matched_idx:
                pred_used[ci] = True
        else:
            gt_status.append("hint_only")

    gt = gt.copy()
    gt["match_status"] = gt_status

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    no_hint_gt = gt[gt["match_status"] == "no_hint"]
    hint_only_gt = gt[gt["match_status"] == "hint_only"]
    if len(no_hint_gt):
        no_hint_gt.to_file(OUT_DIR / f"{grid}_no_hint_gt.gpkg", driver="GPKG")
    if len(hint_only_gt):
        hint_only_gt.to_file(OUT_DIR / f"{grid}_hint_only_gt.gpkg", driver="GPKG")

    n_matched = sum(s == "matched" for s in gt_status)
    n_no_hint = sum(s == "no_hint" for s in gt_status)
    n_hint_only = sum(s == "hint_only" for s in gt_status)
    n_fn = n_no_hint + n_hint_only
    return {
        "grid": grid, "n_gt": len(gt), "n_pred": len(preds),
        "n_matched": n_matched, "n_fn_total": n_fn,
        "n_fn_no_hint": n_no_hint, "n_fn_hint_only": n_hint_only,
        "pct_fn_no_hint": round(100.0 * n_no_hint / n_fn, 2) if n_fn else 0.0,
    }


def main() -> None:
    rows = [analyze(g) for g in SANDTON]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("=== V4.2 @ conf=0.15, 25 Sandton grids — FN breakdown ===")
    print(df.to_string(index=False))
    print()
    tot_gt = df["n_gt"].sum()
    tot_matched = df["n_matched"].sum()
    tot_fn = df["n_fn_total"].sum()
    tot_fn_no_hint = df["n_fn_no_hint"].sum()
    tot_fn_hint_only = df["n_fn_hint_only"].sum()
    print(f"TOTAL GT:           {tot_gt}")
    print(f"  matched (TP):     {tot_matched} ({100*tot_matched/tot_gt:.1f}%)")
    print(f"  FN total:         {tot_fn} ({100*tot_fn/tot_gt:.1f}%)")
    print(f"    FN no_hint:     {tot_fn_no_hint} ({100*tot_fn_no_hint/tot_gt:.1f}% of GT, "
          f"{100*tot_fn_no_hint/tot_fn:.1f}% of FN)")
    print(f"    FN hint_only:   {tot_fn_hint_only} ({100*tot_fn_hint_only/tot_gt:.1f}% of GT, "
          f"{100*tot_fn_hint_only/tot_fn:.1f}% of FN)")
    print(f"[write] {OUT_CSV}")
    print(f"[write] {OUT_DIR}/<grid>_{{no_hint_gt,hint_only_gt}}.gpkg")


if __name__ == "__main__":
    main()
