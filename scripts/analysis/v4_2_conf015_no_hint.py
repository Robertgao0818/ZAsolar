"""V4.2 @ conf=0.15 on 25 Sandton grids: how many predictions have no GT hint?

For each grid: count predictions (at post_conf>=0.15) that do NOT intersect any
polygon in the current annotation source. These are candidate new finds —
either genuine missed panels (future GT additions) or low-conf FPs.

Outputs:
  - results/analysis/v4_2_conf015_sandton_no_hint.csv  (per-grid counts)
  - results/analysis/v4_2_conf015_sandton_no_hint/<grid>_no_hint.gpkg  (new preds only)
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
METRIC_CRS = "EPSG:32735"
RESULTS_DIR = PROJECT / "results/johannesburg/v4_2_jhb_ft_aerial_2023_conf015"
OUT_DIR = PROJECT / "results/analysis/v4_2_conf015_sandton_no_hint"
OUT_CSV = PROJECT / "results/analysis/v4_2_conf015_sandton_no_hint.csv"

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
    if len(preds) == 0:
        return {"grid": grid, "n_gt": len(gt), "n_pred": 0,
                "n_with_hint": 0, "n_no_hint": 0, "pct_no_hint": 0.0}

    sidx = gt.sindex
    has_hint = []
    for i, pg in enumerate(preds.geometry):
        cand = list(sidx.intersection(pg.bounds))
        hit = False
        for ci in cand:
            if pg.intersects(gt.geometry.iloc[ci]):
                hit = True
                break
        has_hint.append(hit)

    preds = preds.copy()
    preds["has_gt_hint"] = has_hint
    no_hint = preds[~preds["has_gt_hint"]].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_gpkg = OUT_DIR / f"{grid}_no_hint.gpkg"
    if len(no_hint):
        no_hint.to_file(out_gpkg, driver="GPKG")

    n_pred = len(preds)
    n_with = int(sum(has_hint))
    n_no = n_pred - n_with
    return {
        "grid": grid, "n_gt": len(gt), "n_pred": n_pred,
        "n_with_hint": n_with, "n_no_hint": n_no,
        "pct_no_hint": round(100.0 * n_no / n_pred, 2) if n_pred else 0.0,
        "mean_conf_no_hint": round(float(no_hint["confidence"].mean()), 4) if len(no_hint) else 0.0,
    }


def main() -> None:
    rows = [analyze(g) for g in SANDTON]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("=== V4.2 @ conf=0.15, 25 Sandton grids ===")
    print(df.to_string(index=False))
    print()
    tot_pred = df["n_pred"].sum()
    tot_with = df["n_with_hint"].sum()
    tot_no = df["n_no_hint"].sum()
    print(f"TOTAL: pred={tot_pred}, with_hint={tot_with}, no_hint={tot_no} "
          f"({100*tot_no/tot_pred:.1f}%)")
    print(f"[write] {OUT_CSV}")
    print(f"[write] {OUT_DIR}/<grid>_no_hint.gpkg")


if __name__ == "__main__":
    main()
