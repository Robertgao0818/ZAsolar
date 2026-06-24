#!/usr/bin/env python
"""Score the *delivered* Cape Town census inventory against ground truth.

This is the Cape-Town analogue of the JHB post-census per-grid validation
table (`jhb_382grid_unified_A_perdet_sam/combined/spatial_eval_per_grid.csv`).
Where the calibration appendix (`ct_census_calibration_appendix.html`) reported
Tier-1 metrics from *dedicated* evaluation inference runs (the wave-1 face and
the Li held-out face), this script takes the FINAL delivered census deliverable
(`ct_full_inventory_*_merged.gpkg`, 111,801 installations) and re-evaluates it
on exactly the census cells that carry ground truth — answering "does the
artifact we shipped reproduce the locked-config numbers on its own GT cells?".

GT-bearing cells = Gao SAM2/V4 annotations under `data/annotations/Capetown/`
whose digit-preserving CPT cell (via `data/ct_grid_crosswalk_g_to_cpt.csv`)
falls inside the 2,083-cell aerial_2025 census grid. (The Li held-out face is
the eastern Cape Flats and is NOT inside this census footprint, which is why
the §6 Li numbers cannot be reproduced here — that face is reported separately
in the appendix.)

Metric semantics are reused verbatim from the production kernels so the numbers
are directly comparable to the appendix:
  * validity / area-cap filtering  -> core.polygon_validation.clean_metric_gdf
  * per-grid union-area R/P/F1      -> mirrors area_aggregate_eval.evaluate_run
  * Tier-1 aggregate (σ_Bw / RMSE / bulk / agg-F1 / thru0 β / R²)
                                    -> core.area_metrics.summarize
  * cov50 (count recall)           -> mirrors li_count_recall_sweep (GT polygon
                                       with >=50% area covered by pred union)

The delivered inventory is the IoU>=0.10 union-merge. Union *area* is identical
to the per-detection set (unary_union is idempotent under pre-merging), so the
area metrics below are merge-mode invariant; only polygon counts differ.

Outputs (under --out-dir):
  ct_census_gt_per_grid.csv     one row per GT-bearing census cell
  ct_census_gt_summary.csv      single Tier-1 aggregate row (+ cov50)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from core.region_registry import get_region_config
from core.polygon_validation import (
    clean_metric_gdf,
    _load_first_layer,
)
from core.area_metrics import summarize
from scripts.analysis.area_aggregate_eval import _discover_gt

REPO = Path(__file__).resolve().parent.parent.parent
RUN_LABEL = "ct_census_merged_vs_gt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inventory",
        default="results/analysis/ct_census_output_table/"
                "ct_full_inventory_2026-06-21_merged.gpkg",
        help="delivered census inventory gpkg (merged deliverable, EPSG:32734)")
    ap.add_argument("--gt-dir", default="data/annotations/Capetown")
    ap.add_argument("--crosswalk", default="data/ct_grid_crosswalk_g_to_cpt.csv")
    ap.add_argument("--task-grid", default="data/task_grid_cpt.gpkg")
    ap.add_argument("--grid-col", default="source_grid",
                    help="census cell id column in the inventory")
    ap.add_argument("--out-dir", default="results/analysis/ct_census_gt_eval")
    args = ap.parse_args()

    region = get_region_config("cape_town")
    crs = region.crs_metric  # EPSG:32734, looked up (never hardcoded)

    inv_path = (REPO / args.inventory) if not Path(args.inventory).is_absolute() \
        else Path(args.inventory)
    print(f"[read] inventory {inv_path}")
    inv = gpd.read_file(inv_path)
    if inv.crs is None or str(inv.crs) != crs:
        inv = inv.to_crs(crs)
    print(f"[read] {len(inv):,} polygons | crs={inv.crs}")

    # G -> CPT crosswalk + census cell membership.
    cw = pd.read_csv(REPO / args.crosswalk)
    g2c = dict(zip(cw["g_id"], cw["cpt_id"]))
    tg = gpd.read_file(REPO / args.task_grid, columns=["gridcell_id"])
    census_cells = set(tg["gridcell_id"].astype(str))

    # Enumerate Gao GT grids, pick best gpkg per grid (SAM2 > V4 > reviewed).
    gt_dir = REPO / args.gt_dir
    gids = sorted({m.group(1) for f in gt_dir.glob("G*.gpkg")
                   if (m := re.match(r"(G\d+)", f.name))})
    print(f"[gt] {len(gids)} Gao GT grids on disk")

    rows: list[dict] = []
    cov_total = cov_hit = 0
    skipped: list[str] = []
    for gid in gids:
        cpt = g2c.get(gid)
        if not isinstance(cpt, str) or cpt not in census_cells:
            skipped.append(f"{gid}(no-census-cell)")
            continue
        gt_path = _discover_gt(gt_dir, gid)
        if gt_path is None:
            skipped.append(f"{gid}(no-gt-file)")
            continue

        # --- prediction side: the delivered census polygons in this cell ---
        slice_ = inv[inv[args.grid_col].astype(str) == cpt]
        pred_clean, pred_drop = clean_metric_gdf(
            slice_, metric_crs=crs, drop_zero_area=True)
        pred_geoms = list(pred_clean.geometry)
        pred_u = unary_union(pred_geoms) if pred_geoms else None
        pred_union_m2 = float(pred_u.area) if pred_u is not None else 0.0
        n_pred = len(pred_geoms)

        # --- GT side: one cleaned frame for union area AND cov50 ---
        # CT SAM2 annotations are EPSG:4326; 2 grids (G1685/G1688) ship without
        # a written CRS tag — default naive geometry to the dominant 4326.
        gt_raw = _load_first_layer(gt_path, None)
        n_raw = len(gt_raw)
        if gt_raw.crs is None:
            gt_raw = gt_raw.set_crs("EPSG:4326")
        gt_clean, gt_drop = clean_metric_gdf(
            gt_raw, metric_crs=crs, drop_zero_area=True)
        gt_geoms = list(gt_clean.geometry)
        # Corrupt-GT guard: a mixed-CRS / damaged file (e.g. G1688 carries both
        # lon/lat and UTM coords) loses most polygons to the validity + area cap
        # filter, leaving a phantom tiny-area GT that blows up the ratio. Skip
        # any file where <50% of a non-trivial polygon set survives cleaning.
        if n_raw >= 10 and len(gt_geoms) < 0.5 * n_raw:
            skipped.append(f"{gid}(corrupt-gt {len(gt_geoms)}/{n_raw} kept)")
            continue
        gt_u = unary_union(gt_geoms) if gt_geoms else None
        if gt_u is None or gt_u.area <= 0:
            skipped.append(f"{gid}(empty-gt)")
            continue
        n_gt = len(gt_geoms)
        gt_union_m2 = float(gt_u.area)
        inter_m2 = float(pred_u.intersection(gt_u).area) if pred_u is not None else 0.0
        abs_err = pred_union_m2 - gt_union_m2
        area_R = inter_m2 / gt_union_m2
        area_P = inter_m2 / pred_union_m2 if pred_union_m2 > 0 else 0.0
        area_F1 = (2 * area_R * area_P / (area_R + area_P)
                   if (area_R + area_P) > 0 else 0.0)

        # --- cov50: per-GT-polygon >=50% covered by the prediction union ---
        for gpoly in gt_geoms:
            ga = gpoly.area
            if ga <= 0:
                continue
            cov_total += 1
            if pred_u is not None and gpoly.intersection(pred_u).area / ga >= 0.5:
                cov_hit += 1

        rows.append({
            "region": "cape_town",
            "model_run": RUN_LABEL,
            "model_version": "unifiedA_census_perdet+cls",
            "imagery_layer": "aerial_2025",
            "grid_id": cpt,
            "legacy_gao_id": gid,
            "gt_source": gt_path.name,
            "n_pred": n_pred,
            "n_gt": n_gt,
            "pred_total_m2": round(pred_union_m2, 2),
            "gt_total_m2": round(gt_union_m2, 2),
            "inter_m2": round(inter_m2, 2),
            "area_R": round(area_R, 4),
            "area_P": round(area_P, 4),
            "area_F1": round(area_F1, 4),
            "abs_error_m2": round(abs_err, 2),
            "signed_rel_error": round(abs_err / gt_union_m2, 4),
            "abs_rel_error": round(abs(abs_err) / gt_union_m2, 4),
            "pred_gt_ratio": round(pred_union_m2 / gt_union_m2, 4),
            "n_dropped_pred": pred_drop,
            "n_dropped_gt": gt_drop,
        })

    print(f"[eval] {len(rows)} GT-bearing census cells scored | "
          f"{len(skipped)} skipped")
    if skipped:
        print("       skipped:", ", ".join(skipped))

    summary = summarize(rows)
    cov50 = cov_hit / cov_total if cov_total else float("nan")
    if summary:
        summary[0]["cov50_count_recall"] = round(cov50, 4)
        summary[0]["n_gt_polys"] = cov_total

    # Robustness row: σ_Bw is sensitive to tiny-GT cells (a few m² of over-paint
    # swings the ratio), so also report the suite on cells with GT >= 500 m².
    # This is diagnostic context, NOT the headline; the headline is all valid
    # GT-bearing census cells.
    big_summary = summarize([{**r, "model_run": RUN_LABEL + "_gt_ge500"}
                             for r in rows if r["gt_total_m2"] >= 500])
    summary.extend(big_summary)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ct_census_gt_per_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "ct_census_gt_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    def show(s, title):
        print(f"\n=== {title} (n={s['n_grids']} cells) ===")
        print(f"  agg-F1            {s['agg_area_F1']}")
        print(f"  bulk ratio        {s['bulk_pred_gt_ratio']}")
        print(f"  σ_Bw (primary)    {s['std_ratio_Bw']}")
        print(f"  log-σ (robust)    {s['std_logratio']}")
        print(f"  RMSE (m²)         {s['rmse_m2']}")
        print(f"  thru0 β           {s['thru0_slope']}")
        print(f"  R² (thru0 / OLS)  {s['thru0_r2']} / {s['ols_r2']}")
        print(f"  within ±20%       {s['frac_grids_within_pm20pct']}")

    show(summary[0], "CT delivered census vs GT — all valid GT cells")
    print(f"  cov50             {round(cov50, 4)}  (on {cov_total:,} GT polys)")
    if len(summary) > 1:
        show(summary[1], "CT delivered census vs GT — GT >= 500 m² (robust)")
    print(f"\n[write] {out_dir}/ct_census_gt_per_grid.csv  ({len(rows)} rows)")
    print(f"[write] {out_dir}/ct_census_gt_summary.csv")


if __name__ == "__main__":
    main()
