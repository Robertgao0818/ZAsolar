#!/usr/bin/env python3
"""Cross-domain evaluation of unified_reviewall_A on 6 new Vexcel cities.

Scores the production detector (unified_reviewall_A, per-detection + SAM2.1
mask+box, polygon-conf c=0.925) on 60 Li-RA-annotated Vexcel grids across
pretoria / bloemfontein / durban / east_london / gqeberha / pietermaritzburg
to quantify generalization from the JHB/CT training domain.

Reuses ``area_aggregate_eval.summarize`` (the canonical Tier-1 metric suite)
and ``_read_polys_geom`` verbatim, so the area-aggregate numbers are identical
to the in-domain JHB baseline. GT is Li sub-array / panel-level (A2 / T2) — the
primary signal is area-aggregate + presence; polygon F1 is diagnostic only and
is expected to be depressed by the carved (non-installation-merged) GT.

Outputs (under --output-dir):
  per_grid.csv          one row per non-empty grid (area + polygon metrics)
  per_city_tier1.csv    Tier-1 summary per city
  overall_tier1.csv     Tier-1 summary pooled across all cities
  empty_grid_fp.csv     false-positive probe on the 10 zero-GT grids
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
from shapely import STRtree
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.area_aggregate_eval import (  # noqa: E402
    _read_polys_geom,
    summarize,
)
from core.region_registry import get_region_config  # noqa: E402
from core.negative_pool_leakage import mined_grids_for_region  # noqa: E402

CITIES = [
    "pretoria", "bloemfontein", "durban",
    "east_london", "gqeberha", "pietermaritzburg",
]
RUN = "unified_reviewall_A_perdet_sam_maskbox_xdomain_c0925"
MODEL_VERSION = "exp_unified_reviewall_A+sam2_maskbox"
LAYER = {
    "pretoria": "vexcel_ortho_2026_02",
    "bloemfontein": "vexcel_ortho_2026_01",
    "durban": "vexcel_ortho_2025_12_2026_01",
    "east_london": "vexcel_ortho_2026_01",
    "gqeberha": "vexcel_ortho_2026_01",
    "pietermaritzburg": "vexcel_ortho_2025_12",
}
MAX_PLAUSIBLE_POLY_M2 = 20_000.0


def _clean_polys(path: Path, metric_crs: str, layer=None) -> list:
    """Return a list of individual valid polygon geometries in metric CRS."""
    g = gpd.read_file(path, layer=layer)
    if g.empty:
        return []
    if g.crs is None:
        return []
    g = g.to_crs(metric_crs)
    out = []
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or not geom.is_valid:
            continue
        if geom.area <= 0 or geom.area > MAX_PLAUSIBLE_POLY_M2:
            continue
        out.append(geom)
    return out


def _polygon_prf(preds: list, gts: list, iou_thr: float = 0.5) -> dict:
    """Greedy 1-1 polygon matching at IoU>=thr. Plain (NOT installation-merge)
    matching — honest diagnostic against carved sub-array GT."""
    if not gts and not preds:
        return {"tp": 0, "fp": 0, "fn": 0, "P": None, "R": None, "F1": None}
    if not preds:
        return {"tp": 0, "fp": 0, "fn": len(gts), "P": 0.0, "R": 0.0, "F1": 0.0}
    if not gts:
        return {"tp": 0, "fp": len(preds), "fn": 0, "P": 0.0, "R": 0.0, "F1": 0.0}
    tree = STRtree(gts)
    matched_gt = set()
    tp = 0
    # match highest-overlap first: collect candidate (iou, pi, gi) then greedy
    cands = []
    for pi, p in enumerate(preds):
        for gi in tree.query(p):
            gi = int(gi)
            inter = p.intersection(gts[gi]).area
            if inter <= 0:
                continue
            union = p.area + gts[gi].area - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_thr:
                cands.append((iou, pi, gi))
    cands.sort(reverse=True)
    matched_pred = set()
    for iou, pi, gi in cands:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": round(P, 4),
            "R": round(R, 4), "F1": round(F1, 4)}


def eval_city(city: str, pred_root: Path, gt_root: Path,
              exclude_mined_hn: bool = True):
    region_cfg = get_region_config(city)
    metric_crs = region_cfg.crs_metric
    rows, empties = [], []
    if not pred_root.exists():
        return rows, empties
    grids = sorted(p.name for p in pred_root.iterdir() if p.is_dir())
    # eval-leakage guard: drop grids mined into the HN pool with at least one
    # training_eligible row (a retrain that saw those HN chips makes any
    # cross-domain claim on them contaminated). Provenance-only rows
    # (training_eligible=false, e.g. the xdomain empty-probe FPs) are NOT
    # excluded — no model trains on them, so those grids stay clean eval
    # surfaces (consistent with the training_eligible gate; see
    # core.negative_pool_leakage.mined_grid_keys).
    if exclude_mined_hn:
        mined = mined_grids_for_region(city)
        excluded = [g for g in grids if g in mined]
        if excluded:
            print(f"[{city}] excluding {len(excluded)} HN-mined grid(s) from "
                  f"eval surface: {excluded}")
        grids = [g for g in grids if g not in mined]
    for g in grids:
        pred_path = pred_root / g / "predictions_metric.gpkg"
        gt_path = gt_root / f"{g}.gpkg"
        if not pred_path.exists() or not gt_path.exists():
            continue
        n_pred, _ps, _pm, _pd, pred_u = _read_polys_geom(pred_path, metric_crs, layer=None)
        n_gt, _gs, _gm, _gd, gt_u = _read_polys_geom(gt_path, metric_crs, layer=None)
        pred_area = float(pred_u.area) if pred_u is not None else 0.0
        if gt_u is None or gt_u.area <= 0:
            empties.append({"region": city, "grid_id": g, "n_pred": n_pred,
                            "pred_area_m2": round(pred_area, 2)})
            continue
        gt_area = float(gt_u.area)
        inter = float(pred_u.intersection(gt_u).area) if pred_u is not None else 0.0
        area_R = inter / gt_area
        area_P = inter / pred_area if pred_area > 0 else 0.0
        area_F1 = 2 * area_R * area_P / (area_R + area_P) if (area_R + area_P) else 0.0
        abs_err = pred_area - gt_area
        # polygon-level diagnostic (plain greedy IoU>=0.5)
        prf = _polygon_prf(_clean_polys(pred_path, metric_crs),
                           _clean_polys(gt_path, metric_crs), 0.5)
        rows.append({
            "region": city, "model_run": RUN,
            "model_version": MODEL_VERSION, "imagery_layer": LAYER.get(city, ""),
            "grid_id": g,
            "n_pred": n_pred, "n_gt": n_gt,
            "pred_total_m2": round(pred_area, 2), "gt_total_m2": round(gt_area, 2),
            "inter_m2": round(inter, 2),
            "area_R": round(area_R, 4), "area_P": round(area_P, 4),
            "area_F1": round(area_F1, 4),
            "abs_error_m2": round(abs_err, 2),
            "signed_rel_error": round(abs_err / gt_area, 4),
            "abs_rel_error": round(abs(abs_err) / gt_area, 4),
            "pred_gt_ratio": round(pred_area / gt_area, 4),
            "poly_tp": prf["tp"], "poly_fp": prf["fp"], "poly_fn": prf["fn"],
            "poly_P": prf["P"], "poly_R": prf["R"], "poly_F1_iou50": prf["F1"],
        })
    return rows, empties


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=REPO / "results" / "vexcel")
    ap.add_argument("--gt-root", type=Path, default=REPO / "data" / "annotations" / "Vexcel")
    ap.add_argument("--run-subdir", default=RUN,
                    help="per-city subdir under results-root/<city>/ holding <grid>/predictions_metric.gpkg")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "analysis" / "xdomain60")
    ap.add_argument("--cities", nargs="+", default=CITIES)
    ap.add_argument("--include-mined-hn", action="store_true",
                    help="do NOT exclude HN-mined grids (default: exclude them "
                         "to keep the cross-domain eval surface leakage-free)")
    args = ap.parse_args()

    all_rows, all_empty = [], []
    for city in args.cities:
        pred_root = args.results_root / city / args.run_subdir
        gt_root = args.gt_root / city
        rows, empties = eval_city(city, pred_root, gt_root,
                                  exclude_mined_hn=not args.include_mined_hn)
        all_rows.extend(rows)
        all_empty.extend(empties)
        print(f"[{city}] non-empty grids={len(rows)} empty-GT grids={len(empties)}")

    _write_csv(args.output_dir / "per_grid.csv", all_rows)

    per_city = summarize(all_rows)
    _write_csv(args.output_dir / "per_city_tier1.csv", per_city)

    pooled = [{**r, "region": "ALL", "model_run": RUN} for r in all_rows]
    overall = summarize(pooled)
    _write_csv(args.output_dir / "overall_tier1.csv", overall)

    _write_csv(args.output_dir / "empty_grid_fp.csv", all_empty)

    # console Tier-1 view
    cols = ["region", "n_grids", "agg_area_F1", "mean_per_grid_F1",
            "bulk_pred_gt_ratio", "std_ratio_Bw", "std_logratio", "rmse_m2",
            "thru0_slope", "ols_r2"]
    def fmt(s):
        return " ".join(f"{str(s.get(c, '')):>10}"[:11] for c in cols)
    print("\n=== Tier-1 per city ===")
    print(" ".join(f"{c:>10}"[:11] for c in cols))
    for s in per_city:
        print(fmt(s))
    print("--- overall ---")
    for s in overall:
        print(fmt(s))

    # empty-grid FP probe
    n_empty = len(all_empty)
    fp_polys = sum(e["n_pred"] for e in all_empty)
    fp_area = sum(e["pred_area_m2"] for e in all_empty)
    n_clean = sum(1 for e in all_empty if e["n_pred"] == 0)
    print(f"\n=== empty-GT FP probe (n={n_empty}) ===")
    print(f"grids with 0 FP (perfect specificity): {n_clean}/{n_empty}")
    print(f"total FP polygons: {fp_polys}  total FP area: {fp_area:.0f} m²  "
          f"mean FP polys/grid: {fp_polys / n_empty:.2f}" if n_empty else "")
    print(f"\nWrote outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
