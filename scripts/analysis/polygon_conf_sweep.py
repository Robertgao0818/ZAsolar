"""Per-model polygon-confidence sweep on existing predictions_metric.gpkg.

Reuses area_aggregate_eval's Tier-1 metric machinery verbatim (set-theoretic
unary_union, EPSG:32734 metric reproject, >20000 m² drop, std_ratio_Bw, RMSE,
agg_area_F1, OLS R²/thru0_β) and only adds a per-polygon `confidence >= t`
filter to the prediction side before computing per-grid union area. GT is
loaded once per grid via the same auto-discovery path.

Baseline point (t = global min confidence, i.e. no filter) must reproduce the
fixed-0.85 v4_canonical table from per_run_summary.csv.

Local-only. Does not re-run inference. Does not commit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.ops import unary_union

# Reuse the canonical eval internals so the metric definition is identical.
from scripts.analysis.area_aggregate_eval import (
    REPO_ROOT,
    _MAX_PLAUSIBLE_POLY_M2,
    _geometry_finite,
    _gt_spec_for,
    _load_run_grids,
    _read_polys_geom,
    summarize,
)
from core.region_registry import get_model_run, get_region_config


def _read_pred_polys_filtered(gpkg_path: Path, metric_crs: str, conf_min: float):
    """Like _read_polys_geom but keeps only polygons with confidence >= conf_min.

    Returns (n_kept, sum_area_m2, max_poly_m2, n_dropped, union_geom_or_None).
    Mirrors _read_polys_geom's validity / finite / >20000 m² / area>0 filtering
    exactly, then adds the confidence gate.
    """
    available = [row[0] for row in pyogrio.list_layers(gpkg_path)]
    chosen = available[0] if available else None
    read_kwargs = {"layer": chosen} if chosen else {}
    gdf = gpd.read_file(gpkg_path, **read_kwargs)
    if gdf.empty:
        return 0, 0.0, 0.0, 0, None
    # confidence gate (apply before geometry filtering so n_dropped semantics
    # match the >20000 m² / invalid drops the eval already reports)
    conf_col = "confidence" if "confidence" in gdf.columns else (
        "score" if "score" in gdf.columns else None)
    if conf_col is not None:
        gdf = gdf[gdf[conf_col] >= conf_min]
    if gdf.empty:
        return 0, 0.0, 0.0, 0, None
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.apply(_geometry_finite)]
    if gdf.empty:
        return 0, 0.0, 0.0, 0, None
    if gdf.crs is None or str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    areas = gdf.geometry.area
    keep_mask = areas <= _MAX_PLAUSIBLE_POLY_M2
    n_dropped = int((~keep_mask).sum())
    kept_mask = keep_mask & (areas > 0)
    kept_geoms = [g for g, k in zip(gdf.geometry, kept_mask) if k]
    if not kept_geoms:
        return 0, 0.0, 0.0, n_dropped, None
    sum_area = float(sum(g.area for g in kept_geoms))
    max_area = float(max(g.area for g in kept_geoms))
    u = unary_union(kept_geoms)
    return len(kept_geoms), sum_area, max_area, n_dropped, u


def evaluate_run_at_threshold(region_key: str, run_id: str, conf_min: float) -> list[dict]:
    """Per-grid rows for one (run, threshold), schema-compatible with summarize()."""
    region_cfg = get_region_config(region_key)
    mr = get_model_run(region_key, run_id)
    metric_crs = region_cfg.crs_metric

    rows: list[dict] = []
    for grid_id, pred_path in _load_run_grids(region_key, run_id):
        gt_spec = _gt_spec_for(region_cfg, grid_id)
        if gt_spec is None:
            continue
        gt_path, gt_layer = gt_spec
        try:
            n_pred, pred_sum_m2, pred_max, pred_drop, pred_u = _read_pred_polys_filtered(
                pred_path, metric_crs, conf_min)
            n_gt, gt_sum_m2, gt_max, gt_drop, gt_u = _read_polys_geom(
                gt_path, metric_crs, layer=gt_layer)
        except Exception as exc:
            print(f"[warn] {region_key}/{run_id}/{grid_id} @t={conf_min}: {exc}")
            continue
        if gt_u is None or gt_u.area <= 0:
            continue
        pred_union_m2 = float(pred_u.area) if pred_u is not None else 0.0
        gt_union_m2 = float(gt_u.area)
        inter_m2 = float(pred_u.intersection(gt_u).area) if pred_u is not None else 0.0
        abs_err = pred_union_m2 - gt_union_m2
        area_R = inter_m2 / gt_union_m2
        area_P = inter_m2 / pred_union_m2 if pred_union_m2 > 0 else 0.0
        area_F1 = (2 * area_R * area_P / (area_R + area_P)
                   if (area_R + area_P) > 0 else 0.0)
        rows.append({
            "region": region_key,
            "model_run": run_id,
            "model_version": mr.model_version,
            "imagery_layer": mr.imagery_layer,
            "grid_id": grid_id,
            "gt_source": gt_path.name,
            "n_pred": n_pred,
            "pred_total_m2": round(pred_union_m2, 2),
            "pred_sum_m2": round(pred_sum_m2, 2),
            "pred_overlap_factor": round(pred_sum_m2 / pred_union_m2, 4)
                if pred_union_m2 > 0 else None,
            "pred_max_poly_m2": round(pred_max, 2),
            "n_gt": n_gt,
            "gt_total_m2": round(gt_union_m2, 2),
            "gt_sum_m2": round(gt_sum_m2, 2),
            "gt_max_poly_m2": round(gt_max, 2),
            "n_dropped_pred": pred_drop,
            "n_dropped_gt": gt_drop,
            "inter_m2": round(inter_m2, 2),
            "area_R": round(area_R, 4),
            "area_P": round(area_P, 4),
            "area_F1": round(area_F1, 4),
            "abs_error_m2": round(abs_err, 2),
            "signed_rel_error": round(abs_err / gt_union_m2, 4),
            "abs_rel_error": round(abs(abs_err) / gt_union_m2, 4),
            "pred_gt_ratio": round(pred_union_m2 / gt_union_m2, 4),
        })
    return rows


def _global_min_conf(region_key: str, run_id: str) -> float:
    """Smallest per-polygon confidence over all grids in the run (= no-filter point)."""
    mvals = []
    for _gid, pred_path in _load_run_grids(region_key, run_id):
        g = gpd.read_file(pred_path)
        if len(g) == 0:
            continue
        col = "confidence" if "confidence" in g.columns else "score"
        mvals.append(float(g[col].min()))
    return min(mvals) if mvals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="cape_town")
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.875, 0.90, 0.925, 0.95, 0.97, 0.99])
    ap.add_argument("--output-dir", default="results/analysis/polygon_conf_sweep")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    per_grid_rows: list[dict] = []
    for run_id in args.run:
        baseline_t = _global_min_conf(args.region, run_id)
        # baseline (no filter) first, then the requested thresholds above it.
        ts = [baseline_t] + [t for t in sorted(args.thresholds) if t > baseline_t]
        print(f"[sweep] {run_id}: baseline(min)={baseline_t:.4f}  ts={[round(t,4) for t in ts]}")
        for t in ts:
            grid_rows = evaluate_run_at_threshold(args.region, run_id, t)
            for gr in grid_rows:
                gr2 = dict(gr)
                gr2["conf_threshold"] = round(t, 4)
                per_grid_rows.append(gr2)
            summ = summarize(grid_rows)
            for s in summ:
                s2 = dict(s)
                s2["conf_threshold"] = round(t, 4)
                s2["is_baseline"] = (t == baseline_t)
                sweep_rows.append(s2)

    # write
    pg_path = out_dir / "sweep_per_grid.csv"
    summ_path = out_dir / "sweep_summary.csv"
    if per_grid_rows:
        with open(pg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_grid_rows[0].keys()))
            w.writeheader(); w.writerows(per_grid_rows)
    if sweep_rows:
        # stable column order: put conf_threshold / is_baseline up front
        front = ["region", "model_run", "conf_threshold", "is_baseline", "n_grids"]
        rest = [k for k in sweep_rows[0].keys() if k not in front]
        cols = front + rest
        with open(summ_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sweep_rows:
                w.writerow({k: r.get(k) for k in cols})
    print(f"=== wrote {len(sweep_rows)} sweep rows -> {summ_path}")
    print(f"=== wrote {len(per_grid_rows)} per-grid rows -> {pg_path}")

    # console view
    print()
    hdr = f"{'run':24s} {'t':>6} {'base':>5} {'n':>3} {'bulk':>6} {'σ_Bw':>6} {'RMSE':>8} {'aggF1':>6} {'pgF1':>6} {'R²':>6}"
    print(hdr); print("-" * len(hdr))
    for r in sweep_rows:
        print(f"{r['model_run']:24s} {r['conf_threshold']:>6.3f} "
              f"{('Y' if r['is_baseline'] else ''):>5} {r['n_grids']:>3} "
              f"{(r['bulk_pred_gt_ratio'] or 0):>6.3f} "
              f"{(r['std_ratio_Bw'] or 0):>6.3f} "
              f"{(r['rmse_m2'] or 0):>8.1f} "
              f"{(r['agg_area_F1'] or 0):>6.3f} "
              f"{(r['mean_per_grid_F1'] or 0):>6.3f} "
              f"{(r['ols_r2'] or 0):>6.3f}")


if __name__ == "__main__":
    main()
