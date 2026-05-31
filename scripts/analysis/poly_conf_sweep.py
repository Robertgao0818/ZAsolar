"""poly_conf_sweep.py — polygon-confidence threshold sweep on a registered run.

For a registered model_run, sweep a polygon-level confidence threshold c over
`predictions_metric.gpkg` (filtering polygons with confidence < c BEFORE the
set-theoretic union), recomputing the Tier-1 aggregate metrics at each c.
GT is held constant. Ranks operating points by `sigma_Bw + rmse_m2/1e5`
subject to a bulk-ratio sanity gate (default [0.5, 2.0]).

This is the "consumption-time" lever documented in
docs/experiments/2026-05-14-jhb-cbd25-3model-sam.md: predictions retain
low-confidence polygons on disk; a downstream threshold picks the operating
point. Metric definitions match scripts/analysis/area_aggregate_eval.py
(std_ratio_Bw = B-weighted dispersion; rmse_m2 over per-grid A_h - B_h).

Originally run ad-hoc on the pod; recreated 2026-05-29 to score the
solar_zerov2 Phase 0 pilot fairly (same sweep the production models got).

Run:
  python scripts/analysis/poly_conf_sweep.py --region johannesburg \\
      --run phase0_dinov3sat_m2f_perdet \\
      --gt-root data/annotations_channel2_clean \\
      --output results/analysis/phase0_dinov3_m2f_eval/poly_conf_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.ops import unary_union

from area_aggregate_eval import (  # type: ignore
    _MAX_PLAUSIBLE_POLY_M2,
    _geometry_finite,
    _gt_spec_for,
    _load_run_grids,
)
from core.region_registry import get_region_config


def _load_polys(gpkg_path: Path, metric_crs: str, layer: str | None, want_conf: bool):
    """Return (geoms, confidences|None) after the same filters area_aggregate_eval uses."""
    available = [row[0] for row in pyogrio.list_layers(gpkg_path)]
    chosen = layer if (layer and layer in available) else (available[0] if available else None)
    kwargs = {"layer": chosen} if chosen else {}
    gdf = gpd.read_file(gpkg_path, **kwargs)
    if gdf.empty:
        return [], (np.array([]) if want_conf else None)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.apply(_geometry_finite)]
    if gdf.empty:
        return [], (np.array([]) if want_conf else None)
    if gdf.crs is None or str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    areas = gdf.geometry.area
    keep = (areas <= _MAX_PLAUSIBLE_POLY_M2) & (areas > 0)
    gdf = gdf[keep]
    geoms = list(gdf.geometry)
    confs = None
    if want_conf:
        if "confidence" in gdf.columns:
            confs = gdf["confidence"].to_numpy(dtype=float)
        else:
            confs = np.ones(len(geoms), dtype=float)
    return geoms, confs


def _agg(As: np.ndarray, Bs: np.ndarray, Is: np.ndarray) -> dict:
    """Aggregate per-grid (A_h, B_h, inter_h) into Tier-1 metrics — matches
    summarize() in area_aggregate_eval.py."""
    pred_total, gt_total, inter_total = float(As.sum()), float(Bs.sum()), float(Is.sum())
    R = inter_total / gt_total if gt_total else 0.0
    P = inter_total / pred_total if pred_total else 0.0
    F1 = 2 * R * P / (R + P) if (R + P) else 0.0
    ratios = As / np.where(Bs > 0, Bs, 1.0)
    mean_ratio = float(ratios.mean())
    w = Bs / gt_total if gt_total else np.zeros_like(Bs)
    sigma_Bw = float(np.sqrt((w * (ratios - mean_ratio) ** 2).sum())) if gt_total else float("nan")
    valid = (As > 0) & (Bs > 0)
    sigma_log = float(np.log(As[valid] / Bs[valid]).std(ddof=1)) if valid.sum() >= 2 else float("nan")
    eps = As - Bs
    rmse = float(np.sqrt((eps ** 2).mean())) if len(eps) else float("nan")
    beta = float((Bs * As).sum() / (Bs ** 2).sum()) if (Bs ** 2).sum() > 0 else float("nan")
    ss_res = float(((As - beta * Bs) ** 2).sum())
    ss_tot = float(((As - As.mean()) ** 2).sum())
    beta_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    bulk = pred_total / gt_total if gt_total else float("nan")
    return dict(A_total_m2=pred_total, B_total_m2=gt_total, bulk=bulk, agg_R=R, agg_P=P,
                agg_F1=F1, sigma_Bw=sigma_Bw, sigma_log=sigma_log, rmse_m2=rmse,
                thru0_beta=beta, thru0_r2=beta_r2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt-root", type=Path, default=None)
    ap.add_argument("--gt-pattern", default="{grid}/{grid}_clean_gt.gpkg")
    ap.add_argument("--c-min", type=float, default=0.50)
    ap.add_argument("--c-max", type=float, default=0.97)
    ap.add_argument("--c-step", type=float, default=0.025)
    ap.add_argument("--gate-lo", type=float, default=0.5)
    ap.add_argument("--gate-hi", type=float, default=2.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    region_cfg = get_region_config(args.region)
    metric_crs = region_cfg.crs_metric

    # Preload per-grid GT (constant) + pred geoms with confidence (filtered once).
    grids = []
    for grid_id, pred_path in _load_run_grids(args.region, args.run):
        gt_spec = _gt_spec_for(region_cfg, grid_id, gt_root_override=args.gt_root,
                               gt_pattern=args.gt_pattern)
        if gt_spec is None:
            print(f"[skip] {grid_id}: no GT")
            continue
        gt_path, gt_layer = gt_spec
        gt_geoms, _ = _load_polys(gt_path, metric_crs, gt_layer, want_conf=False)
        if not gt_geoms:
            continue
        gt_u = unary_union(gt_geoms)
        if gt_u.area <= 0:
            continue
        pred_geoms, confs = _load_polys(pred_path, metric_crs, None, want_conf=True)
        grids.append((grid_id, gt_u, float(gt_u.area), pred_geoms, confs))
    print(f"[loaded] {len(grids)} grids for {args.region}/{args.run}")

    cs = np.arange(args.c_min, args.c_max + 1e-9, args.c_step)
    rows = []
    for c in cs:
        As, Bs, Is = [], [], []
        for grid_id, gt_u, gt_area, pred_geoms, confs in grids:
            sel = [g for g, cf in zip(pred_geoms, confs) if cf >= c]
            if sel:
                pu = unary_union(sel)
                A = float(pu.area)
                inter = float(pu.intersection(gt_u).area)
            else:
                A, inter = 0.0, 0.0
            As.append(A); Bs.append(gt_area); Is.append(inter)
        m = _agg(np.array(As), np.array(Bs), np.array(Is))
        in_gate = args.gate_lo <= m["bulk"] <= args.gate_hi
        rank_key = m["sigma_Bw"] + m["rmse_m2"] / 1e5
        rows.append(dict(c=round(float(c), 4), n_grids=len(grids), **{k: round(v, 6) for k, v in m.items()},
                         in_gate=in_gate, rank_key=round(rank_key, 6)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    gated = [r for r in rows if r["in_gate"]]
    best = min(gated, key=lambda r: r["rank_key"]) if gated else None
    print(f"\n=== {args.region}/{args.run} sweep -> {args.output} ===")
    hdr = f"{'c':>6}{'bulk':>8}{'F1':>8}{'σ_Bw':>9}{'RMSE':>9}{'β0':>8}{'gate':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['c']:>6}{r['bulk']:>8.3f}{r['agg_F1']:>8.3f}{r['sigma_Bw']:>9.3f}"
              f"{r['rmse_m2']:>9.1f}{r['thru0_beta']:>8.3f}{'Y' if r['in_gate'] else 'n':>6}")
    if best:
        print(f"\n>>> BEST (min σ_Bw+RMSE/1e5 in gate): c={best['c']} "
              f"bulk={best['bulk']:.3f} F1={best['agg_F1']:.3f} "
              f"σ_Bw={best['sigma_Bw']:.3f} RMSE={best['rmse_m2']:.1f}")


if __name__ == "__main__":
    main()
