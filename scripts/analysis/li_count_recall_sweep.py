"""Count-recall (cov50) sweep companion to polygon_conf_sweep.py.

polygon_conf_sweep.py reports Tier-1 area metrics (bulk / σ_Bw / RMSE / aggF1 /
pgF1 / R²) but NOT count recall. This helper adds the missing column: for each
(run, threshold) it computes

    cov50 = (# GT polygons whose own area is >=50% covered by the UNION of kept
             predictions) / (total # GT polygons),

summed over all grids in the run. Semantics are kept identical to the area
sweep by reusing the canonical internals:
  - grid/GT discovery via area_aggregate_eval (_load_run_grids, _gt_spec_for)
  - kept-prediction UNION via polygon_conf_sweep._read_pred_polys_filtered
    (same confidence gate + validity / finite / >20000 m² / area>0 drops)
  - GT polygons loaded individually with the SAME validity/finite/>20000 m²
    filtering, reprojected to region crs_metric.

Threshold list mirrors Task A: a baseline (global-min-conf, ~no filter) row is
auto-prepended per run, then the requested thresholds above it.

Local-only. Does not re-run inference. Does not commit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import pyogrio

from core.region_registry import get_region_config

# Canonical eval internals (identical semantics to the area sweep).
from scripts.analysis.area_aggregate_eval import (
    REPO_ROOT,
    _MAX_PLAUSIBLE_POLY_M2,
    _geometry_finite,
    _gt_spec_for,
    _load_run_grids,
    _read_polys_geom,  # noqa: F401  (kept for symmetry / explicit reuse contract)
)
from scripts.analysis.polygon_conf_sweep import (
    _global_min_conf,
    _read_pred_polys_filtered,
)


def _load_gt_polys_individual(gpkg_path: Path, gt_layer: str | None,
                              metric_crs: str) -> list:
    """Individual GT polygon geoms (NOT a union), filtered + reprojected the
    same way _read_polys_geom filters: valid, finite, area in (0, 20000] m²."""
    available = [row[0] for row in pyogrio.list_layers(gpkg_path)]
    chosen = gt_layer if (gt_layer and gt_layer in available) else (
        available[0] if available else None)
    read_kwargs = {"layer": chosen} if chosen else {}
    gdf = gpd.read_file(gpkg_path, **read_kwargs)
    if gdf.empty:
        return []
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.apply(_geometry_finite)]
    if gdf.empty:
        return []
    if gdf.crs is None or str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    areas = gdf.geometry.area
    keep = (areas > 0) & (areas <= _MAX_PLAUSIBLE_POLY_M2)
    return [g for g, k in zip(gdf.geometry, keep) if k]


def cov50_for_run_at_threshold(region_key: str, run_id: str, conf_min: float):
    """Return (n_gt_total, n_covered, n_pred_kept) summed across grids."""
    region_cfg = get_region_config(region_key)
    metric_crs = region_cfg.crs_metric

    n_gt_total = 0
    n_covered = 0
    n_pred_kept = 0
    for grid_id, pred_path in _load_run_grids(region_key, run_id):
        gt_spec = _gt_spec_for(region_cfg, grid_id)
        if gt_spec is None:
            continue
        gt_path, gt_layer = gt_spec
        try:
            n_kept, _sum, _max, _drop, pred_u = _read_pred_polys_filtered(
                pred_path, metric_crs, conf_min)
            gt_polys = _load_gt_polys_individual(gt_path, gt_layer, metric_crs)
        except Exception as exc:
            print(f"[warn] {region_key}/{run_id}/{grid_id} @t={conf_min}: {exc}")
            continue
        n_pred_kept += n_kept
        for gp in gt_polys:
            ga = gp.area
            if ga <= 0:
                continue
            n_gt_total += 1
            if pred_u is None:
                continue
            inter = gp.intersection(pred_u).area
            if inter / ga >= 0.5:
                n_covered += 1
    return n_gt_total, n_covered, n_pred_kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="cape_town")
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.85, 0.875, 0.90, 0.92, 0.95, 0.97])
    ap.add_argument("--output-dir", default="results/analysis/polygon_conf_sweep_li")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() \
        else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run_id in args.run:
        baseline_t = _global_min_conf(args.region, run_id)
        ts = [baseline_t] + [t for t in sorted(args.thresholds) if t > baseline_t]
        print(f"[cov50] {run_id}: baseline(min)={baseline_t:.4f}  "
              f"ts={[round(t, 4) for t in ts]}")
        # pred count at baseline = denominator for pct_preds_kept
        base_n_pred = None
        for t in ts:
            n_gt, n_cov, n_kept = cov50_for_run_at_threshold(args.region, run_id, t)
            if base_n_pred is None:
                base_n_pred = n_kept
            cov50 = (n_cov / n_gt) if n_gt > 0 else 0.0
            pct_kept = (n_kept / base_n_pred) if base_n_pred else 0.0
            rows.append({
                "run": run_id,
                "conf_threshold": round(t, 4),
                "is_baseline": (t == baseline_t),
                "n_gt_total": n_gt,
                "n_covered": n_cov,
                "cov50_recall": round(cov50, 4),
                "pct_preds_kept": round(pct_kept, 4),
            })

    out_path = out_dir / "count_recall_cov50.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"=== wrote {len(rows)} cov50 rows -> {out_path}")

    print()
    hdr = (f"{'run':24s} {'t':>6} {'base':>5} {'n_gt':>5} {'n_cov':>5} "
           f"{'cov50':>6} {'pct_kept':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:24s} {r['conf_threshold']:>6.3f} "
              f"{('Y' if r['is_baseline'] else ''):>5} "
              f"{r['n_gt_total']:>5} {r['n_covered']:>5} "
              f"{r['cov50_recall']:>6.3f} {r['pct_preds_kept']:>8.3f}")


if __name__ == "__main__":
    main()
