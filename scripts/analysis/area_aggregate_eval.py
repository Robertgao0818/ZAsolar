"""Aggregate area evaluation: DeepSolar-style total m² comparison.

For each (region, model_run, grid), sums predicted installation area
against ground-truth total area and reports absolute / relative error.
Errors within a grid cancel between FP and FN — this is the
"聚合抵消" metric DeepSolar used against utility/EIA totals.

Outputs:
  - <output-dir>/per_grid.csv        — one row per (run, grid)
  - <output-dir>/per_run_summary.csv — region-level MAE / MRE / bulk ratio

Example:
  python scripts/analysis/area_aggregate_eval.py
  python scripts/analysis/area_aggregate_eval.py --region johannesburg --skip-deprecated
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import math

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.ops import unary_union

from core.region_registry import (
    get_annotations_dir_for_grid,
    get_model_run,
    get_region_config,
    list_model_runs,
    list_regions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# Any single solar installation polygon larger than this in metric area is
# almost certainly a corrupted geometry (broken coord, grid-tile outline, etc.)
# and will distort aggregate sums. Residential installs are <~200 m²,
# commercial <~5000 m². 20 000 m² is a generous upper bound.
_MAX_PLAUSIBLE_POLY_M2 = 20_000.0


def _geometry_finite(geom) -> bool:
    """Reject polygons with non-finite coordinates (NaN, inf, denormals)."""
    try:
        minx, miny, maxx, maxy = geom.bounds
    except Exception:
        return False
    for v in (minx, miny, maxx, maxy):
        if not math.isfinite(v) or abs(v) > 1e18:
            return False
    return True


def _sum_area_m2(
    gpkg_path: Path, metric_crs: str, layer: str | None
) -> tuple[int, float, float, int]:
    """Return (n_features_kept, total_area_m2, max_poly_m2, n_dropped).

    Drops polygons with invalid geometries, non-finite coords, or polygon
    area exceeding _MAX_PLAUSIBLE_POLY_M2. Reprojects to ``metric_crs``.
    If ``layer`` is specified but missing, falls back to the first layer.
    """
    available = [row[0] for row in pyogrio.list_layers(gpkg_path)]
    chosen: str | None = layer if layer and layer in available else None
    if chosen is None and available:
        chosen = available[0]
    read_kwargs: dict[str, object] = {}
    if chosen:
        read_kwargs["layer"] = chosen
    gdf = gpd.read_file(gpkg_path, **read_kwargs)
    if gdf.empty:
        return 0, 0.0, 0.0, 0
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.apply(_geometry_finite)]
    if gdf.empty:
        return 0, 0.0, 0.0, 0
    if gdf.crs is None or str(gdf.crs) != metric_crs:
        gdf = gdf.to_crs(metric_crs)
    areas = gdf.geometry.area
    keep_mask = areas <= _MAX_PLAUSIBLE_POLY_M2
    n_dropped = int((~keep_mask).sum())
    kept = areas[keep_mask]
    if kept.empty:
        return 0, 0.0, 0.0, n_dropped
    return len(kept), float(kept.sum()), float(kept.max()), n_dropped


def _read_polys_geom(
    gpkg_path: Path, metric_crs: str, layer: str | None
):
    """Like _sum_area_m2 but also returns the unary_union geometry needed
    for set-theoretic R/P/F1 / IoU. Required for per-detection outputs where
    overlapping polygons would inflate naive sum (overlap factor ~1.6 on
    train20_val5_hn per-det).

    Returns: (n_kept, sum_area_m2, max_poly_m2, n_dropped, union_geom_or_None)
    """
    available = [row[0] for row in pyogrio.list_layers(gpkg_path)]
    chosen: str | None = layer if layer and layer in available else None
    if chosen is None and available:
        chosen = available[0]
    read_kwargs: dict[str, object] = {}
    if chosen:
        read_kwargs["layer"] = chosen
    gdf = gpd.read_file(gpkg_path, **read_kwargs)
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


_GT_PRIORITY_SUFFIXES = ("_SAM2_", "_V4_", "_reviewed", "")


def _discover_gt(annotations_dir: Path, grid_id: str) -> Path | None:
    """Pick the best GT gpkg for a grid via filename heuristics.

    Preference: SAM2 > V4-reviewed > reviewed > plain <grid>.gpkg,
    latest date when multiple candidates share a category.
    """
    if not annotations_dir.exists():
        return None
    candidates = sorted(annotations_dir.glob(f"{grid_id}*.gpkg"))
    if not candidates:
        return None
    def rank(p: Path) -> tuple[int, str]:
        name = p.name
        for i, tag in enumerate(_GT_PRIORITY_SUFFIXES):
            if tag and tag in name:
                return (i, name)
        return (len(_GT_PRIORITY_SUFFIXES), name)
    candidates.sort(key=rank)
    # Within same rank, the latest date suffix sorts last alphabetically.
    best_rank = rank(candidates[0])[0]
    same = [p for p in candidates if rank(p)[0] == best_rank]
    return max(same, key=lambda p: p.name)


def _load_run_grids(region_key: str, run_id: str) -> list[tuple[str, Path]]:
    """Return (grid_id, predictions_metric.gpkg path) pairs actually present on disk."""
    mr = get_model_run(region_key, run_id)
    run_dir = _resolve(mr.results_path)
    if not run_dir.exists():
        return []
    pairs: list[tuple[str, Path]] = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        pred_gpkg = sub / "predictions_metric.gpkg"
        if pred_gpkg.exists():
            pairs.append((sub.name, pred_gpkg))
    return pairs


def _gt_spec_for(
    region_cfg,
    grid_id: str,
    gt_root_override: Path | None = None,
    gt_pattern: str = "{grid}/{grid}_clean_gt.gpkg",
) -> tuple[Path, str | None] | None:
    if gt_root_override is not None:
        candidate = gt_root_override / gt_pattern.format(grid=grid_id)
        if candidate.exists():
            return candidate, None
        return None

    entry = region_cfg.grids.get(grid_id) or {}
    src = entry.get("annotation_source")
    layer = entry.get("annotation_layer")
    gt_path = _resolve(src) if src else None
    if gt_path is None or not gt_path.exists():
        # Fall back to auto-discovery. Route through the annotation scheme that
        # owns this grid_id (e.g. L-prefix Li grids -> Capetown_Li), so multi-
        # scheme regions don't all glob the primary annotations dir.
        scheme_dir = get_annotations_dir_for_grid(region_cfg.key, grid_id)
        annotations_dir = scheme_dir or _resolve(region_cfg.paths.annotations_dir)
        gt_path = _discover_gt(annotations_dir, grid_id)
        layer = None  # _sum_area_m2 will pick the first available layer
    if gt_path is None:
        return None
    return gt_path, layer


def evaluate_run(
    region_key: str,
    run_id: str,
    gt_root_override: Path | None = None,
    gt_pattern: str = "{grid}/{grid}_clean_gt.gpkg",
) -> list[dict]:
    region_cfg = get_region_config(region_key)
    mr = get_model_run(region_key, run_id)
    metric_crs = region_cfg.crs_metric

    rows: list[dict] = []
    for grid_id, pred_path in _load_run_grids(region_key, run_id):
        gt_spec = _gt_spec_for(region_cfg, grid_id,
                               gt_root_override=gt_root_override,
                               gt_pattern=gt_pattern)
        if gt_spec is None:
            continue
        gt_path, gt_layer = gt_spec
        try:
            n_pred, pred_sum_m2, pred_max, pred_drop, pred_u = _read_polys_geom(
                pred_path, metric_crs, layer=None)
            n_gt, gt_sum_m2, gt_max, gt_drop, gt_u = _read_polys_geom(
                gt_path, metric_crs, layer=gt_layer)
        except Exception as exc:
            print(f"[warn] {region_key}/{run_id}/{grid_id}: {exc}")
            continue
        if gt_drop or pred_drop:
            print(f"[filter] {region_key}/{run_id}/{grid_id}: "
                  f"dropped {pred_drop} pred + {gt_drop} gt polygons "
                  f"(invalid or > {_MAX_PLAUSIBLE_POLY_M2:.0f} m²)")
        if gt_u is None or gt_u.area <= 0:
            continue
        # Set-theoretic union areas (canonical for inventory / R/P/F1).
        pred_union_m2 = float(pred_u.area) if pred_u is not None else 0.0
        gt_union_m2 = float(gt_u.area)
        inter_m2 = float(pred_u.intersection(gt_u).area) if pred_u is not None else 0.0
        abs_err = pred_union_m2 - gt_union_m2
        # Per-grid R/P/F1 (pixel set-theoretic on union geometries).
        area_R = inter_m2 / gt_union_m2
        area_P = inter_m2 / pred_union_m2 if pred_union_m2 > 0 else 0.0
        area_F1 = (
            2 * area_R * area_P / (area_R + area_P) if (area_R + area_P) > 0 else 0.0
        )
        rows.append({
            "region": region_key,
            "model_run": run_id,
            "model_version": mr.model_version,
            "imagery_layer": mr.imagery_layer,
            "grid_id": grid_id,
            "gt_source": gt_path.name,
            "n_pred": n_pred,
            # Union (canonical) and sum (overlap-blind, kept for diagnosis)
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
            # Tier 1a — classic detection
            "area_R": round(area_R, 4),
            "area_P": round(area_P, 4),
            "area_F1": round(area_F1, 4),
            # Tier 1b — level
            "abs_error_m2": round(abs_err, 2),
            "signed_rel_error": round(abs_err / gt_union_m2, 4),
            "abs_rel_error": round(abs(abs_err) / gt_union_m2, 4),
            "pred_gt_ratio": round(pred_union_m2 / gt_union_m2, 4),
        })
    return rows


def _ols_regression(xs: list[float], ys: list[float]) -> dict:
    """Simple OLS: y = slope * x + intercept. Returns slope, intercept, R²
    (coefficient of determination against the mean-of-y baseline).

    R² here is the classical goodness-of-fit — closest analog to DeepSolar's
    tract-level predicted-vs-utility regression R². R² can be negative when
    the fit is worse than predicting the mean.
    """
    n = len(xs)
    if n < 2:
        return {"slope": None, "intercept": None, "r2": None}
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    if ss_xx == 0 or ss_yy == 0:
        return {"slope": None, "intercept": None, "r2": None}
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_yy
    return {"slope": slope, "intercept": intercept, "r2": r2}


def _bootstrap_ci(values, statfn, n_boot: int = 500, ci: float = 0.95,
                  seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a single statistic of a 1-D sample.
    Returns (lo, hi). Returns (nan, nan) on n < 3."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = np.array([statfn(arr[s]) for s in idx])
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def summarize(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        buckets.setdefault((r["region"], r["model_run"]), []).append(r)

    out: list[dict] = []
    for (region, run), items in sorted(buckets.items()):
        n_grids = len(items)
        Bs = np.array([r["gt_total_m2"] for r in items], float)
        As = np.array([r["pred_total_m2"] for r in items], float)
        Is = np.array([r["inter_m2"] for r in items], float)
        ratios = As / np.where(Bs > 0, Bs, 1.0)
        eps = As - Bs
        pg_F1 = np.array([r["area_F1"] for r in items], float)

        pred_total = float(As.sum())
        gt_total = float(Bs.sum())
        inter_total = float(Is.sum())

        # ---- Tier 1a: classic detection ----
        agg_R = inter_total / gt_total if gt_total > 0 else None
        agg_P = inter_total / pred_total if pred_total > 0 else None
        if agg_R is not None and agg_P is not None and (agg_R + agg_P) > 0:
            agg_F1 = 2 * agg_R * agg_P / (agg_R + agg_P)
        else:
            agg_F1 = None
        mean_pg_F1 = float(pg_F1.mean()) if len(pg_F1) else None
        f1_lo, f1_hi = _bootstrap_ci(pg_F1, lambda v: float(v.mean()))

        # ---- Tier 1b: legacy aggregate stats ----
        mae = statistics.fmean(abs(r["abs_error_m2"]) for r in items)
        mre = statistics.fmean(r["abs_rel_error"] for r in items)
        signed_mre = statistics.fmean(r["signed_rel_error"] for r in items)
        within_20 = sum(1 for r in items if 0.8 <= r["pred_gt_ratio"] <= 1.2) / n_grids

        # ---- Tier 1c: dispersion ----
        mean_ratio = float(ratios.mean())
        std_ratio = float(ratios.std(ddof=1)) if n_grids >= 2 else float("nan")
        sigma_lo, sigma_hi = _bootstrap_ci(ratios, lambda v: float(v.std(ddof=1)))
        # B-weighted dispersion — the user-validated paper-relevant metric.
        if gt_total > 0:
            w = Bs / gt_total
            std_ratio_Bw = float(np.sqrt((w * (ratios - mean_ratio) ** 2).sum()))
            cv_ratio_Bw = std_ratio_Bw / mean_ratio if mean_ratio != 0 else float("nan")
        else:
            std_ratio_Bw = float("nan"); cv_ratio_Bw = float("nan")
        # Log-ratio: relative-error scale, robust to small-B blow-ups.
        valid = (As > 0) & (Bs > 0)
        if valid.sum() >= 2:
            log_ratios = np.log(As[valid] / Bs[valid])
            std_logratio = float(log_ratios.std(ddof=1))
        else:
            std_logratio = float("nan")

        # ---- Tier 1d: absolute residuals ----
        rmse = float(np.sqrt((eps ** 2).mean())) if n_grids else float("nan")
        rmse_lo, rmse_hi = _bootstrap_ci(np.abs(eps),
                                         lambda v: float(np.sqrt((v ** 2).mean())))

        # ---- Tier 1e: regression diagnostic (DeepSolar-style) ----
        reg = _ols_regression(Bs.tolist(), As.tolist())
        # Through-origin variant — calibration-fixable with a single multiplier.
        if (Bs ** 2).sum() > 0 and len(Bs) >= 2:
            beta_o = float((Bs * As).sum() / (Bs ** 2).sum())
            ss_res_o = float(((As - beta_o * Bs) ** 2).sum())
            ss_tot_o = float(((As - As.mean()) ** 2).sum())
            r2_o = 1.0 - ss_res_o / ss_tot_o if ss_tot_o > 0 else float("nan")
        else:
            beta_o, r2_o = float("nan"), float("nan")

        out.append({
            "region": region,
            "model_run": run,
            "model_version": items[0]["model_version"],
            "imagery_layer": items[0]["imagery_layer"],
            "n_grids": n_grids,
            "pred_total_m2": round(pred_total, 2),
            "gt_total_m2": round(gt_total, 2),
            "inter_total_m2": round(inter_total, 2),
            # Tier 1a — classic detection
            "agg_area_R": round(agg_R, 4) if agg_R is not None else None,
            "agg_area_P": round(agg_P, 4) if agg_P is not None else None,
            "agg_area_F1": round(agg_F1, 4) if agg_F1 is not None else None,
            "mean_per_grid_F1": round(mean_pg_F1, 4) if mean_pg_F1 is not None else None,
            "f1_pg_CI95_lo": round(f1_lo, 4),
            "f1_pg_CI95_hi": round(f1_hi, 4),
            # Tier 1b — level + legacy
            "bulk_pred_gt_ratio": round(pred_total / gt_total, 4) if gt_total else None,
            "bulk_signed_rel_error": round((pred_total - gt_total) / gt_total, 4) if gt_total else None,
            "mae_m2_per_grid": round(mae, 2),
            "mre_per_grid": round(mre, 4),
            "signed_mre_per_grid": round(signed_mre, 4),
            "frac_grids_within_pm20pct": round(within_20, 3),
            # Tier 1c — dispersion (paper-relevant primary)
            "std_ratio": round(std_ratio, 4),
            "std_ratio_CI95_lo": round(sigma_lo, 4),
            "std_ratio_CI95_hi": round(sigma_hi, 4),
            "std_ratio_Bw": round(std_ratio_Bw, 4),
            "cv_ratio_Bw": round(cv_ratio_Bw, 4),
            "std_logratio": round(std_logratio, 4),
            # Tier 1d — absolute residuals (inventory error)
            "rmse_m2": round(rmse, 2),
            "rmse_CI95_lo": round(rmse_lo, 2),
            "rmse_CI95_hi": round(rmse_hi, 2),
            # Tier 1e — calibration diagnostic
            "ols_slope": round(reg["slope"], 4) if reg["slope"] is not None else None,
            "ols_intercept_m2": round(reg["intercept"], 2) if reg["intercept"] is not None else None,
            "ols_r2": round(reg["r2"], 4) if reg["r2"] is not None else None,
            "thru0_slope": round(beta_o, 4),
            "thru0_r2": round(r2_o, 4),
        })
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", nargs="+", help="Restrict to these regions")
    parser.add_argument("--run", nargs="+", help="Restrict to these model_run IDs")
    parser.add_argument("--skip-deprecated", action="store_true",
                        help="Skip model_runs flagged deprecated=true in regions.yaml")
    parser.add_argument("--output-dir", default="results/analysis/area_aggregate")
    parser.add_argument("--gt-root", type=Path, default=None,
                        help="Override GT lookup root. When set, GT for grid G is "
                             "expected at <gt-root>/<gt-pattern>.")
    parser.add_argument("--gt-pattern", default="{grid}/{grid}_clean_gt.gpkg",
                        help="Path template under --gt-root; {grid} is substituted.")
    args = parser.parse_args()

    regions = args.region or list_regions()
    all_rows: list[dict] = []

    # We read the raw yaml for the `deprecated` flag — ModelRunConfig dataclass
    # does not expose it.
    import yaml
    raw = yaml.safe_load(open(REPO_ROOT / "configs" / "datasets" / "regions.yaml"))

    for region_key in regions:
        runs = list_model_runs(region_key)
        for run_id in runs:
            if args.run and run_id not in args.run:
                continue
            run_raw = raw["regions"][region_key].get("model_runs", {}).get(run_id, {})
            if args.skip_deprecated and run_raw.get("deprecated"):
                print(f"[skip-deprecated] {region_key}/{run_id}")
                continue
            print(f"[eval] {region_key}/{run_id} ...", flush=True)
            rows = evaluate_run(region_key, run_id,
                                gt_root_override=args.gt_root,
                                gt_pattern=args.gt_pattern)
            print(f"        {len(rows)} grids matched")
            all_rows.extend(rows)

    summary = summarize(all_rows)
    out_dir = _resolve(args.output_dir)
    _write_csv(out_dir / "per_grid.csv", all_rows)
    _write_csv(out_dir / "per_run_summary.csv", summary)

    print()
    print(f"=== Wrote {len(all_rows)} per-grid rows -> {out_dir}/per_grid.csv")
    print(f"=== Wrote {len(summary)} per-run rows   -> {out_dir}/per_run_summary.csv")
    print()
    if summary:
        # Deploy-decision view: F1 (sanity), σ_Bw + RMSE (paper-relevant primary),
        # bulk (level diagnostic). OLS R² stays as calibration-fixable indicator.
        header = (
            f"{'region':<14} {'model_run':<38} {'n':>3} "
            f"{'F1':>6} {'pgF1':>6} {'bulk':>6} "
            f"{'σ_Bw':>6} {'log-σ':>6} {'RMSE':>8} {'thru0_β':>8} {'R²':>6}"
        )
        print(header)
        print("-" * len(header))
        for s in summary:
            def f(v, w=6, p=3, sign=False):
                if v is None:
                    return f"{'-':>{w}}"
                fmt = f"{{:>+{w}.{p}f}}" if sign else f"{{:>{w}.{p}f}}"
                return fmt.format(v)
            print(
                f"{s['region']:<14} {s['model_run']:<38} {s['n_grids']:>3} "
                f"{f(s['agg_area_F1'], 6, 3)} "
                f"{f(s['mean_per_grid_F1'], 6, 3)} "
                f"{f(s['bulk_pred_gt_ratio'], 6, 3)} "
                f"{f(s['std_ratio_Bw'], 6, 3)} "
                f"{f(s['std_logratio'], 6, 3)} "
                f"{f(s['rmse_m2'], 8, 1)} "
                f"{f(s['thru0_slope'], 8, 3)} "
                f"{f(s['ols_r2'], 6, 3)}"
            )


if __name__ == "__main__":
    main()
