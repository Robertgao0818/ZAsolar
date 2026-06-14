"""Tier-1 aggregate-area statistics kernel (V1.4 Channel-3 main judge).

This module holds the *pure* statistical core extracted verbatim from
``scripts/analysis/area_aggregate_eval.py`` on 2026-06-12. The file-system /
GeoPackage I/O (``evaluate_run`` and friends) stays in that script; only the
side-effect-free summarisation lives here so it can be unit-tested and reused
by the three sibling callers without dragging in geopandas I/O.

The functions are byte-for-byte the same logic as the original ``summarize`` —
no numeric, default, or rounding changes. ``area_aggregate_eval.summarize``
re-exports the symbol here for backward compatibility.

Input contract — ``summarize(rows)`` consumes a list of *per-grid* dicts, one
per ``(region, model_run, grid)``, each carrying these keys (units in m² unless
noted):

  - ``region``           (str)   — region key; first grouping axis.
  - ``model_run``        (str)   — model-run id; second grouping axis.
  - ``model_version``    (str)   — copied through from the first item in a bucket.
  - ``imagery_layer``    (str)   — copied through from the first item in a bucket.
  - ``gt_total_m2``      (float) — set-theoretic GT union area for the grid (B).
  - ``pred_total_m2``    (float) — set-theoretic prediction union area (A).
  - ``inter_m2``         (float) — area of A ∩ B for the grid.
  - ``area_F1``          (float) — per-grid pixel-set F1 (precomputed upstream).
  - ``abs_error_m2``     (float) — A − B for the grid (signed level error).
  - ``abs_rel_error``    (float) — |A − B| / B.
  - ``signed_rel_error`` (float) — (A − B) / B.
  - ``pred_gt_ratio``    (float) — A / B.

These rows are produced by ``area_aggregate_eval.evaluate_run`` and the
``_agg`` helpers in ``per_grid_dispersion_audit.py`` / ``poly_conf_sweep.py``.
``summarize`` returns one summary dict per ``(region, model_run)`` bucket with
the full Tier-1 suite (σ_Bw / RMSE / agg_F1 / through-origin β / R² / OLS /
bootstrap CIs). The bucket dicts and their keys/rounding mirror the original
``per_run_summary.csv`` schema exactly.
"""

from __future__ import annotations

import statistics

import numpy as np


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
