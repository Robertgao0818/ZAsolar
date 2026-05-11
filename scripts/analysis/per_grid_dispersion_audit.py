"""Per-grid A_h vs B_h dispersion: stress-test bulk_ratio vs σ(A/B) framing.

Compares V3-C+SAM(mask+box) and train20_val5_hn per-det+SAM(mask+box) on
JHB CBD 25 grid (Vexcel 2024, clean_gt). Outputs per-grid table, scatter
PNG, and dispersion summary (mean / σ / R² of A vs B regression).

Motivation: |A|/|B| as a level metric punishes recall-driven bulk growth
even at constant precision. For the paper LHS (hexagon-level kW), the
relevant quality measure is *cross-grid spread* of A_h/B_h (heteroscedastic
bias), not aggregate level. A model with mean=1.2 σ small can be better
for paper-use than mean=1.0 σ big.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import unary_union

ROOT = Path("/home/gaosh/projects/ZAsolar")
OUTDIR = ROOT / "docs/experiments/per_hexagon_dispersion_2026-05-10"
OUTDIR.mkdir(parents=True, exist_ok=True)

GRIDS = [
    "G0772","G0773","G0774","G0775","G0776",
    "G0814","G0815","G0816","G0817","G0818",
    "G0853","G0854","G0855","G0856","G0857",
    "G0888","G0889","G0890","G0891","G0892",
    "G0922","G0923","G0924","G0925","G0926",
]

GT_TPL = "data/annotations_channel2_clean/{grid}/{grid}_clean_gt.gpkg"
LAYERS = {
    "V3-C+SAM": "results/johannesburg/v3c_sam_maskbox_vexcel_2024/{grid}/predictions_metric.gpkg",
    "train20_val5_hn per-det+SAM": (
        "results/analysis/v3c_failed_weight_compare/perdet/"
        "train20_val5_hn_perdet_sam_maskbox/{grid}/predictions_metric.gpkg"
    ),
}
METRIC_CRS = "EPSG:32735"
MAX_PLAUSIBLE = 20_000.0  # m², reject corrupted polygons


def union_geom(p: Path):
    """Return unary_union of valid polygons. Empty Polygon if no data."""
    if not p.exists():
        return None
    g = gpd.read_file(p)
    if g.empty:
        return None
    g = g[g.geometry.notna() & g.geometry.is_valid]
    if str(g.crs) != METRIC_CRS:
        g = g.to_crs(METRIC_CRS)
    geoms = [x for x in g.geometry if x and not x.is_empty
             and 0 < x.area <= MAX_PLAUSIBLE]
    if not geoms:
        return None
    return unary_union(geoms)


def union_area(p: Path) -> float:
    u = union_geom(p)
    return float(u.area) if u is not None else 0.0


rows: list[dict] = []
for grid in GRIDS:
    gt_p = ROOT / GT_TPL.format(grid=grid)
    gt_u = union_geom(gt_p)
    B = float(gt_u.area) if gt_u is not None else 0.0
    row = {"grid": grid, "B_m2": round(B, 2)}
    for name, tpl in LAYERS.items():
        pred_u = union_geom(ROOT / tpl.format(grid=grid))
        A = float(pred_u.area) if pred_u is not None else 0.0
        # Pixel-level set-theoretic intersection for R/P/F1
        if pred_u is not None and gt_u is not None:
            inter = float(pred_u.intersection(gt_u).area)
        else:
            inter = 0.0
        row[f"A_{name}"] = round(A, 2)
        row[f"inter_{name}"] = round(inter, 2)
        row[f"ratio_{name}"] = round(A / B, 4) if B > 0 else float("nan")
        row[f"area_R_{name}"] = round(inter / B, 4) if B > 0 else float("nan")
        row[f"area_P_{name}"] = round(inter / A, 4) if A > 0 else float("nan")
        if A > 0 and B > 0 and (inter / A + inter / B) > 0:
            r, p = inter / B, inter / A
            row[f"area_F1_{name}"] = round(2 * r * p / (r + p), 4)
        else:
            row[f"area_F1_{name}"] = float("nan")
    rows.append(row)

# ---------------------------------------------------------------------------
# Stats per layer
def ols(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 2: return None
    mx, my = xs.mean(), ys.mean()
    sxx = ((xs - mx)**2).sum(); sxy = ((xs - mx)*(ys - my)).sum()
    syy = ((ys - my)**2).sum()
    if sxx == 0 or syy == 0: return None
    slope = sxy / sxx
    intercept = my - slope * mx
    yhat = slope * xs + intercept
    ss_res = ((ys - yhat)**2).sum()
    r2 = 1.0 - ss_res / syy
    return slope, intercept, r2


def thru_origin(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    sxx = (xs * xs).sum()
    if sxx == 0: return None
    beta = (xs * ys).sum() / sxx  # slope of y = beta * x
    yhat = beta * xs
    ss_res = ((ys - yhat)**2).sum()
    syy = ((ys - ys.mean())**2).sum()
    r2 = 1.0 - ss_res / syy if syy > 0 else float("nan")
    return beta, r2


print("=" * 92)
print("Per-grid dispersion audit (JHB CBD 25 grid, Vexcel 2024, clean_gt)")
print("=" * 92)

def bootstrap_ci(values: np.ndarray, statfn, n_boot: int = 500, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 3: return (float("nan"), float("nan"))
    samples = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statfn(values[s]) for s in samples])
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


stats_rows: list[dict] = []
for name in LAYERS:
    Bs = np.array([r["B_m2"] for r in rows], float)
    As = np.array([r[f"A_{name}"] for r in rows], float)
    inters = np.array([r[f"inter_{name}"] for r in rows], float)
    ratios = np.array([r[f"ratio_{name}"] for r in rows], float)
    eps = As - Bs                              # signed absolute residual, m²
    bulk = As.sum() / Bs.sum()                 # area-weighted mean ratio

    # Aggregate area R/P/F1 (sum-then-divide, the standard pixel-set-theoretic version)
    agg_R = inters.sum() / Bs.sum() if Bs.sum() > 0 else float("nan")
    agg_P = inters.sum() / As.sum() if As.sum() > 0 else float("nan")
    agg_F1 = 2 * agg_R * agg_P / (agg_R + agg_P) if (agg_R + agg_P) > 0 else float("nan")
    # Per-grid mean F1 (each grid 1 vote)
    pg_F1 = np.array([r[f"area_F1_{name}"] for r in rows], float)
    pg_F1 = pg_F1[~np.isnan(pg_F1)]
    mean_pg_F1 = float(pg_F1.mean()) if len(pg_F1) else float("nan")

    # Unweighted ratio dispersion (each grid = 1 point)
    mean_ratio = float(ratios.mean())
    std_ratio = float(ratios.std(ddof=1))
    cv_ratio = std_ratio / mean_ratio
    median_ratio = float(np.median(ratios))
    iqr_ratio = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    # B-weighted ratio dispersion (large-B grids dominate, the user's framing)
    w = Bs / Bs.sum()
    weighted_mean = float((w * ratios).sum())   # equals bulk
    weighted_var = float((w * (ratios - weighted_mean) ** 2).sum())
    std_ratio_w = float(np.sqrt(weighted_var))
    cv_ratio_w = std_ratio_w / weighted_mean

    # Log-ratio dispersion (each grid = 1 point, but log compresses small-B blow-ups)
    log_ratios = np.log(ratios)
    std_logratio = float(log_ratios.std(ddof=1))

    # Absolute residual stats — the inventory-prediction error budget
    rmse = float(np.sqrt((eps ** 2).mean()))    # m² per grid
    mae_abs = float(np.abs(eps).mean())
    sum_abs_eps = float(np.abs(eps).sum())
    bias = float(eps.mean())                    # signed
    # Total inventory error: |Σε| / Σ B (cancellation-aware)
    aggregate_signed_rel = float(eps.sum() / Bs.sum())

    sl, ic, r2 = ols(Bs.tolist(), As.tolist())
    beta_o, r2_o = thru_origin(Bs.tolist(), As.tolist())

    # Bootstrap CI for the dispersion metrics that drive deploy decisions
    sigma_unw_ci = bootstrap_ci(ratios, lambda v: float(v.std(ddof=1)))
    rmse_ci = bootstrap_ci(np.abs(eps), lambda v: float(np.sqrt((v ** 2).mean())))
    f1_ci = bootstrap_ci(pg_F1, lambda v: float(v.mean())) if len(pg_F1) else (float("nan"), float("nan"))

    stats_rows.append({
        "layer": name,
        # ---- Tier 1a: classic detection metrics (aggregate over all grids) ----
        "agg_area_R": round(agg_R, 4),
        "agg_area_P": round(agg_P, 4),
        "agg_area_F1": round(agg_F1, 4),
        "mean_per_grid_F1": round(mean_pg_F1, 4),
        "f1_pg_CI95_lo": round(f1_ci[0], 4),
        "f1_pg_CI95_hi": round(f1_ci[1], 4),
        # ---- Tier 1b: level + aggregate bias ----
        "bulk_AoverB": round(bulk, 4),
        "agg_signed_rel": round(aggregate_signed_rel, 4),
        # ---- Tier 1c: dispersion (paper-relevant) ----
        "std_ratio": round(std_ratio, 4),
        "std_ratio_CI95_lo": round(sigma_unw_ci[0], 4),
        "std_ratio_CI95_hi": round(sigma_unw_ci[1], 4),
        "std_ratio_Bw": round(std_ratio_w, 4),
        "cv_ratio_Bw": round(cv_ratio_w, 4),
        "std_logratio": round(std_logratio, 4),
        # ---- Tier 1d: absolute residuals ----
        "rmse_m2": round(rmse, 1),
        "rmse_CI95_lo": round(rmse_ci[0], 1),
        "rmse_CI95_hi": round(rmse_ci[1], 1),
        "mae_m2": round(mae_abs, 1),
        # ---- Tier 1e: regression diagnostic ----
        "ols_slope": round(sl, 4),
        "ols_intercept": round(ic, 2),
        "ols_R2": round(r2, 4),
        "thru0_slope": round(beta_o, 4),
        "thru0_R2": round(r2_o, 4),
        # Distributional details (kept for reference)
        "mean_ratio": round(mean_ratio, 4),
        "median_ratio": round(median_ratio, 4),
        "iqr_ratio": round(iqr_ratio, 4),
        "cv_ratio": round(cv_ratio, 4),
        "sum_abs_eps_m2": round(sum_abs_eps, 1),
        "bias_m2": round(bias, 1),
    })

w = 22
print(f"\n{'metric':<{w}}", *[f"{name:>30}" for name in LAYERS])
print("-" * 90)
print("# Tier 1a — classic detection (area pixel-set-theoretic)")
for k in ["agg_area_R","agg_area_P","agg_area_F1","mean_per_grid_F1",
          "f1_pg_CI95_lo","f1_pg_CI95_hi"]:
    print(f"{k:<{w}}", *[f"{s[k]:>30}" for s in stats_rows])
print("# Tier 1b — level / aggregate bias")
for k in ["bulk_AoverB","agg_signed_rel"]:
    print(f"{k:<{w}}", *[f"{s[k]:>30}" for s in stats_rows])
print("# Tier 1c — dispersion (paper-relevant)")
for k in ["std_ratio","std_ratio_CI95_lo","std_ratio_CI95_hi",
          "std_ratio_Bw","cv_ratio_Bw","std_logratio"]:
    print(f"{k:<{w}}", *[f"{s[k]:>30}" for s in stats_rows])
print("# Tier 1d — absolute residuals (inventory error)")
for k in ["rmse_m2","rmse_CI95_lo","rmse_CI95_hi","mae_m2"]:
    print(f"{k:<{w}}", *[f"{s[k]:>30}" for s in stats_rows])
print("# Tier 1e — calibration diagnostic")
for k in ["ols_slope","ols_intercept","ols_R2","thru0_slope","thru0_R2"]:
    print(f"{k:<{w}}", *[f"{s[k]:>30}" for s in stats_rows])

# ---------------------------------------------------------------------------
# CSV output
csv_path = OUTDIR / "per_grid.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n[saved] {csv_path}")

stats_path = OUTDIR / "summary.csv"
with stats_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
    writer.writeheader()
    writer.writerows(stats_rows)
print(f"[saved] {stats_path}")

# ---------------------------------------------------------------------------
# Scatter plots
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
colors = ["#1f77b4", "#d62728"]
for k, (name, _) in enumerate(LAYERS.items()):
    Bs = np.array([r["B_m2"] for r in rows])
    As = np.array([r[f"A_{name}"] for r in rows])
    sl, ic, r2 = ols(Bs, As)
    beta_o, r2_o = thru_origin(Bs, As)

    a = ax[k]
    a.scatter(Bs, As, s=60, alpha=0.75, color=colors[k], edgecolor="white", linewidth=0.5)
    for x, y, g in zip(Bs, As, [r["grid"] for r in rows]):
        a.annotate(g[1:], (x, y), fontsize=6.5, color="#333", xytext=(3, 3), textcoords="offset points")
    lim = max(Bs.max(), As.max()) * 1.05
    xs = np.linspace(0, lim, 100)
    a.plot(xs, xs, "--", color="black", lw=1, alpha=0.55, label="y = x (perfect)")
    a.plot(xs, sl * xs + ic, "-", color=colors[k], lw=1.4,
           label=f"OLS: y={sl:.2f}x+{ic:.0f}, R²={r2:.3f}")
    a.plot(xs, beta_o * xs, ":", color=colors[k], lw=1.2,
           label=f"thru-0: y={beta_o:.2f}x, R²={r2_o:.3f}")

    bulk = As.sum() / Bs.sum()
    mean_r = (As / Bs).mean(); std_r = (As / Bs).std(ddof=1)
    a.set_title(f"{name}\nbulk={bulk:.3f}, mean(A/B)={mean_r:.3f}, σ={std_r:.3f}",
                fontsize=11)
    a.set_xlabel("B_h = GT total area, m²")
    a.set_ylabel("A_h = predicted total area, m²")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="lower right")
    a.set_xlim(0, lim); a.set_ylim(0, lim)

plt.suptitle("JHB CBD 25-grid: per-grid pred vs GT area "
             "(clean_gt, Vexcel 2024)", fontsize=12, y=0.995)
plt.tight_layout()
out_png = OUTDIR / "scatter_pred_vs_gt.png"
plt.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"[saved] {out_png}")

# ---------------------------------------------------------------------------
# Per-grid ratio strip plot — visualize σ
fig2, ax2 = plt.subplots(figsize=(12, 5))
for k, (name, _) in enumerate(LAYERS.items()):
    rs = np.array([r[f"ratio_{name}"] for r in rows])
    xs = np.full_like(rs, k, dtype=float) + np.random.RandomState(0).uniform(-0.12, 0.12, len(rs))
    ax2.scatter(xs, rs, s=60, alpha=0.75, color=colors[k], edgecolor="white", linewidth=0.5)
    ax2.hlines(rs.mean(), k - 0.25, k + 0.25, colors=colors[k], lw=2, label=f"mean={rs.mean():.3f}")
    ax2.hlines(rs.mean() + rs.std(ddof=1), k - 0.18, k + 0.18,
               colors=colors[k], lw=1, linestyles="dashed")
    ax2.hlines(rs.mean() - rs.std(ddof=1), k - 0.18, k + 0.18,
               colors=colors[k], lw=1, linestyles="dashed")
ax2.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="A/B = 1")
ax2.set_xticks(range(len(LAYERS)))
ax2.set_xticklabels(list(LAYERS.keys()))
ax2.set_ylabel("A_h / B_h (per-grid ratio)")
ax2.set_title("Per-grid pred/GT ratio dispersion across 25 JHB CBD grids")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)
out_png2 = OUTDIR / "ratio_strip.png"
plt.savefig(out_png2, dpi=140, bbox_inches="tight")
print(f"[saved] {out_png2}")
