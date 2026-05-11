"""Fit hexagon-level kW = β·area calibration on Cape Town SSEG.

The economic paper's LHS is hexagon-level cumulative installed solar capacity
(kW) at H3 res-9. We hand them ∑(predicted polygon area, m²) per hexagon as a
proxy. This script fits the β that converts area → kW and reports R²,
residual structure, and residential / commercial subgroup behavior — the
metric tier (5) from the dispersion audit.

Inputs
------
- data/sseg_registration_geo.csv        SSEG registry, pre-indexed at H3 res-9
- results/cape_town/v3c_targeted_hn_aerial_2025/<grid>/predictions_metric.gpkg
- data/task_grid.gpkg                   coverage bbox per grid

Outputs (docs/experiments/sseg_kw_calibration_2026-05-10/)
- per_h3.csv                            per-hexagon (kW_truth, area_pred, n_*)
- summary.csv                           regression fits (overall, residential,
                                        commercial, log-log)
- scatter.png                           area vs kW scatter, fit line
- residual_diagnostic.png               residuals vs hexagon size & kW
"""
from __future__ import annotations

import csv
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path("/home/gaosh/projects/ZAsolar")
OUTDIR = ROOT / "docs/experiments/sseg_kw_calibration_2026-05-10"
OUTDIR.mkdir(parents=True, exist_ok=True)

CT_RUN_DIR = ROOT / "results/cape_town/v3c_targeted_hn_aerial_2025"
SSEG_CSV = ROOT / "data/sseg_registration_geo.csv"
TASK_GRID = ROOT / "data/task_grid.gpkg"

H3_RES = 9
WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:32734"  # CT region UTM
MAX_PLAUSIBLE_M2 = 20_000.0
COMMISSIONED_PREFIX = "1.0 Approved grid-tied installation commissioned"


# ---------------------------------------------------------------------------
# 1. SSEG → per-hexagon kW truth
print("=" * 90)
print("[1] Loading SSEG and aggregating per H3 res-9 hexagon...")
sseg = pd.read_csv(SSEG_CSV, low_memory=False)
print(f"    raw rows: {len(sseg)}")

mask = (
    sseg["status"].astype(str).str.startswith(COMMISSIONED_PREFIX)
    & sseg["generator_capacity_(va)"].notna()
    & (pd.to_numeric(sseg["generator_capacity_(va)"], errors="coerce") > 0)
    & sseg["h3_level_9"].notna()
    & sseg["latitude"].notna()
)
sseg_c = sseg.loc[mask].copy()
sseg_c["capacity_kw"] = pd.to_numeric(sseg_c["generator_capacity_(va)"]) / 1000.0  # VA ≈ W
print(f"    commissioned + capacity + h3 + loc: {len(sseg_c)}")

# Filter to date_commissioned ≤ 2025-06 (aerial 2025 mid-vintage)
def parse_year(s: str) -> int | None:
    try:
        return pd.to_datetime(s, errors="coerce").year
    except Exception:
        return None
sseg_c["commit_year"] = sseg_c["date_commissioned"].map(parse_year)
sseg_c = sseg_c[sseg_c["commit_year"].fillna(2099) <= 2025]
print(f"    after vintage filter (≤ 2025): {len(sseg_c)}")
print(f"    customer split: {sseg_c['customer_type'].value_counts().to_dict()}")

per_h3_truth = (
    sseg_c.groupby("h3_level_9")
    .agg(
        kw_total=("capacity_kw", "sum"),
        n_install=("capacity_kw", "size"),
        kw_residential=("capacity_kw", lambda s: s[sseg_c.loc[s.index, "customer_type"] == "Residential"].sum()),
        kw_commercial=("capacity_kw", lambda s: s[sseg_c.loc[s.index, "customer_type"].str.startswith("Commercial")].sum()),
        n_residential=("customer_type", lambda s: (s == "Residential").sum()),
        n_commercial=("customer_type", lambda s: s.str.startswith("Commercial").sum()),
    )
    .reset_index()
)
print(f"    {len(per_h3_truth)} hexagons hold ≥1 commissioned install")
print(f"    total kW: {per_h3_truth['kw_total'].sum():.0f}")


# ---------------------------------------------------------------------------
# 2. Predictions → per-hexagon area
print("\n[2] Aggregating CT predictions to H3 res-9 hexagons...")
grid_dirs = sorted(CT_RUN_DIR.iterdir())
grid_ids = [d.name for d in grid_dirs if d.is_dir() and d.name.startswith("G")]
print(f"    {len(grid_ids)} prediction grids found")

# Coverage bbox: union of the task-grid polygons for these grid IDs
task_grid_gdf = gpd.read_file(TASK_GRID)
covered = task_grid_gdf[task_grid_gdf["Name"].isin(grid_ids)]
if len(covered) == 0:
    raise SystemExit("No task-grid match for CT prediction grids.")
coverage_4326 = unary_union(covered.geometry.tolist())
print(f"    coverage area: {coverage_4326.area * 111e3 * 111e3 / 1e6:.1f} km² (rough)")

# For each hexagon centroid, check if inside coverage. Pre-compute hexagon
# centroids per H3 cell on demand.
def cell_centroid(h: str) -> tuple[float, float]:
    lat, lng = h3.cell_to_latlng(h)
    return lng, lat

from shapely.geometry import Point

per_h3_pred: dict[str, dict] = {}
for grid_id in grid_ids:
    pred_p = CT_RUN_DIR / grid_id / "predictions_metric.gpkg"
    if not pred_p.exists():
        continue
    g = gpd.read_file(pred_p)
    if g.empty:
        continue
    g = g[g.geometry.notna() & g.geometry.is_valid]
    # Compute area in metric CRS
    g_metric = g if str(g.crs) == METRIC_CRS else g.to_crs(METRIC_CRS)
    g_metric["area_m2"] = g_metric.geometry.area
    g_metric = g_metric[(g_metric["area_m2"] > 0) & (g_metric["area_m2"] <= MAX_PLAUSIBLE_M2)]
    if g_metric.empty:
        continue
    # Get centroids in WGS84 for H3 lookup
    g_4326 = g_metric.to_crs(WGS84)
    cent = g_4326.geometry.centroid
    for area_m2, c in zip(g_metric["area_m2"].tolist(), cent):
        h = h3.latlng_to_cell(c.y, c.x, H3_RES)
        rec = per_h3_pred.setdefault(h, {"area_m2": 0.0, "n_poly": 0})
        rec["area_m2"] += float(area_m2)
        rec["n_poly"] += 1
print(f"    {len(per_h3_pred)} hexagons with ≥1 prediction polygon")


# ---------------------------------------------------------------------------
# 3. Build per-hexagon table; restrict to hexagons whose centroid is inside coverage
print("\n[3] Joining truth + predictions, restricting to coverage...")
truth_idx = {h: row for h, row in zip(per_h3_truth["h3_level_9"], per_h3_truth.itertuples(index=False))}
all_h3 = set(truth_idx) | set(per_h3_pred)

rows = []
for h in all_h3:
    lng, lat = cell_centroid(h)
    if not coverage_4326.contains(Point(lng, lat)):
        continue
    t = truth_idx.get(h)
    p = per_h3_pred.get(h, {"area_m2": 0.0, "n_poly": 0})
    rows.append({
        "h3": h,
        "lat": lat, "lng": lng,
        "kw_total": float(t.kw_total) if t else 0.0,
        "kw_residential": float(t.kw_residential) if t else 0.0,
        "kw_commercial": float(t.kw_commercial) if t else 0.0,
        "n_install": int(t.n_install) if t else 0,
        "n_residential": int(t.n_residential) if t else 0,
        "n_commercial": int(t.n_commercial) if t else 0,
        "area_m2": p["area_m2"],
        "n_poly": p["n_poly"],
    })
df = pd.DataFrame(rows)
print(f"    {len(df)} hexagons inside coverage")
print(f"    with truth+pred: {(df['kw_total']>0).sum()} truth-positive, "
      f"{(df['area_m2']>0).sum()} pred-positive, "
      f"{((df['kw_total']>0)&(df['area_m2']>0)).sum()} both")

# Save per-hexagon CSV
out_csv = OUTDIR / "per_h3.csv"
df.to_csv(out_csv, index=False)
print(f"    [saved] {out_csv}")


# ---------------------------------------------------------------------------
# 4. Regression fits
def ols(x, y, with_intercept=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 2: return None
    if with_intercept:
        mx, my = x.mean(), y.mean()
        sxx = ((x - mx) ** 2).sum(); sxy = ((x - mx) * (y - my)).sum()
        syy = ((y - my) ** 2).sum()
        if sxx == 0 or syy == 0: return None
        slope = sxy / sxx
        intercept = my - slope * mx
    else:
        sxx = (x * x).sum()
        if sxx == 0: return None
        slope = (x * y).sum() / sxx
        intercept = 0.0
        syy = ((y - y.mean()) ** 2).sum()
    yhat = slope * x + intercept
    ss_res = ((y - yhat) ** 2).sum()
    r2 = 1.0 - ss_res / syy if syy > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / n))
    return {"slope": slope, "intercept": intercept, "r2": r2, "rmse": rmse, "n": n}


print("\n[4] Fitting kW = β · area calibrations...")

fits = []

def record(label, x, y, with_intercept=True, log=False):
    sub = pd.DataFrame({"x": x, "y": y}).query("x>0 and y>0") if log else pd.DataFrame({"x": x, "y": y})
    if log:
        sub["x"] = np.log(sub["x"]); sub["y"] = np.log(sub["y"])
    f = ols(sub["x"].values, sub["y"].values, with_intercept=with_intercept)
    if f is None:
        print(f"    {label}: insufficient data"); return
    print(f"    {label:<46} n={f['n']:>4d} slope={f['slope']:.5f} "
          f"intercept={f['intercept']:.3f} R²={f['r2']:.4f} RMSE={f['rmse']:.3f}")
    fits.append({"label": label, **f})

both_pos = df.query("kw_total > 0 and area_m2 > 0")
truth_pos = df.query("kw_total > 0")
pred_pos = df.query("area_m2 > 0")
print(f"    [data] both-pos hexagons: {len(both_pos)}")

record("kW ~ area, intercept (both>0)", both_pos["area_m2"], both_pos["kw_total"])
record("kW ~ area, thru-0 (both>0)", both_pos["area_m2"], both_pos["kw_total"], with_intercept=False)
record("log kW ~ log area (both>0)", both_pos["area_m2"], both_pos["kw_total"], log=True)
record("kW ~ area, intercept (truth>0)", truth_pos["area_m2"], truth_pos["kw_total"])

# Residential / commercial — drop hexagons with ANY commercial when fitting residential-only,
# else mixed-customer hexagons inflate residual scatter.
res_only = df.query("kw_residential > 0 and n_commercial == 0 and area_m2 > 0")
com_only = df.query("kw_commercial > 0 and n_residential == 0 and area_m2 > 0")
mixed = df.query("kw_residential > 0 and n_commercial >= 1 and area_m2 > 0")
record("residential-only hexagons", res_only["area_m2"], res_only["kw_residential"])
record("commercial-only hexagons", com_only["area_m2"], com_only["kw_commercial"])
record("mixed (resi+commercial)", mixed["area_m2"], mixed["kw_total"])

summary_df = pd.DataFrame(fits)
summary_csv = OUTDIR / "summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"    [saved] {summary_csv}")


# ---------------------------------------------------------------------------
# 5. Plots
print("\n[5] Rendering scatter + diagnostics...")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# (a) area vs kW with main fit
sub = df.query("kw_total > 0 and area_m2 > 0")
fit = ols(sub["area_m2"].values, sub["kw_total"].values)
ax[0].scatter(sub["area_m2"], sub["kw_total"], s=18, alpha=0.5,
              c="#1f77b4", edgecolor="white", linewidth=0.3)
xs = np.linspace(0, sub["area_m2"].max() * 1.05, 200)
ax[0].plot(xs, fit["slope"] * xs + fit["intercept"], "-",
           color="#d62728", lw=1.6,
           label=f"OLS: y={fit['slope']:.4f}x+{fit['intercept']:.2f}\n"
                 f"R²={fit['r2']:.3f}, n={fit['n']}")
fit0 = ols(sub["area_m2"].values, sub["kw_total"].values, with_intercept=False)
ax[0].plot(xs, fit0["slope"] * xs, ":", color="#d62728", lw=1.2,
           label=f"thru-0: y={fit0['slope']:.4f}x, R²={fit0['r2']:.3f}")
ax[0].set_xlabel("Predicted polygon area sum (m²) per H3 res-9 hexagon")
ax[0].set_ylabel("SSEG installed capacity (kW)")
ax[0].set_title("Area → kW calibration (Cape Town, V3-C aerial 2025)")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9, loc="upper left")

# (b) log-log
ax[1].scatter(sub["area_m2"], sub["kw_total"], s=18, alpha=0.5,
              c="#1f77b4", edgecolor="white", linewidth=0.3)
flog = ols(np.log(sub["area_m2"].values), np.log(sub["kw_total"].values))
xs2 = np.linspace(np.log(sub["area_m2"].min()), np.log(sub["area_m2"].max()), 200)
ax[1].plot(np.exp(xs2), np.exp(flog["slope"] * xs2 + flog["intercept"]), "-",
           color="#d62728", lw=1.6,
           label=f"log fit: log(y)={flog['slope']:.3f} log(x)+{flog['intercept']:.2f}\n"
                 f"R²={flog['r2']:.3f}")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("Pred area sum (m²)"); ax[1].set_ylabel("SSEG kW")
ax[1].set_title("Log-log calibration")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=9, loc="upper left")
plt.tight_layout()
out = OUTDIR / "scatter.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); print(f"    [saved] {out}")
plt.close(fig)

# (c) Residential vs commercial scatter
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(res_only["area_m2"], res_only["kw_residential"],
            s=18, alpha=0.5, c="#2ca02c", label=f"residential-only (n={len(res_only)})")
ax2.scatter(com_only["area_m2"], com_only["kw_commercial"],
            s=22, alpha=0.7, c="#d62728", marker="^",
            label=f"commercial-only (n={len(com_only)})")
ax2.scatter(mixed["area_m2"], mixed["kw_total"],
            s=18, alpha=0.5, c="#7f7f7f", label=f"mixed (n={len(mixed)})")
ax2.set_xlabel("Pred area sum (m²)"); ax2.set_ylabel("SSEG kW")
ax2.set_title("Calibration by customer-type composition")
ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
out2 = OUTDIR / "by_customer_type.png"
plt.savefig(out2, dpi=140, bbox_inches="tight"); print(f"    [saved] {out2}")
plt.close(fig2)

# (d) Residual diagnostic
fig3, ax3 = plt.subplots(1, 2, figsize=(13, 5))
resid = sub["kw_total"].values - (fit["slope"] * sub["area_m2"].values + fit["intercept"])
ax3[0].scatter(sub["area_m2"], resid, s=18, alpha=0.5, c="#9467bd")
ax3[0].axhline(0, color="black", linestyle="--", alpha=0.5)
ax3[0].set_xlabel("Pred area (m²)"); ax3[0].set_ylabel("Residual (kW)")
ax3[0].set_title("Residual vs hexagon size")
ax3[0].grid(alpha=0.3)

ax3[1].hist(resid, bins=50, color="#9467bd", alpha=0.7, edgecolor="white")
ax3[1].axvline(0, color="black", linestyle="--", alpha=0.5)
ax3[1].set_xlabel("Residual (kW)"); ax3[1].set_ylabel("count")
ax3[1].set_title(f"Residual distribution (mean={resid.mean():.2f}, σ={resid.std():.2f})")
ax3[1].grid(alpha=0.3)
plt.tight_layout()
out3 = OUTDIR / "residual_diagnostic.png"
plt.savefig(out3, dpi=140, bbox_inches="tight"); print(f"    [saved] {out3}")
plt.close(fig3)

# ---------------------------------------------------------------------------
# 6. Coverage diagnostic — how many SSEG kW are we actually capturing?
print("\n[6] Coverage diagnostics:")
total_truth_kw = df["kw_total"].sum()
covered_truth_kw = both_pos["kw_total"].sum()
truth_only_kw = (truth_pos[truth_pos["area_m2"] == 0]["kw_total"]).sum()
pred_only_area = (pred_pos[pred_pos["kw_total"] == 0]["area_m2"]).sum()
print(f"    total SSEG kW in coverage:        {total_truth_kw:.0f}")
print(f"    SSEG kW in hexagons with pred:    {covered_truth_kw:.0f} ({covered_truth_kw/total_truth_kw:.1%})")
print(f"    SSEG kW in hexagons without pred: {truth_only_kw:.0f} ({truth_only_kw/total_truth_kw:.1%})")
print(f"    pred area in pred-only hexagons:  {pred_only_area:.0f} m² (no SSEG → likely missing-truth or FP)")
