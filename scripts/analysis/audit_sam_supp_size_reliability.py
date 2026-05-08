#!/usr/bin/env python3
"""Per-area-bucket reliability audit: V3C-reviewed-correct vs human SAM-added.

Goal: find an area threshold above which SAM-added polygons stop being
trustworthy enough for strict pixel-BCE mask supervision (compared to
V3C-reviewed-correct, which has known halo bias).

Inputs (per grid in JHB CBD 25):
  results/johannesburg/v3c_vexcel_2024_ch1_sample/<grid>/review/<grid>_reviewed.gpkg
    source = NaN          -> V3C prediction (review_status in correct/edit/delete)
    source = sam_fn_review -> human-drawn FN via SAM (all review_status=correct)

Outputs:
  results/analysis/sam_supp_audit/per_polygon.csv
  results/analysis/sam_supp_audit/bucket_summary.csv
  results/analysis/sam_supp_audit/scatter_area_vs_mrr_fill.png
  results/analysis/sam_supp_audit/dist_by_bucket.png

Pools:
  V3C  = reviewed.gpkg rows with source NaN AND review_status=='correct'
  SAM  = reviewed.gpkg rows with source=='sam_fn_review' (all correct by construction)

Reliability metrics (per polygon):
  mrr_fill   = area / minimum_rotated_rectangle_area     (1.0 = perfect rect)
  compact    = 4*pi*area / perimeter^2                   (0.785 = square)
  solidity   = area / convex_hull_area                   (precomputed in input)

Lower values on any of these in the upper area buckets => degraded reliability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_ROOT = PROJECT_ROOT / "results" / "johannesburg" / "v3c_vexcel_2024_ch1_sample"
OUT_ROOT = PROJECT_ROOT / "results" / "analysis" / "sam_supp_audit"
TARGET_CRS = "EPSG:32735"

GRIDS_25 = [
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
]

AREA_BUCKETS = [0, 10, 20, 40, 80, 150, 300, 600, np.inf]
BUCKET_LABELS = ["<10", "10-20", "20-40", "40-80", "80-150", "150-300", "300-600", "≥600"]


def load_grid(grid: str) -> gpd.GeoDataFrame:
    p = REVIEW_ROOT / grid / "review" / f"{grid}_reviewed.gpkg"
    if not p.exists():
        return gpd.GeoDataFrame()
    g = gpd.read_file(p)
    if g.crs is None or str(g.crs) != TARGET_CRS:
        g = g.to_crs(TARGET_CRS)
    g["grid"] = grid
    return g


def assign_pool(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src = g["source"].fillna("v3c_pred")
    pool = np.where(src == "sam_fn_review", "SAM_added",
                    np.where((src == "v3c_pred") & (g["review_status"] == "correct"),
                             "V3C_correct", None))
    g = g.assign(pool=pool)
    return g[g["pool"].notna()].reset_index(drop=True)


def compute_metrics(g: gpd.GeoDataFrame) -> pd.DataFrame:
    geom = g.geometry
    # MRR fill = area / minimum_rotated_rectangle_area (metric CRS, m^2)
    mrr_area = geom.minimum_rotated_rectangle().area.replace(0, np.nan)
    mrr_fill = (geom.area / mrr_area).clip(0, 1)
    perim = geom.length.replace(0, np.nan)
    compact = (4 * np.pi * geom.area / (perim ** 2)).clip(0, 1)
    out = pd.DataFrame({
        "grid": g["grid"].values,
        "pool": g["pool"].values,
        "area_m2": geom.area.values,
        "perimeter_m": perim.values,
        "mrr_fill": mrr_fill.values,
        "compactness": compact.values,
        "solidity": g["solidity"].values if "solidity" in g.columns else np.nan,
        "elongation": g["elongation"].values if "elongation" in g.columns else np.nan,
    })
    return out


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["area_bucket"] = pd.cut(df["area_m2"], bins=AREA_BUCKETS, labels=BUCKET_LABELS,
                               include_lowest=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bucket, pool), sub in df.groupby(["area_bucket", "pool"], observed=True):
        rows.append({
            "area_bucket": bucket,
            "pool": pool,
            "n": len(sub),
            "mrr_fill_p10": sub["mrr_fill"].quantile(0.10),
            "mrr_fill_p50": sub["mrr_fill"].median(),
            "mrr_fill_p90": sub["mrr_fill"].quantile(0.90),
            "mrr_fill_mean": sub["mrr_fill"].mean(),
            "compact_p50": sub["compactness"].median(),
            "solidity_p50": sub["solidity"].median(),
            "solidity_p10": sub["solidity"].quantile(0.10),
        })
    return pd.DataFrame(rows).sort_values(["area_bucket", "pool"])


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for pool, color, marker in [("V3C_correct", "#1f77b4", "o"),
                                 ("SAM_added", "#d62728", "^")]:
        sub = df[df["pool"] == pool]
        ax.scatter(sub["area_m2"], sub["mrr_fill"], s=8, alpha=0.35,
                   c=color, marker=marker, label=f"{pool} (n={len(sub)})")
    ax.set_xscale("log")
    ax.set_xlabel("Polygon area (m²) — log scale")
    ax.set_ylabel("MRR fill = area / minimum_rotated_rectangle_area")
    ax.set_title("Shape quality vs polygon area, JHB CBD 25 grid")
    ax.axhline(0.85, color="gray", ls="--", lw=0.8, label="MRR fill = 0.85")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_area_vs_mrr_fill.png", dpi=140)
    plt.close(fig)

    # Bucketed boxplot of MRR fill
    df_b = bucketize(df).dropna(subset=["area_bucket"])
    fig, ax = plt.subplots(figsize=(11, 5))
    bucket_order = [b for b in BUCKET_LABELS if b in df_b["area_bucket"].cat.categories]
    width = 0.38
    positions_v3c = np.arange(len(bucket_order)) - width / 2
    positions_sam = np.arange(len(bucket_order)) + width / 2

    def vals(pool):
        return [df_b[(df_b["area_bucket"] == b) & (df_b["pool"] == pool)]["mrr_fill"].dropna().values
                for b in bucket_order]

    bp1 = ax.boxplot(vals("V3C_correct"), positions=positions_v3c, widths=width,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(vals("SAM_added"), positions=positions_sam, widths=width,
                     patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#1f77b4"); patch.set_alpha(0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#d62728"); patch.set_alpha(0.6)

    ax.set_xticks(np.arange(len(bucket_order)))
    ax.set_xticklabels(bucket_order)
    ax.set_xlabel("area bucket (m²)")
    ax.set_ylabel("MRR fill")
    ax.set_title("MRR fill per area bucket — V3C correct (blue) vs SAM added (red)")
    ax.axhline(0.85, color="gray", ls="--", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "dist_by_bucket.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for grid in GRIDS_25:
        g = load_grid(grid)
        if g.empty:
            print(f"[skip] {grid}: missing reviewed.gpkg")
            continue
        g = assign_pool(g)
        parts.append(compute_metrics(g))
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["mrr_fill", "area_m2"])
    df.to_csv(args.out_dir / "per_polygon.csv", index=False)

    df_b = bucketize(df)
    summary = summarize(df_b)
    summary.to_csv(args.out_dir / "bucket_summary.csv", index=False)

    make_plots(df, args.out_dir)

    print("=== Pool sizes ===")
    print(df["pool"].value_counts().to_string())
    print("\n=== Bucket summary (mrr_fill) ===")
    pivot = summary.pivot_table(
        index="area_bucket", columns="pool",
        values=["n", "mrr_fill_p50", "mrr_fill_p10"], observed=True)
    print(pivot.to_string())
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
