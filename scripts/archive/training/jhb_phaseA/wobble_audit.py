"""Edge-wobble audit: SAM vs V3C-correct boundary roughness on JHB CBD 25 grid.

Wobble metric (Codex's proposal, replaces MRR fill which was confounded by
step-shaped trapezoid arrays):

    edge_wobble = perimeter(polygon) / perimeter(simplify(polygon, tol=1m))

Douglas-Peucker simplification at 1 m tolerance smooths out edge wobble
(which is < 1 m) but preserves real step structure (step heights ≥ 1 m).
So:
    ratio ≈ 1.00 => clean, axis-aligned edges
    ratio ≥ 1.20 => moderate wobble
    ratio ≥ 1.50 => heavy edge ragging

Compares two pools:
    SAM_added  -> all <grid>_sam_added.gpkg polygons
    V3C_corr   -> reviewed.gpkg with review_status=correct

Output: per-polygon CSV + per-bucket aggregate. Confirms whether SAM has
wobbly-but-correct edges and V3C has smooth-but-haloed edges, which would
support boundary_ignore as the right primitive for SAM and exclusion for V3C.

Usage:
    python scripts/training/jhb_phaseA/wobble_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REVIEW_ROOT = PROJECT_ROOT / "results/johannesburg/v3c_vexcel_2024_ch1_sample"
OUT_DIR = PROJECT_ROOT / "results/analysis/jhb_phaseA_prep"
SIMPLIFY_TOL_M = 1.0
TARGET_CRS = "EPSG:32735"

GRIDS = [
    "G0772","G0773","G0774","G0775","G0776","G0814","G0815","G0816","G0817",
    "G0818","G0853","G0854","G0855","G0856","G0857","G0888","G0889","G0890",
    "G0891","G0892","G0922","G0923","G0924","G0925","G0926",
]

AREA_BUCKETS = [
    ("<10",     0,    10),
    ("10-20",   10,   20),
    ("20-40",   20,   40),
    ("40-80",   40,   80),
    ("80-150",  80,   150),
    ("150-300", 150,  300),
    ("300-600", 300,  600),
    (">=600",   600,  1e9),
]


def _bucket(a):
    for lbl, lo, hi in AREA_BUCKETS:
        if lo <= a < hi:
            return lbl
    return ">=600"


def _wobble(geom) -> float:
    """perimeter / perimeter_simplified (DP tol = 1 m). 1.0 = clean."""
    if geom is None or geom.is_empty:
        return float("nan")
    p_raw = float(geom.length)
    if p_raw == 0:
        return float("nan")
    p_simp = float(geom.simplify(SIMPLIFY_TOL_M, preserve_topology=True).length)
    if p_simp == 0:
        return float("nan")
    return p_raw / p_simp


def load_sam(grid: str) -> gpd.GeoDataFrame:
    p = REVIEW_ROOT / grid / "review" / f"{grid}_sam_added.gpkg"
    if not p.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    g = gpd.read_file(p)
    if str(g.crs) != TARGET_CRS:
        g = g.to_crs(TARGET_CRS)
    return g


def load_v3c_correct(grid: str) -> gpd.GeoDataFrame:
    p = REVIEW_ROOT / grid / "review" / f"{grid}_reviewed.gpkg"
    if not p.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    g = gpd.read_file(p)
    if "review_status" in g.columns:
        g = g[g.review_status == "correct"].reset_index(drop=True)
    if str(g.crs) != TARGET_CRS:
        g = g.to_crs(TARGET_CRS)
    return g


def main():
    rows = []
    for grid in GRIDS:
        sam = load_sam(grid)
        v3c = load_v3c_correct(grid)
        for source, gdf in [("sam_added", sam), ("v3c_correct", v3c)]:
            if gdf.empty:
                continue
            for i, geom in enumerate(gdf.geometry):
                if geom is None or geom.is_empty:
                    continue
                a = float(geom.area)
                rows.append({
                    "grid": grid,
                    "source": source,
                    "idx": i,
                    "area_m2": a,
                    "perimeter_m": float(geom.length),
                    "wobble": _wobble(geom),
                    "bucket": _bucket(a),
                })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_poly = OUT_DIR / "wobble_per_polygon.csv"
    df.to_csv(per_poly, index=False)
    print(f"[SAVE] {per_poly} ({len(df)} rows)")

    # Aggregate by source x bucket
    agg = df.groupby(["source", "bucket"]).agg(
        n=("wobble", "size"),
        wobble_p25=("wobble", lambda s: float(np.nanpercentile(s, 25))),
        wobble_p50=("wobble", lambda s: float(np.nanpercentile(s, 50))),
        wobble_p75=("wobble", lambda s: float(np.nanpercentile(s, 75))),
        wobble_mean=("wobble", "mean"),
    ).reset_index()
    bucket_order = [lbl for lbl, _, _ in AREA_BUCKETS]
    agg["bucket"] = pd.Categorical(agg["bucket"], categories=bucket_order, ordered=True)
    agg = agg.sort_values(["source", "bucket"]).reset_index(drop=True)
    bucket_csv = OUT_DIR / "wobble_bucket_summary.csv"
    agg.to_csv(bucket_csv, index=False)
    print(f"[SAVE] {bucket_csv}")

    # Pretty-print head-to-head
    print("\n[BUCKET COMPARISON] (lower wobble = cleaner edge)")
    pivot = agg.pivot(index="bucket", columns="source",
                      values=["n", "wobble_p50"])
    print(pivot.to_string())

    # Global summary
    print("\n[GLOBAL]")
    for src in ("sam_added", "v3c_correct"):
        sub = df[df.source == src]
        n = len(sub)
        med = float(np.nanpercentile(sub.wobble, 50))
        p75 = float(np.nanpercentile(sub.wobble, 75))
        p90 = float(np.nanpercentile(sub.wobble, 90))
        print(f"  {src:13s}: n={n:5d}  wobble p50={med:.3f}  p75={p75:.3f}  p90={p90:.3f}")


if __name__ == "__main__":
    main()
