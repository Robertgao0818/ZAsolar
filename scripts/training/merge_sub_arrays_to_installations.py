#!/usr/bin/env python3
"""Spatial-merge sub-array clean GT polygons into installation-level GT.

Background:  ``data/annotations_channel2_clean/`` was built for Channel 2
exhaustive-recall evaluation, so it carries SAM_supp补标 at sub-array
granularity (median ~27 m² per polygon, 80% of the corpus). The
``train20_val5_hn`` experiment trained a detector against this GT directly
and the model's output statistics shifted to sub-array — which under
pixel-or finalize collapses into over-merged envelopes (Ch3 +49pp area
over-count) while under per-detection multiplies polygons (Ch2 polygon F1
diverges from area_F1).  See
``docs/experiments/exp_train20_val5_hn_negative_result.md``.

This script produces an installation-level GT for **training** (evaluation
GT stays as-is). Algorithm:

  1. Buffer each clean-GT polygon by ``--buffer-m`` metres (default 3 m).
  2. Connected-component union: any two polygons whose buffered geometries
     intersect collapse to one cluster.
  3. Output: one polygon per cluster (un-buffered ``unary_union`` of the
     originals), with provenance columns recording component count and
     source-mix.

A polygon already at installation granularity (an isolated small array on a
residential roof) emerges as its own one-component cluster — the merge does
not inflate it. Only polygons that share a roof and sit within
``--buffer-m`` of each other collapse.

Default buffer (3 m) was picked on a 5-grid prototype:
  G0816: 111 → 26 clusters,  median 60 → 164 m²
  G0817:  90 → 17 clusters,  median 70 → 64 m²
  G0925:  79 → 19 clusters,  median 60 → 179 m²
  G0772:  34 → 20 clusters,  median 31 → 43 m²  (already coarse)
  G0814: 136 → 53 clusters,  median 24 → 32 m²  (residential, mostly small)

Usage:
  python scripts/training/merge_sub_arrays_to_installations.py \\
    --src-root data/annotations_channel2_clean \\
    --out-root data/annotations_channel2_clean_merged_buf3m \\
    --buffer-m 3.0 \\
    --report-csv data/annotations_channel2_clean_merged_buf3m/merge_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "data" / "annotations_channel2_clean"
TARGET_CRS = "EPSG:32735"  # JHB metric


def merge_polygons(gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Buffer-then-connected-component union.

    Returns one row per cluster with:
      cluster_id, n_subarrays, source_mix, area_m2, geometry, components
    """
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=[
                "cluster_id",
                "n_subarrays",
                "source_mix",
                "area_m2",
                "components",
                "geometry",
            ],
            geometry="geometry",
            crs=gdf.crs or TARGET_CRS,
        )

    gdf = gdf.reset_index(drop=True)
    bufs = gdf.geometry.buffer(buffer_m)
    n = len(gdf)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sidx = bufs.sindex
    for i in range(n):
        for j in sidx.intersection(bufs.iloc[i].bounds):
            if j > i and bufs.iloc[i].intersects(bufs.iloc[j]):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    rows: list[dict] = []
    cid = 0
    for ids in groups.values():
        merged = unary_union([gdf.geometry.iloc[i] for i in ids])
        if merged.is_empty:
            continue
        sources = sorted(
            set(str(gdf["source"].iloc[i]) for i in ids if "source" in gdf.columns)
        )
        comp_ids = []
        if "annotation_id" in gdf.columns:
            comp_ids = [str(gdf["annotation_id"].iloc[i]) for i in ids]
        elif "clean_id" in gdf.columns:
            comp_ids = [str(gdf["clean_id"].iloc[i]) for i in ids]
        rows.append(
            {
                "cluster_id": cid,
                "n_subarrays": len(ids),
                "source_mix": ";".join(sources) if sources else "",
                "area_m2": float(merged.area),
                "components": ";".join(comp_ids),
                "geometry": merged,
            }
        )
        cid += 1

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def process_grid(
    src_path: Path,
    out_path: Path,
    buffer_m: float,
) -> dict | None:
    if not src_path.exists():
        return None
    src = gpd.read_file(src_path)
    if src.crs is None or str(src.crs) != TARGET_CRS:
        src = src.to_crs(TARGET_CRS) if src.crs else src.set_crs(TARGET_CRS)

    merged = merge_polygons(src, buffer_m=buffer_m)
    grid = src_path.parent.name
    merged["grid"] = grid
    merged["annotation_id"] = [f"{grid}_M_{i}" for i in range(len(merged))]
    merged["label"] = "pv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(out_path, driver="GPKG")

    return {
        "grid": grid,
        "n_input": len(src),
        "n_clusters": len(merged),
        "input_total_m2": float(src.geometry.area.sum()) if len(src) else 0.0,
        "merged_total_m2": float(merged["area_m2"].sum()) if len(merged) else 0.0,
        "input_median_m2": float(src.geometry.area.median()) if len(src) else 0.0,
        "merged_median_m2": float(merged["area_m2"].median()) if len(merged) else 0.0,
        "input_max_m2": float(src.geometry.area.max()) if len(src) else 0.0,
        "merged_max_m2": float(merged["area_m2"].max()) if len(merged) else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--buffer-m", type=float, default=3.0)
    p.add_argument(
        "--grids",
        nargs="*",
        default=None,
        help="Optional subset; default = every G* subdir under --src-root",
    )
    p.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional path for per-grid summary CSV (created next to outputs by default)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.grids:
        grid_dirs = [args.src_root / g for g in args.grids]
    else:
        grid_dirs = sorted(d for d in args.src_root.iterdir() if d.is_dir() and d.name.startswith("G"))

    rows: list[dict] = []
    print(f"Merging {len(grid_dirs)} grid(s) with buffer={args.buffer_m} m → {args.out_root}")
    for d in grid_dirs:
        grid = d.name
        src = d / f"{grid}_clean_gt.gpkg"
        out = args.out_root / grid / f"{grid}_merged_gt.gpkg"
        if args.dry_run:
            if not src.exists():
                print(f"  [skip] {grid}: missing source")
                continue
            sg = gpd.read_file(src)
            if sg.crs is None or str(sg.crs) != TARGET_CRS:
                sg = sg.to_crs(TARGET_CRS) if sg.crs else sg.set_crs(TARGET_CRS)
            merged = merge_polygons(sg, args.buffer_m)
            row = {
                "grid": grid,
                "n_input": len(sg),
                "n_clusters": len(merged),
                "input_total_m2": float(sg.geometry.area.sum()) if len(sg) else 0.0,
                "merged_total_m2": float(merged["area_m2"].sum()) if len(merged) else 0.0,
                "input_median_m2": float(sg.geometry.area.median()) if len(sg) else 0.0,
                "merged_median_m2": float(merged["area_m2"].median()) if len(merged) else 0.0,
                "input_max_m2": float(sg.geometry.area.max()) if len(sg) else 0.0,
                "merged_max_m2": float(merged["area_m2"].max()) if len(merged) else 0.0,
            }
        else:
            row = process_grid(src, out, args.buffer_m)
            if row is None:
                print(f"  [skip] {grid}: missing source")
                continue
        rows.append(row)
        print(
            f"  {grid}: {row['n_input']:4d} → {row['n_clusters']:3d}  "
            f"(median {row['input_median_m2']:6.1f} → {row['merged_median_m2']:6.1f} m², "
            f"sum {row['input_total_m2']:8.0f} → {row['merged_total_m2']:8.0f} m²)"
        )

    if not rows:
        print("[FAIL] no grids processed")
        return 1

    df = pd.DataFrame(rows)
    report_path = (
        args.report_csv
        if args.report_csv is not None
        else args.out_root / "merge_summary.csv"
    )
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)
        print(f"\n[ok] per-grid summary → {report_path}")

    print()
    print(
        f"TOTAL: {df['n_input'].sum()} polygons → {df['n_clusters'].sum()} clusters "
        f"(reduction {1 - df['n_clusters'].sum() / df['n_input'].sum():.1%})"
    )
    print(
        f"  area sum  preserved: input {df['input_total_m2'].sum():.0f} m² → "
        f"merged {df['merged_total_m2'].sum():.0f} m² "
        f"(diff {(df['merged_total_m2'].sum() - df['input_total_m2'].sum()):+.0f} from buffer overlap)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
