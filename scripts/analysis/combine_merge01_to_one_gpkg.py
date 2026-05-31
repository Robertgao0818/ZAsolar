"""Concatenate all 382-grid MERGE@0.1 + c=0.925 gpkg files into one
city-wide GeoPackage (and optional simplified GeoJSON)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METRIC_CRS = "EPSG:32735"
WGS84 = "EPSG:4326"

KEEP_COLS = [
    "source_grid", "confidence", "score", "area_m2",
    "orig_area_m2", "sam_score", "n_merged", "source_tile", "geometry",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path,
                   default=REPO / "results/johannesburg/unified_reviewall_A_perdet_sam_maskbox_vexcel_2024_full382_sam_maskbox")
    p.add_argument("--pred-name", default="predictions_metric_merge01_c0925.gpkg")
    p.add_argument("--out-gpkg", type=Path,
                   default=REPO / "results/analysis/full382_merge01_2026-05-15/jhb_full382_unified_A_merge01_c0925.gpkg")
    p.add_argument("--out-geojson", type=Path,
                   default=REPO / "results/analysis/full382_merge01_2026-05-15/jhb_full382_unified_A_merge01_c0925.geojson")
    p.add_argument("--no-geojson", action="store_true")
    args = p.parse_args()

    args.out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    grid_dirs = sorted([d for d in args.results_root.iterdir() if d.is_dir()])
    frames: list[gpd.GeoDataFrame] = []
    t0 = time.time()
    for i, d in enumerate(grid_dirs):
        f = d / args.pred_name
        if not f.exists():
            continue
        g = gpd.read_file(f)
        if len(g) == 0:
            continue
        g = g.to_crs(METRIC_CRS)
        g["source_grid"] = d.name
        keep = [c for c in KEEP_COLS if c in g.columns or c == "geometry"]
        # Make sure required cols exist (fill missing with NaN)
        for c in KEEP_COLS:
            if c not in g.columns and c != "geometry":
                g[c] = None
        g = g[KEEP_COLS]
        frames.append(g)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(grid_dirs)}] {d.name}: {len(g)} polys")
    print(f"  [{len(grid_dirs)}/{len(grid_dirs)}] read complete in {time.time()-t0:.1f}s")

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                 geometry="geometry", crs=METRIC_CRS)
    print(f"\n[combined] {len(combined)} polygons across {len(frames)} grids")
    print(f"[combined] total area: {combined.geometry.area.sum() / 1e4:.2f} ha = "
          f"{combined.geometry.area.sum():.0f} m²")
    print(f"[combined] score range: [{combined['confidence'].min():.3f}, "
          f"{combined['confidence'].max():.3f}]")

    if args.out_gpkg.exists():
        args.out_gpkg.unlink()
    print(f"[write] gpkg → {args.out_gpkg}")
    t0 = time.time()
    combined.to_file(args.out_gpkg, driver="GPKG", layer="solar_predictions")
    sz = args.out_gpkg.stat().st_size / (1024**2)
    print(f"[write] gpkg done in {time.time()-t0:.1f}s, size {sz:.1f} MiB")

    if not args.no_geojson:
        combined_wgs = combined.to_crs(WGS84)
        if args.out_geojson.exists():
            args.out_geojson.unlink()
        print(f"[write] geojson (WGS84) → {args.out_geojson}")
        t0 = time.time()
        combined_wgs.to_file(args.out_geojson, driver="GeoJSON")
        sz = args.out_geojson.stat().st_size / (1024**2)
        print(f"[write] geojson done in {time.time()-t0:.1f}s, size {sz:.1f} MiB")


if __name__ == "__main__":
    main()
