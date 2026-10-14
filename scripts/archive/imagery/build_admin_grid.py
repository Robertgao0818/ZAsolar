#!/usr/bin/env python3
"""Slice each city's admin polygon into 1 km grid cells, tagging each cell
with Vexcel coverage fraction and Overture building count.

Inputs (per region key from `configs/datasets/vexcel_urban_coverage.yaml`):
  - data/admin_boundaries/<region>.gpkg              (admin polygon)
  - data/vexcel_coverage/<region>_coverage.geojson   (Vexcel footprint)
  - data/buildings/overture/<region>.parquet         (Overture buildings)

Output:
  - data/admin_grids/<region>_admin_grid.gpkg
    Columns: gridcell_id, region_key, admin_grid_prefix, vexcel_coverage_fraction,
             n_buildings, area_sqm, coverage_fraction (admin), is_edge,
             imagery_plan ('vexcel_full'|'vexcel_partial'|'aerial_needed'|'skip'),
             centroid_x, centroid_y, lon, lat, geometry (EPSG:4326)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "vexcel_urban_coverage.yaml"
DEFAULT_ADMIN_DIR = PROJECT_ROOT / "data" / "admin_boundaries"
DEFAULT_VEXCEL_DIR = PROJECT_ROOT / "data" / "vexcel_coverage"
DEFAULT_BUILDINGS_DIR = PROJECT_ROOT / "data" / "buildings" / "overture"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "admin_grids"
WGS84 = "EPSG:4326"


def utm_epsg_for_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return f"EPSG:{epsg}"


def categorize(vexcel_fraction: float, n_buildings: int,
               *, vexcel_full_thresh: float, vexcel_partial_thresh: float,
               min_buildings: int) -> str:
    if vexcel_fraction >= vexcel_full_thresh:
        return "vexcel_full"
    if vexcel_fraction >= vexcel_partial_thresh:
        return "vexcel_partial"
    if n_buildings >= min_buildings:
        return "aerial_needed"
    return "skip"


def build_one(region_key: str, region: dict[str, Any], *,
              admin_dir: Path, vexcel_dir: Path, buildings_dir: Path,
              grid_size_m: int, min_admin_coverage: float,
              vexcel_full_thresh: float, vexcel_partial_thresh: float,
              min_buildings: int) -> gpd.GeoDataFrame:
    prefix = str(region["admin_grid_prefix"]).upper()

    admin = gpd.read_file(admin_dir / f"{region_key}.gpkg").to_crs(WGS84)
    if admin.empty:
        raise RuntimeError(f"{region_key}: empty admin polygon")
    admin_geom_wgs = admin.geometry.union_all()

    c = admin_geom_wgs.centroid
    metric_crs = utm_epsg_for_lonlat(c.x, c.y)
    admin_metric = admin.to_crs(metric_crs).geometry.union_all()

    # Vexcel coverage (optional)
    vexcel_path = vexcel_dir / f"{region_key}_coverage.geojson"
    if vexcel_path.exists():
        vex = gpd.read_file(vexcel_path).to_crs(metric_crs)
        vex_geom = vex.geometry.union_all()
    else:
        vex_geom = None

    # Buildings (optional)
    bld_path = buildings_dir / f"{region_key}.parquet"
    if bld_path.exists():
        bld = gpd.read_parquet(bld_path)
        if bld.crs is None:
            bld = bld.set_crs(WGS84)
        bld = bld.to_crs(metric_crs)
        # Use centroid-in-cell counting: faster than polygon-polygon intersect
        # and accurate enough at 1km grid scale (median building footprint ~10m).
        bld = bld[bld.geometry.notna() & (~bld.geometry.is_empty)].copy()
        bld_centroids = gpd.GeoDataFrame(geometry=bld.geometry.representative_point(), crs=metric_crs)
    else:
        bld_centroids = None

    # Grid extent in metric CRS
    xmin, ymin, xmax, ymax = admin_metric.bounds
    x0 = math.floor(xmin / grid_size_m) * grid_size_m
    y0 = math.floor(ymin / grid_size_m) * grid_size_m
    x1 = math.ceil(xmax / grid_size_m) * grid_size_m
    y1 = math.ceil(ymax / grid_size_m) * grid_size_m

    # Build candidate cells (N→S, W→E for stable IDs)
    full_area = float(grid_size_m * grid_size_m)
    cells: list[dict[str, Any]] = []
    y_values = list(range(int(y1 - grid_size_m), int(y0 - grid_size_m), -grid_size_m))
    x_values = list(range(int(x0), int(x1), grid_size_m))

    for y in y_values:
        for x in x_values:
            cell = box(x, y, x + grid_size_m, y + grid_size_m)
            clipped = cell.intersection(admin_metric)
            if clipped.is_empty:
                continue
            coverage_fraction = float(clipped.area / full_area)
            if coverage_fraction < min_admin_coverage:
                continue
            cells.append({
                "x": x, "y": y,
                "geom_metric": clipped,
                "cell_box_metric": cell,
                "coverage_fraction": coverage_fraction,
                "area_sqm": float(clipped.area),
                "is_edge": coverage_fraction < 0.999,
            })

    # Vexcel fraction per cell (intersect cell box with vex_geom, divide by full cell area)
    if vex_geom is not None:
        for c in cells:
            inter = c["cell_box_metric"].intersection(vex_geom)
            c["vexcel_fraction"] = float(inter.area / full_area) if not inter.is_empty else 0.0
    else:
        for c in cells:
            c["vexcel_fraction"] = 0.0

    # Building count per cell (sjoin centroids)
    if bld_centroids is not None and len(bld_centroids) > 0:
        cells_metric_gdf = gpd.GeoDataFrame(
            [{"_idx": i, "geometry": c["geom_metric"]} for i, c in enumerate(cells)],
            crs=metric_crs,
        )
        joined = gpd.sjoin(bld_centroids, cells_metric_gdf, predicate="within", how="inner")
        counts = joined.groupby("_idx").size().to_dict()
    else:
        counts = {}
    for i, c in enumerate(cells):
        c["n_buildings"] = int(counts.get(i, 0))

    # Build records with categorization + IDs
    records: list[dict[str, Any]] = []
    for seq, c in enumerate(cells, 1):
        grid_id = f"{prefix}{seq:04d}"
        cat = categorize(
            c["vexcel_fraction"], c["n_buildings"],
            vexcel_full_thresh=vexcel_full_thresh,
            vexcel_partial_thresh=vexcel_partial_thresh,
            min_buildings=min_buildings,
        )
        centroid = c["geom_metric"].centroid
        records.append({
            "gridcell_id": grid_id,
            "Name": grid_id,
            "region_key": region_key,
            "admin_grid_prefix": prefix,
            "crs_metric": metric_crs,
            "grid_size_m": grid_size_m,
            "area_sqm": c["area_sqm"],
            "coverage_fraction": c["coverage_fraction"],
            "is_edge": c["is_edge"],
            "vexcel_fraction": round(c["vexcel_fraction"], 4),
            "n_buildings": c["n_buildings"],
            "imagery_plan": cat,
            "centroid_x": float(centroid.x),
            "centroid_y": float(centroid.y),
            "geometry": c["geom_metric"],
        })

    metric_gdf = gpd.GeoDataFrame(records, crs=metric_crs)
    centroids_wgs = gpd.GeoSeries(metric_gdf.geometry.centroid, crs=metric_crs).to_crs(WGS84)
    gdf = metric_gdf.to_crs(WGS84)
    gdf["lon"] = centroids_wgs.x
    gdf["lat"] = centroids_wgs.y
    return gdf


def summarize(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    out = (
        gdf.groupby("imagery_plan")
        .agg(n_grid=("gridcell_id", "count"),
             total_area_km2=("area_sqm", lambda s: s.sum() / 1e6),
             total_buildings=("n_buildings", "sum"))
        .reset_index()
        .sort_values("imagery_plan")
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--admin-dir", type=Path, default=DEFAULT_ADMIN_DIR)
    parser.add_argument("--vexcel-dir", type=Path, default=DEFAULT_VEXCEL_DIR)
    parser.add_argument("--buildings-dir", type=Path, default=DEFAULT_BUILDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regions", nargs="*")
    parser.add_argument("--grid-size-m", type=int, default=1000)
    parser.add_argument("--min-admin-coverage", type=float, default=0.10,
                        help="Drop cells below this share of admin clip (1km^2)")
    parser.add_argument("--vexcel-full-thresh", type=float, default=0.99)
    parser.add_argument("--vexcel-partial-thresh", type=float, default=0.05)
    parser.add_argument("--min-buildings", type=int, default=50,
                        help="Aerial-fallback threshold (buildings per 1km^2 grid cell)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    selected = args.regions or list(config["regions"].keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key in selected:
        out_path = args.output_dir / f"{key}_admin_grid.gpkg"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} exists (use --overwrite)")
            continue

        print(f"[BUILD] {key}: 1km admin grid + Vexcel/Overture tags...")
        gdf = build_one(
            key, config["regions"][key],
            admin_dir=args.admin_dir,
            vexcel_dir=args.vexcel_dir,
            buildings_dir=args.buildings_dir,
            grid_size_m=args.grid_size_m,
            min_admin_coverage=args.min_admin_coverage,
            vexcel_full_thresh=args.vexcel_full_thresh,
            vexcel_partial_thresh=args.vexcel_partial_thresh,
            min_buildings=args.min_buildings,
        )
        gdf.to_file(out_path, driver="GPKG")
        print(f"  → wrote {out_path.relative_to(PROJECT_ROOT)}  n_cells={len(gdf)}")

        summary = summarize(gdf)
        print(summary.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
