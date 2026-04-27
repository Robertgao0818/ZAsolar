#!/usr/bin/env python3
"""Build task-grid GeoPackages for Vexcel South Africa urban coverage.

The grid is generated in a local UTM CRS as 1 km cells, clipped to each
Vexcel coverage bbox, and written back to EPSG:4326 for pipeline use.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "vexcel_task_grids"
WGS84 = "EPSG:4326"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utm_epsg_for_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return f"EPSG:{epsg}"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "regions" not in data:
        raise ValueError(f"{path} must contain a top-level 'regions' mapping")
    return data


def load_coverage_geometry(region_key: str, region: dict[str, Any]) -> gpd.GeoDataFrame:
    coverage_path = region.get("coverage_path")
    if coverage_path:
        path = PROJECT_ROOT / coverage_path
        if path.exists():
            gdf = gpd.read_file(path).to_crs(WGS84)
            if gdf.empty:
                raise RuntimeError(f"Coverage file is empty for {region_key}: {path}")
            return gdf[["geometry"]].copy()

    lon_min, lat_min, lon_max, lat_max = [float(v) for v in region["bbox"]]
    return gpd.GeoDataFrame(
        [{"geometry": box(lon_min, lat_min, lon_max, lat_max)}],
        crs=WGS84,
    )


def build_region_grid(
    region_key: str,
    region: dict[str, Any],
    *,
    grid_size_m: int,
    min_coverage_fraction: float,
) -> gpd.GeoDataFrame:
    coverage_wgs = load_coverage_geometry(region_key, region)
    lon_min, lat_min, lon_max, lat_max = coverage_wgs.total_bounds
    lon_mid = float((lon_min + lon_max) / 2.0)
    lat_mid = float((lat_min + lat_max) / 2.0)
    metric_crs = region.get("crs_metric") or utm_epsg_for_lonlat(lon_mid, lat_mid)

    coverage_metric = coverage_wgs.to_crs(metric_crs)
    coverage_geom = coverage_metric.geometry.union_all()
    xmin, ymin, xmax, ymax = coverage_geom.bounds

    x0 = math.floor(xmin / grid_size_m) * grid_size_m
    y0 = math.floor(ymin / grid_size_m) * grid_size_m
    x1 = math.ceil(xmax / grid_size_m) * grid_size_m
    y1 = math.ceil(ymax / grid_size_m) * grid_size_m

    records: list[dict[str, Any]] = []
    prefix = str(region["grid_prefix"]).upper()
    seq = 1

    # Sort north-to-south, then west-to-east for stable human-readable IDs.
    y_values = list(range(int(y1 - grid_size_m), int(y0 - grid_size_m), -grid_size_m))
    x_values = list(range(int(x0), int(x1), grid_size_m))
    full_area = float(grid_size_m * grid_size_m)

    for y in y_values:
        for x in x_values:
            cell = box(x, y, x + grid_size_m, y + grid_size_m)
            clipped = cell.intersection(coverage_geom)
            if clipped.is_empty:
                continue
            coverage_fraction = float(clipped.area / full_area)
            if coverage_fraction < min_coverage_fraction:
                continue

            grid_id = f"{prefix}{seq:04d}"
            centroid = clipped.centroid
            records.append(
                {
                    "gridcell_id": grid_id,
                    "Name": grid_id,
                    "region_key": region_key,
                    "city": region["city"],
                    "province": region.get("province"),
                    "collection_id": region["collection"],
                    "product": region["product"],
                    "capture_start": region["capture_start"],
                    "capture_end": region["capture_end"],
                    "avg_gsd_m": float(region["avg_gsd_m"]),
                    "coverage_area_km2": float(region["coverage_area_km2"]),
                    "source_bbox": ",".join(str(v) for v in region["bbox"]),
                    "coverage_source": region.get("coverage_path", "bbox"),
                    "crs_metric": metric_crs,
                    "grid_size_m": grid_size_m,
                    "area_sqm": float(clipped.area),
                    "coverage_fraction": coverage_fraction,
                    "is_edge": coverage_fraction < 0.999,
                    "centroid_x": float(centroid.x),
                    "centroid_y": float(centroid.y),
                    "geometry": clipped,
                }
            )
            seq += 1

    if not records:
        raise RuntimeError(f"No task grid cells generated for {region_key}")

    metric_gdf = gpd.GeoDataFrame(records, crs=metric_crs)
    centroids_wgs = gpd.GeoSeries(metric_gdf.geometry.centroid, crs=metric_crs).to_crs(WGS84)
    gdf = metric_gdf.to_crs(WGS84)
    gdf["lon"] = centroids_wgs.x
    gdf["lat"] = centroids_wgs.y
    return gdf


def write_outputs(frames: dict[str, gpd.GeoDataFrame], output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for region_key, gdf in frames.items():
        out_path = output_dir / f"{region_key}_task_grid.gpkg"
        geojson_path = output_dir / f"{region_key}_task_grid.geojson"
        for path in (out_path, geojson_path):
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            if path.exists():
                path.unlink()
        gdf.to_file(out_path, driver="GPKG")
        gdf.to_file(geojson_path, driver="GeoJSON")
        summary_rows.append(
            {
                "region_key": region_key,
                "grid_count": len(gdf),
                "fullish_grid_count": int((gdf["coverage_fraction"] >= 0.99).sum()),
                "area_km2_generated": round(float(gdf["area_sqm"].sum()) / 1_000_000.0, 3),
                "task_grid": display_path(out_path),
                "task_grid_geojson": display_path(geojson_path),
            }
        )
        print(
            f"{region_key}: wrote {len(gdf)} grids to "
            f"{display_path(out_path)}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("region_key")
    summary_path = output_dir / "vexcel_task_grid_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"summary: {display_path(summary_path)}")

    combined = gpd.GeoDataFrame(pd.concat(frames.values(), ignore_index=True), crs=WGS84)
    combined_path = output_dir / "vexcel_task_grids.geojson"
    if combined_path.exists() and not overwrite:
        raise FileExistsError(f"{combined_path} exists; pass --overwrite to replace it")
    if combined_path.exists():
        combined_path.unlink()
    combined.to_file(combined_path, driver="GeoJSON")
    print(f"combined geojson: {display_path(combined_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size-m", type=int, default=None)
    parser.add_argument(
        "--min-coverage-fraction",
        type=float,
        default=0.25,
        help="Drop tiny edge slivers below this share of a 1 km grid.",
    )
    parser.add_argument("--regions", nargs="*", help="Optional subset of region keys")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    grid_size_m = int(args.grid_size_m or config.get("grid_size_m", 1000))
    selected = set(args.regions or config["regions"].keys())

    unknown = selected.difference(config["regions"].keys())
    if unknown:
        raise KeyError(f"Unknown region key(s): {sorted(unknown)}")

    frames = {
        key: build_region_grid(
            key,
            region,
            grid_size_m=grid_size_m,
            min_coverage_fraction=args.min_coverage_fraction,
        )
        for key, region in config["regions"].items()
        if key in selected
    }
    write_outputs(frames, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
