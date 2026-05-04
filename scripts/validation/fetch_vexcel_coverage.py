#!/usr/bin/env python3
"""Fetch Vexcel collection coverage footprints from API 2.0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "vexcel_urban_coverage.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "vexcel_coverage"
DEFAULT_ENV = PROJECT_ROOT / ".env"

sys.path.insert(0, str(PROJECT_ROOT))
from core.vexcel_auth import load_env, resolve_token  # noqa: E402


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "regions" not in data:
        raise ValueError(f"{path} must contain a top-level 'regions' mapping")
    return data


def request_collection(base_url: str, token: str, collection: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/ortho/collections"
    response = requests.get(
        url,
        params={
            "collection": collection,
            "metadata-format": "JSON",
            "srid": 4326,
            "token": token,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("type") != "FeatureCollection":
        raise RuntimeError(f"Unexpected response for {collection}: {data!r}")
    if not data.get("features"):
        raise RuntimeError(f"No coverage features returned for {collection}")
    return data


def feature_collection_to_gdf(
    data: dict[str, Any],
    *,
    region_key: str,
    region: dict[str, Any],
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
    gdf["region_key"] = region_key
    gdf["city"] = region["city"]
    gdf["province"] = region.get("province")
    gdf["configured_collection"] = region["collection"]
    gdf["configured_product"] = region["product"]
    gdf["configured_area_km2"] = float(region["coverage_area_km2"])
    gdf["configured_avg_gsd_m"] = float(region["avg_gsd_m"])
    gdf["configured_capture_start"] = region["capture_start"]
    gdf["configured_capture_end"] = region["capture_end"]
    return gdf


def write_geojson(gdf: gpd.GeoDataFrame, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GeoJSON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regions", nargs="*", help="Optional subset of region keys")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    env = load_env(args.env_file)
    base_url = env.get("VEXCEL_API_BASE", "https://api.vexcelgroup.com/v2")
    token = resolve_token(env, base_url)

    config = load_config(args.config)
    selected = set(args.regions or config["regions"].keys())
    unknown = selected.difference(config["regions"].keys())
    if unknown:
        raise KeyError(f"Unknown region key(s): {sorted(unknown)}")

    frames: list[gpd.GeoDataFrame] = []
    for region_key, region in config["regions"].items():
        if region_key not in selected:
            continue
        data = request_collection(base_url, token, region["collection"])
        gdf = feature_collection_to_gdf(data, region_key=region_key, region=region)

        configured_path = region.get("coverage_path")
        if configured_path:
            out_path = PROJECT_ROOT / configured_path
        else:
            out_path = args.output_dir / f"{region_key}_coverage.geojson"
        write_geojson(gdf, out_path, overwrite=args.overwrite)

        metric = gdf.to_crs(region.get("crs_metric") or "EPSG:3857")
        area_km2 = float(metric.geometry.area.sum()) / 1_000_000.0
        print(
            f"{region_key}: features={len(gdf)} area_km2={area_km2:.1f} "
            f"wrote {display_path(out_path)}"
        )
        frames.append(gdf)

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    combined_path = args.output_dir / "vexcel_coverage_footprints.geojson"
    write_geojson(combined, combined_path, overwrite=args.overwrite)
    print(f"combined: {display_path(combined_path)}")


if __name__ == "__main__":
    main()
