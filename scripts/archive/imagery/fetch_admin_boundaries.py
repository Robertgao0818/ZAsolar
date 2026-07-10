#!/usr/bin/env python3
"""Fetch OSM admin-area polygons for the Vexcel-covered SA cities.

Reads `configs/datasets/vexcel_urban_coverage.yaml` for each region's
`admin_name` + `admin_country` and queries Nominatim to download the
official admin polygon. Output: `data/admin_boundaries/<region_key>.gpkg`
in EPSG:4326 with a province area summary printed at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
import yaml
from shapely.geometry import shape


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "vexcel_urban_coverage.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "admin_boundaries"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ZAsolar/0.1 (research; taoyu.chen@sciencespo.fr)"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def query_nominatim(name: str, country: str) -> dict[str, Any]:
    params = {
        "q": f"{name}, {country}",
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 5,
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    results = r.json()
    if not results:
        raise RuntimeError(f"Nominatim returned 0 results for '{name}, {country}'")
    # Prefer relation hits with polygon geometry
    relations = [x for x in results if x.get("osm_type") == "relation" and "geojson" in x]
    if not relations:
        raise RuntimeError(f"No relation polygons in Nominatim results for '{name}'")
    return relations[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regions", nargs="*", help="Subset of region keys")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    regions = config["regions"]
    selected = args.regions or list(regions.keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for key in selected:
        region = regions[key]
        admin_name = region.get("admin_name")
        country = region.get("admin_country", "South Africa")
        if not admin_name:
            print(f"[SKIP] {key}: no admin_name in config")
            continue

        out_path = args.output_dir / f"{key}.gpkg"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} exists (use --overwrite)")
            continue

        print(f"[FETCH] {key}: {admin_name}")
        time.sleep(1.1)  # respect Nominatim 1 req/sec
        result = query_nominatim(admin_name, country)
        geom = shape(result["geojson"])
        gdf = gpd.GeoDataFrame(
            [{
                "region_key": key,
                "admin_name": admin_name,
                "country": country,
                "osm_type": result["osm_type"],
                "osm_id": result["osm_id"],
                "place_class": result.get("class"),
                "place_type": result.get("type"),
                "display_name": result["display_name"],
                "geometry": geom,
            }],
            crs="EPSG:4326",
        )
        gdf.to_file(out_path, driver="GPKG")

        # Compute area in a UTM CRS (rough centroid-based zone pick).
        c = geom.centroid
        zone = int((c.x + 180.0) // 6.0) + 1
        utm_epsg = 32700 + zone if c.y < 0 else 32600 + zone
        area_km2 = float(gdf.to_crs(f"EPSG:{utm_epsg}").geometry.area.iloc[0]) / 1e6
        summary.append({"region_key": key, "admin_name": admin_name,
                        "osm_id": result["osm_id"], "area_km2": round(area_km2, 1),
                        "path": str(out_path.relative_to(PROJECT_ROOT))})
        print(f"  → wrote {out_path.relative_to(PROJECT_ROOT)}  area={area_km2:.1f} km^2")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    total = sum(s["area_km2"] for s in summary)
    print(f"\nTotal admin area: {total:.1f} km^2")


if __name__ == "__main__":
    main()
