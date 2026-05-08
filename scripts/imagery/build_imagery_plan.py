#!/usr/bin/env python3
"""Build per-grid imagery source plan for full-admin inference.

For each admin grid (data/admin_grids/<region>_admin_grid.gpkg) decide
which source to fetch from based on policy:

  joburg:           vexcel_joburg_2024 if vexcel_*; else coj_aerial_2023
  durban:           vexcel_durban_2026 if vexcel_*;
                    elif lon < extent_lon_max(ethekwini_2023): primary=ethekwini_2023, fallback=ethekwini_2022
                    else: primary=ethekwini_2022
  pretoria/bloem/   vexcel_<region>_<vintage> if vexcel_*; else SKIP
   east_london/
   gqeberha/pmb:
  any region:       SKIP (drop) if imagery_plan == 'skip'

Output: data/imagery_plans/<region>_plan.csv with columns:
  gridcell_id, region_key, primary_source, fallback_source, lon, lat,
  n_buildings, status (pending), source_used (empty)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADMIN_GRID_DIR = PROJECT_ROOT / "data" / "admin_grids"
OUTPUT_DIR = PROJECT_ROOT / "data" / "imagery_plans"
SOURCES_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "aerial_sources.yaml"


REGION_VEXCEL_KEY = {
    "joburg": "vexcel_joburg_2024",
    "durban": "vexcel_durban_2026",
    "pretoria": "vexcel_pretoria_2026",
    "bloemfontein": "vexcel_bloemfontein_2026",
    "east_london": "vexcel_east_london_2026",
    "gqeberha": "vexcel_gqeberha_2026",
    "pietermaritzburg": "vexcel_pietermaritzburg_2025",
}


def assign_sources(region: str, gdf: gpd.GeoDataFrame, sources: dict[str, Any]) -> pd.DataFrame:
    vexcel_key = REGION_VEXCEL_KEY[region]
    rows: list[dict[str, Any]] = []
    durban_2023_lon_max = sources.get("ethekwini_2023", {}).get("extent_lon_max", 30.87)

    for _, r in gdf.iterrows():
        plan = r["imagery_plan"]
        primary = ""
        fallback = ""

        if plan == "skip":
            continue  # drop empty grids entirely

        if plan in ("vexcel_full", "vexcel_partial"):
            primary = vexcel_key
        elif plan == "aerial_needed":
            if region == "joburg":
                primary = "coj_aerial_2023"
            elif region == "durban":
                if r["lon"] < durban_2023_lon_max:
                    primary = "ethekwini_2023"
                    fallback = "ethekwini_2022"
                else:
                    primary = "ethekwini_2022"
            else:
                continue  # other 5 cities: don't do aerial fallback (per policy)
        else:
            continue

        rows.append({
            "gridcell_id": r["gridcell_id"],
            "region_key": region,
            "primary_source": primary,
            "fallback_source": fallback,
            "imagery_plan": plan,
            "lon": float(r["lon"]),
            "lat": float(r["lat"]),
            "n_buildings": int(r["n_buildings"]),
            "vexcel_fraction": float(r["vexcel_fraction"]),
            "status": "pending",
            "source_used": "",
            "downloaded_at": "",
            "error": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="*",
                        default=list(REGION_VEXCEL_KEY.keys()))
    parser.add_argument("--admin-dir", type=Path, default=ADMIN_GRID_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sources-config", type=Path, default=SOURCES_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with args.sources_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sources = cfg["sources"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for region in args.regions:
        admin_path = args.admin_dir / f"{region}_admin_grid.gpkg"
        out_path = args.output_dir / f"{region}_plan.csv"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} exists (--overwrite to replace)")
            continue
        if not admin_path.exists():
            print(f"[SKIP] {region}: no admin grid at {admin_path}")
            continue

        gdf = gpd.read_file(admin_path)
        plan = assign_sources(region, gdf, sources)
        plan.to_csv(out_path, index=False)

        by_source = plan.groupby("primary_source").size().to_dict()
        with_fb = plan[plan.fallback_source != ""].groupby("primary_source").size().to_dict()
        print(f"\n[{region}] plan: {len(plan)} grids → {out_path.relative_to(PROJECT_ROOT)}")
        for src, n in sorted(by_source.items()):
            fb = with_fb.get(src, 0)
            fb_str = f"  (+ {fb} with fallback)" if fb else ""
            print(f"  {src:<35} {n:>5} grids{fb_str}")
        summary.append({"region": region, "n_grids": len(plan),
                        "by_source": dict(by_source)})

    print("\n=== Total ===")
    total = sum(s["n_grids"] for s in summary)
    print(f"7-city plan: {total} grids to fetch")


if __name__ == "__main__":
    main()
