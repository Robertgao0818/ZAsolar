#!/usr/bin/env python
"""Reshape the CT census inventory into the JHB delivered-table format.

The JHB economic-layer deliverable
(`jhb_full382_fpcut_install_dated_*_with_census.csv`) is a per-row solar table
whose CSV column order is:

    source_feature_id, source_grid, centroid_lon, centroid_lat, area_m2,
    confidence, date_provider, date_status, date_is_bound, install_date,
    install_interval_start, install_interval_end, earliest_present_date,
    install_confidence, source_anchor_id, undated_reason

This script maps the CT census inventory onto that exact schema. CT has no
install-date layer yet (CT backdating has not been run), so the whole
install-date block is emitted empty with `undated_reason=ct_backdating_pending`
and `date_status=undated`. `source_feature_id` is a stable 0..N-1 row id that
later serves as the join key when CT backdating fills the date columns — same
contract as JHB.

Granularity:
  --granularity merged   (default) 111,801 de-duplicated installations
                         (IoU>=0.10 cross-detection union; matches the CT
                         census headline count, no double-counting).
  --granularity perdet   170,605 raw per-detection polygons (literal row
                         parity with the JHB per-detection table).

Outputs (same basename, .gpkg keeps geometry @ EPSG:32734, .csv carries
EPSG:4326 centroids and drops geometry), matching the JHB pair.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

# JHB delivered-CSV column order (the "same format as JHB" contract).
JHB_CSV_COLS = [
    "source_feature_id", "source_grid", "centroid_lon", "centroid_lat",
    "area_m2", "confidence", "date_provider", "date_status", "date_is_bound",
    "install_date", "install_interval_start", "install_interval_end",
    "earliest_present_date", "install_confidence", "source_anchor_id",
    "undated_reason",
]
# Empty install-date block for CT (backdating not yet run).
DATE_BLOCK_DEFAULTS = {
    "date_provider": "", "date_status": "undated", "date_is_bound": "",
    "install_date": "", "install_interval_start": "", "install_interval_end": "",
    "earliest_present_date": "", "install_confidence": "",
    "source_anchor_id": "", "undated_reason": "ct_backdating_pending",
}

DEFAULT_INPUTS = {
    "merged": "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/ct_census/"
              "ct_census_inventory_cpt_merged_iou010.gpkg",
    "perdet": "/mnt/c/Users/gaosh/Dropbox/RA_Solar/Gao/ct_census/"
              "ct_census_inventory_cpt.gpkg",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--granularity", choices=["merged", "perdet"],
                    default="merged")
    ap.add_argument("--input", default=None,
                    help="override source gpkg (defaults per granularity)")
    ap.add_argument("--out-dir",
                    default="results/analysis/ct_census_output_table")
    ap.add_argument("--date-tag", required=True,
                    help="date stamp for output basename, e.g. 2026-06-21")
    args = ap.parse_args()

    src_path = args.input or DEFAULT_INPUTS[args.granularity]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"ct_full_inventory_{args.date_tag}_{args.granularity}"

    print(f"[read] {src_path}")
    gdf = gpd.read_file(src_path)
    print(f"[read] {len(gdf):,} features | crs={gdf.crs}")

    # Stable join key (mirrors JHB source_feature_id contract).
    gdf = gdf.reset_index(drop=True)
    gdf["source_feature_id"] = gdf.index.astype(int)
    gdf["source_grid"] = gdf["gridcell_id"]

    # Centroids in EPSG:4326 (gdf is metric EPSG:32734).
    cent = gdf.geometry.centroid.to_crs(4326)
    gdf["centroid_lon"] = cent.x
    gdf["centroid_lat"] = cent.y

    # Empty install-date block.
    for col, val in DATE_BLOCK_DEFAULTS.items():
        gdf[col] = val

    # --- CSV: exact JHB column order, geometry dropped ---
    csv_df = gdf[JHB_CSV_COLS].copy()
    csv_path = out_dir / f"{base}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"[write] {csv_path} ({len(csv_df):,} rows, {len(JHB_CSV_COLS)} cols)")

    # --- GPKG: JHB-equivalent attributes + geometry (no centroid cols) ---
    gpkg_cols = [c for c in JHB_CSV_COLS
                 if c not in ("centroid_lon", "centroid_lat")] + ["geometry"]
    gpkg_gdf = gdf[gpkg_cols].copy()
    gpkg_path = out_dir / f"{base}.gpkg"
    gpkg_gdf.to_file(gpkg_path, layer="ct_solar_inventory", driver="GPKG")
    print(f"[write] {gpkg_path} ({len(gpkg_gdf):,} rows, crs={gpkg_gdf.crs})")


if __name__ == "__main__":
    main()
