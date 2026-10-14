"""Build the per-grid QA sheet for the Li KML batch (L0208..L1841).

One row per staged L<NNNN>.gpkg from this batch:
  Lid, panel_count, geom_invalid_fixed, area_m2_median, area_m2_total,
  tiles_available, density_bucket, notes

tiles_available is resolved via core.grid_utils.resolve_tiles_dir(Lid,
region='cape_town', imagery_layer='aerial_2025'); Li's L-cells are a distinct
physical grid scheme and their tiles are NOT staged locally, so this is
expected False for the whole batch (see final report / mapping note).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import geopandas as gpd

from core import grid_utils as gu

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "annotations" / "Capetown_Li"
QA_CSV = OUT_DIR / "_kml_batch_qa.csv"
MANIFEST = OUT_DIR / "annotation_manifest_li.csv"
METRIC_CRS = "EPSG:32734"  # UTM 34S, Cape Town metric

# This batch = grids ingested from the KML (src_layer_name == kml_placemark).
BATCH_SRC = "kml_placemark"

# Per-grid QA annotations surfaced during ingest review (2026-06-10).
GRID_NOTES = {
    "L0269": "1 empty/degenerate Polygon (G0269_007) dropped at parse; 57 valid kept.",
    "L1787": "large-polygon grid: median ~415 m^2, installation-scale not sub-array.",
}


def batch_lids() -> list[str]:
    lids = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if row["src_layer_name"] == BATCH_SRC:
                lids.append(row["grid_id"])
    return sorted(set(lids))


def density_bucket(n: int) -> str:
    if n >= 20:
        return ">=20"
    if n >= 10:
        return "10-19"
    return "<10"


def main() -> None:
    lids = batch_lids()
    rows = []
    n_tiles_ok = 0
    for lid in lids:
        gpkg = OUT_DIR / f"{lid}.gpkg"
        gdf = gpd.read_file(gpkg, layer="li_ct_gt")
        n = len(gdf)
        # geometry validity (post-stage; the stage step already buffer(0)-fixed)
        invalid = int((~gdf.geometry.is_valid).sum())
        gm = gdf.to_crs(METRIC_CRS)
        areas = gm.geometry.area
        # tiles: resolve by L-id, check directory exists with chunks
        try:
            tdir = gu.resolve_tiles_dir(lid, region="cape_town", imagery_layer="aerial_2025")
            tdir = Path(tdir)
            has = tdir.is_dir() and any(tdir.glob("*_geo.tif"))
        except Exception:
            has = False
        if has:
            n_tiles_ok += 1
        rows.append({
            "Lid": lid,
            "panel_count": n,
            "geom_invalid_fixed": invalid,  # residual invalid after stage buffer(0)
            "area_m2_median": round(float(areas.median()), 2),
            "area_m2_total": round(float(areas.sum()), 2),
            "tiles_available": "yes" if has else "no",
            "density_bucket": density_bucket(n),
            "notes": GRID_NOTES.get(lid, ""),
        })

    with open(QA_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot_panels = sum(r["panel_count"] for r in rows)
    buckets = {b: sum(1 for r in rows if r["density_bucket"] == b) for b in (">=20", "10-19", "<10")}
    print(f"QA sheet -> {QA_CSV}")
    print(f"grids={len(rows)} panels={tot_panels} tiles_available={n_tiles_ok}/{len(rows)}")
    print(f"density buckets: {buckets}")


if __name__ == "__main__":
    main()
