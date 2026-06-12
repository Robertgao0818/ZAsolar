"""Backfill bbox_geo_wkt geometry onto data/negative_pool/manifest.csv rows.

The negative pool was seeded 2026-05-13 from the cls_pv_thermal_v2 subtype
dataset (``bootstrap_from_cls_v2.py``).  Those 678 rows carry no
``bbox_geo_wkt`` geometry, so the negative_pool HN stream extracts 0 chips
today (``pipeline.hn_ops.extract_negative_pool_hn`` skips geometry-less rows).
This script recovers the geometry so the HN stream can actually crop chips.

Join key
--------
The F1-gap plan (docs/plans/2026-06-10-rcnn-f1-gap-review.md C-1) proposed
joining ``source_pred_id`` against
``results/johannesburg/v3c_geid_2024_02/<G>/predictions_metric.gpkg`` by
positional index.  **That positional join is unsafe** — the polygonised
``predictions_metric.gpkg`` rows are NOT in the detector's raw prediction
order, so ``source_pred_id`` (= cls chip ``pred_idx``) does not line up with
gpkg row position (verified: only ~25/678 rows match by position+area).

Instead this script joins on the authoritative ``chip_id`` against the
solar_cls cascade ``manifest.gpkg``, which stores the exact source polygon
geometry (EPSG:32735 metric) for every cls chip keyed by ``chip_id``.  The
negative-pool ``chip_id`` ``johannesburg_<G>_<det>_p<idx>`` maps deterministically
to the cascade ``chip_id`` ``<det>_<G>_p<idx>``.  All 678 rows join cleanly.

Provenance-only vs training-eligible
------------------------------------
Per the plan, geid_2024_02 rows are backfilled for **provenance only** — a
GEID chip must not silently enter a training bundle, because letting the GEID
appearance domain monopolise the HN stream would teach the detector a third
appearance domain only through negatives (imagery-layer balance gate).  This
is recorded by appending a machine-readable ``training_eligible`` column
(``true`` / ``false``); GEID rows land ``false`` until the imagery-layer
balance gate is satisfied at build time.  Appending a column preserves the
existing column order and never mutates any existing row's identity fields
(rule: data/negative_pool/ is append-only, monotonic).

Usage::

    python scripts/training/negative_pool/backfill_geometry.py            # write
    python scripts/training/negative_pool/backfill_geometry.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POOL_ROOT = PROJECT_ROOT / "data" / "negative_pool"
MANIFEST_CSV = POOL_ROOT / "manifest.csv"

# solar_cls cascade manifest.gpkg — authoritative per-chip source geometry.
# Sibling subrepo path (read-only consumer; see CLAUDE.md solar_cls section).
DEFAULT_CASCADE_GPKG = (
    Path.home()
    / "projects"
    / "solar_cls"
    / "data"
    / "cls_pv_nonpv_v3c_v42_cascade"
    / "manifest.gpkg"
)

# Imagery layers that are provenance-only (not training-eligible by default).
# GEID is the third appearance domain; gating it out of training keeps the HN
# stream from being monopolised by one provider's look (imagery-layer balance).
PROVENANCE_ONLY_LAYERS = {"geid_2024_02"}

NEW_COLUMN = "training_eligible"

# negative-pool chip_id  ->  cascade chip_id
#   johannesburg_G0772_v3c_p0000  ->  v3c_G0772_p0000
POOL_CHIP_RE = re.compile(
    r"^(?P<region>[a-z_]+)_(?P<grid>G\d+)_(?P<det>v3c|v4_2)_p(?P<pred>\d+)$"
)


def pool_chip_to_cascade_chip(pool_chip_id: str) -> str | None:
    m = POOL_CHIP_RE.match(pool_chip_id)
    if not m:
        return None
    return f"{m.group('det')}_{m.group('grid')}_p{m.group('pred')}"


def load_cascade_geometry(cascade_gpkg: Path) -> dict[str, str]:
    """Return {cascade_chip_id: bbox_geo_wkt(EPSG:4326)}.

    The cascade gpkg geometry is the source polygon in EPSG:32735; we reproject
    to EPSG:4326 and store the polygon's *bounding box* as WKT (the HN cropper
    in hn_ops centres a chip on the bbox centroid).
    """
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(cascade_gpkg)
    gdf_4326 = gdf.to_crs("EPSG:4326")
    out: dict[str, str] = {}
    for chip_id, geom in zip(gdf["chip_id"], gdf_4326.geometry):
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        out[chip_id] = box(minx, miny, maxx, maxy).wkt
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cascade-gpkg",
        type=Path,
        default=DEFAULT_CASCADE_GPKG,
        help="solar_cls cascade manifest.gpkg (authoritative chip geometry)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report backfill counts without writing the manifest",
    )
    args = parser.parse_args()

    if not MANIFEST_CSV.exists():
        print(f"ERROR: {MANIFEST_CSV} not found", file=sys.stderr)
        return 1
    if not args.cascade_gpkg.exists():
        print(
            f"ERROR: cascade gpkg not found: {args.cascade_gpkg}\n"
            f"  (solar_cls subrepo data; geometry cannot be recovered without it)",
            file=sys.stderr,
        )
        return 1

    geo_by_cascade = load_cascade_geometry(args.cascade_gpkg)
    print(f"Loaded {len(geo_by_cascade)} chip geometries from cascade gpkg")

    with MANIFEST_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if NEW_COLUMN not in fieldnames:
        fieldnames = fieldnames + [NEW_COLUMN]

    filled = 0
    already = 0
    no_match = 0
    eligible = 0
    provenance_only = 0
    no_match_examples: list[str] = []

    for row in rows:
        # never overwrite an existing geometry (append-only identity rule)
        if (row.get("bbox_geo_wkt") or "").strip():
            already += 1
        else:
            cascade_id = pool_chip_to_cascade_chip(row["chip_id"])
            wkt = geo_by_cascade.get(cascade_id) if cascade_id else None
            if wkt is None:
                no_match += 1
                if len(no_match_examples) < 5:
                    no_match_examples.append(row["chip_id"])
            else:
                row["bbox_geo_wkt"] = wkt
                filled += 1

        # training-eligibility flag (idempotent; never downgrade an existing
        # explicit 'true', only fill blanks)
        existing_flag = (row.get(NEW_COLUMN) or "").strip().lower()
        if existing_flag in ("true", "false"):
            pass
        else:
            is_prov_only = row.get("imagery_layer") in PROVENANCE_ONLY_LAYERS
            row[NEW_COLUMN] = "false" if is_prov_only else "true"
        if row.get(NEW_COLUMN) == "true":
            eligible += 1
        else:
            provenance_only += 1

    print("--- backfill summary ---")
    print(f"  rows total:                 {len(rows)}")
    print(f"  geometry filled (new):      {filled}")
    print(f"  geometry already present:   {already}")
    print(f"  no cascade match (skipped): {no_match}")
    if no_match_examples:
        print(f"    examples: {no_match_examples}")
    print(f"  training_eligible=true:     {eligible}")
    print(f"  training_eligible=false (provenance-only, "
          f"{sorted(PROVENANCE_ONLY_LAYERS)}): {provenance_only}")

    if args.dry_run:
        print("(dry-run: manifest not modified)")
        return 0

    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # ensure every row has the new key
            row.setdefault(NEW_COLUMN, "")
            writer.writerow(row)
    print(f"Wrote {MANIFEST_CSV.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
