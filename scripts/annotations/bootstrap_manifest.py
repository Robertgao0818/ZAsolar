#!/usr/bin/env python3
"""Bootstrap the annotation manifest from existing GPKG files.

Reads the same GRID_SOURCES as export_coco_dataset.py, produces one CSV row
per valid annotation polygon with quality tier T2 (unreviewed).

Usage:
    python scripts/bootstrap_manifest.py [--output data/annotations/annotation_manifest.csv]
"""
import argparse
from pathlib import Path

import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"

GRID_SOURCES = {
    "G1238": {
        "file": ANNOTATIONS_DIR / "G1238_SAM2_260320.gpkg",
        "layer": "SAM_Residential_merged",
        "filter": None,
    },
    "G1189": {
        "file": ANNOTATIONS_DIR / "G1189_SAM2_260320.gpkg",
        "layer": "sam_residential_g1189_mosa_109_rgb255105180",
        "filter": None,
    },
    "G1190": {
        "file": ANNOTATIONS_DIR / "G1190_SAM2_260320.gpkg",
        "layer": "SAM_Residential_20260320_221905",
        "filter": None,
    },
}

MANIFEST_COLUMNS = [
    "grid_id",
    "annotation_id",
    "source_file",
    "source_layer",
    "source_id",
    "source_name",
    "label_definition",
    "quality_tier",
    "review_status",
    "issue_type",
    "split_scope",
    "label_source",
    "semantic_confidence",
]


def bootstrap(output_path: Path) -> None:
    rows = []
    for grid_id, src in GRID_SOURCES.items():
        gdf = gpd.read_file(str(src["file"]), layer=src["layer"])
        if src["filter"] is not None:
            gdf = src["filter"](gdf).copy()
        # Same validity filter as export_coco_dataset.py
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        gdf = gdf.reset_index(drop=True)

        rel_file = src["file"].relative_to(ANNOTATIONS_DIR)

        for idx, row in gdf.iterrows():
            # Source ID: panel_id for G1238, hex id for others
            if "panel_id" in row.index:
                source_id = str(row["panel_id"])
            elif "id" in row.index:
                source_id = str(row["id"])
            else:
                source_id = ""

            source_name = str(row.get("Name", "")) if "Name" in row.index else ""

            rows.append({
                "grid_id": grid_id,
                "annotation_id": f"{grid_id}_{idx:03d}",
                "source_file": str(rel_file),
                "source_layer": src["layer"],
                "source_id": source_id,
                "source_name": source_name,
                "label_definition": "installation_footprint",
                "quality_tier": "T2",
                "review_status": "unreviewed",
                "issue_type": "",
                "split_scope": "",
                "label_source": "human_manual_sam_assisted",
                "semantic_confidence": "A2",
            })

        print(f"[MANIFEST] {grid_id}: {len(gdf)} annotations")

    # Write CSV
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[MANIFEST] Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap annotation manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=ANNOTATIONS_DIR / "annotation_manifest.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    bootstrap(args.output)


if __name__ == "__main__":
    main()
