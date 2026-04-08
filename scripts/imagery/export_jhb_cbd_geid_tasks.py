#!/usr/bin/env python3
"""Export Johannesburg CBD grid bounds as a GEID task CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JHB_GRID_PATH = PROJECT_ROOT / "data" / "jhb_task_grid.gpkg"
CBD_GRID_IDS = [
    "G0772", "G0773", "G0774", "G0775", "G0776",
    "G0814", "G0815", "G0816", "G0817", "G0818",
    "G0853", "G0854", "G0855", "G0856", "G0857",
    "G0888", "G0889", "G0890", "G0891", "G0892",
    "G0922", "G0923", "G0924", "G0925", "G0926",
]


def normalize_grid_id(value: str) -> str:
    raw = str(value).strip().upper()
    if raw.startswith("G") and raw[1:].isdigit():
        return f"G{int(raw[1:]):04d}"
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Joburg CBD 25-grid bounds for Google Earth Images Downloader."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--save-root",
        default=r"D:\ZAsolar\joburg_cbd_geid",
        help="Windows directory root passed to GEID's 'Save to' field.",
    )
    parser.add_argument(
        "--task-root",
        default=r"D:\ZAsolar\joburg_cbd_geid\tasks",
        help="Windows directory root for .geid task files.",
    )
    parser.add_argument(
        "--zoom-from",
        type=int,
        default=9,
        help="Default GEID 'From zoom level'.",
    )
    parser.add_argument(
        "--zoom-to",
        type=int,
        default=12,
        help="Default GEID 'To zoom level'.",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Optional historical imagery date (YYYY-MM-DD). Leave blank for current imagery.",
    )
    parser.add_argument(
        "--map-type",
        default="",
        help="Optional map type label to select in GEID. Leave blank to keep current selection.",
    )
    parser.add_argument(
        "--grid-id",
        nargs="+",
        default=None,
        help="Optional subset of CBD grid IDs to export.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to write after filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gdf = gpd.read_file(JHB_GRID_PATH)
    ordered_ids = [normalize_grid_id(gid) for gid in (args.grid_id or CBD_GRID_IDS)]
    wanted = set(ordered_ids)
    selected = gdf[gdf["gridcell_id"].isin(wanted)].copy()
    if len(selected) != len(wanted):
        missing = sorted(wanted - set(selected["gridcell_id"]))
        raise SystemExit(f"Missing CBD grids in {JHB_GRID_PATH}: {missing}")

    if args.limit is not None:
        ordered_ids = ordered_ids[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "grid_id",
                "task_name",
                "save_to",
                "map_type",
                "date",
                "zoom_from",
                "zoom_to",
                "left_longitude",
                "right_longitude",
                "top_latitude",
                "bottom_latitude",
            ],
        )
        writer.writeheader()

        for grid_id in ordered_ids:
            row = selected.loc[selected["gridcell_id"] == grid_id].iloc[0]
            xmin, ymin, xmax, ymax = row.geometry.bounds
            writer.writerow(
                {
                    "grid_id": grid_id,
                    "task_name": f"{grid_id}.geid",
                    "save_to": rf"{args.save_root}\{grid_id}",
                    "map_type": args.map_type,
                    "date": args.date,
                    "zoom_from": args.zoom_from,
                    "zoom_to": args.zoom_to,
                    "left_longitude": f"{xmin:.12f}",
                    "right_longitude": f"{xmax:.12f}",
                    "top_latitude": f"{ymax:.12f}",
                    "bottom_latitude": f"{ymin:.12f}",
                }
            )

    print(f"Wrote {args.output} with {len(ordered_ids)} Joburg CBD grid(s).")


if __name__ == "__main__":
    main()
