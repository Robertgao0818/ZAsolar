#!/usr/bin/env python3
"""Export Johannesburg grid bounds as a GEID task CSV.

Generates a CSV consumed by ``scripts/imagery/windows/run_geid_tasks.ps1`` to
drive the Google Earth Images Downloader (GEID) GUI. Accepts arbitrary grid IDs
from ``data/jhb_task_grid.gpkg`` — no hardcoded category lists.

Usage:
  # subset by id
  python scripts/imagery/export_jhb_geid_tasks.py \\
      --output tasks.csv --grid-id G0772 G0773 G0888

  # subset from a newline-separated file
  python scripts/imagery/export_jhb_geid_tasks.py \\
      --output tasks.csv --grid-ids-file batch1_remaining.txt

  # full coverage of jhb_task_grid.gpkg
  python scripts/imagery/export_jhb_geid_tasks.py --output tasks.csv --all
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JHB_GRID_PATH = PROJECT_ROOT / "data" / "jhb_task_grid.gpkg"


def normalize_grid_id(value: str) -> str:
    raw = str(value).strip().upper()
    if raw.startswith("G") and raw[1:].isdigit():
        return f"G{int(raw[1:]):04d}"
    return raw


def load_grid_ids_file(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split("#", 1)[0].strip()
        if token:
            ids.append(token)
    if not ids:
        raise SystemExit(f"No grid IDs found in {path}")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Joburg grid bounds as a GEID task CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--save-root",
        default=r"D:\ZAsolar\joburg_geid",
        help="Windows directory root passed to GEID's 'Save to' field.",
    )
    parser.add_argument(
        "--task-root",
        default=r"D:\ZAsolar\joburg_geid\tasks",
        help="Windows directory root for .geid task files (used to build task_name path).",
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

    grid_group = parser.add_mutually_exclusive_group(required=True)
    grid_group.add_argument(
        "--grid-id",
        nargs="+",
        default=None,
        help="One or more grid IDs to export (e.g. --grid-id G0772 G0773).",
    )
    grid_group.add_argument(
        "--grid-ids-file",
        type=Path,
        default=None,
        help="Path to newline-separated grid ID list (# comments allowed).",
    )
    grid_group.add_argument(
        "--all",
        action="store_true",
        help="Export every grid in data/jhb_task_grid.gpkg.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rows written after filtering.",
    )
    return parser.parse_args()


def resolve_grid_ids(args: argparse.Namespace, gdf: gpd.GeoDataFrame) -> list[str]:
    if args.all:
        return [normalize_grid_id(gid) for gid in gdf["gridcell_id"].tolist()]
    if args.grid_ids_file is not None:
        return [normalize_grid_id(gid) for gid in load_grid_ids_file(args.grid_ids_file)]
    return [normalize_grid_id(gid) for gid in args.grid_id]


def main() -> None:
    args = parse_args()

    gdf = gpd.read_file(JHB_GRID_PATH)
    ordered_ids = resolve_grid_ids(args, gdf)

    wanted = set(ordered_ids)
    selected = gdf[gdf["gridcell_id"].isin(wanted)].copy()
    if len(selected) != len(wanted):
        missing = sorted(wanted - set(selected["gridcell_id"]))
        raise SystemExit(f"Missing grids in {JHB_GRID_PATH}: {missing}")

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
                    "task_name": rf"{args.task_root}\{grid_id}.geid",
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

    print(f"Wrote {args.output} with {len(ordered_ids)} Joburg grid(s).")


if __name__ == "__main__":
    main()
