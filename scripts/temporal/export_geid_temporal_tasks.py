#!/usr/bin/env python3
# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/scripts/temporal/export_geid_temporal_tasks.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
"""Export GEID historical download tasks from temporal PV anchors.

The output CSV is compatible with
`geid_reverse_engineering/python/geid_historical_cli_batch.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.temporal.geid_temporal_common import (
    build_geid_task_rows,
    read_csv_rows,
    write_csv_rows,
    years_to_dates,
)

DEFAULT_ANCHORS = PROJECT_ROOT / "data" / "geid_temporal" / "anchors.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "geid_temporal" / "geid_tasks.csv"
DEFAULT_SAVE_ROOT = Path.home() / "zasolar_data" / "geid_raw" / "temporal_anchor_presence"
TASK_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dates", nargs="*", help="Explicit requested dates, YYYY-MM-DD.")
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"), help="Inclusive year range if --dates is not given.")
    parser.add_argument("--date-md", default="06-15", help="Month-day for --years mode. Default: 06-15")
    parser.add_argument(
        "--save-root",
        default=str(DEFAULT_SAVE_ROOT),
        help="Root for downloaded GEID task folders. Defaults to WSL canonical ~/zasolar_data. "
        "Pass a Windows/UNC path explicitly only for Windows downloader staging.",
    )
    parser.add_argument(
        "--save-root-win",
        dest="save_root",
        help="Deprecated alias for --save-root; retained for old command lines.",
    )
    parser.add_argument("--zoom-from", type=int, default=21)
    parser.add_argument("--zoom-to", type=int, default=21)
    parser.add_argument("--limit-anchors", type=int, help="Optional cap for smoke/dry-run task exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.anchors_csv.exists():
        raise SystemExit(f"Anchor CSV not found: {args.anchors_csv}")
    if args.dates:
        dates = args.dates
    elif args.years:
        dates = years_to_dates(args.years[0], args.years[1], args.date_md)
    else:
        raise SystemExit("Provide either --dates or --years START END")

    anchors = read_csv_rows(args.anchors_csv)
    if args.limit_anchors:
        anchors = anchors[: args.limit_anchors]
    if not anchors:
        raise SystemExit("No anchors found.")

    rows = build_geid_task_rows(
        anchors,
        dates,
        save_root_win=args.save_root,
        zoom_from=args.zoom_from,
        zoom_to=args.zoom_to,
    )
    write_csv_rows(args.output, rows, TASK_FIELDS)
    print(f"Wrote {len(rows)} GEID tasks from {len(anchors)} anchors × {len(dates)} dates -> {args.output}")
    print("Run with: python /home/gaosh/projects/geid_reverse_engineering/python/geid_historical_cli_batch.py --tasks-csv", args.output)


if __name__ == "__main__":
    main()
