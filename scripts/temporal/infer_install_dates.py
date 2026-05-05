#!/usr/bin/env python3
# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/scripts/temporal/infer_install_dates.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
"""Infer PV installation intervals from binary historical presence observations.

Input CSV schema is intentionally simple and model-agnostic. Required columns:
`anchor_id` plus either `capture_date`, `actual_capture_date`, `requested_date`,
or `date`; and either explicit `pv_present` or a score column such as
`pv_score`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.temporal.geid_temporal_common import (
    InstallInterval,
    infer_install_interval,
    observation_from_row,
    write_csv_rows,
)

DEFAULT_INPUT = PROJECT_ROOT / "data" / "geid_temporal" / "presence_timeseries.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "geid_temporal" / "install_intervals.csv"
OUTPUT_FIELDS = [
    "anchor_id",
    "status",
    "latest_absent_date",
    "earliest_present_date",
    "install_interval_start",
    "install_interval_end",
    "n_observations",
    "n_absent",
    "n_present",
    "confidence",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--presence-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for score-only rows.")
    parser.add_argument(
        "--min-consecutive-present",
        type=int,
        default=1,
        help="Require this many consecutive present observations before accepting an appearance breakpoint.",
    )
    return parser.parse_args()


def read_observations(path: Path, *, threshold: float):
    grouped = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader, start=2):
            obs = observation_from_row(row, row_idx=row_idx, threshold=threshold)
            if obs is None:
                print(f"[WARN] skipping row {row_idx}: missing anchor/date", file=sys.stderr)
                continue
            grouped[obs.anchor_id].append(obs)
    return grouped


def main() -> None:
    args = parse_args()
    if not args.presence_csv.exists():
        raise SystemExit(f"Presence CSV not found: {args.presence_csv}")
    if args.min_consecutive_present < 1:
        raise SystemExit("--min-consecutive-present must be >= 1")

    grouped = read_observations(args.presence_csv, threshold=args.threshold)
    if not grouped:
        raise SystemExit("No usable observations found.")
    intervals: list[InstallInterval] = []
    for anchor_id in sorted(grouped):
        intervals.append(
            infer_install_interval(
                anchor_id,
                grouped[anchor_id],
                min_consecutive_present=args.min_consecutive_present,
            )
        )
    write_csv_rows(args.output, [i.as_row() for i in intervals], OUTPUT_FIELDS)

    by_status: dict[str, int] = defaultdict(int)
    for interval in intervals:
        by_status[interval.status] += 1
    print(f"Wrote {len(intervals)} install intervals -> {args.output}")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
