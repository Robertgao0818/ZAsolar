#!/usr/bin/env python3
"""Parse GEID vintage-probe download results into a vintage-availability table.

After running scripts/validation/probe_geid_vintages.py and feeding its output
through scripts/validation/run_geid_vintage_probe.sh (headless WSL CLI driver),
GEID writes per-year folders under
``<save_root>/<region>/<anchor_id>/<year>/<task_name>/<z>/<x>/gesh_x_y_z.jpg``.

GEID's CLI does **not** fail when the requested vintage is missing — it
returns the **closest available** historical layer. Each tile JPEG embeds the
true capture date in its comment field (``*AD*YYYY:MM:DD*``). Availability
must be judged from that comment, not from file count or HTTP status.

This script walks the output tree, extracts each tile's actual capture date,
and decides per (anchor × requested-year) cell:

  - ``available``     : at least one tile's actual year == requested year
  - ``drift``         : tiles present, but their actual year != requested year
                        (GEID returned the nearest earlier/later layer)
  - ``empty``         : the year folder has no JPEG tiles

Outputs:

  vintage_availability.csv  — per (anchor × year): status, n_tiles, dominant
                              actual capture date, all unique actual dates
  vintage_pivot.csv         — rows = anchor, cols = requested year,
                              values = ``Y`` (exact) / ``~`` (drift) / ``.`` (empty)
                              and a parallel pivot of dominant actual dates
  vintage_anchor_dates.csv  — unique actual capture dates discovered per anchor
                              (the real value of the probe — these are the GEID
                              vintages available at each location)
  vintage_city_summary.csv  — per region: unique actual dates, year coverage span

Usage:
  python scripts/validation/parse_geid_probe_results.py \\
      --geid-root /mnt/d/ZAsolar/geid_raw/vintage_probe \\
      --anchors-csv data/geid_vintage_probe/probe_anchors.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANCHORS = PROJECT_ROOT / "data" / "geid_vintage_probe" / "probe_anchors.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "geid_vintage_probe" / "results"
DEFAULT_GEID_ROOT_WSL = Path("/mnt/d/ZAsolar/geid_raw/vintage_probe")
JPEG_SOI = b"\xff\xd8\xff"
COMMENT_RE = re.compile(rb"\*AD\*(\d{4}):(\d{2}):(\d{2})\*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geid-root", type=Path, default=DEFAULT_GEID_ROOT_WSL)
    parser.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1024,
        help="Minimum JPEG size (bytes) to count as 'real' imagery. Default 1024.",
    )
    return parser.parse_args()


def extract_capture_date(jpg_path: Path) -> str | None:
    """Return YYYY-MM-DD from JPEG comment field, or None if missing."""
    try:
        with jpg_path.open("rb") as f:
            head = f.read(2048)
    except OSError:
        return None
    if not head.startswith(JPEG_SOI):
        return None
    m = COMMENT_RE.search(head)
    if not m:
        return None
    return f"{m.group(1).decode()}-{m.group(2).decode()}-{m.group(3).decode()}"


def summarise_year_folder(folder: Path, requested_year: int, min_bytes: int) -> dict[str, object]:
    if not folder.exists():
        return {
            "status": "missing_folder",
            "n_tiles": 0,
            "dominant_actual_date": "",
            "all_actual_dates": "",
        }
    jpgs = [p for p in folder.rglob("*.jpg") if p.is_file() and p.stat().st_size >= min_bytes]
    if not jpgs:
        return {
            "status": "empty",
            "n_tiles": 0,
            "dominant_actual_date": "",
            "all_actual_dates": "",
        }
    actual_dates = [extract_capture_date(p) for p in jpgs]
    actual_dates = [d for d in actual_dates if d]
    if not actual_dates:
        return {
            "status": "no_date_metadata",
            "n_tiles": len(jpgs),
            "dominant_actual_date": "",
            "all_actual_dates": "",
        }
    counter = Counter(actual_dates)
    dominant = counter.most_common(1)[0][0]
    unique_sorted = sorted(set(actual_dates))
    actual_years = {int(d[:4]) for d in actual_dates}
    if requested_year in actual_years:
        status = "available"
    else:
        status = "drift"
    return {
        "status": status,
        "n_tiles": len(jpgs),
        "dominant_actual_date": dominant,
        "all_actual_dates": ";".join(unique_sorted),
    }


def main() -> None:
    args = parse_args()
    if not args.geid_root.exists():
        sys.exit(f"GEID root not found: {args.geid_root}")
    if not args.anchors_csv.exists():
        sys.exit(f"Anchors CSV not found: {args.anchors_csv}")

    anchors = pd.read_csv(args.anchors_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    seen_years: set[int] = set()
    for _, anchor in anchors.iterrows():
        anchor_dir = args.geid_root / anchor["region_key"] / anchor["anchor_id"]
        if not anchor_dir.exists():
            continue
        for year_folder in sorted(p for p in anchor_dir.iterdir() if p.is_dir()):
            try:
                year = int(year_folder.name)
            except ValueError:
                continue
            seen_years.add(year)
            summary = summarise_year_folder(year_folder, year, args.min_bytes)
            rows.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "region_key": anchor["region_key"],
                    "stratum": anchor["stratum"],
                    "requested_year": year,
                    **summary,
                }
            )

    if not rows:
        sys.exit(f"No year folders found under {args.geid_root}.")

    avail = pd.DataFrame(rows).sort_values(["region_key", "anchor_id", "requested_year"])
    avail_path = args.output_dir / "vintage_availability.csv"
    avail.to_csv(avail_path, index=False)

    def mark(s: str) -> str:
        return {"available": "Y", "drift": "~"}.get(s, ".")

    pivot_status = (
        avail.assign(mark=avail["status"].map(mark))
        .pivot_table(
            index=["region_key", "anchor_id", "stratum"],
            columns="requested_year",
            values="mark",
            aggfunc="first",
        )
        .fillna("?")
    )
    pivot_status_path = args.output_dir / "vintage_pivot.csv"
    pivot_status.to_csv(pivot_status_path)

    pivot_dates = (
        avail.pivot_table(
            index=["region_key", "anchor_id", "stratum"],
            columns="requested_year",
            values="dominant_actual_date",
            aggfunc="first",
        )
        .fillna("")
    )
    pivot_dates_path = args.output_dir / "vintage_pivot_dates.csv"
    pivot_dates.to_csv(pivot_dates_path)

    # Unique actual capture dates discovered per anchor — this is the real signal
    anchor_dates_rows = []
    for (region, anchor_id, stratum), grp in avail.groupby(["region_key", "anchor_id", "stratum"]):
        all_dates: set[str] = set()
        for d_str in grp["all_actual_dates"]:
            if d_str:
                all_dates.update(d_str.split(";"))
        unique = sorted(all_dates)
        years = sorted({int(d[:4]) for d in unique})
        anchor_dates_rows.append(
            {
                "region_key": region,
                "anchor_id": anchor_id,
                "stratum": stratum,
                "n_unique_dates": len(unique),
                "n_unique_years": len(years),
                "year_min": years[0] if years else "",
                "year_max": years[-1] if years else "",
                "all_capture_dates": ";".join(unique),
                "all_years": ";".join(str(y) for y in years),
            }
        )
    anchor_dates_df = pd.DataFrame(anchor_dates_rows).sort_values(["region_key", "anchor_id"])
    anchor_dates_path = args.output_dir / "vintage_anchor_dates.csv"
    anchor_dates_df.to_csv(anchor_dates_path, index=False)

    by_city = (
        anchor_dates_df.groupby("region_key")
        .agg(
            anchors=("anchor_id", "nunique"),
            mean_unique_dates=("n_unique_dates", "mean"),
            mean_unique_years=("n_unique_years", "mean"),
            year_min=("year_min", lambda s: min(int(x) for x in s if x != "")),
            year_max=("year_max", lambda s: max(int(x) for x in s if x != "")),
        )
        .reset_index()
    )

    # Year-level coverage: how many anchors had at least one exact-year hit per requested year
    year_coverage = (
        avail.assign(hit=(avail["status"] == "available").astype(int))
        .groupby(["region_key", "requested_year"])
        .agg(anchors=("anchor_id", "nunique"), exact_hits=("hit", "sum"))
        .reset_index()
    )
    year_coverage_path = args.output_dir / "vintage_year_coverage.csv"
    year_coverage.to_csv(year_coverage_path, index=False)

    summary_path = args.output_dir / "vintage_city_summary.csv"
    by_city.to_csv(summary_path, index=False)

    years = sorted(seen_years)
    print(f"Years probed: {years[0]}–{years[-1]} ({len(years)} years)")
    print(f"Anchors found: {len(anchor_dates_df)}")
    print()
    print("Outputs:")
    print(f"  per-cell:      {avail_path.relative_to(PROJECT_ROOT)}")
    print(f"  pivot status:  {pivot_status_path.relative_to(PROJECT_ROOT)}")
    print(f"  pivot dates:   {pivot_dates_path.relative_to(PROJECT_ROOT)}")
    print(f"  anchor dates:  {anchor_dates_path.relative_to(PROJECT_ROOT)}")
    print(f"  year coverage: {year_coverage_path.relative_to(PROJECT_ROOT)}")
    print(f"  city summary:  {summary_path.relative_to(PROJECT_ROOT)}")
    print()
    print(by_city.to_string(index=False))


if __name__ == "__main__":
    main()
