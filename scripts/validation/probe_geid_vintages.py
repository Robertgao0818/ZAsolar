#!/usr/bin/env python3
"""Generate GEID probe tasks to map historical vintage availability per Vexcel city.

The install-date back-dating sub-line requires a GEID time-stack per detected
panel. Before investing in patch-classifier training, we need to know how deep
GEID's history goes in each Vexcel city. This script generates a task CSV that
the existing Windows runner (scripts/imagery/windows/run_geid_tasks.ps1)
consumes — one tiny ~200m bbox per (city, stratum) anchor, replicated across
candidate years.

Anchors are sampled from the OSM-stratified Vexcel eval sample so the RA's
exhaustive annotation work (which uses the same grids) and the time-stack probe
share locations — the same anchors can later seed install-date ground truth.

Pair with parse_geid_probe_results.py to summarise which years actually
returned imagery.

Usage:
  python scripts/validation/probe_geid_vintages.py \
      --years 2014 2025 \
      --anchors-per-stratum 1 \
      --output data/geid_vintage_probe/probe_tasks.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_CSV = (
    PROJECT_ROOT
    / "data"
    / "vexcel_eval_samples"
    / "vexcel_eval_grids_osm_stratified_seed42_per_region10.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "geid_vintage_probe" / "probe_tasks.csv"
DEFAULT_ANCHORS = PROJECT_ROOT / "data" / "geid_vintage_probe" / "probe_anchors.csv"
DEFAULT_SAVE_ROOT = r"D:\ZAsolar\geid_raw\vintage_probe"
DEFAULT_BBOX_HALF_M = 100.0
DEFAULT_PROBE_DATE_MONTH_DAY = "06-15"
DEFAULT_ZOOM_FROM = 18
DEFAULT_ZOOM_TO = 19

CITY_SHORT = {
    "bloemfontein": "BFN",
    "durban": "DBN",
    "east_london": "ELS",
    "gqeberha": "GQB",
    "pietermaritzburg": "PMB",
    "pretoria": "PTA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="GEID probe task CSV.")
    parser.add_argument("--anchors-csv", type=Path, default=DEFAULT_ANCHORS, help="Anchor manifest CSV.")
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT, help="Windows root passed to GEID 'Save to' field.")
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=[2014, 2025],
        help="Inclusive year range (start end) to probe.",
    )
    parser.add_argument(
        "--anchors-per-stratum",
        type=int,
        default=1,
        help="How many anchors to take from each (region, stratum). Default 1.",
    )
    parser.add_argument(
        "--bbox-half-m",
        type=float,
        default=DEFAULT_BBOX_HALF_M,
        help="Half-side of probe bbox in metres (default 100m → 200m × 200m square).",
    )
    parser.add_argument(
        "--probe-date-md",
        default=DEFAULT_PROBE_DATE_MONTH_DAY,
        help="Month-day (MM-DD) used for every probe year. Default 06-15.",
    )
    parser.add_argument("--zoom-from", type=int, default=DEFAULT_ZOOM_FROM)
    parser.add_argument("--zoom-to", type=int, default=DEFAULT_ZOOM_TO)
    parser.add_argument(
        "--regions",
        nargs="*",
        help="Optional subset of region keys (default = all 6 Vexcel cities present in sample).",
    )
    return parser.parse_args()


def metric_bbox_to_wgs84(
    *, centroid_x: float, centroid_y: float, half_m: float, crs_metric: str
) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)
    minx_m, miny_m = centroid_x - half_m, centroid_y - half_m
    maxx_m, maxy_m = centroid_x + half_m, centroid_y + half_m
    lon_min, lat_min = transformer.transform(minx_m, miny_m)
    lon_max, lat_max = transformer.transform(maxx_m, maxy_m)
    return lon_min, lat_min, lon_max, lat_max


def select_anchors(sample: pd.DataFrame, n_per_stratum: int) -> pd.DataFrame:
    sample = sample.sort_values(["region_key", "suggested_stratum", "candidate_rank", "gridcell_id"])
    anchors = (
        sample.groupby(["region_key", "suggested_stratum"], group_keys=False)
        .head(n_per_stratum)
        .reset_index(drop=True)
    )
    return anchors


def main() -> None:
    args = parse_args()
    if args.years[0] > args.years[1]:
        sys.exit("--years START must be <= END")

    sample = pd.read_csv(args.sample_csv)
    if args.regions:
        sample = sample.loc[sample["region_key"].isin(set(args.regions))].copy()
    if sample.empty:
        sys.exit("Sample CSV is empty after filtering.")

    anchors = select_anchors(sample, args.anchors_per_stratum)

    args.anchors_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    anchor_rows = []
    task_rows = []
    years = list(range(args.years[0], args.years[1] + 1))

    for _, row in anchors.iterrows():
        region = row["region_key"]
        short = CITY_SHORT.get(region, region[:3].upper())
        anchor_id = f"{short}_{row['suggested_stratum'][:3]}_{row['gridcell_id']}"

        lon_min, lat_min, lon_max, lat_max = metric_bbox_to_wgs84(
            centroid_x=float(row["centroid_x"]),
            centroid_y=float(row["centroid_y"]),
            half_m=args.bbox_half_m,
            crs_metric=str(row["crs_metric"]),
        )

        anchor_rows.append(
            {
                "anchor_id": anchor_id,
                "region_key": region,
                "city": row["city"],
                "stratum": row["suggested_stratum"],
                "source_gridcell_id": row["gridcell_id"],
                "centroid_lon": float(row["lon"]),
                "centroid_lat": float(row["lat"]),
                "crs_metric": row["crs_metric"],
                "bbox_half_m": args.bbox_half_m,
                "lon_min": lon_min,
                "lat_min": lat_min,
                "lon_max": lon_max,
                "lat_max": lat_max,
                "vexcel_collection_id": row.get("collection_id", ""),
                "vexcel_capture_start": row.get("capture_start", ""),
                "vexcel_capture_end": row.get("capture_end", ""),
            }
        )

        for year in years:
            task_name = f"{anchor_id}_{year}"
            save_to = rf"{args.save_root}\{region}\{anchor_id}\{year}"
            task_rows.append(
                {
                    "grid_id": anchor_id,
                    "task_name": task_name,
                    "save_to": save_to,
                    "map_type": "",
                    "date": f"{year}-{args.probe_date_md}",
                    "zoom_from": args.zoom_from,
                    "zoom_to": args.zoom_to,
                    "left_longitude": f"{lon_min:.10f}",
                    "right_longitude": f"{lon_max:.10f}",
                    "top_latitude": f"{lat_max:.10f}",
                    "bottom_latitude": f"{lat_min:.10f}",
                }
            )

    pd.DataFrame(anchor_rows).to_csv(args.anchors_csv, index=False)

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
        writer.writerows(task_rows)

    n_anchors = len(anchor_rows)
    print(
        f"Wrote {len(task_rows)} probe tasks across {n_anchors} anchors × "
        f"{len(years)} years ({years[0]}–{years[-1]})."
    )
    print(f"  anchors: {args.anchors_csv.relative_to(PROJECT_ROOT)}")
    print(f"  tasks:   {args.output.relative_to(PROJECT_ROOT)}")
    print(f"  GEID save root (Windows): {args.save_root}")


if __name__ == "__main__":
    main()
