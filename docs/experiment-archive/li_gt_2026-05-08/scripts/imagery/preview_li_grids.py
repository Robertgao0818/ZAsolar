"""
Preview Li grids to check which have valid imagery.

Reads grid bounds from cape_town_grid_Li.gpkg and fetches WMS previews.
Reuses all logic from grid_preview_batch.py.

Usage:
  python scripts/imagery/preview_li_grids.py --start-after G1846 --count 100
  python scripts/imagery/preview_li_grids.py --grid-ids G1875 G1876 G1877
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.imagery.grid_preview_batch import (
    PreviewJob,
    build_parser,
    process_job,
    annotate_thumbnail,
    write_contact_sheet,
    DEFAULT_PREVIEW_SIZE,
    DEFAULT_WHITE_THRESHOLD,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

LI_GRID_GPKG = Path("data/cape_town_grid_Li.gpkg")


def load_li_jobs(
    grid_ids: list[str] | None = None,
    start_after: str | None = None,
    count: int = 100,
) -> list[PreviewJob]:
    gdf = gpd.read_file(LI_GRID_GPKG)
    # Filter numeric G-grids and sort
    mask = gdf["Name"].str.match(r"^G\d+$")
    gdf = gdf[mask].copy()
    gdf["_num"] = gdf["Name"].str[1:].astype(int)
    gdf = gdf.sort_values("_num").reset_index(drop=True)

    if grid_ids:
        wanted = {g.upper() for g in grid_ids}
        gdf = gdf[gdf["Name"].isin(wanted)]
    elif start_after:
        start_num = int(start_after.upper()[1:])
        gdf = gdf[gdf["_num"] > start_num].head(count)

    jobs = []
    for _, row in gdf.iterrows():
        xmin, ymin, xmax, ymax = row.geometry.bounds
        jobs.append(PreviewJob(grid_id=row["Name"], bbox=(xmin, ymin, xmax, ymax)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Preview Li grids for imagery screening")
    parser.add_argument("--grid-ids", nargs="+", help="Explicit grid IDs")
    parser.add_argument("--start-after", default="G1846", help="Start after this grid ID")
    parser.add_argument("--count", type=int, default=100, help="Number of grids to preview")
    parser.add_argument("--preview-size", type=int, default=DEFAULT_PREVIEW_SIZE)
    parser.add_argument("--thumb-size", type=int, default=220)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--white-threshold", type=int, default=DEFAULT_WHITE_THRESHOLD)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output-dir", default="results/grid_previews/li_batch_001")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    jobs = load_li_jobs(
        grid_ids=args.grid_ids,
        start_after=args.start_after,
        count=args.count,
    )
    if not jobs:
        print("No grids selected")
        return

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LI PREVIEW] {len(jobs)} grids: {jobs[0].grid_id} -> {jobs[-1].grid_id}")

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                process_job, job, preview_dir,
                args.preview_size, args.preview_size,
                args.white_threshold, args.timeout,
                args.retries, args.refresh,
            ): job
            for job in jobs
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.append(result)
            status = result["status"]
            ratio = result["valid_imagery_ratio"]
            hint = result["imagery_hint"]
            print(f"  [{idx}/{len(jobs)}] {result['grid_id']}: {status} "
                  f"valid={ratio:.1%} ({hint})")

    # Add thumbnails
    for row in rows:
        if row.get("image"):
            row["thumb"] = annotate_thumbnail(
                row["image"], row["grid_id"],
                row["valid_imagery_ratio"], args.thumb_size,
            )

    # Save metrics CSV
    metrics_rows = [{
        "grid_id": r["grid_id"],
        "status": r["status"],
        "valid_imagery_ratio": r["valid_imagery_ratio"],
        "white_ratio": r["white_ratio"],
        "mean_brightness": r["mean_brightness"],
        "imagery_hint": r["imagery_hint"],
        "error": r.get("error", ""),
    } for r in rows]
    df = pd.DataFrame(metrics_rows).sort_values("grid_id")
    csv_path = output_dir / "grid_preview_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] {csv_path}")

    # Contact sheets
    by_grid = sorted(rows, key=lambda r: r["grid_id"])
    write_contact_sheet(by_grid, output_dir / "contact_sheet_by_grid.jpg",
                        args.thumb_size, args.columns)

    by_ratio = sorted(rows, key=lambda r: -r["valid_imagery_ratio"])
    write_contact_sheet(by_ratio, output_dir / "contact_sheet_by_valid_ratio.jpg",
                        args.thumb_size, args.columns)

    # Summary
    substantial = sum(1 for r in rows if r["imagery_hint"] == "substantial")
    partial = sum(1 for r in rows if r["imagery_hint"] == "partial")
    mostly_blank = sum(1 for r in rows if r["imagery_hint"] == "mostly_blank")
    likely_blank = sum(1 for r in rows if r["imagery_hint"] == "likely_blank")
    errors = sum(1 for r in rows if r["status"] == "error")

    print(f"\n[SUMMARY] Total: {len(rows)}")
    print(f"  substantial (>=60%): {substantial}")
    print(f"  partial (25-60%):    {partial}")
    print(f"  mostly_blank (5-25%): {mostly_blank}")
    print(f"  likely_blank (<5%):  {likely_blank}")
    print(f"  errors:              {errors}")


if __name__ == "__main__":
    main()
