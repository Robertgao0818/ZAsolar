#!/usr/bin/env python3
"""Sample Vexcel task grids for generalization evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID_DIR = PROJECT_ROOT / "data" / "vexcel_task_grids"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "vexcel_eval_samples"
REQUIRED_COLUMNS = {"gridcell_id", "region_key", "coverage_fraction", "geometry"}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_grids(grid_dir: Path, regions: set[str] | None) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for path in sorted(grid_dir.glob("*_task_grid.gpkg")):
        region_key = path.name.removesuffix("_task_grid.gpkg")
        if regions is not None and region_key not in regions:
            continue
        gdf = gpd.read_file(path)
        missing = REQUIRED_COLUMNS.difference(gdf.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frames.append(gdf)

    if not frames:
        raise FileNotFoundError(f"No task grid GPKGs found in {grid_dir}")

    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def allocate_total(df: pd.DataFrame, total: int) -> dict[str, int]:
    counts = df.groupby("region_key").size().sort_index()
    raw = counts / counts.sum() * total
    base = raw.astype(int)
    remainder = total - int(base.sum())
    order = (raw - base).sort_values(ascending=False).index
    allocation = base.to_dict()
    for key in order[:remainder]:
        allocation[key] += 1
    return {key: int(value) for key, value in allocation.items()}


def sample_grids(
    grids: gpd.GeoDataFrame,
    *,
    seed: int,
    per_region: int | None,
    total: int | None,
    min_coverage_fraction: float,
) -> gpd.GeoDataFrame:
    eligible = grids.loc[grids["coverage_fraction"].astype(float) >= min_coverage_fraction].copy()
    if eligible.empty:
        raise RuntimeError("No eligible grids after coverage-fraction filtering")

    if total is not None:
        allocation = allocate_total(eligible, total)
    else:
        if per_region is None:
            raise ValueError("Either --per-region or --total is required")
        allocation = {
            region_key: min(per_region, len(part))
            for region_key, part in eligible.groupby("region_key")
        }

    samples: list[gpd.GeoDataFrame] = []
    for region_key, n in sorted(allocation.items()):
        part = eligible.loc[eligible["region_key"] == region_key]
        if n <= 0:
            continue
        if n > len(part):
            raise ValueError(f"Requested {n} grids for {region_key}, only {len(part)} eligible")
        samples.append(part.sample(n=n, random_state=seed))

    if not samples:
        raise RuntimeError("Sampling produced no rows")

    sample = gpd.GeoDataFrame(pd.concat(samples, ignore_index=True), crs=eligible.crs)
    sample = sample.sort_values(["region_key", "gridcell_id"]).reset_index(drop=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sample.insert(0, "sample_id", [f"VXEV{i:04d}" for i in range(1, len(sample) + 1)])
    sample["sample_seed"] = seed
    sample["selected_at_utc"] = now
    sample["grid_type"] = ""
    sample["human_annotation_status"] = "pending"
    sample["machine_detection_status"] = "pending"
    sample["reviewed_prediction_status"] = "pending"
    return sample


def output_stem(seed: int, per_region: int | None, total: int | None) -> str:
    if total is not None:
        return f"vexcel_eval_grids_seed{seed}_total{total}"
    return f"vexcel_eval_grids_seed{seed}_per_region{per_region}"


def write_sample(sample: gpd.GeoDataFrame, output_dir: Path, stem: str, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    gpkg_path = output_dir / f"{stem}.gpkg"
    geojson_path = output_dir / f"{stem}.geojson"

    for path in (csv_path, gpkg_path, geojson_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        if path.exists():
            path.unlink()

    sample.drop(columns="geometry").to_csv(csv_path, index=False)
    sample.to_file(gpkg_path, driver="GPKG")
    sample.to_file(geojson_path, driver="GeoJSON")
    print(f"wrote {len(sample)} sampled grids")
    print(f"csv:  {display_path(csv_path)}")
    print(f"gpkg: {display_path(gpkg_path)}")
    print(f"geojson: {display_path(geojson_path)}")
    print(sample.groupby("region_key").size().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regions", nargs="*", help="Optional subset of region keys")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-region", type=int, default=10)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument(
        "--min-coverage-fraction",
        type=float,
        default=0.75,
        help="Default keeps most sampled grids close to a full 1 km2 cell.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.total is not None and args.per_region != 10:
        raise ValueError("--total and an explicit --per-region should not be combined")

    regions = set(args.regions) if args.regions else None
    grids = load_grids(args.grid_dir, regions)
    sample = sample_grids(
        grids,
        seed=args.seed,
        per_region=None if args.total is not None else args.per_region,
        total=args.total,
        min_coverage_fraction=args.min_coverage_fraction,
    )
    write_sample(
        sample,
        args.output_dir,
        output_stem(args.seed, None if args.total is not None else args.per_region, args.total),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
