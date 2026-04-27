#!/usr/bin/env python3
"""Download and package Vexcel eval-sample imagery for RA annotation."""

from __future__ import annotations

import argparse
import csv
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = PROJECT_ROOT / ".env"
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "vexcel_eval_samples" / "vexcel_eval_grids_seed42_per_region10.csv"
DEFAULT_GRID_DIR = PROJECT_ROOT / "data" / "vexcel_task_grids"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tiles" / "vexcel_eval_60_seed42"
DEFAULT_DROPBOX_ROOT = Path("/mnt/c/Users/gaosh/Dropbox/RA_Solar/Vexcel_Eval_60_Seed42_20260426")
DEFAULT_TILE_SIZE_DEG = 0.0048
TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def iter_tile_geometries(geom, tile_size_deg: float):
    minx, miny, maxx, maxy = geom.bounds
    col = 0
    x = minx
    while x < maxx:
        txmax = min(x + tile_size_deg, maxx)
        row = 0
        y_top = maxy
        while y_top > miny:
            tymin = max(y_top - tile_size_deg, miny)
            tile_box = box(x, tymin, txmax, y_top)
            clipped = tile_box.intersection(geom)
            if not clipped.is_empty and clipped.area > 1e-12:
                yield col, row, clipped
            y_top -= tile_size_deg
            row += 1
        x += tile_size_deg
        col += 1


def iter_fixed_grid_geometries(geom, n_cols: int, n_rows: int):
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    if n_cols <= 0 or n_rows <= 0:
        raise ValueError("n_cols and n_rows must be positive")
    for col in range(n_cols):
        txmin = minx + width * col / n_cols
        txmax = minx + width * (col + 1) / n_cols
        for row in range(n_rows):
            tymax = maxy - height * row / n_rows
            tymin = maxy - height * (row + 1) / n_rows
            tile_box = box(txmin, tymin, txmax, tymax)
            clipped = tile_box.intersection(geom)
            if not clipped.is_empty and clipped.area > 1e-12:
                yield col, row, clipped


def tiff_like(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as f:
        return f.read(4) in TIFF_SIGNATURES


def fetch_tiff(
    *,
    session: requests.Session,
    base_url: str,
    token: str,
    collection: str,
    wkt: str,
    out_path: Path,
    layer: str,
    retries: int,
    overwrite: bool,
) -> dict[str, Any]:
    if overwrite and out_path.exists():
        out_path.unlink()
    if tiff_like(out_path):
        return {"status": "skipped", "bytes": out_path.stat().st_size, "error": ""}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".part")
    params = {
        "layer": layer,
        "collection": collection,
        "wkt": wkt,
        "srid": "4326",
        "image-format": "tiff",
        "token": token,
    }

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = session.get(f"{base_url.rstrip('/')}/ortho/extract", params=params, timeout=180)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:240]}"
                raise RuntimeError(last_error)
            if response.content[:4] not in TIFF_SIGNATURES:
                last_error = f"non-TIFF response: {response.text[:240]}"
                raise RuntimeError(last_error)
            temp_path.write_bytes(response.content)
            temp_path.replace(out_path)
            return {"status": "downloaded", "bytes": out_path.stat().st_size, "error": ""}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                time.sleep(min(30, 3 * attempt))

    return {"status": "failed", "bytes": 0, "error": last_error}


def fetch_tile_task(task: dict[str, Any]) -> dict[str, Any]:
    result = fetch_tiff(
        session=requests.Session(),
        base_url=task["base_url"],
        token=task["token"],
        collection=task["collection_id"],
        wkt=task["wkt"],
        out_path=task["out_path"],
        layer=task["layer"],
        retries=task["retries"],
        overwrite=task["overwrite"],
    )
    row = {
        "sample_order": task["sample_order"],
        "region_key": task["region_key"],
        "gridcell_id": task["gridcell_id"],
        "collection_id": task["collection_id"],
        "tile_col": task["tile_col"],
        "tile_row": task["tile_row"],
        "tile_path": str(task["out_path"].relative_to(task["output_root"])),
        "status": result["status"],
        "bytes": result["bytes"],
        "error": result["error"],
        "tile_scheme": task["tile_scheme"],
        "wkt": task["wkt"],
    }
    return row


def load_grid_geometries(sample: pd.DataFrame, grid_dir: Path) -> dict[tuple[str, str], Any]:
    geometries: dict[tuple[str, str], Any] = {}
    for region_key in sorted(sample["region_key"].unique()):
        path = grid_dir / f"{region_key}_task_grid.gpkg"
        gdf = gpd.read_file(path)
        wanted = set(sample.loc[sample["region_key"] == region_key, "gridcell_id"])
        for _, row in gdf.loc[gdf["gridcell_id"].isin(wanted)].iterrows():
            geometries[(region_key, row["gridcell_id"])] = row.geometry
    missing = [
        (row.region_key, row.gridcell_id)
        for row in sample.itertuples(index=False)
        if (row.region_key, row.gridcell_id) not in geometries
    ]
    if missing:
        raise RuntimeError(f"Missing grid geometries: {missing[:10]}")
    return geometries


def write_region_boundaries(sample: pd.DataFrame, grid_dir: Path, output_root: Path) -> None:
    for region_key in sorted(sample["region_key"].unique()):
        path = grid_dir / f"{region_key}_task_grid.gpkg"
        gdf = gpd.read_file(path)
        wanted = set(sample.loc[sample["region_key"] == region_key, "gridcell_id"])
        subset = gdf.loc[gdf["gridcell_id"].isin(wanted)].copy()
        out_dir = output_root / "regions" / region_key
        out_dir.mkdir(parents=True, exist_ok=True)
        subset.to_file(out_dir / f"{region_key}_sample_grids.geojson", driver="GeoJSON")


def download_sample(args: argparse.Namespace) -> pd.DataFrame:
    env = load_env(args.env_file)
    token = env.get("VEXCEL_TOKEN")
    if not token:
        raise RuntimeError(f"VEXCEL_TOKEN not found in {args.env_file}")
    base_url = env.get("VEXCEL_API_BASE", "https://api.vexcelgroup.com/v2")

    sample = pd.read_csv(args.sample_csv)
    if args.regions:
        sample = sample.loc[sample["region_key"].isin(set(args.regions))].copy()
    if args.grid_ids:
        sample = sample.loc[sample["gridcell_id"].isin(set(args.grid_ids))].copy()
    if args.max_grids is not None:
        sample = sample.head(args.max_grids).copy()
    if sample.empty:
        raise RuntimeError("No sample rows selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output_root / "sample_manifest.csv", index=False)
    write_region_boundaries(sample, args.grid_dir, args.output_root)
    geometries = load_grid_geometries(sample, args.grid_dir)

    session = requests.Session()
    manifest_rows: list[dict[str, Any]] = []
    total_downloaded = total_skipped = total_failed = 0

    if args.workers > 1:
        tasks: list[dict[str, Any]] = []
        for i, row in enumerate(sample.itertuples(index=False), start=1):
            geom = geometries[(row.region_key, row.gridcell_id)]
            if args.tile_grid:
                n_cols, n_rows = args.tile_grid
                tiles = list(iter_fixed_grid_geometries(geom, n_cols, n_rows))
                tile_scheme = f"fixed_grid_{n_cols}x{n_rows}"
            else:
                tiles = list(iter_tile_geometries(geom, args.tile_size_deg))
                tile_scheme = f"degree_step_{args.tile_size_deg:g}"
            print(f"[queue {i}/{len(sample)}] {row.region_key}/{row.gridcell_id}: {len(tiles)} tiles", flush=True)
            if args.overwrite_tiles:
                grid_dir = args.output_root / "regions" / row.region_key / row.gridcell_id
                for stale_tile in grid_dir.glob(f"{row.gridcell_id}_*_vexcel.tif"):
                    stale_tile.unlink()
            for col, tile_row, tile_geom in tiles:
                tile_name = f"{row.gridcell_id}_{col:02d}_{tile_row:02d}_vexcel.tif"
                out_path = args.output_root / "regions" / row.region_key / row.gridcell_id / tile_name
                tasks.append(
                    {
                        "sample_order": i,
                        "region_key": row.region_key,
                        "gridcell_id": row.gridcell_id,
                        "collection_id": row.collection_id,
                        "tile_col": col,
                        "tile_row": tile_row,
                        "out_path": out_path,
                        "output_root": args.output_root,
                        "base_url": base_url,
                        "token": token,
                        "wkt": tile_geom.wkt,
                        "layer": args.layer,
                        "retries": args.retries,
                        "overwrite": args.overwrite_tiles,
                        "tile_scheme": tile_scheme,
                    }
                )

        print(f"Downloading {len(tasks)} tiles with workers={args.workers}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_tile_task, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                manifest_row = future.result()
                manifest_rows.append(manifest_row)
                if manifest_row["status"] == "downloaded":
                    total_downloaded += 1
                elif manifest_row["status"] == "skipped":
                    total_skipped += 1
                else:
                    total_failed += 1
                if done % 10 == 0 or total_failed or done == len(tasks):
                    print(
                        f"    progress: {done}/{len(tasks)} "
                        f"downloaded={total_downloaded} skipped={total_skipped} failed={total_failed}",
                        flush=True,
                    )

        manifest = pd.DataFrame(manifest_rows)
        manifest = manifest.sort_values(["sample_order", "tile_col", "tile_row"]).drop(columns="sample_order")
        manifest.to_csv(args.output_root / "tile_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        return manifest

    for i, row in enumerate(sample.itertuples(index=False), start=1):
        geom = geometries[(row.region_key, row.gridcell_id)]
        if args.tile_grid:
            n_cols, n_rows = args.tile_grid
            tiles = list(iter_fixed_grid_geometries(geom, n_cols, n_rows))
            tile_scheme = f"fixed_grid_{n_cols}x{n_rows}"
        else:
            tiles = list(iter_tile_geometries(geom, args.tile_size_deg))
            tile_scheme = f"degree_step_{args.tile_size_deg:g}"
        print(f"[{i}/{len(sample)}] {row.region_key}/{row.gridcell_id}: {len(tiles)} tiles", flush=True)
        if args.overwrite_tiles:
            grid_dir = args.output_root / "regions" / row.region_key / row.gridcell_id
            for stale_tile in grid_dir.glob(f"{row.gridcell_id}_*_vexcel.tif"):
                stale_tile.unlink()
        for col, tile_row, tile_geom in tiles:
            tile_name = f"{row.gridcell_id}_{col:02d}_{tile_row:02d}_vexcel.tif"
            out_path = args.output_root / "regions" / row.region_key / row.gridcell_id / tile_name
            result = fetch_tiff(
                session=session,
                base_url=base_url,
                token=token,
                collection=row.collection_id,
                wkt=tile_geom.wkt,
                out_path=out_path,
                layer=args.layer,
                retries=args.retries,
                overwrite=args.overwrite_tiles,
            )
            if result["status"] == "downloaded":
                total_downloaded += 1
            elif result["status"] == "skipped":
                total_skipped += 1
            else:
                total_failed += 1
            manifest_rows.append(
                {
                    "region_key": row.region_key,
                    "gridcell_id": row.gridcell_id,
                    "collection_id": row.collection_id,
                    "tile_col": col,
                    "tile_row": tile_row,
                    "tile_path": str(out_path.relative_to(args.output_root)),
                    "status": result["status"],
                    "bytes": result["bytes"],
                    "error": result["error"],
                    "tile_scheme": tile_scheme,
                    "wkt": tile_geom.wkt,
                }
            )
        print(
            f"    totals: downloaded={total_downloaded} skipped={total_skipped} failed={total_failed}",
            flush=True,
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_root / "tile_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return manifest


def zip_region(region_dir: Path, zip_path: Path, *, overwrite: bool) -> None:
    if zip_path.exists() and not overwrite:
        print(f"[zip skip] {display_path(zip_path)} exists")
        return
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(region_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(region_dir.parent))


def write_readme(package_root: Path) -> None:
    text = """# Vexcel Eval 60 Seed42

This package contains the 60-grid Vexcel generalization evaluation sample.

Folder layout:
- regions/<region>/<grid_id>/*.tif: georeferenced Vexcel ortho tiles for annotation.
- regions/<region>/<region>_sample_grids.geojson: sampled grid boundaries for that region.
- sample_manifest.csv: one row per sampled grid.
- tile_manifest.csv: one row per downloaded tile.
- zips/<region>.zip: same region folder packaged for RA download.

Annotation target: reviewed prediction footprints / visible rooftop solar footprints,
following the project annotation spec and V1.3 semantics.
"""
    (package_root / "README.md").write_text(text, encoding="utf-8")


def package_outputs(args: argparse.Namespace) -> None:
    args.dropbox_root.mkdir(parents=True, exist_ok=True)
    write_readme(args.output_root)
    for name in ("README.md", "sample_manifest.csv", "tile_manifest.csv"):
        src = args.output_root / name
        if src.exists():
            (args.dropbox_root / name).write_bytes(src.read_bytes())

    # Copy lightweight region grid GeoJSONs and build one zip per region in Dropbox.
    zips_dir = args.dropbox_root / "zips"
    for region_dir in sorted((args.output_root / "regions").iterdir()):
        if not region_dir.is_dir():
            continue
        target_region_dir = args.dropbox_root / "regions" / region_dir.name
        target_region_dir.mkdir(parents=True, exist_ok=True)
        for geojson in region_dir.glob("*_sample_grids.geojson"):
            (target_region_dir / geojson.name).write_bytes(geojson.read_bytes())
        zip_region(region_dir, zips_dir / f"{region_dir.name}.zip", overwrite=args.overwrite_zips)
        print(f"[zip] {region_dir.name}: {display_path(zips_dir / f'{region_dir.name}.zip')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dropbox-root", type=Path, default=DEFAULT_DROPBOX_ROOT)
    parser.add_argument("--tile-size-deg", type=float, default=DEFAULT_TILE_SIZE_DEG)
    parser.add_argument(
        "--tile-grid",
        nargs=2,
        type=int,
        metavar=("COLS", "ROWS"),
        help="Split each task-grid geometry into a fixed COLS x ROWS layout instead of using --tile-size-deg.",
    )
    parser.add_argument("--layer", default="urban")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--regions", nargs="*")
    parser.add_argument("--grid-ids", nargs="*")
    parser.add_argument("--max-grids", type=int)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--overwrite-tiles", action="store_true")
    parser.add_argument("--overwrite-zips", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent tile downloads (default: 1)")
    args = parser.parse_args()

    if not args.package_only:
        manifest = download_sample(args)
        failed = int((manifest["status"] == "failed").sum())
        if failed:
            raise RuntimeError(f"{failed} tile downloads failed; see {args.output_root / 'tile_manifest.csv'}")
    if not args.download_only:
        package_outputs(args)

    print(f"output:  {display_path(args.output_root)}")
    print(f"dropbox: {args.dropbox_root}")


if __name__ == "__main__":
    main()
