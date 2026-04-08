#!/usr/bin/env python3
"""Stitch GEID-python downloader output into a single EPSG:4326 GeoTIFF.

Layout (per grid):
    <root>/<GRID>/manifest.json
    <root>/<GRID>/<GRID>/<zoom>/<x>/ges_<x>_<y>_<zoom>.jpg

Differs from scripts/imagery/stitch_geid.py (Windows GEID UI output) which
relies on a `<task>_list1.txt` bounds file that the python downloader does
not produce. Bounds are recomputed from the standard Google XYZ formula.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

TILE_SIZE = 256
FILENAME_RE = re.compile(r"^ges_(\d+)_(\d+)_(\d+)\.jpg$", re.IGNORECASE)


def stitch_grid(grid_dir: Path, output: Path, force: bool) -> bool:
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[skip] no manifest: {grid_dir}")
        return False
    manifest = json.loads(manifest_path.read_text())
    grid_id = manifest["grid_id"]
    zoom = int(manifest["zoom"])

    if output.exists() and not force:
        print(f"[skip] exists: {output}")
        return True

    tile_root = grid_dir / grid_id / str(zoom)
    if not tile_root.exists():
        print(f"[skip] no tiles dir: {tile_root}")
        return False

    tiles: list[tuple[int, int, Path]] = []
    for x_dir in tile_root.iterdir():
        if not x_dir.is_dir():
            continue
        for jpg in x_dir.glob("ges_*.jpg"):
            m = FILENAME_RE.match(jpg.name)
            if not m:
                continue
            tiles.append((int(m.group(1)), int(m.group(2)), jpg))

    if not tiles:
        print(f"[skip] no jpgs in {tile_root}")
        return False

    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    min_x, max_x = xs[0], xs[-1]
    min_y, max_y = ys[0], ys[-1]
    cols = max_x - min_x + 1
    rows = max_y - min_y + 1

    expected = cols * rows
    if len(tiles) != expected:
        print(f"[warn] {grid_id}: have {len(tiles)} tiles, expected full grid {expected} ({cols}x{rows}) — missing tiles will be black")

    # GEID tile addressing is not standard slippy XYZ — derive bounds from
    # the manifest bbox and assume the tile_range covers it uniformly.
    bbox = manifest["bbox_lon_lat"]  # [west, south, east, north]
    tr = manifest["tile_range"]
    full_cols = tr["x"][1] - tr["x"][0] + 1
    full_rows = tr["y"][1] - tr["y"][0] + 1
    if (full_cols, full_rows) != (cols, rows):
        print(f"[warn] {grid_id}: on-disk extent {cols}x{rows} != manifest {full_cols}x{full_rows}; using manifest")
        cols, rows = full_cols, full_rows
        min_x, min_y = tr["x"][0], tr["y"][0]
        max_y = tr["y"][1]
    width = cols * TILE_SIZE
    height = rows * TILE_SIZE
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "JPEG",
        "photometric": "YCBCR",
        "interleave": "pixel",
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(output, "w", **profile) as dst:
        for x, y, jpg in tiles:
            col_off = (x - min_x) * TILE_SIZE
            row_off = (max_y - y) * TILE_SIZE  # GEID y grows northward
            try:
                arr = np.asarray(Image.open(jpg).convert("RGB"))
            except Exception as e:
                print(f"[warn] failed to read {jpg}: {e}")
                continue
            if arr.shape[:2] != (TILE_SIZE, TILE_SIZE):
                arr = np.asarray(Image.fromarray(arr).resize((TILE_SIZE, TILE_SIZE)))
            # rasterio expects (bands, h, w)
            data = np.transpose(arr, (2, 0, 1))
            window = rasterio.windows.Window(col_off, row_off, TILE_SIZE, TILE_SIZE)
            dst.write(data, window=window)

    print(f"[ok] wrote {output} ({width}x{height}, {len(tiles)} tiles)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="GEID-python root, e.g. /mnt/d/ZAsolar/geid_raw/joburg_geid_python")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output dir for <GRID>_mosaic.tif")
    ap.add_argument("--grids", nargs="*", help="Optional subset of grid IDs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    grid_dirs = sorted(p for p in args.root.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    if args.grids:
        wanted = set(args.grids)
        grid_dirs = [p for p in grid_dirs if p.name in wanted]

    print(f"Stitching {len(grid_dirs)} grids -> {args.output_dir}")
    n_ok = 0
    for gd in grid_dirs:
        out = args.output_dir / f"{gd.name}_mosaic.tif"
        if stitch_grid(gd, out, args.force):
            n_ok += 1
    print(f"Done: {n_ok}/{len(grid_dirs)} stitched")


if __name__ == "__main__":
    main()
