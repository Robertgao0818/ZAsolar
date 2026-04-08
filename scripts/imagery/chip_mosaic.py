#!/usr/bin/env python3
"""Chip a single GeoTIFF mosaic into the per-grid chip layout expected by
detect_and_evaluate.py: <output_root>/<GRID>/<GRID>_<row>_<col>_geo.tif

Designed for Joburg GEID mosaics (~7936x6912 EPSG:4326). Each chip is a
fixed-size window written as a separate GeoTIFF preserving CRS/transform.
Edge chips are clipped (smaller than CHIP_SIZE) — matches the Capetown
convention where the bottom/right row/col may be partial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from rasterio.windows import Window

CHIP_SIZE = 2000


def chip_one(mosaic: Path, out_root: Path, grid_id: str, force: bool) -> int:
    out_dir = out_root / grid_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with rasterio.open(mosaic) as src:
        rows = (src.height + CHIP_SIZE - 1) // CHIP_SIZE
        cols = (src.width + CHIP_SIZE - 1) // CHIP_SIZE
        for r in range(rows):
            for c in range(cols):
                out_path = out_dir / f"{grid_id}_{r}_{c}_geo.tif"
                if out_path.exists() and not force:
                    n += 1
                    continue
                w = min(CHIP_SIZE, src.width - c * CHIP_SIZE)
                h = min(CHIP_SIZE, src.height - r * CHIP_SIZE)
                window = Window(c * CHIP_SIZE, r * CHIP_SIZE, w, h)
                transform = src.window_transform(window)
                profile = src.profile.copy()
                profile.update(
                    width=w,
                    height=h,
                    transform=transform,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    compress="JPEG",
                    photometric="YCBCR",
                    interleave="pixel",
                )
                profile.pop("BIGTIFF", None)
                data = src.read(window=window)
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path, help="Dir containing <GRID>_mosaic.tif files")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Output tiles root, e.g. /workspace/tiles_joburg")
    ap.add_argument("--grids", nargs="*", help="Optional subset of grid IDs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    mosaics = sorted(args.input_dir.glob("*_mosaic.tif"))
    if args.grids:
        wanted = set(args.grids)
        mosaics = [m for m in mosaics if m.stem.replace("_mosaic", "") in wanted]

    print(f"Chipping {len(mosaics)} mosaics -> {args.output_root}")
    total = 0
    for m in mosaics:
        gid = m.stem.replace("_mosaic", "")
        n = chip_one(m, args.output_root, gid, args.force)
        total += n
        print(f"  {gid}: {n} chips")
    print(f"Done: {len(mosaics)} grids, {total} chips total")


if __name__ == "__main__":
    main()
