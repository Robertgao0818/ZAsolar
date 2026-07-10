#!/usr/bin/env python3
"""B1 TTA falsification pilot — step 2: pre-resample tiles to 1.5x / 2.0x.

Only out-of-envelope upscale views are sanctioned for the pilot (train-time
scale jitter covers 0.8-1.2x; flips/rot90 are trained equivariances — see
docs/plans/2026-06-10-rcnn-f1-gap-review.md B1). This script upsamples each
chunked GeoTIFF by an integer-free scale factor, adjusting the geotransform
so detect_direct.py windows carry correct geo coordinates downstream.

Layout mirrors the source so SOLAR_TILES_ROOT fast-path resolution works:
  <out-root>/<region>/<imagery_layer>/<grid>/<chunk>.tif

Window-overlap note (pre-registered): detect_direct slides 400 px windows.
At 2.0x a window covers 200 native px (13.4 m); the pilot targets sub-array
misses (median sqrt-area ~70-80 native px). Run detect_direct with
--overlap 0.5 on magnified views so the physical stride (100/133 native px
at 2.0x/1.5x) keeps any target up to ~half a window fully contained in at
least one window. Chunk borders remain partial-window territory, same as
at 1.0x (chunks are >= 4k px; border band is negligible and noted in the
report).

CPU-only; run pod-side next to the tiles.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from affine import Affine


def resample_tif(src_path: Path, dst_path: Path, factor: float) -> tuple[int, int]:
    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        nh, nw = int(round(h * factor)), int(round(w * factor))
        arr = src.read(
            indexes=[1, 2, 3],
            out_shape=(3, nh, nw),
            resampling=Resampling.bilinear,
        )
        transform = src.transform * Affine.scale(w / nw, h / nh)
        profile = {
            "driver": "GTiff",
            "height": nh,
            "width": nw,
            "count": 3,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": transform,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "JPEG",
            "jpeg_quality": 95,
            "photometric": "YCBCR",
        }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8))
    return nh, nw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-dir", type=Path, required=True,
                    help="source layer dir, e.g. /workspace/tiles/johannesburg/vexcel_2024")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="output SOLAR_TILES_ROOT, e.g. /dev/shm/tta_tiles_x15")
    ap.add_argument("--region", default="johannesburg")
    ap.add_argument("--imagery-layer", default="vexcel_2024")
    ap.add_argument("--factor", type=float, required=True, choices=[1.5, 2.0])
    ap.add_argument("--grids", nargs="+", required=True)
    args = ap.parse_args()

    out_layer = args.out_root / args.region / args.imagery_layer
    for grid in args.grids:
        grid_dir = args.tiles_dir / grid
        tifs = sorted(grid_dir.glob(f"{grid}_*.tif"))
        if not tifs:
            raise SystemExit(f"no tiles for {grid} under {grid_dir}")
        t0 = time.time()
        for tif in tifs:
            nh, nw = resample_tif(tif, out_layer / grid / tif.name, args.factor)
        print(f"[{grid}] {len(tifs)} chunks resampled x{args.factor} "
              f"(last {nh}x{nw}) in {time.time()-t0:.0f}s")
    print(f"[done] SOLAR_TILES_ROOT={args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
